"""
causal_lm.py
=============

GPTForCausalLM -- wraps the raw ``GPTModel`` with a HuggingFace-style
forward interface: accepts ``input_ids`` and an optional ``labels``
tensor, computing loss internally when labels are provided.

Overview
--------
HuggingFace's ``*ForCausalLM`` classes (``GPT2LMHeadModel``,
``LlamaForCausalLM``, ...) all share one contract: pass in token ids,
optionally pass in labels, get back an object exposing ``.logits`` and
``.loss``. ``GPTForCausalLM`` reproduces that contract on top of a plain
``GPTModel`` (imported unchanged from ``gpt_model.py``), which buys two
things:

1. **Compatible checkpoints.** Tooling that expects the HF save/load
   shape (a ``config.json`` + a weights file) can load a model trained
   with this class with no translation layer.
2. **A simpler training loop.** ``outputs = model(x, labels=y)`` then
   ``outputs.loss.backward()`` -- one call instead of computing logits
   and loss separately, with the loss logic living in exactly one place.

Quick reference
----------------
    Want to...                          | Do this
    -------------------------------------|----------------------------------
    Start from scratch (random weights)  | ``GPTForCausalLM(config)``
    Resume/reuse a trained model         | ``GPTForCausalLM.from_pretrained(dir)``
    Persist a trained model              | ``model.save_pretrained(dir)``

``__init__`` and ``from_pretrained`` both return a ``GPTForCausalLM``;
the only difference is whether the weights are freshly initialized or
loaded from disk. The forward / generate / save API is identical either
way.

Example
-------
>>> from config import GPTConfig
>>> config = GPTConfig.tiny_debug()
>>> model = GPTForCausalLM(config)
>>>
>>> # inference -- no labels, loss is None
>>> input_ids = torch.randint(0, config.vocab_size, (1, 8))
>>> output = model(input_ids)
>>> output.logits.shape
torch.Size([1, 8, 100])
>>> output.loss is None
True
>>>
>>> # training -- labels given, loss computed
>>> labels = torch.randint(0, config.vocab_size, (1, 8))
>>> output = model(input_ids, labels=labels)
>>> output.loss.item() > 0
True
>>>
>>> # generation
>>> model.eval()
>>> generated = model.generate(input_ids, max_new_tokens=20)
>>>
>>> # checkpointing
>>> model.save_pretrained("checkpoints/run1")
>>> restored = GPTForCausalLM.from_pretrained("checkpoints/run1", map_location="cpu")
"""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .gpt_model import GPTModel  # the model this class wraps, not reimplements


@dataclass
class CausalLMOutput:
    """
    Return type for ``GPTForCausalLM.forward()``.

    A plain dataclass rather than a tuple, so callers read
    ``outputs.loss`` / ``outputs.logits`` by name instead of by
    position -- matching HuggingFace's output-object convention.

    Attributes
    ----------
    logits : torch.Tensor
        Shape ``(batch, seq_len, vocab_size)``. Always present.
    loss : torch.Tensor or None
        Scalar cross-entropy loss, computed only when ``labels`` was
        passed to ``forward()``. ``None`` if no labels were given (e.g.
        during pure inference/generation, where there's no "correct
        answer" to compute loss against).
    """
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class GPTForCausalLM(nn.Module):
    """
    Wraps ``GPTModel`` with a HuggingFace-style forward interface.

    This is a thin ``nn.Module`` around ``GPTModel``: it delegates the
    forward pass to ``self.model`` and adds loss computation,
    autoregressive generation, and checkpointing on top. It does not
    reimplement the transformer itself.

    Parameters
    ----------
    config : GPTConfig
        The validated config object describing the model architecture
        (vocab size, context length, number of layers, etc.).
    
    Examples
    --------
    >>> config = GPTConfig.tiny_debug()
    >>> model = GPTForCausalLM(config)
    >>>
    >>> # inference -- no labels, loss is None
    >>> input_ids = torch.randint(0, config.vocab_size, (1, 8))
    >>> output = model(input_ids)
    >>> output.logits.shape
    torch.Size([1, 8, 100])
    >>> output.loss is None
    True
    >>>
    >>> # training -- labels given, loss computed
    >>> labels = torch.randint(0, config.vocab_size, (1, 8))
    >>> output = model(input_ids, labels=labels)
    >>> output.loss.item() > 0
    True
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        # GPTModel expects a plain dict, not a GPTConfig object, so
        # convert here rather than touching GPTModel itself.
        self.model = GPTModel(dataclasses.asdict(config))

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> CausalLMOutput:
        """
        Run the model forward, optionally computing loss against labels.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape ``(batch, seq_len)``. Token ids.
        labels : torch.Tensor or None
            Shape ``(batch, seq_len)``, same shape as ``input_ids``. If
            given, cross-entropy loss is computed between the model's
            predictions and these labels. If ``None``, no loss is
            computed (pure inference mode).

        Returns
        -------
        CausalLMOutput
            ``.logits`` always populated. ``.loss`` populated only if
            ``labels`` was provided, else ``None``.
        """
        logits = self.model(input_ids)
        if labels is not None:
            # F.cross_entropy is the *functional* form -- it computes
            # the loss directly from (logits, labels). nn.CrossEntropyLoss
            # is a *module class* -- calling it like a function would
            # pass the tensors as constructor args instead of running
            # anything, so F.cross_entropy is the correct choice here.
            loss = F.cross_entropy(
                logits.flatten(0, 1), labels.flatten()
            )
        else:
            loss = None
        return CausalLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int, context_size: int | None = None
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation.

        Decorated with ``@torch.no_grad()`` since generation should
        never build a graph or accumulate gradients.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape ``(batch, seq_len)``. Starting sequence(s).
        max_new_tokens : int
            Number of tokens to generate.
        context_size : int or None
            Max context window fed to the model at each step. Defaults
            to ``self.config.context_length`` if not given.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len + max_new_tokens)``.

        Notes
        -----
        Decoding is greedy (always picks the single most likely next
        token via ``argmax``, no sampling/temperature), so output is
        fully deterministic for a given model and prompt. Swap in
        sampling here later if non-deterministic generation is needed.
        """
        context_size = context_size or self.config.context_length

        for _ in range(max_new_tokens):
            # Crop to the last context_size tokens -- the model can't
            # see further back than its context window.
            idx_cond = input_ids[:, -context_size:]
            output = self.forward(idx_cond)
            # Only the last position's logits predict the *next* token.
            logits = output.logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, idx_next), dim=1)

        return input_ids

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model's config and weights to ``save_directory``, so
        it can later be restored with ``from_pretrained``.

        Writes two files:
            ``save_directory/config.json`` -- architecture config
            ``save_directory/model.pt``    -- state dict (weights)

        Parameters
        ----------
        save_directory : str
            Directory to write to. Created (including parents) if it
            doesn't already exist.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "config.json", "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2)

        torch.save(self.state_dict(), save_dir / "model.pt")

    @classmethod
    def from_pretrained(cls, load_directory: str, map_location: str | None = None) -> "GPTForCausalLM":
        """
        Reconstruct a GPTForCausalLM from a directory previously
        written by ``save_pretrained`` -- rebuilding the exact
        architecture from the saved config before loading weights into
        it, rather than requiring the caller to separately track and
        pass a matching config by hand.

        Parameters
        ----------
        load_directory : str
            Directory containing ``config.json`` and ``model.pt``, as
            written by ``save_pretrained``.
        map_location : str or None
            Passed through to ``torch.load`` -- e.g. ``"cpu"`` to force
            loading onto CPU even if the checkpoint was saved from a
            GPU tensor.

        Returns
        -------
        GPTForCausalLM
            A model with trained weights loaded, in ``eval()`` mode.
            Call ``.train()`` afterward if resuming training rather
            than running inference.
        """
        load_dir = Path(load_directory)

        with open(load_dir / "config.json") as f:
            config = GPTConfig(**json.load(f))

        model = cls(config)
        state_dict = torch.load(load_dir / "model.pt", map_location=map_location)
        model.load_state_dict(state_dict)
        model.eval()

        return model