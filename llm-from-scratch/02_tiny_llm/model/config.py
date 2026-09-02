"""
config.py
==========

GPTConfig -- a typed, validated configuration object for the tiny GPT model.

WHY THIS EXISTS (read before implementing)
--------------------------------------------
Up to now, every model in this repo took a plain dict: `cfg["emb_dim"]`.
That works, but has two real problems:

    1. No validation -- passing `emb_dim=100, n_heads=3` silently builds a
       broken model (100 isn't divisible by 3), and you only find out
       when `MultiHeadAttention.__init__` raises an unrelated-looking
       AssertionError three files away.
    2. No IDE support -- `cfg["emb_dim"]` gives you no autocomplete and a
       typo like `cfg["emd_dim"]` fails at runtime, possibly deep inside
       a training run, rather than immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass
class GPTConfig:
    """
    Configuration for the tiny GPT model.

    Attributes
    ----------
    vocab_size : int
        Number of unique tokens the model can represent. Must be a
        positive integer -- this becomes the size of both the token
        embedding table and the final output layer.
    context_length : int
        Maximum sequence length the model can process in one forward
        pass. This bounds the size of the positional embedding table and
        the causal attention mask. Must be a positive integer.
    emb_dim : int
        The model's hidden dimension (d_model) -- the size of the vector
        representing each token throughout the network. Must be a
        positive integer, AND must be evenly divisible by `n_heads` (see
        why in MultiHeadAttention -- head_dim = emb_dim // n_heads must
        be a whole number, or splitting into heads is impossible).
    n_heads : int
        Number of attention heads in each MultiHeadAttention block. Must
        be a positive integer, and emb_dim % n_heads must equal 0.
    n_layers : int
        Number of TransformerBlocks stacked in the model. Must be a
        positive integer.
    drop_rate : float
        Dropout probability applied throughout the model (embeddings,
        attention weights, shortcut connections). Must be in [0.0, 1.0)
        -- note: exclusive of 1.0, since a dropout rate of exactly 1.0
        would zero out 100% of activations, which is never useful.
    qkv_bias : bool
        Whether the Q/K/V linear projections in attention include a bias
        term. Defaults to False (matches modern practice; GPT-2 itself
        used True -- relevant if you later load pretrained GPT-2 weights,
        see 02_tiny_llm/load_pretrained_weights.py, which will need
        qkv_bias=True to match).

    HINT: raise ValueError (not assert) for these checks -- assertions
    can be stripped out by Python's -O optimization flag, but config
    validation should never be silently skippable.
    """

    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float = 0.1
    qkv_bias: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")

        if self.emb_dim <= 0:
            raise ValueError(f"emb_dim must be positive, got {self.emb_dim}")

        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")

        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")

        if self.emb_dim % self.n_heads != 0:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
            )

        if self.drop_rate < 0.0 or self.drop_rate >= 1.0:
            raise ValueError(f"drop_rate must be in [0.0, 1.0), got {self.drop_rate}")


    @property
    def head_dim(self) -> int:
        """
        Dimension of each individual attention head: emb_dim // n_heads.

        Exposed as a computed property (not a stored field) deliberately
        -- it's fully determined by emb_dim and n_heads, so storing it
        separately would create a second source of truth that could
        drift out of sync if emb_dim or n_heads changed after
        construction.
        """
        return self.emb_dim // self.n_heads

    @classmethod
    def gpt2_small(cls) -> "GPTConfig":
        """
        Convenience constructor matching GPT-2 small's published
        architecture (124M parameters): vocab_size=50257,
        context_length=1024, emb_dim=768, n_heads=12, n_layers=12,
        drop_rate=0.1, qkv_bias=True.

        Useful both as a sanity-check config (does your model actually
        build correctly at real GPT-2 scale?) and later when loading
        real pretrained GPT-2 weights, where every dimension must match
        exactly.
        """
        return cls(
            vocab_size = 50257,
            context_length=1024,
            emb_dim=768,
            n_heads=12,
            n_layers=12,
            drop_rate=0.1,
            qkv_bias=True,
        )

    @classmethod
    def tiny_debug(cls) -> "GPTConfig":
        """
        A deliberately tiny config for fast local testing -- small enough
        that a forward pass, or even a few training steps, runs in
        well under a second on CPU.

        Suggested values: vocab_size=100, context_length=16, emb_dim=32,
        n_heads=4, n_layers=2, drop_rate=0.0 (0.0 specifically, so tests
        checking exact output values aren't affected by dropout's
        randomness).
        """
        return cls(
            vocab_size=100,
            context_length=16,
            emb_dim=32,
            n_heads=4,
            n_layers=2,
            drop_rate=0.0,
        )
