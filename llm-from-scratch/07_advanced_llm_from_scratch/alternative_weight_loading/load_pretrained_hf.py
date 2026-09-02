"""
load_pretrained_weights.py
============================

Loads OpenAI GPT-2 (small, 124M) pretrained weights from HuggingFace's
`transformers` library into this repo's custom `GPTModel` / `GPTForCausalLM`.

WHY THIS ISN'T A PLAIN `load_state_dict()`
--------------------------------------------
1. Different parameter names. HF uses `wte`, `h.0.attn.c_attn.weight`,
   etc; this repo uses `tok_emb`, `trf_blocks[0].attn.W_query`, etc.
2. HF's Linear-like layers (`Conv1D`) store weights as
   (in_features, out_features) -- the TRANSPOSE of PyTorch's
   `nn.Linear`, which expects (out_features, in_features). Every
   copied weight matrix needs `.T`.
3. GPT-2 fuses Q, K, V into one `c_attn` weight/bias of width
   `3 * emb_dim`. This repo keeps them as three separate `nn.Linear`
   layers (`W_query`, `W_key`, `W_value`), so `c_attn` has to be split
   into three equal chunks along the output dimension before copying.
4. GPT-2 ties the output head to the token embedding (`wte`). This
   repo's `out_head` is a separate `nn.Linear`, so we copy (not share)
   the embedding weights into it to match GPT-2's behavior numerically.

Usage
-----
    from transformers import GPT2LMHeadModel
    from model.config import GPTConfig
    from model.gpt_model import GPTModel
    from model.load_pretrained_weights import load_weights_into_gpt

    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")  # "gpt2" == small, 124M
    config = GPTConfig.gpt2_small()
    model = GPTModel(dataclasses.asdict(config))
    load_weights_into_gpt(model, hf_model)

    # or, if you want the GPTForCausalLM wrapper:
    from model.causal_lm import GPTForCausalLM
    clm = GPTForCausalLM(config)
    load_weights_into_gpt(clm.model, hf_model)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from model.gpt_model import GPTModel


def assign(target: nn.Parameter, source: torch.Tensor, name: str = "") -> nn.Parameter:
    """
    Copies `source` into `target`, raising immediately on any shape
    mismatch rather than letting PyTorch silently broadcast or fail
    later during a forward pass with a confusing error far from the
    real cause.
    """
    if target.shape != source.shape:
        raise ValueError(
            f"Shape mismatch for '{name}': target {tuple(target.shape)} "
            f"!= source {tuple(source.shape)}"
        )
    return torch.nn.Parameter(source.clone().detach())


def load_weights_into_gpt(gpt: GPTModel, hf_model) -> None:
    """
    Copies weights from a HuggingFace `GPT2LMHeadModel` (or `GPT2Model`)
    into this repo's `GPTModel`, in place.

    Parameters
    ----------
    gpt : GPTModel
        Destination model. Its config (vocab_size, context_length,
        emb_dim, n_heads, n_layers, qkv_bias) MUST match the source
        checkpoint exactly -- use `GPTConfig.gpt2_small()` with
        `qkv_bias=True` for the standard 124M checkpoint.
    hf_model : transformers.GPT2LMHeadModel or transformers.GPT2Model
        The loaded HuggingFace model to copy weights FROM. Both
        `GPT2LMHeadModel` (which wraps `.transformer`) and the bare
        `GPT2Model` are accepted.

    Raises
    ------
    ValueError
        If any weight's shape doesn't match -- almost always means the
        `gpt` config doesn't match the checkpoint (e.g. n_heads, n_layers,
        or qkv_bias differ from GPT-2 small's actual architecture).
    """
    # GPT2LMHeadModel wraps the transformer body under `.transformer`;
    # GPT2Model IS the transformer body. Normalize to the body either way.
    hf_transformer = getattr(hf_model, "transformer", hf_model)

    # --- Embeddings -------------------------------------------------
    gpt.pos_emb.weight = assign(
        gpt.pos_emb.weight, hf_transformer.wpe.weight, "pos_emb.weight"
    )
    gpt.tok_emb.weight = assign(
        gpt.tok_emb.weight, hf_transformer.wte.weight, "tok_emb.weight"
    )

    for i, block in enumerate(gpt.trf_blocks):
        hf_block = hf_transformer.h[i]

        # --- Attention: fused c_attn -> split Q, K, V ---------------
        # HF Conv1D weight shape: (in_features, 3*out_features).
        # .T gives (3*out_features, in_features); split along dim 0
        # into three (out_features, in_features) chunks matching
        # nn.Linear's expected weight shape.
        q_w, k_w, v_w = np.split(
            hf_block.attn.c_attn.weight.detach().numpy().T, 3, axis=0
        )
        block.attn.W_query.weight = assign(
            block.attn.W_query.weight, torch.from_numpy(q_w), f"trf_blocks[{i}].attn.W_query.weight"
        )
        block.attn.W_key.weight = assign(
            block.attn.W_key.weight, torch.from_numpy(k_w), f"trf_blocks[{i}].attn.W_key.weight"
        )
        block.attn.W_value.weight = assign(
            block.attn.W_value.weight, torch.from_numpy(v_w), f"trf_blocks[{i}].attn.W_value.weight"
        )

        q_b, k_b, v_b = np.split(
            hf_block.attn.c_attn.bias.detach().numpy(), 3, axis=0
        )
        block.attn.W_query.bias = assign(
            block.attn.W_query.bias, torch.from_numpy(q_b), f"trf_blocks[{i}].attn.W_query.bias"
        )
        block.attn.W_key.bias = assign(
            block.attn.W_key.bias, torch.from_numpy(k_b), f"trf_blocks[{i}].attn.W_key.bias"
        )
        block.attn.W_value.bias = assign(
            block.attn.W_value.bias, torch.from_numpy(v_b), f"trf_blocks[{i}].attn.W_value.bias"
        )

        # --- Attention output projection -----------------------------
        block.attn.out_proj.weight = assign(
            block.attn.out_proj.weight,
            hf_block.attn.c_proj.weight.T,
            f"trf_blocks[{i}].attn.out_proj.weight",
        )
        block.attn.out_proj.bias = assign(
            block.attn.out_proj.bias,
            hf_block.attn.c_proj.bias,
            f"trf_blocks[{i}].attn.out_proj.bias",
        )

        # --- FeedForward: net[0] = fc_in, net[2] = fc_out -------------
        block.ffn.net[0].weight = assign(
            block.ffn.net[0].weight,
            hf_block.mlp.c_fc.weight.T,
            f"trf_blocks[{i}].ffn.net[0].weight",
        )
        block.ffn.net[0].bias = assign(
            block.ffn.net[0].bias,
            hf_block.mlp.c_fc.bias,
            f"trf_blocks[{i}].ffn.net[0].bias",
        )
        block.ffn.net[2].weight = assign(
            block.ffn.net[2].weight,
            hf_block.mlp.c_proj.weight.T,
            f"trf_blocks[{i}].ffn.net[2].weight",
        )
        block.ffn.net[2].bias = assign(
            block.ffn.net[2].bias,
            hf_block.mlp.c_proj.bias,
            f"trf_blocks[{i}].ffn.net[2].bias",
        )

        # --- LayerNorms (HF: weight/bias -> this repo: scale/shift) ---
        block.norm1.scale = assign(
            block.norm1.scale, hf_block.ln_1.weight, f"trf_blocks[{i}].norm1.scale"
        )
        block.norm1.shift = assign(
            block.norm1.shift, hf_block.ln_1.bias, f"trf_blocks[{i}].norm1.shift"
        )
        block.norm2.scale = assign(
            block.norm2.scale, hf_block.ln_2.weight, f"trf_blocks[{i}].norm2.scale"
        )
        block.norm2.shift = assign(
            block.norm2.shift, hf_block.ln_2.bias, f"trf_blocks[{i}].norm2.shift"
        )

    # --- Final norm + output head ------------------------------------
    gpt.final_norm.scale = assign(
        gpt.final_norm.scale, hf_transformer.ln_f.weight, "final_norm.scale"
    )
    gpt.final_norm.shift = assign(
        gpt.final_norm.shift, hf_transformer.ln_f.bias, "final_norm.shift"
    )

    # GPT-2 ties the output head to the token embedding. This repo's
    # out_head is a separate nn.Linear (no bias, matching GPT-2), so
    # we copy the same weights in rather than aliasing the tensor --
    # keeps this model's parameters independent/save-able on their own.
    gpt.out_head.weight = assign(
        gpt.out_head.weight, hf_transformer.wte.weight, "out_head.weight"
    )