# 07 - Advanced LLM From Scratch

Everything from `04_modern_architecture/` (and optionally `06_moe/`) combined
into one model: RoPE + RMSNorm + SwiGLU + GQA + KV-cache (+ MoE, optional).

This is the "LLaMA-shaped" version of the tiny GPT from `02_tiny_llm/` --
same training/eval scaffolding, different building blocks.

```
model.py       -- the combined architecture
train.py         -- training loop (MLflow-tracked, same pattern as 02_tiny_llm/train.py)
configs/            -- with_moe.yaml / without_moe.yaml variants
```
