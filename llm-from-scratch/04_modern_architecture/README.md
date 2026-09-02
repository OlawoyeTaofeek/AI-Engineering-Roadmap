# 04 - Modern Architecture

Upgrades that separate a 2019 GPT-2-style model from a 2024+ LLaMA-style one.
Each piece is implemented standalone and unit-tested before being combined in
`07_advanced_llm_from_scratch/`.

```
positional/            -- RoPE, NoPE, and a written comparison of the two
normalization/           -- RMSNorm
feedforward/               -- SwiGLU
inference_efficiency/        -- KV-cache, sliding window attention, rolling buffer KV-cache
ablations.md                   -- perplexity/speed impact of swapping each piece in, one at a time
```
