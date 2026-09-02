# Papers

Annotated reading list. One line per paper: what it introduced, and which
folder in this repo implements it.

| Paper | Introduces | Implemented in |
|---|---|---|
| Attention Is All You Need (Vaswani et al., 2017) | Transformer, self-attention, sinusoidal PE | `02_tiny_llm/`, `03_attention_variants/` |
| GPT-2 (Radford et al., 2019) | Decoder-only pretraining at scale | `02_tiny_llm/` |
| RoFormer / RoPE (Su et al., 2021) | Rotary position embeddings | `04_modern_architecture/positional/rope.py` |
| Root Mean Square Layer Normalization (Zhang & Sennrich, 2019) | RMSNorm | `04_modern_architecture/normalization/rmsnorm.py` |
| GLU Variants Improve Transformer (Shazeer, 2020) | SwiGLU | `04_modern_architecture/feedforward/swiglu.py` |
| LLaMA (Touvron et al., 2023) | RoPE + RMSNorm + SwiGLU combined | `07_advanced_llm_from_scratch/` |
| GQA (Ainslie et al., 2023) | Grouped-query attention | `03_attention_variants/implementations/grouped_query_attention.py` |
| Switch Transformer (Fedus et al., 2021) | Sparse MoE routing | `06_moe/` |
| InstructGPT (Ouyang et al., 2022) | SFT + reward modeling + RLHF pipeline | `08_`, `09_`, `10_` |
| PPO (Schulman et al., 2017) | Policy optimization used in RLHF | `10_rlhf_ppo/` |
| DPO (Rafailov et al., 2023) | RLHF alternative without a reward-model rollout loop | `10_rlhf_ppo/notes_ppo_vs_dpo.md` |

Dropping PDFs directly in this folder; keeping the table above in sync.
