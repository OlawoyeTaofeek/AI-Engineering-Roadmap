# Roadmap

Status tracker for every module in this repo, cross-checked stage by stage
against the 8-stage LLM mastery path (architecture → pretraining →
post-training datasets → SFT → preference alignment → evaluation →
quantization → new trends) so nothing gets silently skipped.

**Legend:** ✅ Complete · 🚧 In Progress · ⬚ Planned

---

## 00 — Explanation — ⬚

| # | Item | Status |
|---|---|---|
| 01 | Attention mechanisms write-up | ⬚ |
| 02 | Normalization write-up | ⬚ |
| 03 | Training objective write-up | ⬚ |
| 04 | Architecture overview diagram | ⬚ |

## 01 — Papers — ⬚

| # | Item | Status |
|---|---|---|
| 01 | Annotated reading list committed | ⬚ |

## 02 — Tiny LLM — 🚧 *(Stage 1: LLM Architecture)*

| # | Item | Status | Notes |
|---|---|---|---|
| 01 | Self-attention | ✅ | |
| 02 | Causal attention | ✅ | |
| 03 | Multi-head attention | ✅ | |
| 04 | LayerNorm | ✅ | |
| 05 | FeedForward block | ✅ | |
| 06 | TransformerBlock (pre-norm) | ✅ | |
| 07 | Full GPTModel skeleton | ✅ | |
| 08 | Text generation loop (greedy) | ✅ | |
| 09 | GPTConfig (typed, validated config) | 🚧 | Skeleton + 18 tests written, implementation pending |
| 10 | GPTForCausalLM (HF-style forward/loss/generate wrapper) | 🚧 | Skeleton + 12 tests written, implementation pending |
| 11 | BPE tokenizer from scratch | 🚧 | Skeleton + 8 tests written, implementation pending |
| 12 | Sampling techniques (temperature/top-k/top-p) | 🚧 | Skeleton + 14 tests written, implementation pending |
| 13 | Training loop with train/val loss tracking | ⬚ | Full LR warmup + decay, grad clipping |
| 14 | Load pretrained GPT-2 weights | ⬚ | Mechanics proven correct at small scale in dev; real download untested — no network in dev sandbox |
| 15 | Fine-tune for classification | ⬚ | |
| 16 | Full test suite passing in CI | ⬚ | |

## 03 — Attention Variants — ⬚

| # | Item | Status |
|---|---|---|
| 01 | Grouped-query attention (GQA) | ⬚ |
| 02 | Multi-query attention (MQA) | ⬚ |
| 03 | Vanilla Multi-Head Attention (MHA) | ⬚ |
| 04 | Sliding Window Attention (SWA) | ⬚ |
| 05 | Gated Attention | ⬚ |
| 06 | Gated DeltaNet Attention | ⬚ |
| 07 | Benchmark script (perplexity / speed / memory) | ⬚ |
| 08 | Comparison charts | ⬚ |
| 09 | Interactive playground (temperature/top-k/top-p, probability charts) | ⬚ | Depends on `02`'s `sampling.py` |

## 04 — Modern Architecture — ⬚

| # | Item | Status | Notes |
|---|---|---|---|
| 01 | RoPE | ⬚ | Mechanics proven correct in dev — relative-position-invariance verified numerically; module implementation pending |
| 02 | NoPE | ⬚ | |
| 03 | RMSNorm | ⬚ | |
| 04 | SwiGLU | ⬚ | |
| 05 | KV-cache | ⬚ | |
| 06 | Sliding window attention | ⬚ | |
| 07 | Rolling buffer KV-cache | ⬚ | |
| 08 | Ablation notes (impact of each change) | ⬚ | |

## 05 — Scaling Up — 🚧 *(Stage 2: Pretraining Models)*

| # | Item | Status | Notes |
|---|---|---|---|
| 01 | Mixed precision training | ⬚ | |
| 02 | Gradient accumulation | ⬚ | |
| 03 | DDP training script | ⬚ | |
| 04 | FSDP training script | ⬚ | |
| 05 | Scaling law notes | ⬚ | |
| 06 | Data preparation | ✅ | See `data_collection/` — books module fully built + tested |
| 07 | Monitoring dashboard beyond basic MLflow logging | ⬚ | |

## 06 — Mixture of Experts — ⬚

| # | Item | Status |
|---|---|---|
| 01 | MoE layer | ⬚ |
| 02 | Router | ⬚ |
| 03 | Load-balancing loss | ⬚ |
| 04 | Tests | ⬚ |

## 07 — Advanced LLM From Scratch — ⬚

| # | Item | Status |
|---|---|---|
| 01 | Combine RoPE + RMSNorm + SwiGLU + GQA + KV-cache (+ optional MoE) | ⬚ |

## 08 — Post-Training Datasets — 🚧 *(Stage 3)*

| # | Item | Status | Notes |
|---|---|---|---|
| 01 | Chat templates + loss masking | 🚧 | Skeleton written, implementation pending |
| 02 | Synthetic data generation (self-instruct / OSS-Instruct style) | 🚧 | Skeleton written |
| 03 | Data enhancement (paraphrasing, difficulty variation) | 🚧 | Skeleton written |
| 04 | Quality filtering (instruction/response pairs) | 🚧 | Skeleton written — distinct from the pretraining-corpus cleaner |

## 09 — Instruction Fine-Tuning / SFT — ⬚ *(Stage 4)*

| # | Item | Status | Notes |
|---|---|---|---|
| 01 | Prompt formatting + loss masking | ⬚ | Depends on `08_post_training_datasets/chat_templates.py` |
| 02 | SFT training script | ⬚ | |
| 03 | Eval harness | ⬚ | Depends on `11_evaluation/` |

## 10 — Preference Alignment — 🚧 *(Stage 5)*

| # | Item | Status | Notes |
|---|---|---|---|
| 01 | Reward model architecture | ⬚ | |
| 02 | Preference dataset loader | ⬚ | |
| 03 | Reward model training script | ⬚ | |
| 04 | Rejection sampling | 🚧 | Skeleton written |
| 05 | DPO (Direct Preference Optimization) | 🚧 | Skeleton written — previously only a comparison note existed |
| 06 | PPO trainer | ⬚ | |
| 07 | PPO rollout collection | ⬚ | |
| 08 | Notes: PPO vs. DPO | ✅ | |

## 11 — Evaluation — 🚧 *(Stage 6)*

| # | Item | Status |
|---|---|---|
| 01 | Automated benchmarks (perplexity + multiple-choice) | 🚧 Skeleton written |
| 02 | Human evaluation tooling (A/B comparison) | 🚧 Skeleton written |
| 03 | Model-based evaluation (LLM-as-judge) | 🚧 Skeleton written |
| 04 | Feedback signal aggregation (MLflow-tracked, cross-checkpoint) | 🚧 Skeleton written |

## 12 — Serving — ⬚

| # | Item | Status | Notes |
|---|---|---|---|
| 01 | Custom FastAPI inference server | ⬚ | |
| 02 | Request batching | ⬚ | |
| 03 | Streaming responses | ⬚ | |
| 04 | vLLM deployment | ⬚ | |
| 05 | Ollama / GGUF deployment | ⬚ | Mechanics scoped: HF export → `convert_hf_to_gguf.py` → `llama-quantize` → Modelfile |
| 06 | Baseten deployment | ⬚ | |
| 07 | Load testing / throughput benchmarks | ⬚ | |

## 13 — Quantization — 🚧 *(Stage 7)*

| # | Item | Status |
|---|---|---|
| 01 | Base techniques (symmetric/asymmetric, per-tensor/per-channel, from scratch) | 🚧 Skeleton written |
| 02 | GGUF / llama.cpp block-wise (K-quant) quantization | 🚧 Skeleton written |
| 03 | GPTQ & AWQ (calibration-based, via existing libraries + comparison) | 🚧 Skeleton written |
| 04 | SmoothQuant & ZeroQuant (activation quantization) | 🚧 Skeleton written |

## 14 — New Trends — 🚧 *(Stage 8)*

| # | Item | Status |
|---|---|---|
| 01 | Model merging (linear + task arithmetic) | 🚧 Skeleton written |
| 02 | Multimodal extension (image encoder + projector) | 🚧 Skeleton written |
| 03 | Interpretability (attention viz, max-activating examples, logit lens) | 🚧 Skeleton written |
| 04 | Test-time compute (self-consistency, best-of-n, tree search) | 🚧 Skeleton written |

## Data Collection

| # | Item | Status | Notes |
|---|---|---|---|
| 01 | Books module (Gutenberg, Internet Archive, Wikipedia) | ✅ | Fully implemented, 34/34 tests passing |
| 02 | Nairaland scraper | ⬚ | Discussed, not yet scaffolded |

## Infra

| # | Item | Status |
|---|---|---|
| 01 | Repo scaffold | ✅ |
| 02 | `requirements.txt` pinned + verified | ⬚ |
| 03 | Dockerfile built + tested | ⬚ |
| 04 | MLflow tracking wired into all `train.py` scripts | ⬚ |
| 05 | Git LFS configured for checkpoints | ⬚ |
| 06 | CI passing on `main` | ⬚ |

---

**Totals:** 15 modules · 2 fully complete (Data Collection's books module,
PPO vs. DPO notes) · 8 with skeletons/tests in progress · rest planned.

Want to help? Pick any ⬚ item and open a PR — see [`CONTRIBUTING.md`](./CONTRIBUTING.md).
