# Roadmap

Legend: `[ ]` not started &nbsp; `[~]` in progress &nbsp; `[x]` done

This roadmap follows the 8-stage LLM mastery path (architecture -> pretraining
-> post-training datasets -> SFT -> preference alignment -> evaluation ->
quantization -> new trends), cross-checked stage by stage so nothing gets
silently skipped.

## 00 - Explanation
- [ ] Attention mechanisms write-up
- [ ] Normalization write-up
- [ ] Training objective write-up
- [ ] Architecture overview diagram

## 01 - Papers
- [ ] Annotated reading list committed

## 02 - Tiny LLM (Stage 1: LLM Architecture)
- [x] Self-attention
- [x] Causal attention
- [x] Multi-head attention
- [x] LayerNorm
- [x] FeedForward block
- [x] TransformerBlock (pre-norm)
- [x] Full GPTModel skeleton
- [x] Text generation loop (greedy)
- [~] GPTConfig (typed, validated config) -- skeleton + 18 tests written, implementation pending
- [~] GPTForCausalLM (HF-style forward/loss/generate wrapper) -- skeleton + 12 tests written, implementation pending
- [~] BPE tokenizer from scratch -- skeleton + 8 tests written, implementation pending
- [~] Sampling techniques (temperature/top-k/top-p) -- skeleton + 14 tests written, implementation pending
- [ ] Training loop with train/val loss tracking (full LR warmup+decay, grad clipping)
- [ ] Load pretrained GPT-2 weights (mechanics proven correct at small scale in conversation; real download untested -- no network in dev sandbox)
- [ ] Fine-tune for classification
- [ ] Full test suite passing in CI

## 03 - Attention Variants
- [ ] Grouped-query attention (GQA)
- [ ] Multi-query attention (MQA)
- [ ] Vanilla Multi Head Attention (MHA)
- [ ] Sliding Window Attention (SWA)
- [ ] Gated Attention
- [ ] Gated DeltaNet Attention
- [ ] Benchmark script (perplexity / speed / memory)
- [ ] Comparison charts
- [ ] Interactive playground (temperature / top-k / top-p, probability charts) -- depends on 02's sampling.py

## 04 - Modern Architecture
- [ ] RoPE (mechanics proven correct in conversation -- relative-position-invariance verified numerically; module implementation pending)
- [ ] NoPE
- [ ] RMSNorm
- [ ] SwiGLU
- [ ] KV-cache
- [ ] Sliding window attention
- [ ] Rolling buffer KV-cache
- [ ] Ablation notes (impact of each change)

## 05 - Scaling Up (Stage 2: Pre-training Models)
- [ ] Mixed precision training
- [ ] Gradient accumulation
- [ ] DDP training script
- [ ] FSDP training script
- [ ] Scaling law notes
- [x] Data preparation -- see data_collection/ (books module fully built + tested)
- [ ] Monitoring dashboard beyond basic MLflow logging

## 06 - Mixture of Experts
- [ ] MoE layer
- [ ] Router
- [ ] Load-balancing loss
- [ ] Tests

## 07 - Advanced LLM From Scratch
- [ ] Combine RoPE + RMSNorm + SwiGLU + GQA + KV-cache (+ optional MoE)

## 08 - Post-Training Datasets (Stage 3) -- NEW, was fully missing
- [~] Chat templates + loss masking -- skeleton written, implementation pending
- [~] Synthetic data generation (self-instruct / OSS-Instruct style) -- skeleton written
- [~] Data enhancement (paraphrasing, difficulty variation) -- skeleton written
- [~] Quality filtering (instruction/response pairs, distinct from pretraining-corpus cleaner) -- skeleton written

## 09 - Instruction Fine-Tuning / SFT (Stage 4) (renumbered from 08)
- [ ] Prompt formatting + loss masking -- now depends on 08_post_training_datasets/chat_templates.py
- [ ] SFT training script
- [ ] Eval harness -- now depends on 11_evaluation/

## 10 - Preference Alignment (Stage 5) (merged/renumbered from old 09_reward_modeling + 10_rlhf_ppo)
- [ ] Reward model architecture
- [ ] Preference dataset loader
- [ ] Reward model training script
- [~] Rejection sampling -- skeleton written, NEW
- [~] DPO (Direct Preference Optimization) -- skeleton written, NEW (previously only a comparison note existed)
- [ ] PPO trainer
- [ ] PPO rollout collection
- [x] Notes: PPO vs DPO

## 11 - Evaluation (Stage 6) -- NEW, was fully missing
- [~] Automated benchmarks (perplexity on fixed eval set + multiple-choice) -- skeleton written
- [~] Human evaluation tooling (A/B comparison) -- skeleton written
- [~] Model-based evaluation (LLM-as-judge) -- skeleton written
- [~] Feedback signal aggregation (MLflow-tracked, cross-checkpoint comparison) -- skeleton written

## 12 - Serving (renumbered from 11)
- [ ] Custom FastAPI inference server
- [ ] Request batching
- [ ] Streaming responses
- [ ] vLLM deployment
- [ ] Ollama / GGUF deployment (mechanics covered in conversation: HF export -> convert_hf_to_gguf.py -> llama-quantize -> Modelfile)
- [ ] Baseten deployment
- [ ] Load testing / throughput benchmarks

## 13 - Quantization (Stage 7) -- NEW, was fully missing
- [~] Base techniques (symmetric/asymmetric, per-tensor/per-channel, from scratch) -- skeleton written
- [~] GGUF / llama.cpp block-wise (K-quant) quantization -- skeleton written
- [~] GPTQ & AWQ (calibration-based, via existing libraries + comparison) -- skeleton written
- [~] SmoothQuant & ZeroQuant (activation quantization) -- skeleton written

## 14 - New Trends (Stage 8) -- NEW, was fully missing
- [~] Model merging (linear + task arithmetic) -- skeleton written
- [~] Multimodal extension (image encoder + projector) -- skeleton written
- [~] Interpretability (attention visualization, max-activating examples, logit lens) -- skeleton written
- [~] Test-time compute (self-consistency, best-of-n, tree search) -- skeleton written

## Data Collection
- [x] Books module (Gutenberg, Internet Archive, Wikipedia) -- fully implemented, 34/34 tests passing
- [ ] Nairaland scraper (discussed, not yet scaffolded)

## Infra
- [x] Repo scaffold
- [ ] requirements.txt pinned + verified
- [ ] Dockerfile built + tested
- [ ] MLflow tracking wired into all train.py scripts
- [ ] Git LFS configured for checkpoints
- [ ] CI passing on main
