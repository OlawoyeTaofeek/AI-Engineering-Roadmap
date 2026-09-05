<p align="center">
  <img src="./assets/llm-from-scratch-banner.svg" alt="LLM From Scratch — Foundations to Serving" width="100%">
</p>

<p align="center">
  <a href="./ROADMAP.md"><img alt="progress" src="https://img.shields.io/badge/progress-tracked%20in%20ROADMAP.md-2563eb?style=flat-square"></a>
  <img alt="modules" src="https://img.shields.io/badge/modules-15-2563eb?style=flat-square">
  <img alt="part" src="https://img.shields.io/badge/part-1%20of%202-2563eb?style=flat-square">
  <a href="./LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-2563eb?style=flat-square"></a>
</p>

> Build a large language model from first principles: tensors and autograd, up
> through a trained, quantized, RLHF-aligned model served behind a real API.
> Every stage is implemented by hand first — no `import transformers`
> shortcuts until you've built the thing you're importing — then compared
> against the production approach, so the tradeoffs are visible instead of
> assumed.

```
Attention, from scratch          ->  torch.nn.MultiheadAttention
A hand-rolled training loop      ->  HuggingFace Trainer
Manual RLHF reward loop          ->  TRL / trlx
```

This is **Part 1** of a two-part series. Part 2,
[`AI-EngineeringRoadmap`](../AI-EngineeringRoadmap), picks up once a model
exists and covers RAG, agents, inference optimization, and deployment.

---

## Why from scratch

Frameworks like HuggingFace `transformers` are the right tool for building
*products*. They're a bad tool for *understanding* — the abstraction that
makes them productive is exactly what hides how attention, KV-caching, or
DPO actually work. This repo trades that productivity for transparency: every
notebook here should leave you able to explain, not just call, the thing it
implements.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#eff6ff','primaryTextColor':'#0f172a','primaryBorderColor':'#3b82f6','lineColor':'#3b82f6','fontFamily':'JetBrains Mono','fontSize':'15px'}}}%%
flowchart LR
  classDef build fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#0f172a
  classDef use fill:#bfdbfe,stroke:#2563eb,stroke-width:1.5px,color:#0f172a
  classDef compare fill:#93c5fd,stroke:#1d4ed8,stroke-width:1.5px,color:#0f172a

  B("<b>Build it</b><br/><sub>raw PyTorch tensors, no framework shortcuts</sub>"):::build
  U("<b>Use it</b><br/><sub>the same thing via the production library</sub>"):::use
  C("<b>Compare</b><br/><sub>what the abstraction actually buys you</sub>"):::compare

  B ==>|"e.g. hand-rolled attention"| U
  U ==>|"e.g. torch.nn.MultiheadAttention"| C
  C -.->|"repeats for every module"| B
```

## What you'll have built by the end

- A GPT-style transformer, trained from raw tensors up — no framework model
  classes
- A modern architecture upgrade path: RoPE, RMSNorm, GQA, sliding-window
  attention, MoE
- A model that's been through the full post-training pipeline: SFT, reward
  modeling, DPO/PPO
- The same model quantized and served behind a FastAPI endpoint and a vLLM
  deployment

---

## The shape of the repo

Fifteen modules stack roughly in order. Architecture is the floor; serving
and quantization are the roof. The numbering follows the standard LLM
mastery path (architecture → pretraining → post-training → evaluation →
quantization → new trends), grouped below by the eight stages tracked in
[`ROADMAP.md`](./ROADMAP.md).

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#eff6ff','primaryTextColor':'#0f172a','primaryBorderColor':'#3b82f6','lineColor':'#3b82f6','fontFamily':'JetBrains Mono','fontSize':'13px'},'flowchart':{'curve':'basis','nodeSpacing':22,'rankSpacing':55}}}%%
flowchart LR
  classDef s0 fill:#f8fafc,stroke:#94a3b8,color:#334155
  classDef s1 fill:#eff6ff,stroke:#60a5fa,color:#0f172a
  classDef s2 fill:#dbeafe,stroke:#3b82f6,color:#0f172a
  classDef s3 fill:#bfdbfe,stroke:#2563eb,color:#0f172a
  classDef s4 fill:#93c5fd,stroke:#1d4ed8,color:#0f172a
  classDef s5 fill:#60a5fa,stroke:#1d4ed8,color:#0f172a
  classDef ghost fill:none,stroke:none,color:#0f172a

  subgraph G0[" "]
    direction LR
    E["00 · Explanation"]:::s0
    P["01 · Papers"]:::s0
  end

  subgraph G1["Stage 1 — Architecture"]
    direction LR
    T["02 · Tiny LLM"]:::s1
    AV["03 · Attention Variants"]:::s1
    MA["04 · Modern Architecture"]:::s1
  end

  subgraph G2["Stage 2 — Pretraining"]
    direction LR
    SU["05 · Scaling Up"]:::s2
    MOE["06 · Mixture of Experts"]:::s2
    ADV["07 · Advanced LLM"]:::s2
  end

  subgraph G3["Stages 3–5 — Post-Training"]
    direction LR
    PTD["08 · PT Datasets"]:::s3
    SFT["09 · SFT"]:::s3
    PA["10 · Preference Alignment"]:::s3
  end

  subgraph G4["Stages 6–8 — Ship It"]
    direction LR
    EV["11 · Evaluation"]:::s4
    SRV["12 · Serving"]:::s4
    Q["13 · Quantization"]:::s4
    NT["14 · New Trends"]:::s4
  end

  E --> P --> T
  T --> AV --> MA --> SU
  SU --> MOE --> ADV --> PTD
  PTD --> SFT --> PA --> EV
  EV --> SRV --> Q --> NT
  NT -.-> PART2["Part 2 — AI Engineering Roadmap"]:::s5

  class G0,G1,G2,G3,G4 ghost
```

Skip ahead if you already know the lower layers — but if something near the
top breaks, it's usually because a lower layer wasn't actually solid.

---

## Contents

Click a module to expand what's inside it. Full status detail — including
what's a skeleton vs. fully implemented — lives in [`ROADMAP.md`](./ROADMAP.md).

<details>
<summary><strong>00 · Explanation</strong></summary>

Conceptual write-ups: attention, normalization, the training objective — the
"why" behind what gets built in every module after this one.

</details>

<details>
<summary><strong>01 · Papers</strong></summary>

Source papers referenced throughout, with an annotated index.

</details>

<details>
<summary><strong>02 · Tiny LLM</strong> — Stage 1: LLM Architecture</summary>

First working GPT, built end-to-end: data pipeline, four tokenizers,
attention/normalization/transformer-block architecture, config-driven model
assembly, an HF-style CausalLM wrapper, MLflow-tracked training, OpenAI
GPT-2 (124M) pretrained-weight loading, five sampling strategies plus beam
search, classification fine-tuning, a full `pytest` suite, and two chat UIs.
See [`02_tiny_llm/README.md`](./02_tiny_llm/README.md) for the full writeup.

| Stage | What it is | Where |
|---|---|---|
| Data | Custom `Dataset` + `DataLoader`, tokenized text sliced into overlapping context windows with a configurable stride | `data/` |
| Tokenizers | Byte-level, character-level, word-level, and a from-scratch **BPE** tokenizer, compared side by side | `tokenizers/` |
| Attention | Self-attention → causal (masked) attention → multi-head attention, built up in stages | `model/attention.py` |
| Normalization | `LayerNorm` from raw mean/variance ops | `model/layer_norm.py` |
| Transformer block | Pre-norm → MHA → residual → pre-norm → FFN → residual | `model/transformer_block.py` |
| Config | Typed, validated `GPTConfig` dataclass with `from_pretrained`/`save_pretrained` | `model/config.py` |
| Model assembly | Full `GPTModel`: embeddings → N transformer blocks → final norm → LM head | `model/gpt_model.py` |
| Causal LM wrapper | HF-`GPT2LMHeadModel`-style wrapper — loss-aware forward pass + `.generate()` | `model/causal_lm.py` |
| Training + tracking | Training loop with LR scheduling and hyperparameter sweeps, metrics logged to **MLflow** | `train.py`, `learning_rate_scheuler/`, `hyperparameter_tuning/` |
| Pretrained weights | Converts and loads **OpenAI's original GPT-2 (124M)** TF checkpoint into the from-scratch model | `load_pretrained_weight.py`, `loading_pretrained_weight.ipynb` |
| Decoding | Greedy, temperature scaling, top-k, top-p (nucleus), frequency penalty, and beam search — each its own function/class | `sampling.py`, `generate.py` |
| Fine-tuning | Classification head swap, generalized to any N-class task | `finetune_classification.py` |
| Testing | Full `pytest` suite covering every component above, plus two proof notebooks (`model/test.ipynb`, `tokenizers/tokenizers_test.ipynb`) | `tests/` |
| UI | Chat with the trained-from-scratch model *or* the loaded GPT-2 weights, decoding params exposed as controls | `user_interface/streamlit_ui.py`, `user_interface/chainlit_ui.py` |

Reproduces **GPT-2 124M**, verified against OpenAI's real weights, before
the architecture is upgraded with RoPE/RMSNorm/GQA/MoE starting in `04`.

</details>

<details>
<summary><strong>03 · Attention Variants</strong></summary>

Self / causal / multi-head / GQA / MQA compared side by side, with
benchmarks, comparison charts, and an interactive playground.

</details>

<details>
<summary><strong>04 · Modern Architecture</strong></summary>

RoPE, NoPE, RMSNorm, SwiGLU, KV-cache, sliding window attention, rolling
buffer cache — plus ablation notes on what each change actually bought.

</details>

<details>
<summary><strong>05 · Scaling Up</strong> — Stage 2: Pretraining Models</summary>

Mixed precision, gradient accumulation, distributed training (DDP/FSDP),
scaling law notes.

</details>

<details>
<summary><strong>06 · Mixture of Experts</strong></summary>

MoE layer, router, load-balancing loss.

</details>

<details>
<summary><strong>07 · Advanced LLM From Scratch</strong></summary>

Combines RoPE + RMSNorm + SwiGLU + GQA + KV-cache (+ optional MoE) into one
model.

</details>

<details>
<summary><strong>08 · Post-Training Datasets</strong> — Stage 3</summary>

Chat templates + loss masking, synthetic data generation, data enhancement,
quality filtering — the data prep that everything from `09` onward depends
on.

</details>

<details>
<summary><strong>09 · Instruction Fine-Tuning / SFT</strong> — Stage 4</summary>

Prompt formatting + loss masking, SFT training script, eval harness.

</details>

<details>
<summary><strong>10 · Preference Alignment</strong> — Stage 5</summary>

Reward model architecture + training, preference dataset loader, rejection
sampling, DPO, PPO trainer and rollout collection.

</details>

<details>
<summary><strong>11 · Evaluation</strong> — Stage 6</summary>

Automated benchmarks, human evaluation tooling, LLM-as-judge, feedback
signal aggregation — tracked and compared across checkpoints.

</details>

<details>
<summary><strong>12 · Serving</strong></summary>

Custom FastAPI inference server, request batching, streaming responses,
vLLM deployment, GGUF/Ollama deployment, Baseten deployment, load testing.

</details>

<details>
<summary><strong>13 · Quantization</strong> — Stage 7</summary>

Base techniques (symmetric/asymmetric, per-tensor/per-channel) from scratch,
GGUF/llama.cpp, GPTQ/AWQ, SmoothQuant/ZeroQuant.

</details>

<details>
<summary><strong>14 · New Trends</strong> — Stage 8</summary>

Model merging, multimodal extension, interpretability, test-time compute.

</details>

<details>
<summary><strong>Data Collection</strong></summary>

Public-domain book corpus (Gutenberg, Internet Archive, Wikipedia).

</details>

<details>
<summary><strong>extra/pytorch-basics</strong></summary>

PyTorch/CNN/DNN prerequisites, kept separate from the main track — start
here if tensors and `nn.Module` aren't yet second nature.

</details>

---

## Where to start

If you're comfortable with PyTorch already, start at `02_tiny_llm/`. If not,
`extra/pytorch-basics/pytorch_nn_cnn_transformer_from_scratch.ipynb` builds
tensors → autograd → DNN → CNN → transformer block by block and is the
intended on-ramp.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Or via Docker:

```bash
docker build -t llm-from-scratch .
docker run --gpus all -it llm-from-scratch
```

## Experiment tracking

Training scripts log to MLflow:

```bash
mlflow ui --backend-store-uri ./experiments/mlruns
```

## Running tests

```bash
pytest -v
```

CI runs the suite on every push (see `.github/workflows/tests.yml`).

## Checkpoints

Model weights are **not** committed directly. See
[`checkpoints/README.md`](./checkpoints/README.md) for how weights are
stored and how to reproduce them.

## Contributing

Issues and PRs welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md).
Notebooks should run top-to-bottom without manual patching; strip outputs
before committing.

## License

See [`LICENSE`](./LICENSE).