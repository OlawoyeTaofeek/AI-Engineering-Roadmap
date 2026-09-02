# LLM From Scratch — Foundations to Serving

Build a large language model from first principles: tensors and autograd, up
through a trained, quantized, RLHF-aligned model served behind a real API.
Every stage is implemented by hand first — no `import transformers` shortcuts
until you've built the thing you're importing — then compared against the
production approach, so the tradeoffs are visible instead of assumed.

```
Attention, from scratch          ->  torch.nn.MultiheadAttention
A hand-rolled training loop      ->  HuggingFace Trainer
Manual RLHF reward loop          ->  TRL / trlx
```

This is **Part 1** of a two-part series. Part 2, [`AI-EngineeringRoadmap`](../AI-EngineeringRoadmap),
picks up once a model exists and covers RAG, agents, inference optimization,
and deployment.

## Why from scratch

Frameworks like HuggingFace `transformers` are the right tool for building
*products*. They're a bad tool for *understanding* — the abstraction that
makes them productive is exactly what hides how attention, KV-caching, or
DPO actually work. This repo trades that productivity for transparency: every
notebook here should leave you able to explain, not just call, the thing it
implements.

## What you'll have built by the end

- A GPT-style transformer, trained from raw tensors up — no framework model
  classes
- A modern architecture upgrade path: RoPE, RMSNorm, GQA, sliding-window
  attention, MoE
- A model that's been through the full post-training pipeline: SFT, reward
  modeling, DPO/PPO
- The same model quantized and served behind a FastAPI endpoint and a vLLM
  deployment

## Roadmap

See [`ROADMAP.md`](./ROADMAP.md) for the full checklist of what's done, in
progress, and planned. Status honestly reflects the current state of the
repo, not the target end-state.

## Structure

| Folder | Contents |
|---|---|
| `00_explanation/` | Conceptual write-ups: attention, normalization, training objective |
| `01_papers/` | Source papers referenced throughout, with an annotated index |
| `02_tiny_llm/` | First working GPT: architecture, BPE tokenizer, sampling (temperature/top-k/top-p), training, pretrained-weight loading, classification fine-tuning |
| `03_attention_variants/` | Self / causal / multi-head / GQA / MQA compared, with an interactive playground |
| `04_modern_architecture/` | RoPE, NoPE, RMSNorm, SwiGLU, KV-cache, sliding window, rolling buffer cache |
| `05_scaling_up/` | Mixed precision, gradient accumulation, distributed training (DDP/FSDP) |
| `06_moe/` | Mixture-of-Experts layer, routing, load-balancing loss |
| `07_advanced_llm_from_scratch/` | Full model combining all modern components |
| `08_post_training_datasets/` | Chat templates, synthetic data generation, data enhancement, quality filtering |
| `09_instruction_finetuning_sft/` | Instruction tuning / SFT |
| `10_preference_alignment/` | Reward modeling, rejection sampling, DPO, PPO |
| `11_evaluation/` | Automated benchmarks, human evaluation, LLM-as-judge, tracked feedback signal |
| `12_serving/` | Custom FastAPI inference server, vLLM deployment, Baseten deployment |
| `13_quantization/` | Base techniques, GGUF/llama.cpp, GPTQ/AWQ, SmoothQuant/ZeroQuant |
| `14_new_trends/` | Model merging, multimodal extension, interpretability, test-time compute |
| `data_collection/` | Public-domain book corpus (Gutenberg, Internet Archive, Wikipedia) |
| `extra/pytorch-basics/` | PyTorch/CNN/DNN prerequisites, kept separate from the main LLM track — start here if tensors and `nn.Module` aren't yet second nature |

## Where to start

If you're comfortable with PyTorch already, start at `02_tiny_llm/`. If not,
`extra/pytorch-basics/pytorch_nn_cnn_transformer_from_scratch.ipynb` builds
tensors → autograd → DNN → CNN → transformer block by block and is the
intended on-ramp into `02_tiny_llm/`.

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

Training scripts log to MLflow. Start the local UI with:

```bash
mlflow ui --backend-store-uri ./experiments/mlruns
```

## Running tests

```bash
pytest -v
```

CI runs the suite on every push (see `.github/workflows/tests.yml`).

## Checkpoints

Model weights are **not** committed directly. See [`checkpoints/README.md`](./checkpoints/README.md)
for how weights are stored and how to reproduce them.

## Contributing

Issues and PRs welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md). Notebooks
should run top-to-bottom without manual patching; strip outputs before
committing.

## License

See [`LICENSE`](./LICENSE).