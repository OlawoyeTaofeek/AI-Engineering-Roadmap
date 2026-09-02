# Checkpoints

Model weights are tracked with **Git LFS**, not committed as plain blobs.

## Setup (one-time, per clone)

```bash
git lfs install
```

## Saving a checkpoint

```python
torch.save(model.state_dict(), "checkpoints/tiny_llm_epoch10.pt")
```

Anything matching `*.pt`, `*.pth`, `*.safetensors`, `*.bin` under this repo is
automatically routed through LFS (see `.gitattributes`).

## If you'd rather not store weights at all

Every training script accepts a `--config` pointing at the exact hyperparameters
and data split used, so any checkpoint can be regenerated deterministically
(same seed, same config) instead of stored. Configs live alongside each stage's
training script.

## Loading pretrained GPT-2 weights

See `02_tiny_llm/load_pretrained_weights.py` -- downloads and converts OpenAI's
released GPT-2 weights into this repo's model format. Not stored in the repo;
downloaded on demand and cached under `checkpoints/pretrained/` (gitignored).
