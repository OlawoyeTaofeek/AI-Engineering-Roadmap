# 02 - Tiny LLM

First working GPT, built from primitives:

```
model/attention.py         -- self-attention, causal attention, multi-head attention
model/layer_norm.py         -- LayerNorm from scratch
model/transformer_block.py  -- pre-norm block: attn -> add&norm -> ffn -> add&norm
model/gpt_model.py           -- full model: embeddings -> N blocks -> final norm -> head
data/dataset.py               -- sliding-window Dataset + DataLoader builder
train.py                       -- training loop, MLflow-tracked
generate.py                     -- greedy text generation
load_pretrained_weights.py       -- load OpenAI's released GPT-2 weights into this model
finetune_classification.py        -- fine-tune the pretrained model for classification
tests/                              -- shape/correctness tests for every piece above
```

## Quickstart

```bash
python train.py --config configs/tiny.yaml
python generate.py --checkpoint checkpoints/tiny_llm.pt --prompt "Once upon a time"
pytest tests/ -v
```
