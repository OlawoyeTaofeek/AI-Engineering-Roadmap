# 08 - Instruction Fine-Tuning / SFT

Fine-tune the pretrained base model on (instruction, response) pairs, with
loss computed (masked) only over the response tokens -- not the prompt.

```
dataset_formatting.py   -- prompt templates + response-only loss masking
sft_train.py               -- SFT training loop (MLflow-tracked)
eval/                          -- held-out instruction-following eval
```
