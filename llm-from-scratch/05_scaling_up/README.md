# 05 - Scaling Up

Techniques for training larger models than fit comfortably on a single GPU
at full precision.

```
mixed_precision_training.py   -- torch.autocast + GradScaler
gradient_accumulation.py       -- simulate larger batch sizes under memory limits
distributed/ddp_train.py         -- DistributedDataParallel, multi-GPU single-node (or multi-node)
distributed/fsdp_train.py         -- Fully Sharded Data Parallel, for models too large for DDP
configs/                            -- small -> medium -> large model configs, same architecture
scaling_laws_notes.md                 -- notes on compute-optimal scaling (Chinchilla etc.)
```
