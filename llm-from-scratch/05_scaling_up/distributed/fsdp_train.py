"""Fully Sharded Data Parallel training entrypoint, for models too large to
replicate fully on each GPU (unlike DDP)."""
raise NotImplementedError("TODO: wrap model in FSDP, configure sharding strategy + mixed precision policy")
