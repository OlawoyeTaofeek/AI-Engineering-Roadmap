# Ablations

Swap one component at a time into the `02_tiny_llm` baseline, retrain under
identical conditions, and record the delta.

| Change | Perplexity delta | Step time delta | Notes |
|---|---|---|---|
| LayerNorm -> RMSNorm | TBD | TBD | |
| GELU-MLP -> SwiGLU | TBD | TBD | |
| Learned absolute -> RoPE | TBD | TBD | |
| + KV-cache (inference only) | n/a | TBD | throughput, not perplexity |
| + Sliding window attention | TBD | TBD | |
