# Positional Encoding: RoPE vs NoPE vs Learned Absolute

| | Learned absolute | RoPE | NoPE |
|---|---|---|---|
| Extra parameters | yes (embedding table) | no | no |
| Extrapolates past training length | poorly | well | mixed, task-dependent |
| Where applied | added to token embeddings, once | rotates q/k, every attention layer | nowhere |

_TODO: fill in empirical results once `04_modern_architecture/ablations.md` has numbers._
