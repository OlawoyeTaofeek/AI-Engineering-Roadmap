# 06 - Mixture of Experts (MoE)

Sparse routing: replace a single dense FFN with several "expert" FFNs, and a
learned router that sends each token to only a handful of them -- more total
parameters, similar compute per token.

```
moe_layer.py            -- MoE layer: router + expert FFNs + weighted combination
router.py                 -- top-k routing logic
load_balancing_loss.py      -- auxiliary loss encouraging even expert utilization
tests/                         -- routing correctness, load-balance sanity checks
```
