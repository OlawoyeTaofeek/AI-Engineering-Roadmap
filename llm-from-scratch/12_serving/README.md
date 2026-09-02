# 11 - Serving

Three ways to serve the final model, from "build it yourself" to "use the
standard tools."

```
custom_inference/       -- hand-built FastAPI server: batching, streaming responses
vllm_deployment/           -- serve via vLLM (PagedAttention, continuous batching)
baseten_deployment/          -- deploy to Baseten for managed hosting
load_testing/                   -- throughput/latency benchmarks across all three
```
