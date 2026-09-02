# 13 - Quantization

Reduces model weight precision (typically float32/float16 -> int8/int4)
to shrink memory footprint and speed up inference, at a small quality
cost. This connects directly to the Ollama/GGUF deployment work covered
earlier in this repo's conversation history -- GGUF files ARE quantized
weights, so this stage is what actually produces the numbers behind that
`llama-quantize ... Q4_K_M` command you'll run when deploying.

```
base_techniques.py        -- fundamentals: what quantization IS, symmetric vs
                                asymmetric, per-tensor vs per-channel quantization
gguf_llama_cpp.py            -- the GGUF format + llama.cpp's quantization
                                  scheme specifically (what Q4_K_M etc. actually mean)
gptq_awq.py                    -- GPTQ and AWQ: post-training quantization methods
                                    that calibrate on sample data to minimize
                                    quality loss, rather than naive rounding
smoothquant_zeroquant.py         -- SmoothQuant and ZeroQuant: techniques for
                                      quantizing ACTIVATIONS as well as weights
                                      (harder than weight-only quantization,
                                      since activations have a much wider
                                      dynamic range)
```

## Suggested build order

1. `base_techniques.py` first -- implement basic weight-only int8
   quantization from scratch (round-to-nearest), to understand the core
   mechanic before using any library.
2. `gguf_llama_cpp.py` -- ties directly into deployment; even if you use
   `llama.cpp`'s own converter tool in practice (as shown in the
   deployment conversation), understanding what it's actually doing
   closes the gap between "I ran a command" and "I understand what
   happened."
3. `gptq_awq.py` and `smoothquant_zeroquant.py` -- more advanced,
   calibration-based methods; good to study and reference existing
   library implementations (AutoGPTQ, AutoAWQ) rather than necessarily
   reimplementing from scratch, given the complexity.
