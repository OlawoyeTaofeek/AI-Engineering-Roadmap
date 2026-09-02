"""
smoothquant_zeroquant.py
============================

SmoothQuant and ZeroQuant: quantize ACTIVATIONS in addition to weights
(everything in base_techniques.py, gguf_llama_cpp.py, and gptq_awq.py
quantizes WEIGHTS only, keeping activations in full precision during
inference).

WHY ACTIVATION QUANTIZATION IS HARDER
-------------------------------------------
Weights are static after training -- you can analyze their full
distribution offline and pick good quantization parameters once.
Activations are computed dynamically per input, and their distribution
can have extreme outlier values in certain channels (especially in
larger models) -- naively quantizing them the same way as weights causes
much larger errors.

SmoothQuant's key idea: mathematically "smooth" the difficulty by
migrating quantization difficulty FROM activations TO weights, using a
per-channel scaling factor applied before quantization (since
weight_channel * activation_channel is what actually matters for the
matmul, you can rescale one and inversely rescale the other without
changing the mathematical result, while making both easier to quantize).

TODO
-------
1. Implement `compute_smoothing_scale(activation_samples, weight,
   alpha=0.5)` -- given a calibration batch of activations and the
   corresponding weight matrix, compute a per-channel scale that
   balances quantization difficulty between them (alpha controls how
   much difficulty shifts to weights vs stays with activations).
2. Implement `apply_smoothing(activations, weight, scale)` -- applies
   the computed scale: activations / scale, weight * scale (elementwise
   per channel) -- confirm mathematically that activations @ weight.T
   gives the SAME result before and after smoothing (this invariant is
   the whole point -- smoothing changes quantization-friendliness, not
   the model's actual computation).
3. Implement `quantize_activations(activations, num_bits=8)` reusing
   base_techniques.py's quantize_symmetric, now applied per-forward-pass
   rather than once, offline, like weight quantization is.
4. Compare (via 11_evaluation) end-to-end inference quality with
   weight-only quantization (gptq_awq.py) vs weight+activation
   quantization (this module) at the same nominal bit width -- this
   should make the actual tradeoff (activation quantization gets you
   faster INFERENCE via int8 matmul, at additional quality risk)
   concrete rather than theoretical.
"""

raise NotImplementedError("Implement SmoothQuant-style activation quantization -- see module docstring")
