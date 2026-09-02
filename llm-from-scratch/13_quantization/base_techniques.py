"""
base_techniques.py
======================

Fundamentals of quantization, implemented from scratch on a single
weight tensor before applying to a full model.

THE CORE IDEA
-----------------
A float32 weight uses 32 bits per value. Quantization maps each float
value to a much smaller integer range (e.g. int8: -128 to 127), storing
a SCALE factor (and optionally a ZERO POINT) that lets you approximately
recover the original float value later:

    quantized_value = round(float_value / scale) + zero_point
    dequantized_value = (quantized_value - zero_point) * scale

Symmetric quantization: zero_point = 0, scale chosen so the range
[-max_abs_value, max_abs_value] maps to the full integer range. Simpler,
slightly less precise for asymmetric distributions.

Asymmetric quantization: zero_point != 0, scale AND zero_point chosen so
[min_value, max_value] maps to the full integer range exactly. More
precise for weights that aren't centered around zero, at the cost of
needing to store/apply zero_point too.

Per-tensor vs per-channel: per-tensor uses ONE scale for an entire
weight matrix; per-channel uses a SEPARATE scale per output channel
(row/column, depending on convention). Per-channel is more accurate
(different channels can have very different value ranges) at the cost
of more scale factors to store.

TODO
-------
1. Implement `quantize_symmetric(tensor, num_bits=8) -> (quantized, scale)`
2. Implement `dequantize_symmetric(quantized, scale) -> tensor`
3. Implement `quantize_asymmetric(tensor, num_bits=8) -> (quantized, scale, zero_point)`
4. Implement `dequantize_asymmetric(quantized, scale, zero_point) -> tensor`
5. Implement `measure_quantization_error(original, dequantized) -> float`
   (e.g. mean squared error) -- use this to compare symmetric vs
   asymmetric, and per-tensor vs per-channel, on a real trained weight
   matrix from your 02_tiny_llm model.
6. Implement `quantize_model_weights(model, num_bits=8)` -- applies
   quantization to every linear layer's weight in a full GPTForCausalLM,
   and a matching `dequantize_model_weights` -- then run
   11_evaluation/automated_benchmarks.py's perplexity check on both the
   original and quantized model to measure the REAL quality cost, not
   just a synthetic error metric.
"""

raise NotImplementedError("Implement quantize/dequantize functions -- see module docstring")
