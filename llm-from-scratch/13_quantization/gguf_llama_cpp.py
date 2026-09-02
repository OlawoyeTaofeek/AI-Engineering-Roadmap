"""
gguf_llama_cpp.py
=====================

Explains and implements the GGUF format's quantization scheme
specifically -- what "Q4_K_M" and similar names actually mean, connecting
directly to the `llama-quantize` command used in this repo's deployment
conversation (12_serving, and the Ollama discussion earlier).

BACKGROUND
--------------
GGUF quantization types follow a naming convention:
    Q<bits>_<variant>
    e.g. Q4_0, Q4_K, Q4_K_M, Q5_K_S, Q8_0

- The number (4, 5, 8...) is roughly how many bits per weight, though
  "K-quants" (Q4_K, Q5_K, etc.) use a more sophisticated BLOCK-WISE
  scheme (see below) that achieves better quality than a naive uniform
  N-bit quantization would.
- _0 / _1 suffixes: older, simpler quantization variants (symmetric vs
  asymmetric, roughly).
- _K suffixes: "K-quants" -- weights are grouped into small blocks (e.g.
  32 or 256 values), each block gets ITS OWN scale factor, computed to
  minimize error for that specific block, rather than one scale for the
  entire tensor. This is essentially fine-grained per-channel
  quantization (base_techniques.py's per-channel idea) taken further --
  down to per-BLOCK rather than per-channel.
- _S / _M / _L suffixes on K-quants (Small/Medium/Large): different
  bit-allocation strategies -- e.g. Q4_K_M keeps SOME layers (typically
  attention output and FFN down-projection, which are more sensitive to
  quantization error) at higher precision than others, trading a bit of
  size for meaningfully better quality than uniform Q4_K_S.

TODO
-------
1. Implement `block_wise_quantize(tensor, block_size=32, num_bits=4)` --
   extends base_techniques.py's per-tensor scheme to per-BLOCK: split
   the tensor into chunks of block_size values, quantize each chunk
   independently with its own scale.
2. Compare block_wise_quantize's error (using
   base_techniques.measure_quantization_error) against plain per-tensor
   quantization at the same bit width, on a real weight matrix -- this
   should make concrete WHY K-quants outperform naive uniform
   quantization at the same nominal bit count.
3. Document (as a comment or in this docstring, extend as needed) the
   ACTUAL command-line workflow for producing a real GGUF file from your
   trained model, referencing the convert_hf_to_gguf.py +
   llama-quantize commands already covered in this repo's deployment
   conversation -- this module's from-scratch implementation is for
   UNDERSTANDING the mechanism; producing a real deployable GGUF file
   should still use llama.cpp's actual tooling, not this reimplementation.
"""

raise NotImplementedError("Implement block-wise quantization -- see module docstring")
