"""
gptq_awq.py
===============

GPTQ and AWQ: calibration-based post-training quantization methods that
choose quantization parameters using a small set of REAL sample data,
rather than base_techniques.py's naive round-to-nearest approach --
achieving much better quality at very low bit widths (3-4 bit) than
naive quantization can.

CONCEPTUAL DIFFERENCE FROM base_techniques.py
--------------------------------------------------
Naive quantization (round-to-nearest) treats every weight independently
-- it doesn't consider how quantization ERROR in one weight interacts
with others, or which weights actually matter most for the model's real
outputs.

GPTQ: processes weights column-by-column, and after quantizing each
column, ADJUSTS the remaining not-yet-quantized columns to compensate
for the error just introduced -- using second-order (Hessian-based)
information computed from calibration data, so errors don't just
accumulate independently.

AWQ (Activation-aware Weight Quantization): observes that a SMALL
fraction of weights (correlated with high-activation-magnitude input
channels) matter disproportionately for output quality. AWQ identifies
these salient weight channels using calibration data and PROTECTS them
(keeps higher precision, or scales them favorably) while quantizing the
rest more aggressively.

TODO -- given the complexity of a faithful from-scratch GPTQ/AWQ
implementation, this module is scoped as a STUDY + comparison exercise
rather than a full reimplementation:
-------------------------------------------------------------------------
1. Install and run AutoGPTQ or AutoAWQ (existing, well-tested open-source
   implementations) on your trained 07_advanced_llm_from_scratch model,
   once exported to HuggingFace format (see the deployment conversation).
2. Implement `compare_quantization_methods(model, methods=["naive",
   "gptq", "awq"], eval_fn)` -- runs 11_evaluation/automated_benchmarks.py
   perplexity against each quantized version, producing a table showing
   the REAL quality-vs-compression tradeoff of naive quantization
   (base_techniques.py) vs GPTQ vs AWQ on your own model.
3. OPTIONAL, more advanced: implement a simplified version of GPTQ's
   column-wise error-compensation update yourself, on a single linear
   layer, to build direct intuition for the Hessian-based correction
   step -- treat this as a stretch goal, not a requirement for this
   module to be "done."
"""

raise NotImplementedError("Set up GPTQ/AWQ comparison pipeline -- see module docstring")
