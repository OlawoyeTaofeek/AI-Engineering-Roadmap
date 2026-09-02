"""
model_merging.py
====================

Combines multiple fine-tuned versions of the SAME base model directly at
the WEIGHT level (no additional training), producing a single merged
model that often exhibits properties of both parents.

WHY THIS WORKS AT ALL (surprising at first)
-------------------------------------------------
If model_a and model_b were both fine-tuned starting from the SAME
pretrained checkpoint, their weights stay relatively close to each other
in parameter space (fine-tuning is typically a small perturbation, not a
wholesale change). Simple weight averaging often works reasonably well
BECAUSE of this shared starting point -- it would NOT work merging two
models trained from scratch with different random initializations, since
their weights would encode similar functions in unrelated, incompatible
coordinate systems.

TODO
-------
1. Implement `linear_merge(model_a, model_b, weight=0.5)` -- simplest
   method: merged_weights = weight * model_a_weights + (1-weight) *
   model_b_weights, applied parameter-by-parameter. Requires model_a and
   model_b to have IDENTICAL architecture (same GPTConfig).
2. Implement `task_arithmetic_merge(base_model, finetuned_model_a,
   finetuned_model_b, scale=1.0)` -- computes each fine-tuned model's
   "task vector" (finetuned_weights - base_weights), adds both task
   vectors to the base model's weights instead of averaging the
   fine-tuned weights directly. This tends to work better than naive
   averaging because it isolates what EACH fine-tuning run actually
   changed, rather than blending toward the base model's original
   (pre-fine-tuning) behavior.
3. Compare both merge methods against each individual model using
   11_evaluation/automated_benchmarks.py -- does the merged model
   actually combine both models' strengths, or does it just perform
   worse than either parent? This is genuinely an open empirical
   question per merge, not something to assume works well in advance.
"""

raise NotImplementedError("Implement weight merging methods -- see module docstring")
