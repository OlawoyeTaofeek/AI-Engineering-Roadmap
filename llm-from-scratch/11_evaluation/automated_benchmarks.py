"""
automated_benchmarks.py
===========================

Standard, repeatable benchmark evaluation: perplexity on a FIXED
held-out set, plus a small multiple-choice task benchmark.

WHY A SEPARATE, FIXED EVAL SET MATTERS
-------------------------------------------
Your training loops already compute train/val loss during training
(calc_loss_loader, used throughout 02_tiny_llm). That's necessary but
not sufficient: the val set there is drawn from the SAME distribution/
run as training data, and changes if you change your data pipeline.
A benchmark eval set here should be FIXED and version-controlled, so
perplexity numbers are comparable ACROSS different training runs,
different model versions, and different architecture choices (e.g. the
ablations from 04_modern_architecture/ablations.md) -- you're not
comparing apples to oranges because the eval set shifted underneath you.

TODO
-------
1. Implement `load_fixed_eval_set(path)` -- loads a held-out text file
   that NEVER changes across experiments (check it into the repo, small
   enough to be practical -- a few hundred KB of diverse text is
   plenty).
2. Implement `compute_perplexity(model, eval_set, tokenizer, device)` --
   reuses calc_loss_loader's logic from 02_tiny_llm, then converts to
   perplexity via torch.exp(loss) (exactly as covered in this
   conversation's perplexity explanation).
3. Implement `multiple_choice_eval(model, questions, tokenizer, device)`
   -- for a small hand-written or downloaded multiple-choice benchmark
   (a tiny MMLU-style subset is fine at this repo's scale), compute
   which answer choice the model assigns the HIGHEST probability to
   (compare log-probability of each candidate answer's tokens, similar
   to compute_log_probs() in 10_preference_alignment/dpo/), and check
   whether that matches the correct answer.
4. Implement `run_full_benchmark_suite(model, tokenizer, device) -> dict`
   orchestrating both, returning a dict of metric_name -> value, ready
   to log to MLflow (see feedback_signal.py in this same folder).
"""

raise NotImplementedError("Implement benchmark evaluation functions -- see module docstring")
