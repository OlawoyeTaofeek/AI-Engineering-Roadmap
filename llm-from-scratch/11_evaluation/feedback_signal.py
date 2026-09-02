"""
feedback_signal.py
======================

Aggregates automated_benchmarks.py, human_evaluation.py, and
model_based_evaluation.py results into a single tracked signal per model
checkpoint, logged to MLflow -- reusing this repo's existing experiment-
tracking pattern (every train.py across every stage already logs to
MLflow; this extends that same tracking to EVALUATION runs, not just
training runs).

TODO
-------
1. Implement `log_evaluation_run(model_checkpoint_name, benchmark_results,
   human_eval_results, judge_results)` -- opens an MLflow run tagged with
   the checkpoint being evaluated, logs all metrics together so a single
   MLflow experiment view lets you compare perplexity, benchmark
   accuracy, human win rate, and judge scores side by side across every
   checkpoint you've evaluated.
2. Implement `compare_checkpoints(checkpoint_names) -> pandas.DataFrame`
   -- pulls logged metrics for multiple checkpoints from MLflow and
   returns a comparison table, useful for deciding e.g. "did DPO
   actually improve over the SFT-only checkpoint" or "did quantization
   (13_quantization/) hurt quality enough to matter."
"""

raise NotImplementedError("Implement evaluation-result aggregation and MLflow logging -- see module docstring")
