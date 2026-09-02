"""
human_evaluation.py
=======================

Tooling for collecting human A/B preference judgments between two model
outputs -- the same KIND of data (prompt, response_a, response_b,
preferred) that feeds reward_model.py in 10_preference_alignment/, but
here used purely for EVALUATION (measuring which of two model versions
people prefer), not training.

TODO
-------
1. Implement `generate_comparison_pairs(model_a, model_b, prompts)` --
   for each prompt, generate one response from each model, return
   (prompt, response_a, response_b) triples with a/b order RANDOMIZED
   per pair (avoids raters developing a positional bias toward always
   preferring "the first option").
2. Implement a minimal CLI or simple local web form (a small Gradio app,
   same tool already used for 03_attention_variants/playground/app.py)
   presenting each pair blind (rater doesn't know which is model_a vs
   model_b) and recording which one they preferred.
3. Implement `aggregate_human_preferences(results) -> dict` -- computes
   win rate for each model, with a basic confidence interval given the
   sample size (a simple binomial confidence interval is sufficient at
   this repo's scale).
"""

raise NotImplementedError("Implement human evaluation collection tooling -- see module docstring")
