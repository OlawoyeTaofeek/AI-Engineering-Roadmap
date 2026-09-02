"""
test_time_compute.py
========================

Techniques that improve output quality by spending MORE compute at
INFERENCE time (multiple generations, self-checking, search) rather than
only investing more compute in training -- the idea behind"reasoning"
models generating long chains of thought, and behind techniques like
self-consistency and best-of-N sampling.

TODO
-------
1. Implement `self_consistency(model, prompt, n_samples, tokenizer)` --
   generate n_samples independent completions using SAMPLING (not
   greedy -- reuse sample_next_token from 02_tiny_llm/sampling.py, which
   you'll have built by this point), then return the most COMMON answer
   across all samples (majority vote) -- useful specifically for tasks
   with a single correct final answer (e.g. math problems), where
   different sampled reasoning paths might still converge on the same
   correct result more often than any single greedy attempt gets it
   right.
2. Implement `best_of_n(model, prompt, n_samples, reward_model,
   tokenizer)` -- generate n_samples completions, score each with the
   reward model from 10_preference_alignment/reward_model.py, return the
   highest-scoring one. Note this is architecturally the SAME pattern as
   10_preference_alignment/rejection_sampling/rejection_sampler.py's
   generate_candidates + select_best_candidate -- rejection sampling
   USES this exact technique to build TRAINING data, whereas here it's
   applied directly at INFERENCE time with no further training step.
3. Implement `tree_search_generation(model, prompt, branching_factor,
   depth, scoring_fn)` -- a simple beam-search-style generation that
   explores multiple continuation branches at each step rather than
   committing to one token at a time, keeping only the top-scoring
   branches at each depth -- a lightweight version of the search-based
   approaches used in more advanced reasoning-focused generation setups.
4. Compare all three (plus plain greedy and plain sampling) via
   11_evaluation on a small set of problems with objectively checkable
   answers (e.g. simple arithmetic word problems) -- test-time compute
   techniques should show a measurable accuracy improvement at the cost
   of more inference-time compute per answer; quantify that tradeoff
   directly rather than assuming it.
"""

raise NotImplementedError("Implement self-consistency, best-of-n, and tree search generation -- see module docstring")
