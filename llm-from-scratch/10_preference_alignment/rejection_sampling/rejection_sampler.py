"""
rejection_sampler.py
========================

Rejection sampling: the simplest preference-alignment technique. For
each prompt, generate N candidate responses from the current model,
score all N with the trained reward model, keep only the BEST one, and
fine-tune the model on that (prompt, best_response) pair -- essentially
using the reward model as a filter rather than as a training signal
inside an RL loop (unlike PPO).

WHY START HERE (before DPO or PPO)
---------------------------------------
Rejection sampling requires no new training algorithm at all -- it's
just generation (already built: GPTForCausalLM.generate() /
sample_next_token from 02_tiny_llm) + scoring (reward_model.py in this
folder) + a normal SFT-style fine-tuning step (09_instruction_finetuning_sft's
existing training loop, reused unchanged). This makes it the fastest way
to see SOME preference-alignment effect before tackling DPO's loss
function or PPO's full RL machinery.

TODO
-------
1. Implement `generate_candidates(model, prompt, n, sampling_kwargs)` --
   calls model.generate() n times per prompt with sampling enabled
   (temperature/top-k/top-p from 02_tiny_llm/sampling.py, NOT greedy --
   you need diverse candidates, which greedy decoding can't give you
   since it's deterministic).
2. Implement `select_best_candidate(candidates, reward_model)` -- scores
   every candidate, returns the highest-scoring one.
3. Implement `build_rejection_sampled_dataset(model, reward_model,
   prompts, n_candidates_per_prompt)` -- orchestrates 1+2 across many
   prompts, returning a dataset of (prompt, best_response) pairs in the
   SAME format chat_templates.py (08_post_training_datasets) expects.
4. Fine-tune on the resulting dataset using the EXISTING training loop
   from 09_instruction_finetuning_sft/sft_train.py -- no new training
   code needed, just new data.
"""

raise NotImplementedError("Implement rejection sampling pipeline -- see module docstring")
