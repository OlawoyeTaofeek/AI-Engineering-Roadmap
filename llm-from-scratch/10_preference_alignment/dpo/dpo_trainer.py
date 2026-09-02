"""
dpo_trainer.py
==================

Direct Preference Optimization (DPO): trains directly on (prompt,
chosen, rejected) preference triples, without a separate reward model
or an RL rollout loop -- see notes_ppo_vs_dpo.md in the parent folder
for the full comparison.

THE DPO LOSS (derived from the same Bradley-Terry assumption PPO's
reward model uses, but applied directly to the POLICY instead of a
separate reward model)
-------------------------------------------------------------------------
    loss = -log(sigmoid(
        beta * (log_ratio_chosen - log_ratio_rejected)
    ))

    where log_ratio_x = log P_policy(x | prompt) - log P_reference(x | prompt)

In words: for the CHOSEN response, the policy's log-probability relative
to the frozen reference model (your SFT checkpoint, not updated during
DPO) should INCREASE; for the REJECTED response, it should DECREASE,
relative to each other. beta controls how strongly to enforce this
(higher beta = stronger preference signal, but risks drifting further
from the reference model's behavior).

TODO
-------
1. Implement `compute_log_probs(model, prompt, response)` -- runs the
   model forward, returns the SUM of log-probabilities the model assigns
   to each token in `response` given `prompt` as context. Reuse the same
   "gather target token probabilities" pattern from 02_tiny_llm's loss
   computation (torch.gather over softmax output, or directly from
   F.cross_entropy with reduction='none', summed and negated).
2. Implement `dpo_loss(policy_model, reference_model, prompt, chosen,
   rejected, beta)` following the formula above. reference_model should
   be a FROZEN copy of your SFT checkpoint (requires_grad=False on all
   its parameters) -- only policy_model's weights get updated.
3. Implement a training loop reusing the same four-step update pattern
   (zero_grad, backward, clip, step) from every training loop already
   built in this repo -- the only new part is dpo_loss() itself; the
   surrounding loop mechanics are unchanged.
"""

raise NotImplementedError("Implement DPO loss and training loop -- see module docstring")
