"""PPO update step for RLHF: clipped surrogate policy loss, value function loss,
and a KL penalty against the original SFT model to prevent the policy from
drifting too far and "reward hacking."
"""
raise NotImplementedError("TODO: implement clipped surrogate objective, value loss, "
                           "and per-token KL penalty vs. a frozen reference (SFT) model")
