# PPO vs DPO

PPO (used in InstructGPT-style RLHF):
- Requires a separately trained reward model
- Requires an online rollout loop (generate -> score -> update), which is
  slow and has many moving pieces (value function, KL penalty, clipping)
- More faithful to the original "RL from human feedback" formulation

DPO (Direct Preference Optimization):
- Skips the reward model and the RL rollout loop entirely
- Directly optimizes the policy on preference pairs with a closed-form loss
  derived from the same Bradley-Terry assumption PPO's reward model uses
- Much simpler to implement and more stable to train, at the cost of being
  slightly less flexible than a full RL loop

_TODO: implement a DPO variant alongside PPO here once both reward modeling
and PPO are working, and compare final policy quality._
