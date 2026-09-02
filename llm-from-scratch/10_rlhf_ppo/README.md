# 10 - RLHF via PPO

Use the reward model from `09_reward_modeling/` to fine-tune the SFT model
with Proximal Policy Optimization -- the "historical" RLHF approach (as used
in InstructGPT). See `notes_ppo_vs_dpo.md` for why DPO is now often used
instead in practice.

```
ppo_trainer.py            -- PPO update step: clipped surrogate objective + KL penalty vs SFT model
rollout.py                   -- generate responses with the current policy, score with the reward model
notes_ppo_vs_dpo.md             -- PPO's complexity (separate reward model, RL rollout loop) vs DPO's
                                     simpler direct-preference-optimization objective
```
