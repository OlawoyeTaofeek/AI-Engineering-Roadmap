# 10 - Preference Alignment

Everything that aligns a model to human preferences, AFTER SFT (09).
Three techniques, from simplest to most complex:

```
rejection_sampling/     -- simplest: generate N responses, keep only the
                              reward model's top pick, fine-tune on that
reward_model.py            -- (moved from the old 09_reward_modeling/) trains
preference_dataset.py         the reward model used by BOTH rejection sampling
train_reward_model.py          and PPO
dpo/                              -- Direct Preference Optimization: skips the
                                      reward model + RL rollout loop entirely
ppo/                                -- Proximal Policy Optimization: the full
                                        RL-based approach (InstructGPT-style)
notes_ppo_vs_dpo.md                   -- comparison, see this repo's earlier
                                          conversation on why DPO is now often
                                          preferred in practice
```

## Suggested build order

1. `reward_model.py` + `train_reward_model.py` first -- needed by both
   rejection sampling and PPO.
2. `rejection_sampling/` -- the simplest technique, good first working
   preference-alignment result.
3. `dpo/` -- no reward model rollout loop needed, but still needs
   PREFERENCE data (chosen/rejected pairs) -- can reuse
   `preference_dataset.py`.
4. `ppo/` -- most complex, do this last once the simpler techniques work
   and you want to compare all three head-to-head.
