# 09 - Reward Modeling

Train a reward model on human (or synthetic) preference pairs: given a prompt
and two candidate responses, predict which one is preferred. This reward model
is what PPO optimizes against in `10_rlhf_ppo/`.

```
reward_model.py           -- base model + scalar reward head
preference_dataset.py       -- (prompt, chosen, rejected) pair loader
train_reward_model.py         -- Bradley-Terry pairwise loss training loop
```
