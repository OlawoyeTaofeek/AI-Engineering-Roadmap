"""Train the reward model with the Bradley-Terry pairwise loss:
    loss = -log(sigmoid(reward(chosen) - reward(rejected)))
"""
raise NotImplementedError("TODO: forward both chosen and rejected through the reward model, "
                           "apply -F.logsigmoid(r_chosen - r_rejected).mean()")
