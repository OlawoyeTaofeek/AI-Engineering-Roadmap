"""Reward model: pretrained/SFT backbone + a scalar head on the final token,
predicting a single reward score for a full (prompt, response) sequence.
"""
raise NotImplementedError("TODO: reuse GPTModel backbone, replace out_head with nn.Linear(emb_dim, 1)")
