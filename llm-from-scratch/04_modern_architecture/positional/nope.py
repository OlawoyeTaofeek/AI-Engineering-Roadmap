"""NoPE (No Positional Embeddings): omit explicit positional encoding entirely
and rely on causal masking alone -- the model can, in principle, infer relative
order from the autoregressive structure itself. Useful as an ablation baseline
against RoPE / learned absolute embeddings.
"""

def apply_nope(x):
    return x  # literally a no-op -- the point is what you DON'T add
