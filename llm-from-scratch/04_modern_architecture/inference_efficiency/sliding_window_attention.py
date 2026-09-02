"""Sliding window attention: each token attends only to the most recent W
tokens instead of the full context -- bounds compute/memory per step,
used in Mistral and similar models.
"""
raise NotImplementedError("TODO: extend the causal mask to also mask out "
                           "positions further than `window_size` in the past")
