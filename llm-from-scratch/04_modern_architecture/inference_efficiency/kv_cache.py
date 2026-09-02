"""KV-cache: store computed key/value tensors across generation steps so each
new token only computes attention against a growing cache, instead of
recomputing the full sequence's k/v from scratch on every step.
"""
raise NotImplementedError("TODO: cache k, v per layer; on each new token, "
                           "concat new k/v onto the cache instead of recomputing")
