"""Rolling buffer KV-cache: fixed-size circular buffer for the KV-cache,
paired with sliding window attention, so cache memory stays bounded even
for arbitrarily long generation.
"""
raise NotImplementedError("TODO: circular buffer of size window_size, "
                           "overwrite oldest entries as new tokens arrive")
