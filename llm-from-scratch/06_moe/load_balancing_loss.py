"""Auxiliary load-balancing loss: penalizes the router for sending too many
tokens to too few experts, which otherwise collapses to a small subset of
experts being used at all.
"""
raise NotImplementedError("TODO: implement the Switch Transformer-style auxiliary loss "
                           "(fraction of tokens routed to expert i * mean router prob for expert i)")
