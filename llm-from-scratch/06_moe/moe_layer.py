"""Mixture-of-Experts FFN layer: route each token to its top-k experts,
combine their outputs weighted by router probability.
"""
raise NotImplementedError("TODO: nn.ModuleList of FeedForward experts + Router; "
                           "for each token, run only its selected experts")
