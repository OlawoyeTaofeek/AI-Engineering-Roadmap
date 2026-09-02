"""SwiGLU feed-forward block: gated activation (SiLU(gate) * up) before
projecting back down. hidden_dim is typically ~8/3 * emb_dim (not 4x) to keep
compute comparable to a standard GELU-MLP despite the extra projection.
"""
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
