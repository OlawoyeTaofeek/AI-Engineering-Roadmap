"""RMSNorm: rescales by root-mean-square, skips mean-subtraction and the
learnable shift parameter that LayerNorm has. See 00_explanation/normalization.md
for the full derivation and why this became the default in modern LLMs.
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale
