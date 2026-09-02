"""Rotary Position Embedding (RoPE). Rotates query/key vectors by an angle
proportional to position, applied inside attention (never to values), so that
q . k depends on relative position rather than absolute position.
"""
import torch


def precompute_rope_params(head_dim, context_length, theta_base=10000.0):
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(context_length)
    angles = positions[:, None] * inv_freq[None, :]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    # x: (batch, num_heads, seq_len, head_dim)
    seq_len = x.shape[2]
    cos, sin = cos[:seq_len], sin[:seq_len]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)
