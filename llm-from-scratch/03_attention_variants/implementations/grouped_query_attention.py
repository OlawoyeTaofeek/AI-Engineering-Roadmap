"""Grouped-Query Attention (GQA): fewer key/value heads than query heads,
shared across groups of query heads. Reduces KV-cache size significantly
at inference time vs. standard multi-head attention, at a small quality cost.
"""
import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, num_kv_groups, dropout=0.0):
        super().__init__()
        assert d_out % num_heads == 0
        assert num_heads % num_kv_groups == 0
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        raise NotImplementedError("TODO: project q with num_heads, k/v with num_kv_groups, "
                                   "repeat_interleave k/v to match num_heads before the dot product")
