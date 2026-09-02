import torch
from model.attention import CausalAttention, MultiHeadAttention


def test_causal_attention_output_shape():
    x = torch.randn(2, 4, 16)
    attn = CausalAttention(d_in=16, d_out=8, context_length=4)
    out = attn(x)
    assert out.shape == (2, 4, 8)


def test_multihead_attention_output_shape():
    x = torch.randn(2, 4, 16)
    attn = MultiHeadAttention(d_in=16, d_out=16, context_length=4, num_heads=4)
    out = attn(x)
    assert out.shape == (2, 4, 16)


def test_multihead_requires_divisible_heads():
    try:
        MultiHeadAttention(d_in=16, d_out=15, context_length=4, num_heads=4)
        assert False, "expected AssertionError for non-divisible num_heads"
    except AssertionError:
        pass
