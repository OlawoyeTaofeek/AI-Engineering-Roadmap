import torch
from model.attention import CausalAttention


def test_first_token_only_attends_to_itself():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 8)
    attn = CausalAttention(d_in=8, d_out=8, context_length=3)

    # manually reproduce attn_weights to check the mask, since forward() only returns context vecs
    q, k = attn.W_query(x), attn.W_key(x)
    scores = q @ k.transpose(-2, -1)
    scores.masked_fill_(attn.mask.bool()[:3, :3], -torch.inf)
    weights = torch.softmax(scores / k.shape[-1] ** 0.5, dim=-1)

    assert torch.allclose(weights[0, 0, 1:], torch.zeros(2), atol=1e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 3), atol=1e-5)
