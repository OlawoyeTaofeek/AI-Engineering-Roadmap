import torch
from model.layer_norm import LayerNorm


def test_layernorm_output_mean_and_std():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16) * 5 + 3  # arbitrary scale/offset
    norm = LayerNorm(emb_dim=16)
    out = norm(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 4), atol=1e-5)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(2, 4), atol=1e-4)
