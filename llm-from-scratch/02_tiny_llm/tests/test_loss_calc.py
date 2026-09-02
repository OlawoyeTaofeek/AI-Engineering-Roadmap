import torch
import torch.nn.functional as F


def test_manual_nll_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 10)
    targets = torch.tensor([[1, 4, 7]])

    probas = torch.softmax(logits, dim=-1)
    target_probas = torch.gather(probas, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    manual_loss = -torch.log(target_probas).mean()

    builtin_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    assert torch.allclose(manual_loss, builtin_loss, atol=1e-5)
