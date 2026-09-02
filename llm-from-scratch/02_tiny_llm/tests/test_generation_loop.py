import torch
from model.gpt_model import GPTModel
from generate import generate_text_simple


def test_generation_appends_correct_number_of_tokens():
    cfg = {"vocab_size": 50, "context_length": 8, "emb_dim": 16,
           "n_heads": 2, "n_layers": 2, "drop_rate": 0.0, "qkv_bias": False}
    model = GPTModel(cfg)
    model.eval()

    start = torch.tensor([[1, 2, 3]])
    out = generate_text_simple(model, start, max_new_tokens=5, context_size=cfg["context_length"])
    assert out.shape == (1, 3 + 5)
