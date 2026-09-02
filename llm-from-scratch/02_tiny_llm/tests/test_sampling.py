"""
test_sampling.py
==================

Defines correct behavior for sampling.py. Implement sampling.py until
every test here passes.

Run with:
    pytest 02_tiny_llm/tests/test_sampling.py -v
"""

import torch
import pytest

from sampling import apply_temperature, filter_top_k, filter_top_p, sample_next_token


class TestApplyTemperature:
    def test_temperature_one_is_unchanged(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = apply_temperature(logits, temperature=1.0)
        assert torch.allclose(result, logits)

    def test_low_temperature_increases_magnitude_spread(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = apply_temperature(logits, temperature=0.5)
        # dividing by <1 should INCREASE the spread between values
        assert (result.max() - result.min()) > (logits.max() - logits.min())

    def test_high_temperature_decreases_magnitude_spread(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = apply_temperature(logits, temperature=2.0)
        assert (result.max() - result.min()) < (logits.max() - logits.min())


class TestFilterTopK:
    def test_keeps_exactly_k_finite_values(self):
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        result = filter_top_k(logits, k=2)
        finite_count = torch.isfinite(result).sum().item()
        assert finite_count == 2

    def test_keeps_the_correct_top_values(self):
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        result = filter_top_k(logits, k=2)
        # the two highest values (5.0 and 4.0) should survive, at their
        # ORIGINAL positions (index 1 and index 4)
        assert result[1].item() == 5.0
        assert result[4].item() == 4.0

    def test_excluded_positions_are_negative_infinity(self):
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        result = filter_top_k(logits, k=2)
        assert result[0].item() == float("-inf")
        assert result[2].item() == float("-inf")
        assert result[3].item() == float("-inf")

    def test_k_equals_vocab_size_keeps_everything(self):
        logits = torch.tensor([1.0, 5.0, 3.0])
        result = filter_top_k(logits, k=3)
        assert torch.isfinite(result).all()


class TestFilterTopP:
    def test_high_p_keeps_more_tokens_than_low_p(self):
        logits = torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0])  # one dominant token
        result_low_p = filter_top_p(logits.clone(), p=0.5)
        result_high_p = filter_top_p(logits.clone(), p=0.99)
        low_p_count = torch.isfinite(result_low_p).sum().item()
        high_p_count = torch.isfinite(result_high_p).sum().item()
        assert high_p_count >= low_p_count

    def test_always_keeps_at_least_one_token(self):
        logits = torch.tensor([10.0, 1.0, 1.0, 1.0])
        result = filter_top_p(logits, p=0.01)  # extremely restrictive
        finite_count = torch.isfinite(result).sum().item()
        assert finite_count >= 1

    def test_p_equals_one_keeps_everything(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = filter_top_p(logits, p=1.0)
        assert torch.isfinite(result).all()


class TestSampleNextToken:
    def test_output_shape(self):
        logits = torch.randn(4, 100)  # batch=4, vocab_size=100
        result = sample_next_token(logits)
        assert result.shape == (4, 1)

    def test_output_is_valid_token_ids(self):
        logits = torch.randn(4, 100)
        result = sample_next_token(logits)
        assert (result >= 0).all()
        assert (result < 100).all()

    def test_top_k_of_one_is_effectively_greedy(self):
        # with top_k=1, only the single highest-probability token can be
        # sampled -- so the output should match argmax exactly, every time
        torch.manual_seed(0)
        logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
        expected = torch.argmax(logits, dim=-1, keepdim=True)
        for _ in range(10):  # run several times -- should be deterministic
            result = sample_next_token(logits, top_k=1)
            assert torch.equal(result, expected)

    def test_temperature_and_top_k_combine_without_error(self):
        logits = torch.randn(2, 50)
        result = sample_next_token(logits, temperature=0.8, top_k=10, top_p=0.9)
        assert result.shape == (2, 1)
