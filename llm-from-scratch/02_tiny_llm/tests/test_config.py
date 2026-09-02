"""
test_config.py
================

Defines correct behavior for GPTConfig. These tests are written FIRST,
against the interface described in config.py's docstrings -- implement
GPTConfig until every test here passes.

Run with:
    pytest 02_tiny_llm/tests/test_config.py -v
"""

import pytest

from model.config import GPTConfig


class TestConstruction:
    def test_valid_config_constructs_successfully(self):
        cfg = GPTConfig(
            vocab_size=100, context_length=16, emb_dim=32,
            n_heads=4, n_layers=2, drop_rate=0.1, qkv_bias=False,
        )
        assert cfg.vocab_size == 100
        assert cfg.context_length == 16
        assert cfg.emb_dim == 32
        assert cfg.n_heads == 4
        assert cfg.n_layers == 2
        assert cfg.drop_rate == 0.1
        assert cfg.qkv_bias is False

    def test_qkv_bias_defaults_to_false(self):
        cfg = GPTConfig(vocab_size=100, context_length=16, emb_dim=32, n_heads=4, n_layers=2)
        assert cfg.qkv_bias is False

    def test_drop_rate_has_sensible_default(self):
        cfg = GPTConfig(vocab_size=100, context_length=16, emb_dim=32, n_heads=4, n_layers=2)
        assert 0.0 <= cfg.drop_rate < 1.0


class TestValidation:
    def test_rejects_non_divisible_emb_dim_and_heads(self):
        with pytest.raises(ValueError):
            GPTConfig(vocab_size=100, context_length=16, emb_dim=100, n_heads=3, n_layers=2)

    def test_error_message_includes_actual_values(self):
        with pytest.raises(ValueError, match="100"):
            GPTConfig(vocab_size=100, context_length=16, emb_dim=100, n_heads=3, n_layers=2)

    def test_rejects_zero_vocab_size(self):
        with pytest.raises(ValueError):
            GPTConfig(vocab_size=0, context_length=16, emb_dim=32, n_heads=4, n_layers=2)

    def test_rejects_negative_context_length(self):
        with pytest.raises(ValueError):
            GPTConfig(vocab_size=100, context_length=-1, emb_dim=32, n_heads=4, n_layers=2)

    def test_rejects_zero_n_layers(self):
        with pytest.raises(ValueError):
            GPTConfig(vocab_size=100, context_length=16, emb_dim=32, n_heads=4, n_layers=0)

    def test_rejects_drop_rate_of_exactly_one(self):
        with pytest.raises(ValueError):
            GPTConfig(vocab_size=100, context_length=16, emb_dim=32, n_heads=4, n_layers=2, drop_rate=1.0)

    def test_rejects_negative_drop_rate(self):
        with pytest.raises(ValueError):
            GPTConfig(vocab_size=100, context_length=16, emb_dim=32, n_heads=4, n_layers=2, drop_rate=-0.1)

    def test_accepts_drop_rate_of_exactly_zero(self):
        cfg = GPTConfig(vocab_size=100, context_length=16, emb_dim=32, n_heads=4, n_layers=2, drop_rate=0.0)
        assert cfg.drop_rate == 0.0


class TestHeadDim:
    def test_head_dim_computed_correctly(self):
        cfg = GPTConfig(vocab_size=100, context_length=16, emb_dim=32, n_heads=4, n_layers=2)
        assert cfg.head_dim == 8

    def test_head_dim_updates_if_config_values_differ(self):
        cfg_a = GPTConfig(vocab_size=100, context_length=16, emb_dim=64, n_heads=8, n_layers=2)
        cfg_b = GPTConfig(vocab_size=100, context_length=16, emb_dim=768, n_heads=12, n_layers=2)
        assert cfg_a.head_dim == 8
        assert cfg_b.head_dim == 64


class TestPresetConstructors:
    def test_gpt2_small_matches_published_architecture(self):
        cfg = GPTConfig.gpt2_small()
        assert cfg.vocab_size == 50257
        assert cfg.context_length == 1024
        assert cfg.emb_dim == 768
        assert cfg.n_heads == 12
        assert cfg.n_layers == 12
        assert cfg.qkv_bias is True  # real GPT-2 uses bias in qkv projections

    def test_gpt2_small_head_dim_is_64(self):
        # a well-known fact about GPT-2 small worth confirming your config gets right
        cfg = GPTConfig.gpt2_small()
        assert cfg.head_dim == 64

    def test_tiny_debug_is_small_and_fast(self):
        cfg = GPTConfig.tiny_debug()
        assert cfg.n_layers <= 4
        assert cfg.emb_dim <= 64
        assert cfg.context_length <= 32

    def test_tiny_debug_has_zero_dropout(self):
        # zero dropout matters for tests elsewhere that check exact
        # output values -- dropout's randomness would break those
        cfg = GPTConfig.tiny_debug()
        assert cfg.drop_rate == 0.0

    def test_tiny_debug_is_internally_valid(self):
        # tiny_debug's own values must still satisfy emb_dim % n_heads == 0 --
        # if this fails, it means tiny_debug() was implemented with
        # inconsistent values, not that validation itself is broken
        cfg = GPTConfig.tiny_debug()
        assert cfg.emb_dim % cfg.n_heads == 0
