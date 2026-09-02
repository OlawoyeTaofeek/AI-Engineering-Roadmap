"""
test_causal_lm.py
===================

Defines correct behavior for GPTForCausalLM. Implement causal_lm.py
until every test here passes.

Run with:
    pytest 02_tiny_llm/tests/test_causal_lm.py -v

NOTE: these tests use GPTConfig.tiny_debug() -- meaning test_config.py
must already be passing before these tests can meaningfully run (a
NotImplementedError from GPTConfig will surface here as a fixture
error, not a real failure of causal_lm.py itself). Implement config.py
first.
"""

import torch
import pytest

from model.config import GPTConfig
from model.causal_lm import GPTForCausalLM, CausalLMOutput


@pytest.fixture
def config():
    return GPTConfig.tiny_debug()


@pytest.fixture
def model(config):
    m = GPTForCausalLM(config)
    m.eval()  # deterministic -- tiny_debug() already sets drop_rate=0.0,
              # but .eval() is still good practice for tests checking exact shapes/values
    return m


class TestOutputShapes:
    def test_logits_shape_without_labels(self, model, config):
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        output = model(input_ids)
        assert output.logits.shape == (2, 5, config.vocab_size)

    def test_loss_is_none_without_labels(self, model, config):
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        output = model(input_ids)
        assert output.loss is None

    def test_loss_is_populated_with_labels(self, model, config):
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        labels = torch.randint(0, config.vocab_size, (2, 5))
        output = model(input_ids, labels=labels)
        assert output.loss is not None

    def test_loss_is_scalar(self, model, config):
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        labels = torch.randint(0, config.vocab_size, (2, 5))
        output = model(input_ids, labels=labels)
        assert output.loss.dim() == 0  # scalar tensor, not per-token

    def test_output_type_is_causal_lm_output(self, model, config):
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        output = model(input_ids)
        assert isinstance(output, CausalLMOutput)


class TestLossCorrectness:
    def test_loss_matches_manual_cross_entropy(self, model, config):
        """The loss computed inside forward() must exactly match what
        calling F.cross_entropy manually on the same logits/labels gives
        -- this is the same equivalence check we ran manually earlier in
        this conversation when verifying the 6-step loss derivation."""
        import torch.nn.functional as F

        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        labels = torch.randint(0, config.vocab_size, (2, 5))

        output = model(input_ids, labels=labels)

        with torch.no_grad():
            manual_logits = model(input_ids).logits
            manual_loss = F.cross_entropy(
                manual_logits.flatten(0, 1), labels.flatten()
            )

        assert torch.allclose(output.loss, manual_loss, atol=1e-5)

    def test_loss_is_higher_for_random_untrained_model_near_ln_vocab_size(self, model, config):
        """Sanity check from earlier in this conversation: an untrained
        model's loss should sit close to ln(vocab_size), confirming loss
        computation is wired correctly before any training happens."""
        import math

        input_ids = torch.randint(0, config.vocab_size, (4, 10))
        labels = torch.randint(0, config.vocab_size, (4, 10))
        output = model(input_ids, labels=labels)

        expected_baseline = math.log(config.vocab_size)
        # generous tolerance -- this is a sanity range check, not an exact match
        assert abs(output.loss.item() - expected_baseline) < 2.0

    def test_backward_pass_works(self, model, config):
        """Confirms the loss is actually connected to the model's
        parameters via autograd -- i.e. .backward() populates gradients,
        proving this loss can actually be used to train the model."""
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        labels = torch.randint(0, config.vocab_size, (2, 5))

        output = model(input_ids, labels=labels)
        output.loss.backward()

        # at least one parameter should have received a non-None, non-zero gradient
        found_nonzero_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        assert found_nonzero_grad


class TestGenerate:
    def test_generate_appends_correct_number_of_tokens(self, model, config):
        input_ids = torch.randint(0, config.vocab_size, (1, 3))
        output = model.generate(input_ids, max_new_tokens=5)
        assert output.shape == (1, 3 + 5)

    def test_generate_preserves_original_tokens(self, model, config):
        input_ids = torch.randint(0, config.vocab_size, (1, 3))
        output = model.generate(input_ids, max_new_tokens=5)
        assert torch.equal(output[:, :3], input_ids)

    def test_generate_respects_context_size_cropping(self, model, config):
        # generate well past context_length to confirm the sliding-window
        # crop (idx_cond = idx[:, -context_size:]) doesn't crash
        input_ids = torch.randint(0, config.vocab_size, (1, 3))
        output = model.generate(
            input_ids, max_new_tokens=config.context_length + 5,
            context_size=config.context_length,
        )
        assert output.shape == (1, 3 + config.context_length + 5)

    def test_generate_does_not_require_grad(self, model, config):
        # @torch.no_grad() should mean the output has no grad_fn
        input_ids = torch.randint(0, config.vocab_size, (1, 3))
        output = model.generate(input_ids, max_new_tokens=3)
        assert output.requires_grad is False
