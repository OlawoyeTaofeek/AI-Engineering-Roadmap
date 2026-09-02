"""
test_bpe_tokenizer.py
=======================

Defines correct behavior for BPETokenizer. Implement bpe_tokenizer.py
until every test here passes.

Run with:
    pytest 02_tiny_llm/tests/test_bpe_tokenizer.py -v
"""

import pytest

from tokenizer.bpe_tokenizer import BPETokenizer


SIMPLE_CORPUS = "low low low lower lower newest newest newest widest widest"


class TestTraining:
    def test_vocab_reaches_target_size_or_stops_early(self):
        tok = BPETokenizer()
        tok.train(SIMPLE_CORPUS, vocab_size=50)
        # vocab should reach the target OR stop early if it ran out of
        # pairs to merge -- either is valid, just shouldn't exceed target
        assert len(tok.vocab) <= 50

    def test_merges_are_recorded_in_order(self):
        tok = BPETokenizer()
        tok.train(SIMPLE_CORPUS, vocab_size=30)
        assert len(tok.merges) > 0
        assert all(isinstance(m, tuple) and len(m) == 2 for m in tok.merges)

    def test_base_characters_all_in_vocab(self):
        tok = BPETokenizer()
        tok.train(SIMPLE_CORPUS, vocab_size=30)
        for ch in set(SIMPLE_CORPUS.replace(" ", "")):
            assert ch in tok.vocab

    def test_training_is_deterministic(self):
        tok_a = BPETokenizer()
        tok_a.train(SIMPLE_CORPUS, vocab_size=30)
        tok_b = BPETokenizer()
        tok_b.train(SIMPLE_CORPUS, vocab_size=30)
        assert tok_a.merges == tok_b.merges


class TestEncodeDecodeRoundTrip:
    @pytest.fixture
    def trained_tokenizer(self):
        tok = BPETokenizer()
        tok.train(SIMPLE_CORPUS, vocab_size=40)
        return tok

    def test_encode_returns_list_of_ints(self, trained_tokenizer):
        ids = trained_tokenizer.encode("low")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_round_trip_recovers_original_word(self, trained_tokenizer):
        original = "lower"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == original

    def test_round_trip_on_multiple_words(self, trained_tokenizer):
        original = "low lower newest"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == original

    def test_frequent_word_compresses_to_fewer_tokens_than_chars(self, trained_tokenizer):
        # "low" appeared 3 times in training -- BPE should have learned
        # to merge it into fewer tokens than its raw character count
        ids = trained_tokenizer.encode("low")
        assert len(ids) < len("low")
