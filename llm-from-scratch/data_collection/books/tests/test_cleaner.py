"""
test_cleaner.py
=================

Unit tests for cleaner.py. These require no network access -- every
function under test is a pure function operating on in-memory strings,
so these tests run instantly and should be part of CI.
"""

import pytest

from data_collection.books.cleaner import (
    clean_book_text,
    compute_fingerprint,
    deduplicate_texts,
    is_valid_book_text,
    normalize_whitespace,
    strip_gutenberg_boilerplate,
    strip_ocr_noise,
    strip_urls,
)

SAMPLE_GUTENBERG_TEXT = """\
The Project Gutenberg eBook of Some Old Book

This ebook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever.

*** START OF THE PROJECT GUTENBERG EBOOK SOME OLD BOOK ***

Chapter One

It was a dark and stormy night. The wind howled through the trees, and
somewhere in the distance, a dog barked.

Chapter Two

The morning came, bright and clear, as mornings often do after a storm
has passed.

*** END OF THE PROJECT GUTENBERG EBOOK SOME OLD BOOK ***

This is a copy of a public domain book, provided as-is.
"""


class TestStripGutenbergBoilerplate:
    def test_removes_header_and_footer(self):
        result = strip_gutenberg_boilerplate(SAMPLE_GUTENBERG_TEXT)
        assert "Project Gutenberg" not in result
        assert "This ebook is for the use of anyone" not in result
        assert "This is a copy of a public domain book" not in result

    def test_preserves_actual_content(self):
        result = strip_gutenberg_boilerplate(SAMPLE_GUTENBERG_TEXT)
        assert "It was a dark and stormy night" in result
        assert "The morning came, bright and clear" in result

    def test_returns_original_text_when_markers_absent(self):
        text_without_markers = "Just some plain text with no Gutenberg markers at all."
        result = strip_gutenberg_boilerplate(text_without_markers)
        assert result == text_without_markers

    def test_is_case_insensitive(self):
        text = "*** start of this project gutenberg ebook FOO *** content *** end of this project gutenberg ebook FOO ***"
        result = strip_gutenberg_boilerplate(text)
        assert result.strip() == "content"


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("hello    world") == "hello world"

    def test_collapses_newlines_and_tabs(self):
        assert normalize_whitespace("hello\n\n\tworld") == "hello world"

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize_whitespace("   hello world   ") == "hello world"

    def test_empty_string_stays_empty(self):
        assert normalize_whitespace("") == ""


class TestStripUrls:
    def test_removes_http_url(self):
        result = strip_urls("Check this out: http://example.com/page for more info.")
        assert "http://example.com" not in result
        assert "Check this out:" in result
        assert "for more info." in result

    def test_removes_https_url(self):
        result = strip_urls("Visit https://example.com/page?x=1 today.")
        assert "https://" not in result

    def test_no_urls_unchanged(self):
        text = "No links in this sentence at all."
        assert strip_urls(text) == text


class TestStripOcrNoise:
    def test_removes_isolated_symbol_tokens(self):
        result = strip_ocr_noise("This is | a sentence _ with noise.")
        assert "|" not in result
        assert "_" not in result

    def test_preserves_short_real_words(self):
        result = strip_ocr_noise("I a e o u")
        # single-letter WORDS (alphanumeric) must survive -- only pure
        # punctuation/symbol tokens should be stripped
        assert "I" in result
        assert "a" in result

    def test_preserves_words_with_punctuation_attached(self):
        result = strip_ocr_noise("Wait... really?")
        assert "Wait" in result
        assert "really" in result


class TestCleanBookText:
    def test_gutenberg_pipeline_strips_boilerplate_and_normalizes(self):
        result = clean_book_text(SAMPLE_GUTENBERG_TEXT, source="gutenberg")
        assert "Project Gutenberg" not in result
        assert "\n" not in result  # whitespace normalized
        assert "It was a dark and stormy night" in result

    def test_archive_pipeline_strips_ocr_noise(self):
        raw = "Some scanned text | with _ ocr artifacts."
        result = clean_book_text(raw, source="archive")
        assert "|" not in result
        assert "_" not in result

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            clean_book_text("some text", source="not_a_real_source")


class TestIsValidBookText:
    def test_rejects_empty_text(self):
        assert is_valid_book_text("") is False

    def test_rejects_too_short_text(self):
        assert is_valid_book_text("Just a few words here.", min_words=500) is False

    def test_accepts_sufficiently_long_text(self):
        long_text = " ".join(["word"] * 600)
        assert is_valid_book_text(long_text, min_words=500) is True

    def test_respects_custom_min_words(self):
        text = " ".join(["word"] * 10)
        assert is_valid_book_text(text, min_words=5) is True
        assert is_valid_book_text(text, min_words=20) is False


class TestDeduplication:
    def test_identical_texts_deduplicated(self):
        texts = ["The quick brown fox.", "The quick brown fox.", "A different sentence."]
        result = deduplicate_texts(texts)
        assert len(result) == 2

    def test_preserves_first_occurrence_order(self):
        texts = ["first text here", "second text here", "first text here"]
        result = deduplicate_texts(texts)
        assert result == ["first text here", "second text here"]

    def test_no_duplicates_unchanged(self):
        texts = ["one", "two", "three"]
        assert deduplicate_texts(texts) == texts

    def test_fingerprint_is_deterministic(self):
        text = "Some consistent piece of text."
        assert compute_fingerprint(text) == compute_fingerprint(text)

    def test_fingerprint_differs_for_different_text(self):
        assert compute_fingerprint("text A") != compute_fingerprint("text B")

    def test_fingerprint_uses_prefix_only(self):
        # two texts sharing the same first 500 chars but diverging after
        # should be treated as duplicates by this cheap fingerprint --
        # documenting the known limitation described in cleaner.py
        long_common_prefix = "word " * 200  # well over 500 chars
        text_a = long_common_prefix + "ending A"
        text_b = long_common_prefix + "ending B"
        assert compute_fingerprint(text_a) == compute_fingerprint(text_b)
