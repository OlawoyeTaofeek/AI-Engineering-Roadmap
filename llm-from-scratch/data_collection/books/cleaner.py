"""
cleaner.py
===========

Text-cleaning utilities for raw downloaded book content.

Raw downloads from any source carry artifacts that shouldn't end up in
training data: legal boilerplate repeated identically across every
Gutenberg book, inconsistent whitespace, OCR noise from scanned Archive.org
texts, and near-duplicate content. This module handles all of that.

Design note: every function here is a pure function (text in, text out --
or a bool for validity checks) with no I/O and no network calls. This
makes them trivially unit-testable in isolation from the downloader
modules, and safe to run repeatedly/idempotently over the same text.
"""

from __future__ import annotations

import hashlib
import re

# Project Gutenberg wraps every book's actual content between a start and
# end marker. The exact wording of these markers has changed slightly
# across Gutenberg's history, so this pattern is intentionally loose
# (case-insensitive, tolerant of "THE"/"THIS", tolerant of the trailing
# book title after the marker).
_GUTENBERG_START_PATTERN = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_GUTENBERG_END_PATTERN = re.compile(
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)

# Matches runs of whitespace (spaces, tabs, newlines) of any length.
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Matches URLs, so they can be stripped -- links carry no language-modeling
# value and often break tokenization boundaries in unhelpful ways.
_URL_PATTERN = re.compile(r"https?://\S+")

# OCR output from scanned texts (Internet Archive) frequently contains
# isolated single characters or short garbage tokens surrounded by
# whitespace, from misrecognized page furniture (page numbers, margin
# noise). This pattern is DELIBERATELY conservative -- it only strips
# very short "words" that are pure punctuation/symbols, not short real
# words like "a" or "I", to avoid deleting legitimate short tokens.
#
# NOTE: Python's \w character class includes underscore ('_'), since
# underscore is a valid identifier character in most programming
# contexts. That's the WRONG behavior for OCR noise stripping -- a lone
# underscore is exactly the kind of misread page-rule artifact this
# pattern exists to catch. So the negated class explicitly excludes
# underscore in addition to \w, rather than relying on \w alone.
_OCR_NOISE_PATTERN = re.compile(r"(?<!\S)[^\w\s](?!\S)|(?<!\S)_(?!\S)")


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg's standard legal header and footer,
    returning only the book's actual content.

    Every Gutenberg text includes a licensing notice before and after
    the actual book -- identical (or near-identical) boilerplate that,
    left in, would appear as a near-duplicate block across every single
    book in your dataset, wasting training signal and skewing token
    frequency statistics.

    Parameters
    ----------
    text : str
        Raw text as downloaded from Project Gutenberg.

    Returns
    -------
    str
        Text between the start and end markers, stripped of surrounding
        whitespace. If either marker isn't found (some very old catalog
        entries predate the standardized markers), the original text is
        returned unchanged -- callers should treat this as a signal to
        review that particular book manually rather than silently
        including un-stripped boilerplate.
    """
    start_match = _GUTENBERG_START_PATTERN.search(text)
    end_match = _GUTENBERG_END_PATTERN.search(text)

    if start_match and end_match and start_match.end() < end_match.start():
        return text[start_match.end():end_match.start()].strip()

    return text


def normalize_whitespace(text: str) -> str:
    """
    Collapse all runs of whitespace (including newlines) into single
    spaces, and strip leading/trailing whitespace.

    Note this intentionally removes paragraph-break structure (double
    newlines). For training a causal language model on plain prose, this
    is a reasonable simplification; if you specifically want to preserve
    paragraph boundaries (e.g. for structure-aware training later),
    replace runs of 2+ newlines with a single '\\n\\n' marker before
    calling this function.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def strip_urls(text: str) -> str:
    """Remove URLs from text. Returns text with URLs deleted entirely
    (not replaced with a placeholder), since a bare URL carries no
    natural-language signal for training."""
    return _URL_PATTERN.sub("", text)


def strip_ocr_noise(text: str) -> str:
    """
    Remove isolated single-character punctuation/symbol tokens typical
    of OCR misrecognition in scanned texts (Internet Archive sources).

    This is intentionally conservative: it only removes standalone
    non-alphanumeric single characters (e.g. a lone '|' or '_' left by a
    misread page rule), never real short words. Apply this ONLY to
    OCR-sourced text (Internet Archive), not to Gutenberg's already-clean
    plain text, where it would do nothing useful and adds unnecessary
    processing.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return _OCR_NOISE_PATTERN.sub("", text)


def clean_book_text(text: str, source: str = "gutenberg") -> str:
    """
    Apply the full cleaning pipeline appropriate to a given source.

    Parameters
    ----------
    text : str
        Raw downloaded text.
    source : str
        One of "gutenberg" or "archive". Determines which
        source-specific steps are applied (boilerplate stripping only
        makes sense for Gutenberg; OCR-noise stripping only makes sense
        for Archive.org's scanned texts).

    Returns
    -------
    str
        Cleaned text, ready for validity checking and deduplication.

    Raises
    ------
    ValueError
        If `source` is not a recognized value -- fails loudly rather
        than silently skipping a cleaning step for a typo'd source name.
    """
    if source == "gutenberg":
        text = strip_gutenberg_boilerplate(text)
    elif source == "archive":
        text = strip_ocr_noise(text)
    else:
        raise ValueError(
            f"Unknown source '{source}'. Expected 'gutenberg' or 'archive'."
        )

    text = strip_urls(text)
    text = normalize_whitespace(text)
    return text


def is_valid_book_text(text: str, min_words: int = 500) -> bool:
    """
    Check whether cleaned text looks like a genuine book rather than a
    failed/empty download or a stub page.

    Parameters
    ----------
    text : str
        Already-cleaned text (call `clean_book_text()` first).
    min_words : int
        Minimum word count to be considered valid. Defaults to 500 --
        low enough not to exclude legitimately short public-domain
        works (e.g. essays, novellas), high enough to filter out empty
        or near-empty failed downloads.

    Returns
    -------
    bool
    """
    if not text:
        return False
    word_count = len(text.split())
    return word_count >= min_words


def compute_fingerprint(text: str, prefix_chars: int = 500) -> str:
    """
    Compute a cheap content fingerprint for approximate deduplication.

    Uses a hash of the text's first `prefix_chars` characters rather than
    the full text, on the reasoning that two genuinely different books
    are exceedingly unlikely to share an identical opening, while this
    stays fast even over a large corpus. This catches EXACT-prefix
    duplicates (e.g. the same book downloaded twice from two sources) but
    will NOT catch near-duplicates with different openings (e.g. two
    different translations, or a book with a variant preface).

    For serious large-scale near-duplicate detection (paraphrased or
    partially-overlapping texts), use a MinHash/LSH approach instead, via
    e.g. the `datasketch` library -- deliberately out of scope for this
    lightweight fingerprint, which is meant for the common case of exact
    or near-exact re-downloads.

    Parameters
    ----------
    text : str
        Already-cleaned text.
    prefix_chars : int
        Number of leading characters to hash.

    Returns
    -------
    str
        A hex-encoded SHA-256 hash, suitable for use as a dict/set key.
    """
    prefix = text[:prefix_chars]
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def deduplicate_texts(texts: list[str]) -> list[str]:
    """
    Remove exact/near-exact duplicate texts based on content fingerprint.

    Preserves the original order of first occurrence -- if the same
    fingerprint appears twice, only the first instance is kept.

    Parameters
    ----------
    texts : list[str]
        Already-cleaned texts.

    Returns
    -------
    list[str]
        Deduplicated texts, same relative order as input.
    """
    seen_fingerprints: set[str] = set()
    unique_texts: list[str] = []

    for text in texts:
        fingerprint = compute_fingerprint(text)
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            unique_texts.append(text)

    return unique_texts
