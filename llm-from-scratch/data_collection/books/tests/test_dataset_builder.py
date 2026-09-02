"""
test_dataset_builder.py
=========================

Tests for dataset_builder.py's orchestration logic.

These tests use `unittest.mock` to replace the downloader classes
entirely -- NO network requests are made. This verifies the pipeline's
WIRING (download results flow correctly into cleaning, validation,
JSONL writing, and deduplication) independently of whether Gutenberg or
Archive.org happen to be reachable when tests run. Network-dependent
correctness (e.g. "does this URL pattern still match Gutenberg's real
file layout") should be checked manually against the live sources, since
mocking can't verify that Gutenberg's actual file paths behave as coded.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from data_collection.books.dataset_builder import (
    build_gutenberg_subset,
    deduplicate_corpus,
    report_corpus_stats,
)

SAMPLE_BOOK_TEXT = """\
*** START OF THE PROJECT GUTENBERG EBOOK TEST BOOK ***
""" + ("This is a sentence of the actual book content. " * 100) + """
*** END OF THE PROJECT GUTENBERG EBOOK TEST BOOK ***
"""

TOO_SHORT_BOOK_TEXT = """\
*** START OF THE PROJECT GUTENBERG EBOOK SHORT BOOK ***
Too short to be valid.
*** END OF THE PROJECT GUTENBERG EBOOK SHORT BOOK ***
"""


@pytest.fixture
def tmp_output_path(tmp_path) -> Path:
    return tmp_path / "test_corpus.jsonl"


@pytest.fixture
def tmp_book_files(tmp_path) -> dict[str, Path]:
    """Simulate what GutenbergDownloader.download_books() would have
    already saved to disk, without actually downloading anything."""
    valid_book = tmp_path / "1001.txt"
    valid_book.write_text(SAMPLE_BOOK_TEXT, encoding="utf-8")

    short_book = tmp_path / "1002.txt"
    short_book.write_text(TOO_SHORT_BOOK_TEXT, encoding="utf-8")

    return {"valid": valid_book, "short": short_book}


class TestBuildGutenbergSubset:
    def test_writes_valid_books_only(self, tmp_output_path, tmp_book_files):
        mock_downloader = MagicMock()
        mock_downloader.download_books.return_value = [
            tmp_book_files["valid"],
            tmp_book_files["short"],
        ]

        written = build_gutenberg_subset(
            output_path=tmp_output_path,
            downloader=mock_downloader,
            keyword_ids={1001: "Test Book", 1002: "Short Book"},
            category_label="test_category",
        )

        # only the valid (long-enough) book should have been written --
        # the short one should be filtered out by is_valid_book_text()
        assert written == 1

        with open(tmp_output_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        assert len(records) == 1
        assert records[0]["id"] == "1001"
        assert records[0]["category"] == "test_category"
        assert records[0]["source"] == "gutenberg"
        # boilerplate must be gone from the written text
        assert "PROJECT GUTENBERG" not in records[0]["text"]
        # actual content must be present
        assert "actual book content" in records[0]["text"]

    def test_calls_downloader_with_correct_ids(self, tmp_output_path, tmp_book_files):
        mock_downloader = MagicMock()
        mock_downloader.download_books.return_value = [tmp_book_files["valid"]]

        build_gutenberg_subset(
            output_path=tmp_output_path,
            downloader=mock_downloader,
            keyword_ids={1001: "Test Book"},
            category_label="test_category",
        )

        mock_downloader.download_books.assert_called_once()
        called_ids = mock_downloader.download_books.call_args[0][0]
        assert called_ids == [1001]

    def test_subject_keyword_search_path(self, tmp_output_path, tmp_book_files):
        import pandas as pd

        mock_downloader = MagicMock()
        mock_downloader.fetch_catalog.return_value = pd.DataFrame(
            {"Text#": [1001], "Subjects": ["History -- Africa"], "Language": ["en"]}
        )
        mock_downloader.filter_by_subject.return_value = pd.DataFrame(
            {"Text#": [1001]}
        )
        mock_downloader.download_books.return_value = [tmp_book_files["valid"]]

        written = build_gutenberg_subset(
            output_path=tmp_output_path,
            downloader=mock_downloader,
            subject_keywords=["History"],
            category_label="test_category",
        )

        assert written == 1
        mock_downloader.fetch_catalog.assert_called_once()
        mock_downloader.filter_by_subject.assert_called_once()

    def test_appends_across_multiple_calls(self, tmp_output_path, tmp_book_files):
        mock_downloader = MagicMock()
        mock_downloader.download_books.return_value = [tmp_book_files["valid"]]

        build_gutenberg_subset(
            tmp_output_path, mock_downloader,
            keyword_ids={1001: "Book A"}, category_label="cat_a",
        )
        build_gutenberg_subset(
            tmp_output_path, mock_downloader,
            keyword_ids={1001: "Book A again"}, category_label="cat_b",
        )

        with open(tmp_output_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        # two separate calls, in APPEND mode, should yield two records
        assert len(records) == 2
        assert {r["category"] for r in records} == {"cat_a", "cat_b"}


class TestDeduplicateCorpus:
    def test_removes_exact_duplicate_records(self, tmp_output_path):
        records = [
            {"source": "gutenberg", "id": "1", "category": "x", "text": "duplicate content here"},
            {"source": "gutenberg", "id": "2", "category": "x", "text": "duplicate content here"},
            {"source": "gutenberg", "id": "3", "category": "x", "text": "unique content here"},
        ]
        with open(tmp_output_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        before, after = deduplicate_corpus(tmp_output_path)
        assert before == 3
        assert after == 2

        with open(tmp_output_path, encoding="utf-8") as f:
            remaining = [json.loads(line) for line in f]
        assert len(remaining) == 2
        # first occurrence (id "1") should be kept over the later duplicate (id "2")
        remaining_ids = {r["id"] for r in remaining}
        assert "1" in remaining_ids
        assert "2" not in remaining_ids
        assert "3" in remaining_ids

    def test_no_duplicates_leaves_file_unchanged(self, tmp_output_path):
        records = [
            {"source": "gutenberg", "id": "1", "category": "x", "text": "text one"},
            {"source": "gutenberg", "id": "2", "category": "x", "text": "text two"},
        ]
        with open(tmp_output_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        before, after = deduplicate_corpus(tmp_output_path)
        assert before == after == 2


class TestReportCorpusStats:
    def test_computes_correct_totals(self, tmp_output_path):
        records = [
            {"source": "gutenberg", "id": "1", "category": "history", "text": "one two three"},
            {"source": "gutenberg", "id": "2", "category": "history", "text": "four five"},
            {"source": "archive", "id": "3", "category": "motivational", "text": "six seven eight nine"},
        ]
        with open(tmp_output_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        stats = report_corpus_stats(tmp_output_path)

        assert stats["total_records"] == 3
        assert stats["total_words"] == 3 + 2 + 4  # 9
        assert stats["words_by_category"]["history"] == 5
        assert stats["words_by_category"]["motivational"] == 4
        assert stats["records_by_source"]["gutenberg"] == 2
        assert stats["records_by_source"]["archive"] == 1
