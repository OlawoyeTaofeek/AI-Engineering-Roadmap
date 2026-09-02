"""
dataset_builder.py
====================

Orchestrates the full pipeline: download -> clean -> validate ->
deduplicate -> save, producing a single JSONL training corpus from all
configured book sources.

Output format
-----------------
Each line of the output file is a single JSON object:

    {"source": "gutenberg", "id": "4507", "category": "motivational", "text": "..."}

JSONL (one JSON object per line) is used rather than a single large JSON
array because it:
    * can be appended to incrementally without re-parsing the whole file,
    * can be streamed/read line-by-line without loading the entire
      dataset into memory (important once your corpus reaches the
      hundreds of MB / GB range),
    * matches the format most training-data tooling (including
      HuggingFace `datasets`) expects out of the box.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .archive_downloader import ArchiveDownloader
from .cleaner import clean_book_text, deduplicate_texts, is_valid_book_text
from .config import DEFAULT_PATHS, Paths
from .curated_lists import (
    AFRICA_SUBJECT_KEYWORDS,
    AFRICAN_HISTORY_BOOK_IDS,
    HISTORY_SUBJECT_KEYWORDS,
    MOTIVATIONAL_BOOK_IDS,
)
from .gutenberg_downloader import GutenbergDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _write_jsonl_record(
    file_handle, source: str, item_id: str, category: str, text: str
) -> None:
    """Write a single cleaned record as one JSON line."""
    record = {"source": source, "id": item_id, "category": category, "text": text}
    file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_gutenberg_subset(
    output_path: Path,
    downloader: GutenbergDownloader,
    keyword_ids: dict[int, str] | None = None,
    subject_keywords: list[str] | None = None,
    category_label: str = "uncategorized",
    max_books: int | None = None,
) -> int:
    """
    Download, clean, and append a Gutenberg-sourced subset to a JSONL file.

    Supports two ways of selecting books (can be combined):
        * `keyword_ids`: an explicit dict of {book_id: description}, as
          found in `curated_lists.py` -- use this for topics where
          catalog subject tags are unreliable (e.g. "motivational").
        * `subject_keywords`: a list of subject-tag keywords to search
          the catalog for -- use this for well-tagged, broad topics
          (e.g. "History").

    Parameters
    ----------
    output_path : pathlib.Path
        JSONL file to append cleaned records to. Opened in append mode,
        so calling this function multiple times (e.g. once per category)
        accumulates into the same output file.
    downloader : GutenbergDownloader
        A configured downloader instance (shared across calls so the
        underlying HTTP session and rate limit are consistent).
    keyword_ids : dict[int, str] or None
        Explicit book IDs to download, e.g. `MOTIVATIONAL_BOOK_IDS`.
    subject_keywords : list[str] or None
        Subject-tag keywords to search the catalog for.
    category_label : str
        Label written into each record's "category" field, so downstream
        consumers of the corpus can filter/weight by category (e.g. give
        more weight to "african_history" vs "motivational" when building
        a training mix -- see 05_scaling_up/ for mixing ratio decisions).
    max_books : int or None
        Cap on the number of books to process from `subject_keywords`
        search results (has no effect on `keyword_ids`, since that list
        is already explicit). Useful to keep a first test run small.

    Returns
    -------
    int
        Number of valid, cleaned records written.
    """
    book_ids: list[int] = []

    if keyword_ids:
        book_ids.extend(keyword_ids.keys())

    if subject_keywords:
        catalog = downloader.fetch_catalog()
        matches = downloader.filter_by_subject(catalog, subject_keywords)
        matched_ids = matches["Text#"].astype(int).tolist()
        if max_books is not None:
            matched_ids = matched_ids[:max_books]
        book_ids.extend(matched_ids)

    book_ids = sorted(set(book_ids))  # dedupe across the two selection methods
    logger.info(
        "Building Gutenberg subset '%s': %d candidate book IDs",
        category_label, len(book_ids),
    )

    downloaded_paths = downloader.download_books(book_ids)

    written_count = 0
    with open(output_path, "a", encoding="utf-8") as out_file:
        for path in downloaded_paths:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_book_text(raw_text, source="gutenberg")

            if not is_valid_book_text(cleaned):
                logger.warning("Skipping %s: failed validity check after cleaning", path)
                continue

            book_id = path.stem  # filename without extension, e.g. "4507"
            _write_jsonl_record(out_file, "gutenberg", book_id, category_label, cleaned)
            written_count += 1

    logger.info(
        "Wrote %d valid records for category '%s' to %s",
        written_count, category_label, output_path,
    )
    return written_count


def build_archive_subset(
    output_path: Path,
    downloader: ArchiveDownloader,
    query: str,
    category_label: str,
    max_results: int = 30,
) -> int:
    """
    Search, download, clean, and append an Internet Archive-sourced
    subset to a JSONL file.

    See `ArchiveDownloader.search()` for the public-domain filtering
    applied by default -- this function does NOT override those defaults,
    so review that docstring's licensing caveats before running this at
    scale.

    Parameters
    ----------
    output_path : pathlib.Path
        JSONL file to append cleaned records to (append mode).
    downloader : ArchiveDownloader
        A configured downloader instance.
    query : str
        Search query, e.g. "Africa history colonial".
    category_label : str
        Label written into each record's "category" field.
    max_results : int
        Maximum number of search results to attempt downloading.

    Returns
    -------
    int
        Number of valid, cleaned records written.
    """
    results = downloader.search(query, max_results=max_results)
    logger.info(
        "Building Archive.org subset '%s': %d search results for query=%r",
        category_label, len(results), query,
    )

    identifiers = [item.identifier for item in results]
    downloaded_paths = downloader.download_items(identifiers)

    written_count = 0
    with open(output_path, "a", encoding="utf-8") as out_file:
        for path in downloaded_paths:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_book_text(raw_text, source="archive")

            if not is_valid_book_text(cleaned):
                logger.warning("Skipping %s: failed validity check after cleaning", path)
                continue

            item_id = path.stem.replace("archive_", "")
            _write_jsonl_record(out_file, "archive", item_id, category_label, cleaned)
            written_count += 1

    logger.info(
        "Wrote %d valid records for category '%s' to %s",
        written_count, category_label, output_path,
    )
    return written_count


def deduplicate_corpus(jsonl_path: Path) -> tuple[int, int]:
    """
    Deduplicate an entire JSONL corpus file in place, based on each
    record's cleaned text content.

    Rewrites the file to a temporary path first, then atomically replaces
    the original -- avoids leaving a corrupted/partial file if the
    process is interrupted mid-write.

    Parameters
    ----------
    jsonl_path : pathlib.Path
        The corpus file to deduplicate.

    Returns
    -------
    tuple[int, int]
        (records_before, records_after) counts, so callers can log/report
        how many duplicates were removed.
    """
    with open(jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    texts = [r["text"] for r in records]
    unique_texts = set()
    deduped_records = []
    for record in records:
        # Reuse the same fingerprint logic as cleaner.deduplicate_texts,
        # but keep the full record (not just text) since we need to
        # preserve source/id/category metadata.
        from .cleaner import compute_fingerprint
        fingerprint = compute_fingerprint(record["text"])
        if fingerprint not in unique_texts:
            unique_texts.add(fingerprint)
            deduped_records.append(record)

    tmp_path = jsonl_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in deduped_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(jsonl_path)  # atomic on POSIX systems

    before, after = len(records), len(deduped_records)
    logger.info("Deduplicated corpus: %d -> %d records (%d removed)", before, after, before - after)
    return before, after


def build_full_corpus(
    output_path: str | Path = "data_processed/books/books_corpus.jsonl",
    paths: Paths = DEFAULT_PATHS,
    include_motivational: bool = True,
    include_african_history: bool = True,
    include_general_history: bool = True,
    max_general_history_books: int = 50,
) -> Path:
    """
    Run the full books corpus-building pipeline end to end: download from
    all configured sources, clean, validate, deduplicate, and save as a
    single JSONL file.

    This is the main entry point most callers should use. For finer
    control (e.g. only rebuilding one category), call the individual
    `build_gutenberg_subset()` / `build_archive_subset()` functions
    directly instead.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Where to write the final corpus. Overwritten if it already
        exists (each category is then appended within a single run).
    paths : Paths
        Filesystem configuration for raw downloads.
    include_motivational : bool
        Whether to include the curated public-domain motivational books.
    include_african_history : bool
        Whether to include curated + searched African history content.
    include_general_history : bool
        Whether to include general history books via catalog search.
    max_general_history_books : int
        Cap on how many books the general-history catalog search
        downloads, since that search can match a very large number of
        results otherwise.

    Returns
    -------
    pathlib.Path
        Path to the final, deduplicated JSONL corpus file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()  # start fresh; subsets below append

    gutenberg = GutenbergDownloader(paths=paths)
    archive = ArchiveDownloader(paths=paths)

    total_written = 0

    if include_motivational:
        total_written += build_gutenberg_subset(
            output_path, gutenberg,
            keyword_ids=MOTIVATIONAL_BOOK_IDS,
            category_label="motivational",
        )

    if include_african_history:
        total_written += build_gutenberg_subset(
            output_path, gutenberg,
            keyword_ids=AFRICAN_HISTORY_BOOK_IDS,
            subject_keywords=AFRICA_SUBJECT_KEYWORDS,
            category_label="african_history",
        )
        total_written += build_archive_subset(
            output_path, archive,
            query="Africa history",
            category_label="african_history",
        )

    if include_general_history:
        total_written += build_gutenberg_subset(
            output_path, gutenberg,
            subject_keywords=HISTORY_SUBJECT_KEYWORDS,
            category_label="general_history",
            max_books=max_general_history_books,
        )

    logger.info("Corpus build complete: %d records written before deduplication", total_written)

    deduplicate_corpus(output_path)

    logger.info("Final corpus saved to %s", output_path)
    return output_path


def report_corpus_stats(jsonl_path: Path) -> dict:
    """
    Summarize a built corpus: record count, word count, and breakdown by
    category and source -- useful as a sanity check before using the
    corpus for training (see 00_explanation/training_objective.md for
    why eyeballing your data before training matters).

    Parameters
    ----------
    jsonl_path : pathlib.Path

    Returns
    -------
    dict
        Summary statistics, also logged at INFO level.
    """
    category_word_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    total_words = 0
    total_records = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            word_count = len(record["text"].split())
            total_words += word_count
            total_records += 1
            category_word_counts[record["category"]] = (
                category_word_counts.get(record["category"], 0) + word_count
            )
            source_counts[record["source"]] = source_counts.get(record["source"], 0) + 1

    stats = {
        "total_records": total_records,
        "total_words": total_words,
        "words_by_category": category_word_counts,
        "records_by_source": source_counts,
    }
    logger.info("Corpus stats: %s", json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    corpus_path = build_full_corpus()
    report_corpus_stats(corpus_path)
