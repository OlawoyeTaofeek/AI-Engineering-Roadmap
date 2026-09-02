# data_collection/books

Builds a public-domain book corpus for training data, sourced from
Project Gutenberg, the Internet Archive, and Wikipedia's official bulk
data releases.

## Why these three sources, and not live-site scraping

Every source in this module has an **official, sanctioned bulk-access
method** — a downloadable catalog, a documented search API, or a full
data dump — specifically provided so that automated tools don't need to
crawl live web pages. This module uses those methods exclusively:

| Source | Access method | Why not scrape pages directly |
|---|---|---|
| Project Gutenberg | catalog CSV + predictable per-book download URLs | Gutenberg explicitly discourages page-by-page crawling of volunteer-funded infrastructure |
| Internet Archive | Advanced Search API + per-item OCR text download | Structured API avoids parsing rendered HTML; also lets us filter by rights metadata before downloading |
| Wikipedia | official dataset release / bulk XML dumps | Wikimedia publishes dumps specifically to avoid live-site scraping load |

## Licensing — read this before running anything at scale

- **Project Gutenberg**: every book in its catalog is verified public
  domain (US law). Safe to use and redistribute without restriction.
- **Internet Archive**: hosts a **mix** of public-domain texts and
  copyrighted books available only for controlled digital lending. This
  module's `ArchiveDownloader.search()` defaults to filtering by
  uploader-provided rights metadata and a `year_max=1928` cutoff, but
  metadata is not always accurate. **Spot-check a sample of any batch
  before including it in a dataset you plan to redistribute.**
- **Wikipedia**: text is CC BY-SA 4.0 licensed — free to use for training,
  but redistribution of the raw corpus should retain attribution per the
  license terms if you're publishing the dataset itself (not just a
  model trained on it).

## Installation

```bash
pip install requests pandas pytest
pip install datasets  # only needed for wikipedia_loader.load_via_huggingface()
```

## Quickstart

```python
from data_collection.books.dataset_builder import build_full_corpus, report_corpus_stats

corpus_path = build_full_corpus(
    output_path="data_processed/books/books_corpus.jsonl",
    include_motivational=True,
    include_african_history=True,
    include_general_history=True,
    max_general_history_books=50,
)

report_corpus_stats(corpus_path)
```

This downloads, cleans, validates, and deduplicates books from all three
curated categories, saving a single JSONL corpus file.

## Module layout

```
config.py                -- rate limits, timeouts, output paths (all in one place)
gutenberg_downloader.py    -- catalog fetch + book download, by ID or subject search
archive_downloader.py        -- Internet Archive search + OCR text download
wikipedia_loader.py            -- encyclopedia text via HF datasets or raw dump
cleaner.py                       -- boilerplate/OCR-noise stripping, dedup, validation
curated_lists.py                   -- hand-picked IDs for topics keyword search misses
dataset_builder.py                   -- orchestrates the above into one JSONL corpus
tests/                                  -- unit + mocked integration tests, no live network needed
```

## Using individual pieces

### Download by subject keyword

```python
from data_collection.books.gutenberg_downloader import GutenbergDownloader

downloader = GutenbergDownloader()
catalog = downloader.fetch_catalog()          # one request, full catalog
matches = downloader.filter_by_subject(catalog, ["Africa", "Nigeria"])
paths = downloader.download_books(matches["Text#"].tolist()[:20])
```

### Download hand-curated books (for topics keyword search handles poorly)

```python
from data_collection.books.curated_lists import MOTIVATIONAL_BOOK_IDS

downloader = GutenbergDownloader()
paths = downloader.download_books(list(MOTIVATIONAL_BOOK_IDS.keys()))
```

### Search and download from the Internet Archive

```python
from data_collection.books.archive_downloader import ArchiveDownloader

downloader = ArchiveDownloader()
results = downloader.search("Africa history", year_max=1928)
paths = downloader.download_items([r.identifier for r in results[:10]])
```

### Clean and validate a downloaded book

```python
from data_collection.books.cleaner import clean_book_text, is_valid_book_text

raw_text = paths[0].read_text(encoding="utf-8")
cleaned = clean_book_text(raw_text, source="gutenberg")
if is_valid_book_text(cleaned):
    print("Valid, ready for the corpus.")
```

## Rate limiting

Every downloader respects `config.DEFAULT_POLICY.request_delay_seconds`
(default: 2 seconds) between requests to the same host. Adjust this in
`config.py` if a source's terms specify a different preferred rate —
**don't** lower it without checking the source's stated policy first.

For bulk downloads beyond a few hundred books, use Project Gutenberg's
official rsync mirror instead of this module's HTTP downloader — the
exact command is documented in `gutenberg_downloader.RSYNC_MIRROR_COMMAND`.

## Running tests

```bash
pytest data_collection/books/tests/ -v
```

All 34 tests run with **zero network access** — `test_cleaner.py` tests
pure text-processing functions directly, and `test_dataset_builder.py`
mocks the downloader classes to verify the download → clean → validate →
write pipeline is wired correctly, independent of whether Gutenberg or
Archive.org happen to be reachable at test time.

## Output format

Each line of the output JSONL file is one JSON object:

```json
{"source": "gutenberg", "id": "4507", "category": "motivational", "text": "..."}
```

- `source` — which downloader produced this record (`"gutenberg"` or `"archive"`)
- `id` — the source's native identifier (Gutenberg book ID or Archive.org item identifier)
- `category` — which curated category this came from (`"motivational"`, `"african_history"`, `"general_history"`), so you can weight or filter by category when building a training mix (see `05_scaling_up/` for mixing-ratio guidance elsewhere in this repo)
- `text` — cleaned, deduplicated book text

## Extending this module

- **Add more curated books**: edit `curated_lists.py` — each entry is a
  `{book_id: "Title -- Author (year)"}` pair. Always verify a new ID with
  a single `download_book()` call before adding it to a list that will
  be batch-downloaded.
- **Add a new source entirely**: follow the pattern in
  `gutenberg_downloader.py` / `archive_downloader.py` — a class wrapping
  a `requests.Session`, a `_request_with_retries` helper, and methods
  that return `pathlib.Path` objects for downstream cleaning. Then add a
  corresponding `build_<source>_subset()` function to `dataset_builder.py`.
