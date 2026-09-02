"""
data_collection.books
======================

Tools for building a training-data corpus of public-domain books, sourced
from Project Gutenberg, the Internet Archive, and Wikipedia's official
bulk data releases.

This package deliberately avoids scraping live web pages for book content.
Every source used here provides an official, sanctioned bulk-access method
(a catalog file, a search API, or a data dump) specifically so that large
downloads don't place uncontrolled load on a live web server. See each
submodule's docstring for the specific method used and why.

Submodules
----------
config
    Central configuration: output paths, request timeouts, rate limits.
gutenberg_downloader
    Fetches Project Gutenberg's catalog and downloads book texts by ID.
archive_downloader
    Searches and downloads public-domain texts from the Internet Archive.
wikipedia_loader
    Loads encyclopedia text from Wikipedia's official dataset release.
cleaner
    Strips source-specific boilerplate and normalizes whitespace/encoding.
curated_lists
    Hand-picked Project Gutenberg IDs for public-domain motivational and
    African-history texts, since these are unreliable to find by keyword
    search alone.
dataset_builder
    Orchestrates the above into a single JSONL training corpus.

Quickstart
----------
>>> from data_collection.books.dataset_builder import build_full_corpus
>>> build_full_corpus(output_path="corpus/books.jsonl")

See README.md in this directory for the full usage guide, licensing notes,
and a description of what each data source legally allows.
"""

__version__ = "0.1.0"
