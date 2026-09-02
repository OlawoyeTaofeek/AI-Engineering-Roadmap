"""
wikipedia_loader.py
=====================

Loads encyclopedia-style text from Wikipedia's official bulk data
releases -- deliberately NOT by scraping live wikipedia.org pages.

Why not scrape Wikipedia directly
--------------------------------------
Wikipedia explicitly publishes complete database dumps specifically so
that nobody needs to crawl the live site for bulk text access. Scraping
individual article pages at any real scale would be:

    1. Unnecessary -- the same content is available as a clean, complete,
       officially-sanctioned download.
    2. Harder to clean -- live pages include navigation, infoboxes,
       citation markup, and templates that need to be stripped, whereas
       pre-extracted dumps are already plain article text.
    3. More load on Wikimedia's infrastructure than the maintainers ask
       automated tools to place on it.

This module offers two ways to get Wikipedia text, in order of
recommendation:

    1. `load_via_huggingface()` -- the fast path. Loads an already
       cleaned, already-extracted Wikipedia snapshot via the HuggingFace
       `datasets` library. This is by far the least effort and is
       sufficient for most training-data purposes.
    2. `download_raw_dump()` -- the from-scratch path, for cases where
       you need a specific language edition or dump date not covered by
       the HuggingFace release. Downloads the official compressed XML
       dump directly from dumps.wikimedia.org. Extracting usable text
       from this requires a separate tool (`wikiextractor`), documented
       in this module's docstring for `download_raw_dump`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from .config import DEFAULT_PATHS, DEFAULT_POLICY, Paths, ScrapingPolicy

logger = logging.getLogger(__name__)

# Official Wikimedia dump index. The exact dump date changes over time;
# "latest" is a stable alias Wikimedia maintains pointing at the most
# recent complete dump.
WIKIPEDIA_DUMP_URL_TEMPLATE = (
    "https://dumps.wikimedia.org/{lang}wiki/latest/"
    "{lang}wiki-latest-pages-articles.xml.bz2"
)


def load_via_huggingface(
    language: str = "20220301.en", split: str = "train", streaming: bool = True
):
    """
    Load a pre-cleaned Wikipedia snapshot via the HuggingFace `datasets`
    library. This is the recommended way to get Wikipedia text for
    training data -- no manual extraction step required.

    Requires the `datasets` package: `pip install datasets`.

    Parameters
    ----------
    language : str
        The HuggingFace Wikipedia config name, which encodes both the
        dump date and language code (e.g. "20220301.en" for the English
        Wikipedia snapshot dated 2022-03-01). Check
        https://huggingface.co/datasets/wikipedia for currently available
        configs, since exact dump dates offered do change over time.
    split : str
        Dataset split to load. Wikipedia is released as a single "train"
        split (there's no official train/val/test division -- you create
        your own, as covered elsewhere in this repo's training pipeline).
    streaming : bool
        If True, streams the dataset instead of downloading it fully to
        disk first. Strongly recommended for Wikipedia specifically,
        since the full dataset is tens of gigabytes -- streaming lets you
        take exactly the number of articles you need (see
        `dataset_builder.py`) without storing the rest.

    Returns
    -------
    datasets.Dataset or datasets.IterableDataset
        Each item has (at minimum) 'title' and 'text' fields.

    Examples
    --------
    >>> wiki = load_via_huggingface(streaming=True)
    >>> first_100 = [next(iter(wiki)) for _ in range(100)]
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "load_via_huggingface() requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    logger.info(
        "Loading Wikipedia via HuggingFace datasets (config=%s, streaming=%s)",
        language, streaming,
    )
    return load_dataset("wikipedia", language, split=split, streaming=streaming)


def download_raw_dump(
    language_code: str = "en",
    policy: ScrapingPolicy = DEFAULT_POLICY,
    paths: Paths = DEFAULT_PATHS,
) -> Path:
    """
    Download the full compressed Wikipedia XML dump for a given language
    directly from Wikimedia's official dump server.

    Use this ONLY if `load_via_huggingface()` doesn't cover your needs
    (e.g. you need a language edition or dump date not mirrored there).
    The resulting file is compressed MediaWiki XML markup, NOT plain
    text -- it must be processed with a tool such as `wikiextractor`
    before it's usable as training text:

        pip install wikiextractor
        python -m wikiextractor.WikiExtractor <downloaded_file> \\
            -o extracted_wiki --json

    This produces JSON files with clean, markup-free article text, one
    article per line, ready to feed into `dataset_builder.py`.

    Warning
    -------
    These dumps are large -- the English Wikipedia dump is 20+ GB
    compressed. This function streams the download to disk in chunks to
    avoid loading the whole file into memory, but you still need
    sufficient disk space and a stable connection; the download can take
    a long time depending on your bandwidth.

    Parameters
    ----------
    language_code : str
        Wikipedia language edition code, e.g. "en", "yo" (Yoruba),
        "ig" (Igbo), "ha" (Hausa) -- relevant if you want to extend your
        corpus to Nigerian-language Wikipedia editions alongside English.
    policy : ScrapingPolicy
        Request configuration (timeout only matters for the initial
        connection here; the download itself is streamed).
    paths : Paths
        Filesystem layout; the dump is saved under `paths.raw_dir`.

    Returns
    -------
    pathlib.Path
        Path to the downloaded .xml.bz2 file.
    """
    paths.ensure_exist()
    url = WIKIPEDIA_DUMP_URL_TEMPLATE.format(lang=language_code)
    destination = paths.raw_dir / f"{language_code}wiki-latest-pages-articles.xml.bz2"

    logger.info("Downloading Wikipedia dump (%s) from %s", language_code, url)
    logger.warning(
        "This is a large file (potentially 20+ GB for English). "
        "Ensure sufficient disk space before proceeding."
    )

    with requests.get(
        url, stream=True, timeout=policy.request_timeout_seconds,
        headers={"User-Agent": policy.user_agent},
    ) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                f.write(chunk)

    logger.info("Wikipedia dump saved to %s", destination)
    return destination
