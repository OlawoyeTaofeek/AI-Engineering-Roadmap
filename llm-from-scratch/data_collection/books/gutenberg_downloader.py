"""
gutenberg_downloader.py
========================

Downloads public-domain book texts from Project Gutenberg.

Why Project Gutenberg, and why this approach specifically
-----------------------------------------------------------
Project Gutenberg hosts over 70,000 books whose copyright has expired
(mostly pre-1929 in the US), making them freely usable -- including for
training data -- without any licensing concern. This is the single
easiest large source of clean, legal book text available.

Project Gutenberg explicitly asks that automated tools NOT crawl its main
website page-by-page, since that places unnecessary load on volunteer-run
infrastructure. Instead, they publish:

    1. A single downloadable CATALOG FILE listing every book, its ID,
       title, author, and subject tags -- so you can decide what you want
       BEFORE making any download requests.
    2. Direct, predictable download URLs for each book's plain-text file,
       keyed by that book's ID.

This module uses exactly that sanctioned pattern: fetch the catalog once,
filter it locally (no network calls), then download only the specific
books you actually want, with a delay between each request.

For bulk downloads of hundreds+ of books, Project Gutenberg's rsync
mirror is the officially preferred method and is documented in this
module's `RSYNC_MIRROR_COMMAND` constant -- rsync is far gentler on their
servers than repeated HTTP requests, and resumes automatically if
interrupted.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from .config import DEFAULT_PATHS, DEFAULT_POLICY, Paths, ScrapingPolicy

logger = logging.getLogger(__name__)

# Official catalog of every Project Gutenberg book, refreshed regularly.
# This one request replaces what would otherwise be thousands of individual
# "browse" requests to discover what's available.
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"

# Officially documented rsync mirror command for bulk downloads.
# Not executed by this module automatically -- shown here as the
# recommended path for anyone scaling beyond a few hundred books.
RSYNC_MIRROR_COMMAND = (
    "rsync -av --del ftp@ftp.ibiblio.org::gutenberg ./gutenberg-mirror/ "
    '--include="*/" --include="*.txt" --exclude="*"'
)


@dataclass
class GutenbergBook:
    """A single catalog entry, resolved down to just the fields we need."""

    book_id: int
    title: str
    authors: str
    subjects: str


class GutenbergDownloader:
    """
    Fetches the Project Gutenberg catalog and downloads individual books.

    Parameters
    ----------
    policy : ScrapingPolicy
        Rate-limiting and request configuration. Defaults to the shared
        package-wide policy in `config.py`.
    paths : Paths
        Filesystem layout for where raw downloads are stored.

    Examples
    --------
    >>> downloader = GutenbergDownloader()
    >>> catalog = downloader.fetch_catalog()
    >>> africa_books = downloader.filter_by_subject(catalog, ["Africa", "Nigeria"])
    >>> downloader.download_books(africa_books.book_id.tolist()[:20])
    """

    def __init__(
        self,
        policy: ScrapingPolicy = DEFAULT_POLICY,
        paths: Paths = DEFAULT_PATHS,
    ) -> None:
        self.policy = policy
        self.paths = paths
        self.paths.ensure_exist()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.policy.user_agent})

    # ------------------------------------------------------------------ #
    # Catalog discovery
    # ------------------------------------------------------------------ #

    def fetch_catalog(self) -> pd.DataFrame:
        """
        Download and parse Project Gutenberg's full book catalog.

        This is a SINGLE request that returns metadata for every book in
        the archive, so all subsequent filtering happens locally in
        pandas -- no further network calls needed just to browse.

        Returns
        -------
        pandas.DataFrame
            Columns include (at minimum): 'Text#', 'Title', 'Authors',
            'Subjects', 'LoCC', 'Bookshelves', 'Language'. See Project
            Gutenberg's catalog documentation for the full schema, as
            columns may be added/renamed upstream over time.

        Raises
        ------
        requests.exceptions.RequestException
            If the catalog can't be fetched after `policy.max_retries`
            attempts.
        """
        logger.info("Fetching Project Gutenberg catalog from %s", CATALOG_URL)
        response = self._request_with_retries(CATALOG_URL)
        catalog = pd.read_csv(io.StringIO(response.text))
        logger.info("Catalog loaded: %d books total", len(catalog))
        return catalog

    def filter_by_subject(
        self, catalog: pd.DataFrame, keywords: list[str], language: str = "en"
    ) -> pd.DataFrame:
        """
        Filter a catalog DataFrame down to books matching given subject
        keywords, restricted to a single language by default.

        This runs entirely locally against the already-downloaded catalog
        -- no additional network requests are made here.

        Parameters
        ----------
        catalog : pandas.DataFrame
            The result of `fetch_catalog()`.
        keywords : list[str]
            Case-insensitive substrings to match against the 'Subjects'
            column, e.g. ["Africa", "History", "Colonialism"]. A book
            matches if ANY keyword is found in its subject tags.
        language : str
            ISO language code to restrict results to. Defaults to
            English ("en"), since most downstream cleaning/tokenization
            in this repo assumes English text.

        Returns
        -------
        pandas.DataFrame
            Filtered subset of `catalog`.
        """
        subjects = catalog["Subjects"].fillna("")
        subject_mask = subjects.str.contains("|".join(keywords), case=False, regex=True)

        language_mask = catalog["Language"].fillna("").str.lower() == language.lower()

        filtered = catalog[subject_mask & language_mask]
        logger.info(
            "Filtered catalog: %d books match keywords=%s, language=%s",
            len(filtered), keywords, language,
        )
        return filtered

    # ------------------------------------------------------------------ #
    # Downloading
    # ------------------------------------------------------------------ #

    def download_book(self, book_id: int) -> Path | None:
        """
        Download a single book's plain-text file by its Gutenberg ID.

        Tries the standard "<id>-0.txt" naming pattern first (UTF-8 text,
        Gutenberg's modern default), falling back to "<id>.txt" for older
        catalog entries that predate that naming convention.

        Parameters
        ----------
        book_id : int
            The Project Gutenberg "Text#" identifier.

        Returns
        -------
        pathlib.Path or None
            Path to the saved raw text file, or None if the download
            failed after all retries (logged as a warning, not raised --
            a single missing book shouldn't halt a batch download).
        """
        destination = self.paths.raw_dir / f"{book_id}.txt"
        if destination.exists():
            logger.debug("Book %d already downloaded, skipping", book_id)
            return destination

        candidate_urls = [
            f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
            f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        ]

        for url in candidate_urls:
            try:
                response = self._request_with_retries(url)
            except requests.exceptions.RequestException:
                continue  # try the next candidate URL pattern

            destination.write_text(response.text, encoding="utf-8")
            logger.info("Downloaded book %d -> %s", book_id, destination)
            return destination

        logger.warning("Failed to download book %d from any known URL pattern", book_id)
        return None

    def download_books(self, book_ids: list[int]) -> list[Path]:
        """
        Download multiple books sequentially, respecting the configured
        rate limit between requests.

        Deliberately sequential (not parallelized) by default: Project
        Gutenberg is a volunteer-funded nonprofit, and a slow, polite
        bulk download is the right tradeoff here over raw speed. For
        downloads beyond a few hundred books, use the rsync mirror
        documented in `RSYNC_MIRROR_COMMAND` instead.

        Parameters
        ----------
        book_ids : list[int]
            Gutenberg Text# IDs to download.

        Returns
        -------
        list[pathlib.Path]
            Paths to successfully downloaded files. Books that failed to
            download are omitted (see logs for which ones and why).
        """
        downloaded = []
        for i, book_id in enumerate(book_ids, start=1):
            path = self.download_book(book_id)
            if path is not None:
                downloaded.append(path)
            if i < len(book_ids):
                time.sleep(self.policy.request_delay_seconds)

        logger.info("Downloaded %d/%d requested books", len(downloaded), len(book_ids))
        return downloaded

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _request_with_retries(self, url: str) -> requests.Response:
        """
        GET a URL, retrying transient failures up to `policy.max_retries`
        times with a short backoff, then raising if all attempts fail.
        """
        last_exception: Exception | None = None
        for attempt in range(1, self.policy.max_retries + 1):
            try:
                response = self._session.get(
                    url, timeout=self.policy.request_timeout_seconds
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                last_exception = exc
                logger.debug(
                    "Request to %s failed (attempt %d/%d): %s",
                    url, attempt, self.policy.max_retries, exc,
                )
                if attempt < self.policy.max_retries:
                    time.sleep(1.5 * attempt)  # simple linear backoff

        assert last_exception is not None
        raise last_exception
