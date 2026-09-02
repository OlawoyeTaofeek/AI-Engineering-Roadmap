"""
archive_downloader.py
=======================

Searches and downloads public-domain texts from the Internet Archive.

Why the Internet Archive, and why this approach specifically
----------------------------------------------------------------
The Internet Archive holds large collections of historical and African
texts that Project Gutenberg doesn't -- colonial-era records, out-of-print
regional histories, and scanned texts with OCR-extracted plain text
already available.

This module uses the Internet Archive's OFFICIAL, documented Advanced
Search API (`https://archive.org/advancedsearch.php`) to discover items,
and its predictable per-item download URL pattern to fetch OCR'd text.
Neither of these involves parsing rendered HTML pages -- both are
sanctioned, structured data-access endpoints.

Critical licensing note
--------------------------
Unlike Project Gutenberg (which ONLY lists public-domain works), the
Internet Archive hosts a mix of public-domain texts AND copyrighted books
made available for controlled digital lending. These are NOT
interchangeable for training-data purposes.

This module defaults to restricting searches to items whose metadata
indicates a public-domain or open license
(`rights_filter="(licenseurl:*publicdomain*) OR (licenseurl:*creativecommons*)"`),
but metadata quality varies across items. Always spot-check a sample of
any batch before including it in a dataset you plan to redistribute, and
prefer restricting `year_range` to pre-1929 works if you want a strong,
simple guarantee of US public-domain status regardless of metadata
accuracy.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import DEFAULT_PATHS, DEFAULT_POLICY, Paths, ScrapingPolicy

logger = logging.getLogger(__name__)

SEARCH_API_URL = "https://archive.org/advancedsearch.php"

# A conservative default filter: only items whose declared rights/license
# metadata indicates public domain or a Creative Commons license. This is
# a best-effort filter based on uploader-provided metadata, NOT a legal
# guarantee -- see module docstring above.
PUBLIC_DOMAIN_RIGHTS_FILTER = (
    "(licenseurl:*publicdomain*) OR (licenseurl:*creativecommons*)"
)


@dataclass
class ArchiveItem:
    """A single Internet Archive search result, resolved to key fields."""

    identifier: str
    title: str
    creator: str | None
    year: str | None


class ArchiveDownloader:
    """
    Searches the Internet Archive and downloads OCR'd plain-text content
    for matching items.

    Parameters
    ----------
    policy : ScrapingPolicy
        Rate-limiting and request configuration.
    paths : Paths
        Filesystem layout for raw downloads.

    Examples
    --------
    >>> downloader = ArchiveDownloader()
    >>> results = downloader.search("Africa history", year_max=1928)
    >>> downloader.download_items([r.identifier for r in results[:10]])
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
    # Search
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        mediatype: str = "texts",
        year_max: int | None = 1928,
        restrict_public_domain: bool = True,
        max_results: int = 50,
    ) -> list[ArchiveItem]:
        """
        Search the Internet Archive for text items matching a query.

        Parameters
        ----------
        query : str
            Free-text search query, e.g. "Africa history colonial".
        mediatype : str
            Internet Archive media type to restrict to. "texts" is the
            correct value for books/documents.
        year_max : int or None
            If set, restricts results to items published in or before
            this year. Defaults to 1928 -- the simplest reliable proxy
            for "definitely public domain in the US" regardless of
            metadata accuracy, since works published 1928 or earlier are
            in the public domain under current US copyright law. Set to
            None to disable this filter (NOT recommended unless you're
            prepared to manually verify rights for every result).
        restrict_public_domain : bool
            If True, additionally applies `PUBLIC_DOMAIN_RIGHTS_FILTER`
            based on uploader-provided license metadata. Recommended to
            leave True; this narrows results but reduces false positives.
        max_results : int
            Maximum number of results to return.

        Returns
        -------
        list[ArchiveItem]
        """
        query_parts = [query, f"mediatype:{mediatype}"]
        if year_max is not None:
            query_parts.append(f"year:[0000 TO {year_max}]")
        if restrict_public_domain:
            query_parts.append(f"({PUBLIC_DOMAIN_RIGHTS_FILTER})")

        params = {
            "q": " AND ".join(query_parts),
            "fl[]": ["identifier", "title", "creator", "year"],
            "rows": max_results,
            "output": "json",
        }

        logger.info("Searching Internet Archive: %s", params["q"])
        response = self._request_with_retries(SEARCH_API_URL, params=params)
        docs = response.json()["response"]["docs"]

        items = [
            ArchiveItem(
                identifier=doc["identifier"],
                title=doc.get("title", "(untitled)"),
                creator=doc.get("creator"),
                year=doc.get("year"),
            )
            for doc in docs
        ]
        logger.info("Found %d matching items", len(items))
        return items

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #

    def download_item_text(self, identifier: str) -> Path | None:
        """
        Download the OCR-extracted plain text for a single Archive item.

        Internet Archive items with a text layer typically expose it at a
        predictable URL: `.../download/<identifier>/<identifier>_djvu.txt`.
        Not every item has this file (e.g. items without OCR, or
        image-only scans) -- in that case this returns None rather than
        raising, so batch downloads can continue past gaps.

        Parameters
        ----------
        identifier : str
            The Internet Archive item identifier, as returned by `search()`.

        Returns
        -------
        pathlib.Path or None
        """
        destination = self.paths.raw_dir / f"archive_{identifier}.txt"
        if destination.exists():
            logger.debug("Item %s already downloaded, skipping", identifier)
            return destination

        url = f"https://archive.org/download/{identifier}/{identifier}_djvu.txt"
        try:
            response = self._request_with_retries(url)
        except requests.exceptions.RequestException:
            logger.warning("No OCR text available for item %s", identifier)
            return None

        destination.write_text(response.text, encoding="utf-8")
        logger.info("Downloaded item %s -> %s", identifier, destination)
        return destination

    def download_items(self, identifiers: list[str]) -> list[Path]:
        """
        Download OCR text for multiple Archive items sequentially,
        respecting the configured rate limit between requests.

        Parameters
        ----------
        identifiers : list[str]
            Internet Archive item identifiers.

        Returns
        -------
        list[pathlib.Path]
            Paths to successfully downloaded files.
        """
        downloaded = []
        for i, identifier in enumerate(identifiers, start=1):
            path = self.download_item_text(identifier)
            if path is not None:
                downloaded.append(path)
            if i < len(identifiers):
                time.sleep(self.policy.request_delay_seconds)

        logger.info(
            "Downloaded %d/%d requested items", len(downloaded), len(identifiers)
        )
        return downloaded

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _request_with_retries(
        self, url: str, params: dict | None = None
    ) -> requests.Response:
        """GET with retries -- identical policy to GutenbergDownloader's."""
        last_exception: Exception | None = None
        for attempt in range(1, self.policy.max_retries + 1):
            try:
                response = self._session.get(
                    url, params=params, timeout=self.policy.request_timeout_seconds
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
                    time.sleep(1.5 * attempt)

        assert last_exception is not None
        raise last_exception
