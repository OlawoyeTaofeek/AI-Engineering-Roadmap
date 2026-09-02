"""
config.py
=========

Central configuration for the books data collection pipeline.

Keeping all tunable values in one place (rather than scattered as magic
numbers across downloader modules) makes it easy to:

    * audit exactly how aggressively this pipeline hits external servers,
    * adjust rate limits in one spot if a source asks us to slow down,
    * point the whole pipeline at a different output directory for testing
      vs. a real run, without touching downloader logic.

Every downloader module in this package imports its settings from here
rather than hardcoding them.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ScrapingPolicy:
    """
    Rate-limiting and identification policy applied to every outbound
    request this pipeline makes.

    Attributes
    ----------
    request_delay_seconds : float
        Minimum time to wait between consecutive requests to the SAME
        host. This is the single most important setting for being a good
        citizen: it caps how much load we place on servers we don't own.
    request_timeout_seconds : float
        How long to wait for a single HTTP response before giving up.
        Prevents a single unresponsive request from hanging the pipeline
        indefinitely.
    max_retries : int
        Number of times to retry a failed request (e.g. transient network
        error or a 5xx server error) before giving up on that item.
    user_agent : str
        Sent with every request. Should honestly identify the traffic as
        an automated research/data-collection tool -- NOT spoofed to look
        like a regular browser. Some sites rate-limit or block requests
        with no User-Agent at all, so an honest one is both more polite
        and more reliable than none.
    """

    request_delay_seconds: float = 2.0
    request_timeout_seconds: float = 15.0
    max_retries: int = 3
    user_agent: str = (
        "data-collection-bot/0.1 "
        "(open-source LLM training corpus; contact: <your-email-here>)"
    )


@dataclass(frozen=True)
class Paths:
    """
    Filesystem layout for raw downloads, cleaned output, and logs.

    Kept separate from ScrapingPolicy since these change per-machine /
    per-run, while ScrapingPolicy is closer to a fixed ethical/technical
    contract with the sources we're pulling from.
    """

    project_root: Path = Path(__file__).resolve().parent
    raw_dir: Path = field(default_factory=lambda: Path("data_raw/books"))
    output_dir: Path = field(default_factory=lambda: Path("data_processed/books"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    def ensure_exist(self) -> None:
        """Create all configured directories if they don't already exist."""
        for directory in (self.raw_dir, self.output_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)


# Single shared instances used throughout the package. Import these rather
# than constructing new ScrapingPolicy()/Paths() objects, so a change here
# is guaranteed to apply everywhere.
DEFAULT_POLICY = ScrapingPolicy()
DEFAULT_PATHS = Paths()
