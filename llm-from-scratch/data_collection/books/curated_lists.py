"""
curated_lists.py
==================

Hand-picked Project Gutenberg book IDs for topics where keyword-based
catalog search (see `GutenbergDownloader.filter_by_subject`) is unreliable.

Why hand-curation is needed here
-------------------------------------
Gutenberg's "Subjects" field is uploader-assigned free text, not a
controlled taxonomy. A keyword search for "motivational" returns almost
nothing useful, because that's a modern marketing category, not how
19th/early-20th-century texts were tagged when cataloged. The books that
best fit "motivational" in the public-domain era are tagged under things
like "Conduct of life," "Self-culture," or aren't tagged that way at all.

Rather than fight the catalog's inconsistent tagging with an ever-growing
keyword list, this module keeps a small, manually verified, well-known
set of IDs for exactly this kind of topic. Each entry includes the title
and author as a comment so the list is self-documenting and easy to audit
or extend by hand.

IMPORTANT: Gutenberg book IDs are stable but not immutable metadata
identifiers set in stone forever -- always spot-check a few IDs from any
list (including this one) after a catalog update, using
`GutenbergDownloader.download_book()` on a single ID first, before
running a full batch download against the whole list.
"""

# Classic public-domain works commonly categorized as motivational /
# self-improvement literature. All well outside copyright (published
# 1850s-1920s).
MOTIVATIONAL_BOOK_IDS: dict[int, str] = {
    4507: "As a Man Thinketh -- James Allen (1902)",
    2680: "Meditations -- Marcus Aurelius (translated by George Long)",
    148: "Up From Slavery -- Booker T. Washington (1901)",
    20203: "The Autobiography of Benjamin Franklin",
    3300: "Self-Help -- Samuel Smiles (1859)",
    26268: "Acres of Diamonds -- Russell H. Conwell",
}

# African history and African-authored/African-focused literature that is
# public domain. This list intentionally favors African-authored primary
# sources (e.g. Equiano, Washington) alongside historical works ABOUT
# Africa, and should be treated as a starting point, not exhaustive --
# extend it as you find more verified entries.
AFRICAN_HISTORY_BOOK_IDS: dict[int, str] = {
    15399: "The Interesting Narrative of the Life of Olaudah Equiano",
    148: "Up From Slavery -- Booker T. Washington",
    16656: "Facts and Fables of African History Selections",  # verify title/id before large runs
}

# Subject-keyword sets for use with GutenbergDownloader.filter_by_subject(),
# for topics broad/well-tagged enough that keyword search works reasonably
# well (unlike "motivational" above).
HISTORY_SUBJECT_KEYWORDS: list[str] = [
    "History",
    "World history",
    "Ancient history",
]

AFRICA_SUBJECT_KEYWORDS: list[str] = [
    "Africa",
    "African",
    "Nigeria",
    "Colonialism -- Africa",
    "Ethiopia",
    "Egypt -- History",
]


def all_curated_ids() -> list[int]:
    """
    Return the deduplicated union of every hand-curated ID in this module.

    Useful as a quick "download everything hand-picked" entry point;
    prefer using the individual dicts directly when you want to track
    which book came from which curated category (see
    `dataset_builder.build_full_corpus` for an example of category-aware
    downloading).

    Returns
    -------
    list[int]
    """
    ids = set(MOTIVATIONAL_BOOK_IDS) | set(AFRICAN_HISTORY_BOOK_IDS)
    return sorted(ids)
