"""
Gutenberg Corpus Builder (v2 -- higher quality, English-only)
===============================================================
Replaces the archive.org-wide search with Project Gutenberg via the
free Gutendex API (https://gutendex.com). Gutenberg only hosts books
that are confirmed public domain (in the US) -- no government reports,
journal articles, arxiv preprints, or FOIA documents in the catalogue
at all, which is what was polluting the archive.org results.

Install:
    pip install requests

Run:
    python gutenberg_corpus_builder.py --limit-per-topic 20
"""

import csv
import os
import re
import time

import requests

GUTENDEX_URL = "https://gutendex.com/books/"

# Gutendex searches title/author/subject text. Multiple terms per topic
# widen coverage the same way the archive.org queries did.
TOPICS = {
    "science": ["natural philosophy", "physics", "chemistry", "astronomy"],
    "mathematics": ["mathematics", "geometry", "algebra", "arithmetic"],
    "world_history": ["history of the world", "ancient history", "world war history"],
    "african_history": ["Africa history", "Egypt history", "Ethiopia history", "African kingdoms"],
    "religious_history": ["history of religion", "church history", "comparative religion", "mythology"],
    "relationships": ["marriage", "courtship", "domestic life"],  # expect thin results -- see note below
}


def _get_with_retry(url: str, params: dict = None, timeout: int = 30, max_attempts: int = 4):
    """
    GET with exponential backoff. Free public APIs (Gutendex included)
    occasionally time out or rate-limit under load -- retrying a couple
    times with increasing delay clears most of these transparently,
    without needing to babysit the script.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < max_attempts:
                wait = 2 ** attempt  # 2s, 4s, 8s...
                print(f"    (request failed: {e.__class__.__name__} -- retrying in {wait}s, "
                      f"attempt {attempt}/{max_attempts})")
                time.sleep(wait)
    raise last_exc


def search_gutenberg(term: str, limit: int = 20) -> list:
    """
    Query Gutendex. `languages=en` restricts to English at the API level
    (far more reliable than guessing from OCR text). Copyright filtering
    is implicit: Gutenberg only lists books it has cleared as public
    domain, so there's no need to re-derive that ourselves.
    """
    results = []
    url = GUTENDEX_URL
    params = {"search": term, "languages": "en"}

    while url and len(results) < limit:
        try:
            resp = _get_with_retry(url, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  Giving up on search term '{term}' after retries: {e}")
            break
        data = resp.json()
        results.extend(data["results"])
        url = data.get("next")
        params = None  # `next` already contains full query string
        time.sleep(0.3)

    return results[:limit]


def pick_text_format(formats: dict):
    """Prefer plain UTF-8 text; fall back to any text/plain variant."""
    for key in ("text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain"):
        if key in formats:
            return formats[key]
    for k, v in formats.items():
        if k.startswith("text/plain"):
            return v
    return None


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name)[:80]


def download_book(book: dict, out_dir: str):
    gid = book["id"]
    title = book.get("title", f"book_{gid}")
    url = pick_text_format(book.get("formats", {}))
    if not url:
        return gid, "skipped (no plain text format)", None

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        return gid, f"error: {e}", None

    os.makedirs(out_dir, exist_ok=True)
    filename = f"{gid}_{sanitize_filename(title)}.txt"
    dest = os.path.join(out_dir, filename)
    with open(dest, "wb") as fh:
        fh.write(r.content)
    return gid, "downloaded", dest


def build_corpus(topics: dict = TOPICS, limit_per_topic: int = 20,
                  base_dir: str = "gutenberg_corpus", manifest_path: str = "gutenberg_manifest.csv"):
    manifest_rows = []

    for topic, terms in topics.items():
        topic_dir = os.path.join(base_dir, topic)
        seen_ids = set()
        candidates = []

        for term in terms:
            try:
                books = search_gutenberg(term, limit=limit_per_topic)
            except Exception as e:
                print(f"  Search term '{term}' failed unexpectedly ({e}), skipping it.")
                continue
            for b in books:
                if b["id"] not in seen_ids:
                    seen_ids.add(b["id"])
                    candidates.append(b)

        candidates = candidates[:limit_per_topic]
        print(f"\n=== {topic} === {len(terms)} search terms -> {len(candidates)} candidate books")

        if not candidates:
            print(f"  No English public-domain Gutenberg books found for '{topic}' "
                  f"with the current search terms -- try broadening them.")
            continue

        for book in candidates:
            gid, status, dest = download_book(book, topic_dir)
            authors = ", ".join(a["name"] for a in book.get("authors", [])) or "unknown"
            print(f"  [{gid}] {book.get('title', '?')[:60]:60s} by {authors:30s} {status}")
            manifest_rows.append({
                "topic": topic,
                "gutenberg_id": gid,
                "title": book.get("title", ""),
                "authors": authors,
                "subjects": "; ".join(book.get("subjects", [])),
                "status": status,
                "path": dest or "",
            })

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "topic", "gutenberg_id", "title", "authors", "subjects", "status", "path"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    downloaded = sum(1 for r in manifest_rows if r["status"] == "downloaded")
    print(f"\nDone. {downloaded}/{len(manifest_rows)} books downloaded. Manifest -> {manifest_path}")
    return manifest_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build an English public-domain book corpus via Project Gutenberg")
    parser.add_argument("--limit-per-topic", type=int, default=20)
    parser.add_argument("--base-dir", default="gutenberg_corpus")
    parser.add_argument("--manifest", default="gutenberg_manifest.csv")
    args = parser.parse_args()

    build_corpus(limit_per_topic=args.limit_per_topic, base_dir=args.base_dir, manifest_path=args.manifest)