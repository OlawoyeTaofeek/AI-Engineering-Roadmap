import urllib.request
import zipfile
import os
from pathlib import Path


def download_and_unzip_spam_data(
    primary_url: str = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    backup_url: str = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip",
    zip_path: str = "sms_spam_collection.zip",
    extracted_path: str = "sms_spam_collection",
    data_file_path: str = "sms_spam_collection/SMSSpamCollection.tsv",
) -> Path:
    """
    Download and extract the SMS Spam Collection dataset.

    Tries the original UCI URL first, then falls back to the book author's
    Backblaze B2 mirror (the same fallback used in the official
    rasbt/LLMs-from-scratch repo) if UCI is unreachable, rate-limited,
    or times out.
    """
    data_file_path = Path(data_file_path)

    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return data_file_path

    if not os.path.exists(zip_path):
        for label, url in (("primary (UCI)", primary_url), ("backup (Backblaze)", backup_url)):
            print(f"Trying {label}: {url}")
            try:
                urllib.request.urlretrieve(url, zip_path)
                print(f"Saved to {zip_path}")
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                print(f"{label} failed: {e}")
        else:
            raise urllib.error.URLError(
                "Both primary and backup download URLs failed. "
                "Check your network connection, or manually download the zip "
                f"from {backup_url} and place it at '{zip_path}'."
            )
    else:
        print(f"{zip_path} already exists. Skipping download.")

    print("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)
    except zipfile.BadZipFile as e:
        raise zipfile.BadZipFile(
            f"{zip_path} is not a valid zip file — download may be corrupted. "
            f"Try deleting {zip_path} and re-running."
        ) from e

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    if not original_file_path.exists():
        raise FileNotFoundError(
            f"Expected {original_file_path} after extraction, but it wasn't found. "
            f"Contents of {extracted_path}: {list(Path(extracted_path).iterdir())}"
        )

    original_file_path.rename(data_file_path)
    print(f"File saved as {data_file_path}")

    return data_file_path