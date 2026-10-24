"""Download a small real Yelp sample without downloading the full archive."""

import argparse
import csv
import json
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi

from model import BASE_DIR

DATASET_URL = "https://huggingface.co/datasets/Yelp/yelp_review_full"


def fetch_page(offset, length):
    query = urlencode({"dataset": "Yelp/yelp_review_full", "config": "yelp_review_full",
                       "split": "train", "offset": offset, "length": length})
    request = Request("https://datasets-server.huggingface.co/rows?" + query,
                      headers={"User-Agent": "review-stars-local/1.0"})
    for attempt in range(6):
        try:
            with urlopen(request, timeout=45, context=ssl.create_default_context(cafile=certifi.where())) as response:
                payload = json.load(response)
            rows = payload["rows"]
            if len(rows) != length:
                raise ValueError(f"Expected {length} reviews at offset {offset}; received {len(rows)}.")
            converted = []
            for item in rows:
                if item.get("truncated_cells"):
                    raise ValueError("The dataset service truncated a review. No sample was saved.")
                row = item["row"]
                if not isinstance(row.get("label"), int) or row["label"] not in range(5):
                    raise ValueError("Unexpected dataset labels.")
                converted.append({"text": row["text"], "stars": row["label"] + 1})
            return converted
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 5:
                raise
            delay = 15 * (attempt + 1) if isinstance(error, HTTPError) and error.code == 429 else 2 ** attempt
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000, help="100–50000 reviews (default: 5000)")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data/yelp_sample.csv")
    args = parser.parse_args()
    if not 100 <= args.rows <= 50000:
        parser.error("--rows must be between 100 and 50000")
    if args.output.exists():
        parser.error(f"{args.output} already exists; choose a different --output to preserve it.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".download")
    print(f"Downloading {args.rows:,} real reviews from {DATASET_URL}", flush=True)
    try:
        offsets = list(range(0, args.rows, 100))
        with ThreadPoolExecutor(max_workers=1) as executor, temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=["text", "stars"])
            writer.writeheader()
            pages = executor.map(lambda offset: fetch_page(offset, min(100, args.rows - offset)), offsets)
            count = 0
            for page in pages:
                writer.writerows(page)
                count += len(page)
                if count % 1000 == 0 or count == args.rows:
                    print(f"  {count:,} / {args.rows:,}", flush=True)
        temporary.replace(args.output)
    except (OSError, ValueError, KeyError) as error:
        temporary.unlink(missing_ok=True)
        parser.exit(1, f"Download failed: {error}\nPlease retry when the dataset service is available.\n")
    print(f"Saved {args.output} ({args.output.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
