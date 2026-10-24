"""Download the full public YelpReviewFull benchmark with resumable transfers."""

import argparse
import subprocess
from pathlib import Path

from model import BASE_DIR

DATASET_URL = "https://huggingface.co/datasets/Yelp/yelp_review_full"
DATA_DIR = BASE_DIR / "data/yelp_full"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        parser.error("Install training dependencies: python -m pip install -r requirements-training.txt")
    args.directory.mkdir(parents=True, exist_ok=True)
    for split, expected_rows in (("train", 650000), ("test", 50000)):
        target = args.directory / f"{split}.parquet"
        if target.exists():
            if parquet.ParquetFile(target).metadata.num_rows != expected_rows:
                parser.error(f"Unexpected row count in {target}; move it aside and retry.")
            print(f"Already downloaded: {target}", flush=True)
            continue
        temporary = target.with_suffix(".parquet.download")
        url = f"{DATASET_URL}/resolve/main/yelp_review_full/{split}-00000-of-00001.parquet"
        print(f"Downloading {split}: {expected_rows:,} reviews", flush=True)
        subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--retry-delay", "5",
                        "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "30",
                        "--continue-at", "-", "--output", str(temporary), url], check=True)
        if parquet.ParquetFile(temporary).metadata.num_rows != expected_rows:
            parser.error(f"Unexpected row count in {temporary}; the download was not activated.")
        temporary.replace(target)
    print("Dataset ready. Run: python improve_model.py", flush=True)


if __name__ == "__main__":
    main()
