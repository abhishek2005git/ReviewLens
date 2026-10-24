"""Train on review-level CSV, TSV, or Yelp JSONL; no business merge needed."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from model import BASE_DIR, MODEL_PATH, build_pipeline


def read_reviews(path, max_rows=None):
    texts, stars = [], []
    seen = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        if Path(path).suffix.lower() in {".json", ".jsonl"}:
            rows = (json.loads(line) for line in source if line.strip())
        else:
            rows = csv.DictReader(source, delimiter="\t" if Path(path).suffix.lower() == ".tsv" else ",")
            if not {"text", "stars"}.issubset(rows.fieldnames or []):
                raise ValueError("Dataset must contain text and stars columns.")
        for row_number, row in enumerate(rows, 1):
            if not isinstance(row, dict) or "text" not in row or "stars" not in row:
                raise ValueError(f"Row {row_number}: expected text and stars fields.")
            text = row["text"]
            if not isinstance(text, str):
                raise ValueError(f"Row {row_number}: review text must be a string.")
            if not text.strip() or not any(c.isalpha() for c in text):
                # Public datasets can contain punctuation-only or empty reviews.
                continue
            try:
                star = float(row["stars"])
            except (TypeError, ValueError):
                raise ValueError(f"Row {row_number}: stars must be an integer from 1 to 5.") from None
            if not math.isfinite(star) or star not in (1, 2, 3, 4, 5):
                raise ValueError(f"Row {row_number}: stars must be an integer from 1 to 5.")
            # Prevent duplicate reviews leaking into both evaluation partitions.
            key = " ".join(text.lower().split())
            if key in seen:
                if seen[key] != int(star):
                    raise ValueError(f"Row {row_number}: duplicate review has conflicting ratings.")
                continue
            seen[key] = int(star)
            texts.append(text.strip())
            stars.append(int(star))
            if max_rows is not None and len(texts) >= max_rows:
                break
    if set(stars) != {1, 2, 3, 4, 5} or min(Counter(stars).values(), default=0) < 5:
        raise ValueError("Provide at least 5 distinct reviews for each rating from 1 through 5.")
    return texts, stars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="CSV/TSV with text,stars columns or Yelp review JSONL")
    parser.add_argument("--max-rows", type=int, default=100000, help="Maximum unique reviews to load (default: 100000)")
    args = parser.parse_args()
    if args.max_rows < 25:
        parser.error("--max-rows must be at least 25")
    try:
        texts, stars = read_reviews(args.dataset, args.max_rows)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    x_train, x_test, y_train, y_test = train_test_split(
        texts, stars, test_size=0.2, random_state=42, stratify=stars)
    pipeline = build_pipeline().fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    metrics = {"accuracy": round(accuracy_score(y_test, predicted), 4),
               "mean_absolute_error": round(mean_absolute_error(y_test, predicted), 4),
               "within_one_star": round(sum(abs(actual - int(guess)) <= 1 for actual, guess in zip(y_test, predicted)) / len(y_test), 4),
               "test_rows": len(y_test)}
    # Evaluate on held-out data first, then fit the saved model on all data.
    pipeline.fit(texts, stars)
    mode = "demo" if args.dataset.resolve() == (BASE_DIR / "data/demo_reviews.csv").resolve() else "trained"
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".tmp")
    joblib.dump({"pipeline": pipeline, "mode": mode, "training_rows": len(texts),
                 "metrics": metrics}, temporary)
    temporary.replace(MODEL_PATH)
    report = {"mode": mode, "dataset": args.dataset.name, "training_rows": len(texts),
              "class_counts": dict(sorted(Counter(stars).items())),
              "evaluation": {"split": "stratified 80/20", "random_state": 42, **metrics}}
    MODEL_PATH.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": mode, "training_rows": len(texts), **metrics}, indent=2))
    print(f"Saved {MODEL_PATH}. Restart the app to use the new model.")


if __name__ == "__main__":
    main()
