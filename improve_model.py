"""Train on the full benchmark, tune on validation only, evaluate on its test set."""

import argparse
import gc
import hashlib
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from model import BASE_DIR, MODEL_PATH, build_pipeline


def fingerprint(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).digest()


def clean_rows(rows, excluded=None):
    """Remove duplicate/conflicting texts; remove overlap with other partitions."""
    excluded = set() if excluded is None else excluded
    positions, conflicted = {}, set()
    texts, labels = [], []
    stats = Counter()
    for text, label in rows:
        stats["input_rows"] += 1
        if not isinstance(text, str) or not text.strip() or not any(c.isalpha() for c in text):
            stats["unusable"] += 1
            continue
        if label not in range(5):
            raise ValueError("Expected Yelp labels 0–4.")
        key = fingerprint(text)
        if key in excluded:
            stats["overlap_removed"] += 1
            continue
        if key in conflicted:
            stats["conflicting_rows"] += 1
            continue
        if key in positions:
            previous = positions[key]
            if labels[previous] != label + 1:
                texts[previous] = None
                conflicted.add(key)
                stats["conflicting_rows"] += 2
            else:
                stats["duplicates"] += 1
            continue
        positions[key] = len(texts)
        texts.append(text.strip())
        labels.append(label + 1)
    kept = [i for i, text in enumerate(texts) if text is not None]
    stats["usable_rows"] = len(kept)
    # Include rejected conflicting texts in the exclusion set as well.
    return ([texts[i] for i in kept], np.asarray([labels[i] for i in kept], dtype=np.int32),
            set(positions), dict(stats))


def read_parquet(path, excluded=None):
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        raise RuntimeError("Install requirements-training.txt to read Parquet datasets.") from None
    def rows():
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=10000, columns=["text", "label"]):
            data = batch.to_pydict()
            yield from zip(data["text"], data["label"])
    return clean_rows(rows(), excluded)


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {"accuracy": round(float(accuracy_score(actual, predicted)), 6),
            "mean_absolute_error": round(float(mean_absolute_error(actual, predicted)), 6),
            "within_one_star": round(float(np.mean(np.abs(actual - predicted) <= 1)), 6),
            "rows": len(actual)}


def predict_batches(pipeline, texts, size=2000):
    return np.concatenate([pipeline.predict(texts[start:start + size])
                           for start in range(0, len(texts), size)])


def save_json(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_comparison_baseline(texts, labels):
    """Make the workflow usable even in a fresh checkout without saved models."""
    baseline_path = MODEL_PATH.parent / "baseline_5k.joblib"
    if baseline_path.exists():
        return joblib.load(baseline_path)
    if MODEL_PATH.exists():
        baseline = joblib.load(MODEL_PATH)
    else:
        count = min(5000, len(texts))
        baseline = {"pipeline": build_pipeline().fit(texts[:count], labels[:count]),
                    "mode": "trained", "training_rows": count}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, baseline_path, compress=3)
    return baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=BASE_DIR / "data/yelp_full")
    parser.add_argument("--validation-rows", type=int, default=30000)
    parser.add_argument("--max-features", type=int, default=400000)
    args = parser.parse_args()
    if args.validation_rows < 1000 or args.max_features < 10000:
        parser.error("Use at least 1000 validation rows and 10000 features.")
    started = time.monotonic()
    log = lambda message: print(f"[{(time.monotonic() - started) / 60:.1f} min] {message}", flush=True)
    log("Reading and cleaning the full training partition.")
    texts, labels, training_keys, training_stats = read_parquet(args.directory / "train.parquet")
    log(f"Training data: {len(texts):,} usable reviews. {training_stats}")
    indices, validation_indices = train_test_split(
        np.arange(len(labels)), test_size=args.validation_rows, stratify=labels, random_state=42)
    train_texts = [texts[i] for i in indices]
    validation_texts = [texts[i] for i in validation_indices]
    y_train, y_validation = labels[indices], labels[validation_indices]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode",
                                 min_df=3, max_features=args.max_features, dtype=np.float32)
    log(f"Learning up to {args.max_features:,} word/phrase features on {len(indices):,} training reviews.")
    x_train = vectorizer.fit_transform(train_texts)
    x_validation = vectorizer.transform(validation_texts)
    del train_texts
    gc.collect()
    log(f"Sparse training matrix: {x_train.shape}, {x_train.nnz:,} nonzero features.")
    validation_results, best_accuracy, best_c = [], -1, None
    # The test partition is not read until model settings have been selected.
    for c in (0.25, 1.0, 4.0):
        log(f"Training Linear SVM C={c}; evaluating on validation only.")
        classifier = LinearSVC(C=c, dual="auto", max_iter=3000, tol=0.001, random_state=42)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(x_train, y_train)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        scores = metrics(y_validation, classifier.predict(x_validation))
        validation_results.append({"C": c, "converged": converged, **scores})
        log(f"Validation: {json.dumps(validation_results[-1])}")
        if converged and scores["accuracy"] > best_accuracy:
            best_accuracy, best_c = scores["accuracy"], c
        del classifier
    if best_c is None:
        raise RuntimeError("No candidate converged. Existing app model has been preserved.")
    del x_train, x_validation, validation_texts
    gc.collect()
    log(f"Selected C={best_c} using validation accuracy. Refitting on {len(texts):,} training reviews.")
    pipeline = Pipeline([("tfidf", vectorizer), ("classifier", LinearSVC(
        C=best_c, dual="auto", max_iter=3000, tol=0.001, random_state=42))])
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(texts, labels)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("Final classifier did not converge. Existing app model has been preserved.")
    log("Reading the separate official test partition; removing overlaps with all training data.")
    test_texts, test_labels, _, test_stats = read_parquet(args.directory / "test.parquet", training_keys)
    if len(test_texts) < 1000:
        raise RuntimeError("Too few uncontaminated test reviews. Existing model preserved.")
    predicted = predict_batches(pipeline, test_texts)
    scores = metrics(test_labels, predicted)
    baseline = load_comparison_baseline(texts, labels)
    baseline_predicted = predict_batches(baseline["pipeline"], test_texts)
    baseline_scores = metrics(test_labels, baseline_predicted)
    log(f"Final test: {json.dumps(scores)}. Previous model on SAME test: {json.dumps(baseline_scores)}")
    report = {"mode": "trained", "model_name": "TF-IDF + Linear SVM", "dataset": "Yelp/yelp_review_full",
              "dataset_url": "https://huggingface.co/datasets/Yelp/yelp_review_full",
              "training_rows": len(texts), "training_cleaning": training_stats, "test_cleaning": test_stats,
              "class_counts": {str(k): int(v) for k, v in sorted(Counter(labels).items())},
              "selection": {"validation_rows": args.validation_rows, "random_state": 42,
                            "max_features": args.max_features, "selected_C": best_c,
                            "candidates": validation_results},
              "evaluation": {"split": "official test; duplicates and train overlap removed", **scores,
                             "classification_report": classification_report(test_labels, predicted, output_dict=True),
                             "confusion_matrix": confusion_matrix(test_labels, predicted, labels=[1, 2, 3, 4, 5]).tolist()},
              "baseline_same_test": baseline_scores,
              "accuracy_gain_percentage_points": round(100 * (scores["accuracy"] - baseline_scores["accuracy"]), 4),
              "elapsed_minutes": round((time.monotonic() - started) / 60, 2)}
    save_json(MODEL_PATH.parent / "full_training_report.json", report)
    if scores["accuracy"] <= baseline_scores["accuracy"]:
        raise RuntimeError("Candidate did not beat the baseline. Existing app model preserved; report saved.")
    artifact = {"pipeline": pipeline, "mode": "trained", "model_name": report["model_name"],
                "training_rows": len(texts), "metrics": scores}
    temporary = MODEL_PATH.with_suffix(".tmp")
    joblib.dump(artifact, temporary, compress=3)
    temporary.replace(MODEL_PATH)
    save_json(MODEL_PATH.with_suffix(".json"), report)
    log(f"Improved model activated at {MODEL_PATH}. Restart the app to use it.")


if __name__ == "__main__":
    main()
