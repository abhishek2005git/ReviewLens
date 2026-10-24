"""One shared text-processing pipeline for training and prediction."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "review_model.joblib"
MAX_REVIEW_LENGTH = 5000


def build_pipeline():
    # Keep stop words: words like "not" matter when predicting sentiment.
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                  strip_accents="unicode", max_features=50000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                           random_state=42)),
    ])


def validate_review(value):
    if not isinstance(value, str):
        raise ValueError("Please enter your review as text.")
    review = value.strip()
    if len(review) < 3 or not any(char.isalpha() for char in review):
        raise ValueError("Please enter a review with at least 3 characters and some words.")
    if len(review) > MAX_REVIEW_LENGTH:
        raise ValueError(f"Please keep your review under {MAX_REVIEW_LENGTH:,} characters.")
    return review


def load_model():
    if MODEL_PATH.exists():
        # Only load artifacts produced locally by train.py; pickle is not safe
        # for arbitrary models downloaded from untrusted sources.
        return joblib.load(MODEL_PATH)
    from train import read_reviews
    texts, stars = read_reviews(BASE_DIR / "data" / "demo_reviews.csv")
    return {"pipeline": build_pipeline().fit(texts, stars), "mode": "demo",
            "training_rows": len(texts), "metrics": None}


def predict_review(artifact, review):
    review = validate_review(review)
    pipeline = artifact["pipeline"]
    vector = pipeline.named_steps["tfidf"].transform([review])
    if vector.nnz == 0:
        raise ValueError("The model doesn't recognize these words. Try a more detailed review in English.")
    rating = int(pipeline.predict([review])[0])
    return {"rating": rating, "mode": artifact["mode"]}
