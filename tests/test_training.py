import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from improve_model import clean_rows, fingerprint, load_comparison_baseline, metrics


class FullTrainingTests(unittest.TestCase):
    def test_duplicates_and_all_conflicting_labels_are_removed(self):
        rows = [("Great food", 4), ("great   food", 4), ("Conflicting review", 0),
                ("conflicting review", 3), ("CONFLICTING REVIEW", 0), ("!!!", 2)]
        texts, labels, excluded, stats = clean_rows(rows)
        self.assertEqual(texts, ["Great food"])
        self.assertEqual(labels.tolist(), [5])
        self.assertIn(fingerprint("Conflicting review"), excluded)
        self.assertEqual(stats["usable_rows"], 1)

    def test_test_reviews_overlapping_training_are_excluded(self):
        _, _, training_keys, _ = clean_rows([("A repeated review", 2)])
        texts, labels, _, stats = clean_rows(
            [("A  REPEATED review", 2), ("An independent review", 4)], training_keys)
        self.assertEqual(texts, ["An independent review"])
        self.assertEqual(labels.tolist(), [5])
        self.assertEqual(stats["overlap_removed"], 1)

    def test_metrics_distinguish_exact_adjacent_and_large_errors(self):
        scores = metrics([1, 3, 5], [1, 4, 3])
        self.assertAlmostEqual(scores["accuracy"], 1 / 3, places=5)
        self.assertAlmostEqual(scores["within_one_star"], 2 / 3, places=5)
        self.assertEqual(scores["mean_absolute_error"], 1)

    def test_fresh_checkout_creates_a_comparison_baseline(self):
        texts = [f"A review about food with rating {star} sample {i}"
                 for star in range(1, 6) for i in range(5)]
        labels = [star for star in range(1, 6) for _ in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "review_model.joblib"
            with patch("improve_model.MODEL_PATH", target):
                baseline = load_comparison_baseline(texts, labels)
            self.assertEqual(baseline["training_rows"], 25)
            self.assertTrue(target.with_name("baseline_5k.joblib").exists())
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
