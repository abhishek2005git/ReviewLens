import csv
import tempfile
import unittest
from pathlib import Path

from app import create_app
from model import BASE_DIR, build_pipeline
from train import read_reviews


class ReviewAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts, stars = read_reviews(BASE_DIR / "data/demo_reviews.csv")
        cls.artifact = {"pipeline": build_pipeline().fit(texts, stars),
                        "mode": "demo", "training_rows": len(texts), "metrics": None}

    def setUp(self):
        self.client = create_app(self.artifact).test_client()

    def test_home_and_assets(self):
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn(b"Demo model", home.data)
        for asset in ("app.js", "style.css"):
            response = self.client.get(f"/static/{asset}")
            self.assertEqual(response.status_code, 200)
            response.close()

    def test_prediction_is_integer_star_on_both_sentiments(self):
        bad = self.client.post("/api/predict", json={"review": "Terrible disgusting food. Rude staff. Never coming back."})
        good = self.client.post("/api/predict", json={"review": "Amazing delicious food. Excellent wonderful service. Perfect!"})
        self.assertEqual(bad.status_code, 200)
        self.assertEqual(good.status_code, 200)
        self.assertLess(bad.json["rating"], good.json["rating"])
        for response in (bad, good):
            self.assertIn(response.json["rating"], range(1, 6))
            self.assertEqual(response.json["mode"], "demo")

    def test_invalid_review_inputs(self):
        for value in (None, "", "  ", "12", "123456", "!!!", [], 5, "a" * 5001, "xyzzzyqv"):
            with self.subTest(value=str(value)[:30]):
                response = self.client.post("/api/predict", json={"review": value})
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json)

    def test_request_errors_are_json(self):
        for body in ([], "text", None):
            response = self.client.post("/api/predict", data=__import__('json').dumps(body), content_type="application/json")
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.post("/api/predict", data="plain").status_code, 415)
        malformed = self.client.post("/api/predict", data="{", content_type="application/json")
        self.assertEqual(malformed.status_code, 400)
        self.assertIn("error", malformed.json)
        large = self.client.post("/api/predict", json={"review": "x" * 70000})
        self.assertEqual(large.status_code, 413)
        self.assertIn("error", large.json)

    def test_duplicate_reviews_do_not_leak_into_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            rows = [{"text": f"Review {star} example {i}", "stars": star} for star in range(1, 6) for i in range(5)]
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=["text", "stars"])
                writer.writeheader()
                writer.writerows(rows + [rows[0], {"text": "!!!", "stars": 3},
                                         {"text": "", "stars": 1}])
            texts, stars = read_reviews(path)
            self.assertEqual(len(texts), 25)
            with path.open("a") as output:
                output.write('"Conflicting review",1\n"Conflicting review",2\n')
            with self.assertRaisesRegex(ValueError, "conflicting"):
                read_reviews(path)


if __name__ == "__main__":
    unittest.main()
