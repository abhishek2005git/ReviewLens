import io
import json
import unittest
from unittest.mock import patch

from download_sample import fetch_page


class SampleDownloadTests(unittest.TestCase):
    def test_zero_based_labels_become_one_to_five_stars(self):
        payload = {"rows": [{"row": {"text": f"Review {i}", "label": i},
                             "truncated_cells": []} for i in range(5)]}
        with patch("download_sample.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
            rows = fetch_page(0, 5)
        self.assertEqual([row["stars"] for row in rows], [1, 2, 3, 4, 5])

    def test_partial_page_is_rejected(self):
        with patch("download_sample.urlopen", return_value=io.BytesIO(b'{"rows": []}')):
            with self.assertRaisesRegex(ValueError, "Expected"):
                fetch_page(0, 100)

    def test_truncated_review_is_rejected(self):
        payload = {"rows": [{"row": {"text": "Partial", "label": 0},
                             "truncated_cells": ["text"]}]}
        with patch("download_sample.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaisesRegex(ValueError, "truncated"):
                fetch_page(0, 1)


if __name__ == "__main__":
    unittest.main()
