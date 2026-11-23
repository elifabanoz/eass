import unittest

from eass.scoring import score_layers


class ScoringTests(unittest.TestCase):
    def test_eicar_hash_is_reported_as_test_indicator(self):
        result = score_layers(
            "text", {"hash": {"status": "complete", "matched": True, "source": "eicar-test"}}
        )
        self.assertEqual(result["verdict"], "malicious")
        self.assertIn("eicar-test", result["reasons"][0])

    def test_high_entropy_alone_is_not_malicious(self):
        result = score_layers(
            "pe",
            {
                "pe": {
                    "status": "complete",
                    "sections": [{"name": ".text", "entropy": 7.61, "raw_size": 2048}],
                    "upx_sections": [],
                    "suspicious_imports": [],
                    "timestamp_anomaly": False,
                }
            },
        )
        self.assertEqual(result["score"], 15)
        self.assertEqual(result["verdict"], "clean")
        self.assertIn("7.61", result["reasons"][0])

    def test_archive_is_not_called_clean(self):
        result = score_layers("zip", {"hash": {"status": "complete", "matched": False}})
        self.assertIsNone(result["verdict"])

    def test_pdf_without_match_is_not_called_clean(self):
        result = score_layers("pdf", {"hash": {"status": "complete", "matched": False}})
        self.assertIsNone(result["verdict"])
        self.assertIn("Only hash and YARA", result["reasons"][0])

    def test_layer_failure_is_not_called_clean(self):
        result = score_layers("pe", {"pe": {"status": "error"}})
        self.assertIsNone(result["verdict"])

    def test_ml_signal_is_explained(self):
        result = score_layers(
            "pe",
            {"ml": {"status": "complete", "probability": 0.92, "threshold": 0.8}},
        )
        self.assertEqual(result["verdict"], "suspicious")
        self.assertIn("0.920", result["reasons"][0])


if __name__ == "__main__":
    unittest.main()
