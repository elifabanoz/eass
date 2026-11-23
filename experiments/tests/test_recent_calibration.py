import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.calibrate_recent_model import ember_validation, threshold_for_fpr
from experiments.train_full_more import sample_training


class Extractor:
    dim = 2

    def process_raw_features(self, record):
        return np.asarray([record["id"], record["label"]], dtype=np.float32)


class RecentCalibrationTests(unittest.TestCase):
    def test_validation_reservoir_matches_training_selection(self):
        records = [{"id": index, "label": index % 2, "appeared": "2018-01"} for index in range(80)]
        with patch("experiments.train_full_more.remote_records", return_value=iter(records)):
            full, _, _ = sample_training(Extractor(), per_class=10)
        with patch("experiments.calibrate_recent_model.remote_records", return_value=iter(records)):
            selected, labels, _ = ember_validation(Extractor(), per_class=10)
        indices = np.arange(20)
        _, validation_indices = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=indices >= 10
        )
        np.testing.assert_array_equal(selected, full[validation_indices])
        np.testing.assert_array_equal(labels, validation_indices >= 10)

    def test_fpr_cutoff_handles_ties(self):
        scores = np.asarray([0.1, 0.2, 0.2, 0.9])
        threshold = threshold_for_fpr(scores, 0.25)
        self.assertEqual(int(np.sum(scores >= threshold)), 1)

    def test_fpr_cutoff_rejects_non_finite_scores(self):
        with self.assertRaises(ValueError):
            threshold_for_fpr(np.asarray([0.2, np.nan]), 0.01)


if __name__ == "__main__":
    unittest.main()
