import unittest

import numpy as np

from eass.layers.ml import vector_from_bytes, vector_from_record


class FeatureTests(unittest.TestCase):
    def test_training_and_inference_histograms_match(self):
        data = bytes([0, 0, 1, 255])
        counts = [0] * 256
        counts[0], counts[1], counts[255] = 2, 1, 1
        np.testing.assert_array_equal(
            vector_from_bytes(data), vector_from_record({"histogram": counts})
        )

    def test_empty_histogram_is_rejected(self):
        with self.assertRaises(ValueError):
            vector_from_record({"histogram": [0] * 256})


if __name__ == "__main__":
    unittest.main()
