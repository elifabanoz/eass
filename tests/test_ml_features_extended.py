import unittest

import numpy as np

from eass.layers.ml_features import (
    FEATURE_DIM,
    _strings_from_bytes,
    vector_from_bytes,
    vector_from_record,
)


class ExtendedFeatureTests(unittest.TestCase):
    def test_bytes_match_record_vector(self):
        data = (
            b"MZ" + bytes(range(256)) * 10 + b" C:\\Windows https://example.test HKEY_LOCAL_MACHINE"
        )
        histogram = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).tolist()
        from eass.layers.ml_features import _byte_entropy

        record = {
            "histogram": histogram,
            "byteentropy": (_byte_entropy(data) * 2048).tolist(),
            "strings": _strings_from_bytes(data),
        }
        actual = vector_from_bytes(data)
        expected = vector_from_record(record)
        self.assertEqual(actual.shape, (FEATURE_DIM,))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_empty_and_invalid_records_rejected(self):
        with self.assertRaises(ValueError):
            vector_from_bytes(b"")
        with self.assertRaises(ValueError):
            vector_from_record({"histogram": [0] * 256, "byteentropy": [0] * 256})


if __name__ == "__main__":
    unittest.main()
