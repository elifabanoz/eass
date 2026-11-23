import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eass.pipeline import run_scan


class PipelineTests(unittest.TestCase):
    def test_benign_text_has_layer_results(self):
        data = b"ordinary text"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample"
            path.write_bytes(data)
            result = run_scan(path, "text", hashlib.sha256(data).hexdigest())
        self.assertEqual(result["verdict"], "clean")
        self.assertEqual(result["layers"]["hash"]["status"], "complete")
        self.assertEqual(result["layers"]["yara"]["matches"], [])
        self.assertEqual(result["layers"]["pe"]["status"], "skipped")
        self.assertEqual(result["layers"]["ml"]["status"], "skipped")

    def test_invalid_pe_is_reported_as_incomplete(self):
        data = b"MZ" + bytes(128)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken"
            path.write_bytes(data)
            result = run_scan(path, "pe", hashlib.sha256(data).hexdigest())
        self.assertEqual(result["layers"]["pe"]["status"], "error")
        self.assertIsNone(result["verdict"])

    def test_malwarebazaar_failure_does_not_stop_other_layers(self):
        data = b"ordinary text"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample"
            path.write_bytes(data)
            with patch("eass.pipeline.hashrep.check_malwarebazaar", side_effect=TimeoutError):
                result = run_scan(path, "text", hashlib.sha256(data).hexdigest())
        self.assertEqual(result["layers"]["malwarebazaar"]["status"], "error")
        self.assertEqual(result["layers"]["yara"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
