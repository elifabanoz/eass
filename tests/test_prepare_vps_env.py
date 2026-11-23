import tempfile
import unittest
from pathlib import Path

from scripts.prepare_vps_env import prepare


class VpsEnvironmentTests(unittest.TestCase):
    def test_credentials_are_generated_without_overwriting_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / ".env"
            output = Path(temp) / "vps.env"
            source.write_text("MALWAREBAZAAR_AUTH_KEY=test-key\n", encoding="utf-8")
            prepare("demo.example.com", source, output)
            values = dict(line.split("=", 1) for line in output.read_text().splitlines())
            self.assertEqual(values["MALWAREBAZAAR_AUTH_KEY"], "test-key")
            self.assertGreaterEqual(len(values["EASS_API_KEY"]), 40)
            self.assertEqual(len(values["EASS_DB_PASSWORD"]), 64)
            with self.assertRaises(FileExistsError):
                prepare("demo.example.com", source, output)

    def test_domain_must_be_a_dns_name(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / ".env"
            source.write_text("MALWAREBAZAAR_AUTH_KEY=test-key\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare("http://example.com", source, Path(temp) / "vps.env")


if __name__ == "__main__":
    unittest.main()
