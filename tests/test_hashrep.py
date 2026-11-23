import io
import json
import os
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from eass.layers.hashrep import EICAR_SHA256, check_local, check_malwarebazaar
from eass.update_hashes import attempt_refresh, extract_hashes, refresh


class HashReputationTests(unittest.TestCase):
    def test_local_eicar_indicator(self):
        with tempfile.TemporaryDirectory() as temp:
            result = check_local(EICAR_SHA256, Path(temp) / "missing.txt")
        self.assertEqual(result["source"], "eicar-test")

    def test_export_parser_ignores_invalid_rows(self):
        other = "a" * 64
        rows = io.StringIO(f"first_seen,sha256_hash\n2026-11-17,{other}\ninvalid,short\n")
        self.assertEqual(extract_hashes(rows), {other})

    def test_zipped_hash_export_replaces_local_list(self):
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.writestr("full_sha256.txt", "# MalwareBazaar\n" + "a" * 64 + "\n")
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "hashes.txt"
            target.write_text("b" * 64 + "\n", encoding="utf-8")
            with patch("eass.update_hashes.urllib.request.build_opener") as opener:
                opener.return_value.open.return_value = io.BytesIO(archive_bytes.getvalue())
                count = refresh(
                    "https://mb-api.abuse.ch/v2/files/exports/key/sha256_full.txt.zip", target
                )
            self.assertEqual(count, 1)
            self.assertEqual(target.read_text(encoding="utf-8"), "a" * 64 + "\n")

    def test_failed_refresh_keeps_list_and_waits_an_hour_across_restarts(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "hashes.txt"
            old = "b" * 64 + "\n"
            target.write_text(old, encoding="utf-8")
            now = time.time()
            os.utime(target, (now - 7200, now - 7200))
            url = "https://mb-api.abuse.ch/v2/files/exports/key/sha256_full.txt.zip"
            with patch("eass.update_hashes.refresh", side_effect=RuntimeError("network")) as fetch:
                self.assertTrue(attempt_refresh(url, target, now))
                self.assertFalse(attempt_refresh(url, target, now + 3599))
                self.assertTrue(attempt_refresh(url, target, now + 3600))
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(target.read_text(encoding="utf-8"), old)

    def test_recent_list_delays_first_scheduled_fetch(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "hashes.txt"
            target.write_text("a" * 64 + "\n", encoding="utf-8")
            with patch("eass.update_hashes.refresh") as fetch:
                self.assertFalse(attempt_refresh("https://mb-api.abuse.ch/", target, time.time()))
            fetch.assert_not_called()

    def test_remote_lookup_is_skipped_without_key(self):
        self.assertEqual(check_malwarebazaar("a" * 64, None)["status"], "skipped")

    def test_remote_lookup_sends_only_hash(self):
        response = io.BytesIO(json.dumps({"query_status": "ok"}).encode())
        with patch("urllib.request.urlopen", return_value=response) as opener:
            result = check_malwarebazaar("a" * 64, "test-key")
        request = opener.call_args.args[0]
        self.assertEqual(request.get_header("Auth-key"), "test-key")
        self.assertEqual(request.data, b"query=get_info&hash=" + b"a" * 64)
        self.assertTrue(result["matched"])


if __name__ == "__main__":
    unittest.main()
