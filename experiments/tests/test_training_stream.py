import hashlib
import io
import json
import tarfile
import unittest
from unittest.mock import patch

from experiments.train_full_more import remote_records


class StreamTests(unittest.TestCase):
    def test_stream_checks_archive_hash_after_reading_records(self):
        payload = (json.dumps({"label": 1, "appeared": "2018-01"}) + "\n").encode()
        archive_data = io.BytesIO()
        with tarfile.open(fileobj=archive_data, mode="w:bz2") as archive:
            member = tarfile.TarInfo("ember2018/train_features_0.jsonl")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        compressed = archive_data.getvalue()
        with (
            patch("experiments.train_full_more.urlopen", return_value=io.BytesIO(compressed)),
            patch(
                "experiments.train_full_more.EMBER2018_SHA256",
                hashlib.sha256(compressed).hexdigest(),
            ),
        ):
            records = list(remote_records("train"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["appeared"], "2018-01")


if __name__ == "__main__":
    unittest.main()
