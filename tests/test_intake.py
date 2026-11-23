import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from eass.layers.intake import EmptyFile, FileTooLarge, detect_type, store_upload


class AsyncUpload:
    def __init__(self, data: bytes):
        self.file = BytesIO(data)

    async def read(self, size: int) -> bytes:
        return self.file.read(size)


def complete(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("unexpected suspended coroutine")


class IntakeTests(unittest.TestCase):
    def test_extension_does_not_control_type(self):
        self.assertEqual(detect_type(b"PK\x03\x04other data"), "zip")
        self.assertEqual(detect_type(b"ordinary text"), "text")

    def test_limit_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            upload = AsyncUpload(b"123456")
            with self.assertRaises(FileTooLarge):
                complete(store_upload(upload, Path(temp), 5))
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_stored_file_has_hash_and_type(self):
        with tempfile.TemporaryDirectory() as temp:
            upload = AsyncUpload(b"ordinary text")
            result = complete(store_upload(upload, Path(temp), 100))
            self.assertEqual(result.file_type, "text")
            self.assertEqual(result.size, 13)
            self.assertEqual(len(result.sha256), 64)
            self.assertTrue(result.path.exists())

    def test_empty_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(EmptyFile):
                complete(store_upload(AsyncUpload(b""), Path(temp), 100))
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
