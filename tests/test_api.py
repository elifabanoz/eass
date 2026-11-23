import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from eass.api import create_scan, get_scan
from eass.db import Base


class AsyncUpload:
    filename = "sample.txt"

    def __init__(self, data: bytes):
        self.file = BytesIO(data)

    async def read(self, size: int) -> bytes:
        return self.file.read(size)

    async def close(self) -> None:
        self.file.close()


def complete(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("unexpected suspended coroutine")


class ApiTests(unittest.TestCase):
    def test_create_and_read_queued_scan(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = create_engine("sqlite:///:memory:")
            Base.metadata.create_all(engine)
            settings = SimpleNamespace(quarantine_dir=Path(temp), max_upload_bytes=1024)
            with Session(engine) as session:
                with (
                    patch("eass.api.get_settings", return_value=settings),
                    patch("eass.api.celery_app.send_task") as send_task,
                ):
                    accepted = complete(create_scan(AsyncUpload(b"ordinary text"), session))
                response = get_scan(accepted.scan_id, session)
                self.assertEqual(response.status, "queued")
                self.assertEqual(response.source, "rest")
                self.assertEqual(response.file_type, "text")
                self.assertEqual(response.size, 13)
                self.assertTrue((Path(temp) / accepted.scan_id).exists())
                send_task.assert_called_once_with("eass.scan", args=[accepted.scan_id])
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
