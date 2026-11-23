import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from eass.db import Base
from eass.models import Scan
from eass.worker import scan_task


class WorkerTests(unittest.TestCase):
    def test_worker_records_result_and_removes_file(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = create_engine("sqlite:///:memory:")
            Base.metadata.create_all(engine)
            sessions = sessionmaker(bind=engine, expire_on_commit=False)
            data = b"ordinary text"
            scan_id = "c020b6ac-3850-4e9c-a3a2-866384cb1350"
            path = Path(temp) / scan_id
            path.write_bytes(data)
            with Session(engine) as session:
                session.add(
                    Scan(
                        id=scan_id,
                        filename="sample.txt",
                        file_type="text",
                        size=len(data),
                        sha256=hashlib.sha256(data).hexdigest(),
                        md5=hashlib.md5(data, usedforsecurity=False).hexdigest(),
                    )
                )
                session.commit()
            with (
                patch("eass.worker.SessionLocal", sessions),
                patch(
                    "eass.worker.get_settings",
                    return_value=SimpleNamespace(quarantine_dir=Path(temp)),
                ),
            ):
                scan_task.run(scan_id)
            with Session(engine) as session:
                scan = session.get(Scan, scan_id)
                self.assertEqual(scan.status, "completed")
                self.assertEqual(scan.attempts, 1)
                self.assertEqual(scan.result["verdict"], "clean")
            self.assertFalse(path.exists())
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
