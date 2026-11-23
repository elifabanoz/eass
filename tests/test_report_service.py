import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from eass.api import get_report
from eass.db import Base
from eass.models import Scan, ScanReport
from eass.report.service import schedule_report
from eass.worker import build_report_task


class ReportServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = create_engine("sqlite:///" + (Path(self.temp.name) / "scans.db").as_posix())
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.scan_id = "c020b6ac-3850-4e9c-a3a2-866384cb1350"
        self.result = {
            "verdict": "malicious",
            "score": 100,
            "reasons": ["SHA-256 matched eicar-test"],
            "layers": {"hash": {"status": "complete", "matched": True, "source": "eicar-test"}},
        }
        with self.sessions() as session:
            session.add(
                Scan(
                    id=self.scan_id,
                    status="completed",
                    filename="sample.txt",
                    file_type="text",
                    size=68,
                    sha256="a" * 64,
                    md5="b" * 32,
                    result=self.result,
                )
            )
            session.commit()

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def test_report_is_queued_and_saved_separately(self):
        settings = SimpleNamespace(reports_enabled=True)
        with (
            patch("eass.report.service.get_settings", return_value=settings),
            patch("eass.report.service.SessionLocal", self.sessions),
            patch("eass.worker.celery_app.send_task") as send_task,
        ):
            schedule_report(self.scan_id)
            schedule_report(self.scan_id)
        send_task.assert_called_once_with("eass.build_report", args=[self.scan_id])
        with (
            patch("eass.worker.get_settings", return_value=settings),
            patch("eass.worker.SessionLocal", self.sessions),
            patch("eass.worker.build_report", return_value={"summary": "test"}),
        ):
            build_report_task.run(self.scan_id)
        with self.sessions() as session, patch("eass.api.get_settings", return_value=settings):
            response = get_report(self.scan_id, session)
            scan = session.get(Scan, self.scan_id)
            record = session.get(ScanReport, self.scan_id)
        self.assertEqual(response.status, "done")
        self.assertEqual(response.report, {"summary": "test"})
        self.assertEqual(record.status, "done")
        self.assertEqual(scan.result, self.result)
        self.assertEqual(scan.status, "completed")

    def test_report_queue_failure_does_not_change_scan(self):
        settings = SimpleNamespace(reports_enabled=True)
        with (
            patch("eass.report.service.get_settings", return_value=settings),
            patch("eass.report.service.SessionLocal", self.sessions),
            patch("eass.worker.celery_app.send_task", side_effect=ConnectionError),
        ):
            schedule_report(self.scan_id)
        with self.sessions() as session:
            record = session.get(ScanReport, self.scan_id)
            scan = session.get(Scan, self.scan_id)
        self.assertEqual(record.status, "failed")
        self.assertEqual(scan.result, self.result)
        self.assertEqual(scan.status, "completed")

    def test_disabled_endpoint_does_not_start_report(self):
        with (
            self.sessions() as session,
            patch("eass.api.get_settings", return_value=SimpleNamespace(reports_enabled=False)),
            patch("eass.api.schedule_report") as schedule,
        ):
            response = get_report(self.scan_id, session)
        self.assertEqual(response.status, "disabled")
        schedule.assert_not_called()


if __name__ == "__main__":
    unittest.main()
