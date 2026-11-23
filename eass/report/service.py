import logging

from sqlalchemy.exc import IntegrityError

from eass.config import get_settings
from eass.db import SessionLocal
from eass.models import Scan, ScanReport, utc_now

logger = logging.getLogger(__name__)


def schedule_report(scan_id: str) -> None:
    if not getattr(get_settings(), "reports_enabled", False):
        return
    try:
        with SessionLocal() as session:
            scan = session.get(Scan, scan_id)
            if scan is None or scan.status != "completed" or session.get(ScanReport, scan_id):
                return
            session.add(ScanReport(scan_id=scan_id, status="pending"))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return
        try:
            from eass.worker import celery_app

            celery_app.send_task("eass.build_report", args=[scan_id])
        except Exception as exc:
            with SessionLocal() as session:
                report = session.get(ScanReport, scan_id)
                if report and report.status == "pending":
                    report.status = "failed"
                    report.error = "report queue unavailable"
                    report.finished_at = utc_now()
                    session.commit()
            logger.warning("report queue failed: %s", type(exc).__name__)
    except Exception as exc:
        logger.warning("report scheduling failed: %s", type(exc).__name__)
