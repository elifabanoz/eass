from celery import Celery

from eass.config import get_settings
from eass.db import SessionLocal
from eass.models import Scan, ScanReport, utc_now
from eass.pipeline import run_scan
from eass.report.graph import build_report
from eass.report.input import build_input
from eass.report.service import schedule_report

celery_app = Celery("eass", broker=get_settings().broker_url)
celery_app.conf.update(
    task_ignore_result=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=120,
    worker_prefetch_multiplier=1,
)


@celery_app.task(name="eass.scan")
def scan_task(scan_id: str) -> None:
    path = get_settings().quarantine_dir / scan_id
    completed = False
    with SessionLocal() as session:
        scan = session.get(Scan, scan_id)
        if scan is None:
            path.unlink(missing_ok=True)
            return
        if scan.status in {"completed", "failed"}:
            path.unlink(missing_ok=True)
            return
        if scan.attempts >= 3:
            scan.status = "failed"
            scan.error = "worker retry limit reached"
            scan.finished_at = utc_now()
            session.commit()
            path.unlink(missing_ok=True)
            return
        scan.attempts += 1
        scan.status = "running"
        session.commit()
        try:
            scan.result = run_scan(path, scan.file_type, scan.sha256)
            scan.status = "completed"
            completed = True
        except Exception as exc:
            scan.status = "failed"
            scan.error = type(exc).__name__
        finally:
            scan.finished_at = utc_now()
            session.commit()
            path.unlink(missing_ok=True)
    if completed:
        schedule_report(scan_id)


@celery_app.task(name="eass.build_report", time_limit=180)
def build_report_task(scan_id: str) -> None:
    settings = get_settings()
    if not settings.reports_enabled:
        return
    with SessionLocal() as session:
        scan = session.get(Scan, scan_id)
        record = session.get(ScanReport, scan_id)
        if scan is None or scan.status != "completed" or record is None:
            return
        if record.status != "pending":
            return
    try:
        report_input = build_input(scan)
        payload = build_report(report_input, settings)
        status = "done"
        error = None
    except Exception as exc:
        payload = None
        status = "failed"
        error = type(exc).__name__
    with SessionLocal() as session:
        record = session.get(ScanReport, scan_id)
        if record and record.status == "pending":
            record.status = status
            record.payload = payload
            record.error = error
            record.finished_at = utc_now()
            session.commit()
