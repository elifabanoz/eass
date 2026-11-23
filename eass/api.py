import ntpath
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from eass.access import ApiKeyGate, UploadRateLimit
from eass.body_limit import UploadBodyLimit
from eass.config import get_settings
from eass.db import SessionLocal
from eass.layers.intake import EmptyFile, FileTooLarge, store_upload
from eass.models import Scan, ScanReport
from eass.report.service import schedule_report
from eass.worker import celery_app

app = FastAPI(title="EASS")
app.add_middleware(UploadBodyLimit)
app.add_middleware(UploadRateLimit)
app.add_middleware(ApiKeyGate)


class ScanAccepted(BaseModel):
    scan_id: str


class ScanResponse(BaseModel):
    scan_id: str
    status: str
    source: str
    attempts: int
    filename: str
    file_type: str
    size: int
    sha256: str
    md5: str
    result: dict | None
    error: str | None


class ReportResponse(BaseModel):
    scan_id: str
    status: str
    report: dict | None
    error: str | None


def database():
    with SessionLocal() as session:
        yield session


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/scans", status_code=202, response_model=ScanAccepted)
async def create_scan(
    file: Annotated[UploadFile, File()], session: Annotated[Session, Depends(database)]
) -> ScanAccepted:
    settings = get_settings()
    try:
        stored = await store_upload(file, settings.quarantine_dir, settings.max_upload_bytes)
    except FileTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except EmptyFile as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()

    filename = ntpath.basename(file.filename or "unnamed")[:255]
    scan = Scan(
        id=stored.path.name,
        filename=filename,
        file_type=stored.file_type,
        size=stored.size,
        sha256=stored.sha256,
        md5=stored.md5,
    )
    try:
        session.add(scan)
        session.commit()
    except Exception as exc:
        session.rollback()
        Path(stored.path).unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    try:
        celery_app.send_task("eass.scan", args=[scan.id])
    except Exception as exc:
        scan.status = "failed"
        scan.error = "scan queue unavailable"
        session.commit()
        Path(stored.path).unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="scan queue unavailable") from exc
    return ScanAccepted(scan_id=scan.id)


@app.get("/scans/{scan_id}", response_model=ScanResponse)
def get_scan(scan_id: str, session: Annotated[Session, Depends(database)]) -> ScanResponse:
    scan = session.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")
    return ScanResponse(
        scan_id=scan.id,
        status=scan.status,
        source=scan.source,
        attempts=scan.attempts,
        filename=scan.filename,
        file_type=scan.file_type,
        size=scan.size,
        sha256=scan.sha256,
        md5=scan.md5,
        result=scan.result,
        error=scan.error,
    )


@app.get("/scans/{scan_id}/report", response_model=ReportResponse)
def get_report(scan_id: str, session: Annotated[Session, Depends(database)]) -> ReportResponse:
    scan = session.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")
    if not get_settings().reports_enabled:
        return ReportResponse(scan_id=scan_id, status="disabled", report=None, error=None)
    if scan.status == "failed":
        return ReportResponse(scan_id=scan_id, status="failed", report=None, error="scan failed")
    if scan.status != "completed":
        return ReportResponse(scan_id=scan_id, status="pending", report=None, error=None)
    record = session.get(ScanReport, scan_id)
    if record is None:
        schedule_report(scan_id)
        session.expire_all()
        record = session.get(ScanReport, scan_id)
    return ReportResponse(
        scan_id=scan_id,
        status=record.status if record else "pending",
        report=record.payload if record else None,
        error=record.error if record else None,
    )
