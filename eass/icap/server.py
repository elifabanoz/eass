import asyncio
import io
import logging
import re
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from eass.config import get_settings
from eass.db import SessionLocal
from eass.icap.protocol import (
    MAX_HEADER_BYTES,
    BodyTooLarge,
    Message,
    ProtocolError,
    encode_chunked,
    read_head,
    read_message,
)
from eass.layers.intake import remove_quarantined, store_upload
from eass.models import Scan, utc_now
from eass.pipeline import run_scan
from eass.report.service import schedule_report

logger = logging.getLogger(__name__)
MAX_CONNECTIONS = 8
ISTAG = '"eass-1"'


def _icap_head(status: str, fields: list[tuple[str, str]], *, keep_alive: bool = False) -> bytes:
    lines = [f"ICAP/1.0 {status}", "Service: EASS", f"ISTag: {ISTAG}"]
    lines.extend(f"{name}: {value}" for name, value in fields)
    lines.append("Connection: keep-alive" if keep_alive else "Connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")


async def _send(writer: asyncio.StreamWriter, payload: bytes) -> None:
    writer.write(payload)
    await writer.drain()


async def _status(writer: asyncio.StreamWriter, code: str) -> None:
    await _send(writer, _icap_head(code, [("Encapsulated", "null-body=0")]))


async def _options(writer: asyncio.StreamWriter, keep_alive: bool) -> None:
    await _send(
        writer,
        _icap_head(
            "200 OK",
            [
                ("Methods", "REQMOD, RESPMOD"),
                ("Max-Connections", str(MAX_CONNECTIONS)),
                ("Options-TTL", "3600"),
                ("Allow", "204"),
                ("Preview", "0"),
                ("Transfer-Preview", "*"),
                ("Encapsulated", "null-body=0"),
            ],
            keep_alive=keep_alive,
        ),
    )


async def _allow(writer: asyncio.StreamWriter, message: Message) -> None:
    allowed = {item.strip() for item in message.headers.get("allow", "").split(",")}
    if "204" in allowed:
        await _status(writer, "204 No Content")
        return
    header = message.http_header
    entity = message.body_name if message.has_body else "null-body"
    fields = [("Encapsulated", f"{message.header_name}=0, {entity}={len(header)}")]
    body = encode_chunked(message.body) if message.has_body else b""
    await _send(writer, _icap_head("200 OK", fields) + header + body)


def _safe_reason(reason: str) -> str:
    return re.sub(r"[^A-Za-z0-9 _.,:/+-]", "_", reason)[:96] or "scan blocked"


async def _block(
    writer: asyncio.StreamWriter, scan_id: str, reason: str, *, infection: bool = False
) -> None:
    threat = _safe_reason(reason)
    body = f"Scan blocked\nScan ID: {scan_id}\nReason: {threat}\n".encode()
    http_header = (
        "HTTP/1.1 403 Forbidden\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Cache-Control: no-store\r\n\r\n"
    ).encode("ascii")
    fields = [("X-Scan-ID", scan_id)]
    if infection:
        fields.extend(
            [
                ("X-Infection-Found", f"Type=0; Resolution=2; Threat={threat};"),
                ("X-Violations-Found", f"1, -, {threat}, 0, 0"),
            ]
        )
    else:
        fields.append(("X-EASS-Block-Reason", threat))
    fields.append(("Encapsulated", f"res-hdr=0, res-body={len(http_header)}"))
    await _send(writer, _icap_head("200 OK", fields) + http_header + encode_chunked(body))


def _create_record(stored, method: str) -> None:
    with SessionLocal() as session:
        session.add(
            Scan(
                id=stored.path.name,
                source="icap",
                status="running",
                attempts=1,
                filename="icap-upload" if method == "REQMOD" else "icap-download",
                file_type=stored.file_type,
                size=stored.size,
                sha256=stored.sha256,
                md5=stored.md5,
            )
        )
        session.commit()


def _finish_record(scan_id: str, result: dict | None, error: str | None) -> None:
    with SessionLocal() as session:
        scan = session.get(Scan, scan_id)
        if scan is None:
            raise RuntimeError("ICAP scan record missing")
        scan.status = "failed" if error else "completed"
        scan.result = result
        scan.error = error
        scan.finished_at = utc_now()
        session.commit()


def _run_and_clean(path: Path, file_type: str, sha256: str) -> dict:
    try:
        return run_scan(path, file_type, sha256)
    finally:
        remove_quarantined(path)


class IcapServer:
    def __init__(self) -> None:
        self.active_connections = 0
        self.report_tasks: set[asyncio.Task] = set()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.active_connections >= MAX_CONNECTIONS:
            try:
                await _status(writer, "503 Service Unavailable")
            finally:
                writer.close()
                await writer.wait_closed()
            return
        self.active_connections += 1
        try:
            while await self._process(reader, writer):
                pass
        except BodyTooLarge:
            await _block(writer, str(uuid4()), "file exceeds upload limit")
        except asyncio.IncompleteReadError as exc:
            if exc.partial:
                logger.warning("ICAP request rejected: incomplete read")
                await _status(writer, "400 Bad Request")
        except (ProtocolError, asyncio.LimitOverrunError) as exc:
            logger.warning("ICAP request rejected: %s", type(exc).__name__)
            await _status(writer, "400 Bad Request")
        except TimeoutError:
            logger.warning("ICAP client read timed out")
            await _status(writer, "408 Request Timeout")
        except Exception as exc:
            logger.exception("ICAP request failed: %s", type(exc).__name__)
            await _status(writer, "500 Server Error")
        finally:
            self.active_connections -= 1
            writer.close()
            await writer.wait_closed()

    async def _process(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bool:
        method, headers = await read_head(reader)
        if method == "OPTIONS":
            keep_alive = headers.get("connection", "").lower() != "close"
            await _options(writer, keep_alive)
            return keep_alive
        settings = get_settings()
        message = await read_message(reader, writer, method, headers, settings.max_upload_bytes)
        if not message.body:
            await _allow(writer, message)
            return False
        scan_id = str(uuid4())
        stored = None
        task_started = False
        record_created = False
        record_finished = False
        try:
            upload = UploadFile(file=io.BytesIO(message.body), filename="icap-body")
            stored = await store_upload(upload, settings.quarantine_dir, settings.max_upload_bytes)
            scan_id = stored.path.name
            await asyncio.to_thread(_create_record, stored, method)
            record_created = True
            task = asyncio.create_task(
                asyncio.to_thread(_run_and_clean, stored.path, stored.file_type, stored.sha256)
            )
            task.add_done_callback(
                lambda completed: completed.exception() if not completed.cancelled() else None
            )
            task_started = True
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(task), settings.icap_scan_timeout_seconds
                )
            except TimeoutError:
                logger.warning("ICAP scan timed out: %s", scan_id)
                await asyncio.to_thread(_finish_record, scan_id, None, "scan timeout")
                record_finished = True
                await self._fail(writer, message, scan_id, "scan timeout", settings.icap_fail_mode)
                return False
            except Exception as exc:
                logger.warning("ICAP scan failed: %s", type(exc).__name__)
                await asyncio.to_thread(_finish_record, scan_id, None, type(exc).__name__)
                record_finished = True
                await self._fail(writer, message, scan_id, "scan error", settings.icap_fail_mode)
                return False
            await asyncio.to_thread(_finish_record, scan_id, result, None)
            record_finished = True
            verdict = result.get("verdict")
            if verdict == "malicious" or (
                verdict == "suspicious" and settings.icap_suspicious_action == "block"
            ):
                await _block(
                    writer, scan_id, result["reasons"][0], infection=verdict == "malicious"
                )
            elif verdict in {"clean", "suspicious"}:
                await _allow(writer, message)
            else:
                await self._fail(
                    writer, message, scan_id, "scan inconclusive", settings.icap_fail_mode
                )
            if getattr(settings, "reports_enabled", False):
                task = asyncio.create_task(asyncio.to_thread(schedule_report, scan_id))
                self.report_tasks.add(task)
                task.add_done_callback(self.report_tasks.discard)
        except Exception as exc:
            logger.warning("ICAP scan operation failed: %s", type(exc).__name__)
            if record_created and not record_finished:
                try:
                    await asyncio.to_thread(_finish_record, scan_id, None, type(exc).__name__)
                except Exception:
                    logger.exception("ICAP scan status could not be saved")
            await self._fail(writer, message, scan_id, "scan error", settings.icap_fail_mode)
        finally:
            if stored is not None and not task_started:
                remove_quarantined(stored.path)
        return False

    async def _fail(
        self,
        writer: asyncio.StreamWriter,
        message: Message,
        scan_id: str,
        reason: str,
        mode: str,
    ) -> None:
        if mode == "open":
            await _allow(writer, message)
        else:
            await _block(writer, scan_id, reason)


async def start_server(host: str = "0.0.0.0", port: int | None = None) -> asyncio.Server:
    settings = get_settings()
    if settings.icap_suspicious_action not in {"block", "allow"}:
        raise ValueError("ICAP_SUSPICIOUS_ACTION must be block or allow")
    if settings.icap_fail_mode not in {"closed", "open"}:
        raise ValueError("ICAP_FAIL_MODE must be closed or open")
    if settings.icap_scan_timeout_seconds <= 0:
        raise ValueError("ICAP_SCAN_TIMEOUT_SECONDS must be positive")
    server = IcapServer()
    return await asyncio.start_server(
        server.handle,
        host,
        settings.icap_port if port is None else port,
        limit=MAX_HEADER_BYTES + 4,
    )
