import asyncio
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from eass.db import Base
from eass.icap.protocol import ProtocolError, encode_chunked, parse_encapsulated
from eass.icap.server import start_server
from eass.layers.hashrep import EICAR_PARTS
from eass.models import Scan


def request(method: str, body: bytes, *, allow: bool = True, req_hdr: bool = False) -> bytes:
    header = (
        b"POST /upload HTTP/1.1\r\nHost: example.test\r\n\r\n"
        if method == "REQMOD"
        else b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n\r\n"
    )
    original = b"GET /download HTTP/1.1\r\nHost: example.test\r\n\r\n" if req_hdr else b""
    first = "req-hdr" if method == "REQMOD" else "res-hdr"
    offset = len(original) + len(header)
    parts = []
    if original:
        parts.append("req-hdr=0")
    parts.append(f"{first}={len(original)}")
    parts.append(f"{'req-body' if method == 'REQMOD' else 'res-body'}={offset}")
    lines = [
        f"{method} icap://127.0.0.1/eass ICAP/1.0",
        "Host: 127.0.0.1",
        f"Encapsulated: {', '.join(parts)}",
    ]
    if allow:
        lines.append("Allow: 204")
    return ("\r\n".join(lines) + "\r\n\r\n").encode() + original + header + encode_chunked(body)


class EncapsulatedTests(unittest.TestCase):
    def test_offsets_and_legal_sections(self):
        self.assertEqual(
            parse_encapsulated("req-hdr=0, res-hdr=42, res-body=99", "RESPMOD"),
            [("req-hdr", 0), ("res-hdr", 42), ("res-body", 99)],
        )
        self.assertEqual(
            parse_encapsulated("req-hdr=0, null-body=42", "REQMOD"),
            [("req-hdr", 0), ("null-body", 42)],
        )

    def test_reversed_offsets_are_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_encapsulated("res-hdr=0, res-body=0", "RESPMOD")


class IcapSocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.engine = create_engine("sqlite:///" + (base / "scans.db").as_posix())
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.settings = SimpleNamespace(
            icap_port=0,
            icap_scan_timeout_seconds=10,
            icap_suspicious_action="block",
            icap_fail_mode="closed",
            max_upload_bytes=1024,
            quarantine_dir=base / "quarantine",
            hash_list_path=base / "hashes.txt",
            malwarebazaar_key=None,
            model_dir=base / "models",
        )
        self.patches = [
            patch("eass.icap.server.get_settings", return_value=self.settings),
            patch("eass.icap.server.SessionLocal", self.sessions),
            patch("eass.pipeline.get_settings", return_value=self.settings),
        ]
        for item in self.patches:
            item.start()
        self.server = await start_server("127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        for item in reversed(self.patches):
            item.stop()
        self.engine.dispose()
        self.temp.cleanup()

    async def exchange(self, payload: bytes) -> bytes:
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(payload)
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), 15)
        writer.close()
        await writer.wait_closed()
        return response

    async def test_options(self):
        response = await self.exchange(
            b"OPTIONS icap://127.0.0.1/eass ICAP/1.0\r\n"
            b"Host: 127.0.0.1\r\nConnection: close\r\n\r\n"
        )
        self.assertIn(b"ICAP/1.0 200 OK", response)
        self.assertIn(b"Methods: REQMOD, RESPMOD", response)
        self.assertIn(b"Preview: 0", response)
        self.assertIn(b"Encapsulated: null-body=0", response)

    async def test_options_then_respmod_reuses_connection(self):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(b"OPTIONS icap://127.0.0.1/eass ICAP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        await writer.drain()
        options = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        self.assertIn(b"Connection: keep-alive", options)
        writer.write(request("RESPMOD", b"ordinary text"))
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), 15)
        self.assertIn(b"ICAP/1.0 204 No Content", response)
        writer.close()
        await writer.wait_closed()

    async def test_respmod_eicar_is_blocked_and_recorded(self):
        response = await self.exchange(request("RESPMOD", b"".join(EICAR_PARTS), req_hdr=True))
        self.assertIn(b"HTTP/1.1 403 Forbidden", response)
        self.assertIn(b"X-Infection-Found: Type=0; Resolution=2;", response)
        scan_id = re.search(rb"X-Scan-ID: ([0-9a-f-]+)", response).group(1).decode()
        with Session(self.engine) as session:
            scan = session.get(Scan, scan_id)
            self.assertEqual(scan.source, "icap")
            self.assertEqual(scan.result["verdict"], "malicious")
        self.assertEqual(list(self.settings.quarantine_dir.iterdir()), [])

    async def test_reqmod_eicar_is_blocked(self):
        response = await self.exchange(request("REQMOD", b"".join(EICAR_PARTS)))
        self.assertIn(b"HTTP/1.1 403 Forbidden", response)

    async def test_clean_with_allow_returns_204(self):
        response = await self.exchange(request("RESPMOD", b"ordinary text"))
        self.assertIn(b"ICAP/1.0 204 No Content", response)

    async def test_clean_without_allow_echoes_http_message(self):
        response = await self.exchange(request("REQMOD", b"ordinary text", allow=False))
        self.assertIn(b"ICAP/1.0 200 OK", response)
        self.assertIn(b"POST /upload HTTP/1.1\r\nHost: example.test\r\n\r\n", response)
        self.assertIn(b"ordinary text", response)

    async def test_preview_zero_gets_continue(self):
        whole = request("RESPMOD", b"ordinary text")
        head, _, body = whole.partition(b"\r\n\r\n")
        http_header, _, chunks = body.partition(b"\r\n\r\n")
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(head + b"\r\nPreview: 0\r\n\r\n" + http_header + b"\r\n\r\n0\r\n\r\n")
        await writer.drain()
        interim = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        self.assertEqual(interim, b"ICAP/1.0 100 Continue\r\n\r\n")
        writer.write(chunks)
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), 15)
        self.assertIn(b"ICAP/1.0 204 No Content", response)
        writer.close()
        await writer.wait_closed()

    async def test_preview_ieof_skips_continue(self):
        whole = request("RESPMOD", b"")
        whole = whole.replace(b"\r\n\r\nHTTP/1.1", b"\r\nPreview: 0\r\n\r\nHTTP/1.1")
        response = await self.exchange(whole[:-5] + b"0; ieof\r\n\r\n")
        self.assertIn(b"ICAP/1.0 204 No Content", response)
        self.assertNotIn(b"100 Continue", response)

    async def test_oversize_body_is_blocked(self):
        self.settings.max_upload_bytes = 4
        response = await self.exchange(request("RESPMOD", b"ordinary text"))
        self.assertIn(b"HTTP/1.1 403 Forbidden", response)
        self.assertIn(b"file exceeds upload limit", response)
        self.assertNotIn(b"X-Infection-Found", response)

    async def test_scan_timeout_fails_closed(self):
        self.settings.icap_scan_timeout_seconds = 0.001

        def slow_scan(*_args):
            time.sleep(0.1)
            return {"verdict": "clean", "reasons": ["No indicators found"]}

        with patch("eass.icap.server.run_scan", side_effect=slow_scan):
            response = await self.exchange(request("RESPMOD", b"ordinary text"))
        self.assertIn(b"HTTP/1.1 403 Forbidden", response)
        self.assertIn(b"scan timeout", response)
        await asyncio.sleep(0.15)

    async def test_scan_timeout_can_fail_open(self):
        self.settings.icap_scan_timeout_seconds = 0.001
        self.settings.icap_fail_mode = "open"

        def slow_scan(*_args):
            time.sleep(0.1)
            return {"verdict": "clean", "reasons": ["No indicators found"]}

        with patch("eass.icap.server.run_scan", side_effect=slow_scan):
            response = await self.exchange(request("RESPMOD", b"ordinary text"))
        self.assertIn(b"ICAP/1.0 204 No Content", response)
        await asyncio.sleep(0.15)

    async def test_suspicious_action_can_allow(self):
        self.settings.icap_suspicious_action = "allow"
        result = {"verdict": "suspicious", "reasons": ["test indicator"]}
        with patch("eass.icap.server.run_scan", return_value=result):
            response = await self.exchange(request("RESPMOD", b"ordinary text"))
        self.assertIn(b"ICAP/1.0 204 No Content", response)

    async def test_suspicious_is_blocked_by_default(self):
        result = {"verdict": "suspicious", "reasons": ["test indicator"]}
        with patch("eass.icap.server.run_scan", return_value=result):
            response = await self.exchange(request("RESPMOD", b"ordinary text"))
        self.assertIn(b"HTTP/1.1 403 Forbidden", response)
        self.assertNotIn(b"X-Infection-Found", response)


if __name__ == "__main__":
    unittest.main()
