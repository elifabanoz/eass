import asyncio
import re
from dataclasses import dataclass
from itertools import pairwise
from urllib.parse import urlsplit

MAX_HEADER_BYTES = 64 * 1024
MAX_CHUNK_LINE = 256
READ_TIMEOUT_SECONDS = 30
ENTITY = re.compile(r"(req-hdr|res-hdr|req-body|res-body|null-body)=(0|[1-9][0-9]*)")
TOKEN = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+")


class ProtocolError(ValueError):
    pass


class BodyTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class Message:
    method: str
    headers: dict[str, str]
    req_hdr: bytes
    res_hdr: bytes
    body: bytes
    has_body: bool

    @property
    def http_header(self) -> bytes:
        return self.req_hdr if self.method == "REQMOD" else self.res_hdr

    @property
    def header_name(self) -> str:
        return "req-hdr" if self.method == "REQMOD" else "res-hdr"

    @property
    def body_name(self) -> str:
        return "req-body" if self.method == "REQMOD" else "res-body"


async def _exactly(reader: asyncio.StreamReader, length: int) -> bytes:
    return await asyncio.wait_for(reader.readexactly(length), READ_TIMEOUT_SECONDS)


async def _line(reader: asyncio.StreamReader) -> bytes:
    line = await asyncio.wait_for(reader.readline(), READ_TIMEOUT_SECONDS)
    if not line.endswith(b"\r\n") or len(line) > MAX_CHUNK_LINE:
        raise ProtocolError("invalid chunk line")
    return line[:-2]


async def read_head(reader: asyncio.StreamReader) -> tuple[str, dict[str, str]]:
    raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), READ_TIMEOUT_SECONDS)
    if len(raw) > MAX_HEADER_BYTES:
        raise ProtocolError("ICAP header too large")
    lines = raw[:-4].split(b"\r\n")
    try:
        start = lines[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ProtocolError("invalid ICAP request line") from exc
    fields = start.split(" ")
    if len(fields) != 3 or fields[2] != "ICAP/1.0":
        raise ProtocolError("invalid ICAP request line")
    method, uri, _ = fields
    if method not in {"OPTIONS", "REQMOD", "RESPMOD"}:
        raise ProtocolError("unsupported ICAP method")
    parsed = urlsplit(uri)
    if parsed.scheme != "icap" or not parsed.netloc or parsed.path != "/eass":
        raise ProtocolError("unknown ICAP service")
    headers = {}
    for line in lines[1:]:
        if b":" not in line:
            raise ProtocolError("invalid ICAP header")
        name, value = line.split(b":", 1)
        if not TOKEN.fullmatch(name) or b"\x00" in value or b"\n" in value:
            raise ProtocolError("invalid ICAP header")
        key = name.decode("ascii").lower()
        if key in headers:
            raise ProtocolError("duplicate ICAP header")
        headers[key] = value.strip().decode("latin-1")
    return method, headers


def parse_encapsulated(value: str, method: str) -> list[tuple[str, int]]:
    entries = []
    for part in value.split(","):
        match = ENTITY.fullmatch(part.strip())
        if match is None:
            raise ProtocolError("invalid Encapsulated header")
        entries.append((match.group(1), int(match.group(2))))
    names = tuple(name for name, _ in entries)
    legal = {
        "REQMOD": {("req-hdr", "req-body"), ("req-hdr", "null-body")},
        "RESPMOD": {
            ("res-hdr", "res-body"),
            ("res-hdr", "null-body"),
            ("req-hdr", "res-hdr", "res-body"),
            ("req-hdr", "res-hdr", "null-body"),
        },
    }
    if names not in legal[method] or entries[0][1] != 0:
        raise ProtocolError("invalid Encapsulated sections")
    if any(right[1] <= left[1] for left, right in pairwise(entries)):
        raise ProtocolError("invalid Encapsulated offsets")
    if entries[-1][1] > MAX_HEADER_BYTES:
        raise ProtocolError("encapsulated headers too large")
    return entries


async def read_chunks(reader: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    parts = []
    total = 0
    while True:
        line = await _line(reader)
        size_text, separator, extension = line.partition(b";")
        try:
            size = int(size_text.strip(), 16)
        except ValueError as exc:
            raise ProtocolError("invalid chunk size") from exc
        if size < 0 or (separator and extension.strip().lower() != b"ieof"):
            raise ProtocolError("invalid chunk extension")
        if separator and size:
            raise ProtocolError("ieof must end the preview")
        if size == 0:
            if await _exactly(reader, 2) != b"\r\n":
                raise ProtocolError("invalid final chunk")
            return b"".join(parts), bool(separator)
        total += size
        if total > limit:
            raise BodyTooLarge("file exceeds upload limit")
        parts.append(await _exactly(reader, size))
        if await _exactly(reader, 2) != b"\r\n":
            raise ProtocolError("invalid chunk framing")


async def read_message(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    method: str,
    headers: dict[str, str],
    max_body: int,
) -> Message:
    if "encapsulated" not in headers:
        raise ProtocolError("Encapsulated header required")
    entries = parse_encapsulated(headers["encapsulated"], method)
    offset = entries[-1][1]
    raw_headers = await _exactly(reader, offset)
    sections = {}
    for index, (name, start) in enumerate(entries[:-1]):
        end = entries[index + 1][1]
        section = raw_headers[start:end]
        if not section.endswith(b"\r\n\r\n"):
            raise ProtocolError("invalid encapsulated HTTP header")
        sections[name] = section
    req_hdr = sections.get("req-hdr", b"")
    res_hdr = sections.get("res-hdr", b"")
    if req_hdr and not re.match(rb"[A-Z]+ [^\r\n]+ HTTP/1\.[01]\r\n", req_hdr):
        raise ProtocolError("invalid encapsulated HTTP request")
    if res_hdr and not re.match(rb"HTTP/1\.[01] [0-9]{3} [^\r\n]*\r\n", res_hdr):
        raise ProtocolError("invalid encapsulated HTTP response")
    has_body = entries[-1][0] != "null-body"
    if not has_body:
        return Message(method, headers, req_hdr, res_hdr, b"", False)
    preview = headers.get("preview")
    if preview is not None:
        if not preview.isdecimal() or int(preview) > max_body:
            raise ProtocolError("invalid Preview header")
        first, eof = await read_chunks(reader, max_body)
        if len(first) > int(preview):
            raise ProtocolError("preview exceeds advertised size")
        if eof:
            return Message(method, headers, req_hdr, res_hdr, first, True)
        writer.write(b"ICAP/1.0 100 Continue\r\n\r\n")
        await writer.drain()
        rest, _ = await read_chunks(reader, max_body - len(first))
        return Message(method, headers, req_hdr, res_hdr, first + rest, True)
    body, _ = await read_chunks(reader, max_body)
    return Message(method, headers, req_hdr, res_hdr, body, True)


def encode_chunked(body: bytes) -> bytes:
    if not body:
        return b"0\r\n\r\n"
    return f"{len(body):x}\r\n".encode("ascii") + body + b"\r\n0\r\n\r\n"
