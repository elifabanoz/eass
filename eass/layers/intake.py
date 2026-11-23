import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


@dataclass(frozen=True)
class StoredFile:
    path: Path
    size: int
    sha256: str
    md5: str
    file_type: str


class FileTooLarge(ValueError):
    pass


class EmptyFile(ValueError):
    pass


def remove_quarantined(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        path.chmod(0o600)
        path.unlink(missing_ok=True)


def detect_type(head: bytes) -> str:
    if len(head) >= 64 and head[:2] == b"MZ":
        offset = int.from_bytes(head[0x3C:0x40], "little")
        if offset + 4 <= len(head) and head[offset : offset + 4] == b"PE\x00\x00":
            return "pe"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"\x7fELF"):
        return "elf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head:
        try:
            value = head.decode("utf-8")
        except UnicodeDecodeError:
            return "unknown"
        if all(char.isprintable() or char in "\r\n\t" for char in value):
            return "text"
    return "unknown"


async def store_upload(upload: UploadFile, directory: Path, limit: int) -> StoredFile:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / str(uuid4())
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    head = bytearray()
    try:
        with path.open("xb") as output:
            os.chmod(path, 0o600)
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise FileTooLarge(f"file exceeds {limit} bytes")
                sha256.update(chunk)
                md5.update(chunk)
                if len(head) < 4096:
                    head.extend(chunk[: 4096 - len(head)])
                output.write(chunk)
        if size == 0:
            raise EmptyFile("file is empty")
        os.chmod(path, 0o400)
        return StoredFile(path, size, sha256.hexdigest(), md5.hexdigest(), detect_type(head))
    except BaseException:
        path.unlink(missing_ok=True)
        raise
