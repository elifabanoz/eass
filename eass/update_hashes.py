import argparse
import csv
import io
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from eass.config import get_settings

SHA256 = re.compile(r"[0-9a-fA-F]{64}")
MAX_COMPRESSED_BYTES = 128 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MIN_REFRESH_SECONDS = 3600


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def extract_hashes(lines) -> set[str]:
    values = set()
    for row in csv.reader(lines):
        for field in row:
            value = field.strip()
            if SHA256.fullmatch(value):
                values.add(value.lower())
                break
    return values


def refresh(export_url: str, target: Path) -> int:
    parsed = urllib.parse.urlparse(export_url)
    if parsed.scheme != "https" or parsed.hostname != "mb-api.abuse.ch":
        raise ValueError("export URL must use mb-api.abuse.ch over HTTPS")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.build_opener(NoRedirect).open(export_url, timeout=60) as response:
            if parsed.path.endswith(".zip"):
                with tempfile.TemporaryFile(dir=target.parent) as compressed:
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_COMPRESSED_BYTES:
                            raise ValueError("compressed export is too large")
                        compressed.write(chunk)
                    compressed.seek(0)
                    with zipfile.ZipFile(compressed) as archive:
                        members = [item for item in archive.infolist() if not item.is_dir()]
                        if len(members) != 1 or members[0].file_size > MAX_UNCOMPRESSED_BYTES:
                            raise ValueError("unexpected export archive contents")
                        with archive.open(members[0]) as content:
                            hashes = extract_hashes(io.TextIOWrapper(content, encoding="utf-8-sig"))
            else:
                lines = (line.decode("utf-8-sig", errors="replace") for line in response)
                hashes = extract_hashes(lines)
    except urllib.error.URLError:
        raise RuntimeError("MalwareBazaar export download failed") from None
    if not hashes:
        raise ValueError("export contained no SHA-256 hashes")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            for value in sorted(hashes):
                output.write(value + "\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(hashes)


def seconds_until_due(target: Path, now: float) -> float:
    last_attempt = target.with_name(target.name + ".last_attempt")
    timestamps = []
    if last_attempt.exists():
        timestamps.append(float(last_attempt.read_text(encoding="ascii")))
    if target.exists():
        timestamps.append(target.stat().st_mtime)
    return (
        min(MIN_REFRESH_SECONDS, max(0.0, MIN_REFRESH_SECONDS - (now - max(timestamps))))
        if timestamps
        else 0.0
    )


def attempt_refresh(export_url: str, target: Path, now: float) -> bool:
    if seconds_until_due(target, now) > 0:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    attempt_path = target.with_name(target.name + ".last_attempt")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", dir=target.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(str(now))
        os.replace(temporary, attempt_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    try:
        count = refresh(export_url, target)
    except Exception as exc:
        print(f"Hash refresh failed: {type(exc).__name__}", flush=True)
    else:
        print(f"Stored {count} SHA-256 hashes", flush=True)
    return True


def refresh_loop(export_url: str, target: Path) -> None:
    while True:
        now = time.time()
        if attempt_refresh(export_url, target, now):
            time.sleep(MIN_REFRESH_SECONDS)
        else:
            time.sleep(max(1.0, seconds_until_due(target, now)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url-template",
        default="https://mb-api.abuse.ch/v2/files/exports/{key}/sha256_full.txt.zip",
    )
    parser.add_argument("--output", type=Path, default=get_settings().hash_list_path)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    key = get_settings().malwarebazaar_key
    if not key:
        if args.loop:
            print("Hash refresh skipped: Auth-Key is not configured", flush=True)
            return
        parser.error("MALWAREBAZAAR_AUTH_KEY is required")
    url = args.url_template.format(key=urllib.parse.quote(key, safe=""))
    if args.loop:
        refresh_loop(url, args.output)
    else:
        print(f"Stored {refresh(url, args.output)} SHA-256 hashes")


if __name__ == "__main__":
    main()
