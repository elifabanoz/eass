import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

EICAR_PARTS = (
    b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$",
    b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*",
)
EICAR_SHA256 = hashlib.sha256(b"".join(EICAR_PARTS)).hexdigest()


@lru_cache(maxsize=8)
def _load_hashes(path: Path, mtime_ns: int | None) -> frozenset[str]:
    if not path.exists():
        return frozenset({EICAR_SHA256})
    values = {EICAR_SHA256}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip().lower()
        if len(value) == 64 and all(char in "0123456789abcdef" for char in value):
            values.add(value)
    return frozenset(values)


def load_hashes(path: Path) -> frozenset[str]:
    mtime_ns = path.stat().st_mtime_ns if path.exists() else None
    return _load_hashes(path, mtime_ns)


def check_local(sha256: str, path: Path) -> dict:
    matched = sha256.lower() in load_hashes(path)
    return {
        "status": "complete",
        "matched": matched,
        "source": "eicar-test"
        if sha256.lower() == EICAR_SHA256
        else "local-list"
        if matched
        else None,
    }


def check_malwarebazaar(sha256: str, auth_key: str | None) -> dict:
    if not auth_key:
        return {"status": "skipped", "reason": "Auth-Key is not configured"}
    body = urllib.parse.urlencode({"query": "get_info", "hash": sha256}).encode("ascii")
    request = urllib.request.Request(
        "https://mb-api.abuse.ch/api/v1/",
        data=body,
        headers={"Auth-Key": auth_key, "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.load(response)
    state = payload.get("query_status")
    if state == "ok":
        return {"status": "complete", "matched": True, "source": "malwarebazaar"}
    if state == "hash_not_found":
        return {"status": "complete", "matched": False, "source": "malwarebazaar"}
    raise ValueError(f"unexpected MalwareBazaar status: {state}")
