import json
import os
import socket
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

from eass.layers.hashrep import EICAR_PARTS

API = "http://127.0.0.1:8000"
ICAP = ("127.0.0.1", 1344)
EICAR = b"".join(EICAR_PARTS)


def api_json(path: str, body: bytes | None = None, content_type: str | None = None):
    headers = {}
    if key := os.getenv("EASS_API_KEY"):
        headers["X-API-Key"] = key
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(API + path, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def wait_for(path: str, terminal: set[str], timeout: int = 210):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = api_json(path)
        if response["status"] in terminal:
            return response
        time.sleep(1)
    raise TimeoutError(f"{path} did not reach {terminal}")


def scan_api():
    boundary = uuid4().hex
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="eicar.com.txt"\r\n'
            "Content-Type: text/plain\r\n\r\n"
        ).encode()
        + EICAR
        + f"\r\n--{boundary}--\r\n".encode()
    )
    accepted = api_json("/scans", body, f"multipart/form-data; boundary={boundary}")
    scan_id = accepted["scan_id"]
    scan = wait_for(f"/scans/{scan_id}", {"completed", "failed"})
    if scan["status"] != "completed" or scan["result"]["verdict"] != "malicious":
        raise AssertionError("API EICAR scan did not complete as malicious")
    if scan["result"]["layers"]["hash"]["source"] != "eicar-test":
        raise AssertionError("API EICAR hash was not recognized")
    return scan_id, scan


def scan_icap():
    http_header = b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n\r\n"
    chunk = f"{len(EICAR):x}\r\n".encode() + EICAR + b"\r\n0\r\n\r\n"
    head = (
        "RESPMOD icap://127.0.0.1/eass ICAP/1.0\r\n"
        "Host: 127.0.0.1\r\n"
        "Allow: 204\r\n"
        f"Encapsulated: res-hdr=0, res-body={len(http_header)}\r\n\r\n"
    ).encode()
    with socket.create_connection(ICAP, timeout=35) as connection:
        connection.settimeout(35)
        connection.sendall(head + http_header + chunk)
        parts = []
        while fragment := connection.recv(65536):
            parts.append(fragment)
    response = b"".join(parts)
    if b"ICAP/1.0 200 OK" not in response or b"HTTP/1.1 403 Forbidden" not in response:
        raise AssertionError("ICAP did not block EICAR")
    scan_id = None
    for line in response.split(b"\r\n"):
        if line.lower().startswith(b"x-scan-id:"):
            scan_id = line.split(b":", 1)[1].strip().decode("ascii")
            break
    if not scan_id:
        raise AssertionError("ICAP response has no scan ID")
    scan = wait_for(f"/scans/{scan_id}", {"completed", "failed"})
    if scan["status"] != "completed" or scan["result"]["verdict"] != "malicious":
        raise AssertionError("ICAP EICAR scan did not complete as malicious")
    return scan_id, scan


def main():
    if api_json("/health") != {"status": "ok"}:
        raise AssertionError("API health failed")
    api_id, api_scan = scan_api()
    print(f"API scan completed: {api_id}", flush=True)
    api_report = wait_for(f"/scans/{api_id}/report", {"done", "failed", "disabled"})
    print(f"API report: {api_report['status']}", flush=True)
    icap_id, icap_scan = scan_icap()
    print(f"ICAP scan completed: {icap_id}", flush=True)
    icap_report = wait_for(f"/scans/{icap_id}/report", {"done", "failed", "disabled"})
    print(f"ICAP report: {icap_report['status']}", flush=True)
    for report in (api_report, icap_report):
        if report["status"] != "done":
            raise AssertionError("report did not complete")
        if report["report"]["verdict"] != "malicious":
            raise AssertionError("report changed the scan verdict")
    result = {
        "api": {
            "scan_id": api_id,
            "verdict": api_scan["result"]["verdict"],
            "report_status": api_report["status"],
            "report_parts": api_report["report"]["parts"],
        },
        "icap": {
            "scan_id": icap_id,
            "verdict": icap_scan["result"]["verdict"],
            "report_status": icap_report["status"],
            "report_parts": icap_report["report"]["parts"],
        },
    }
    output = Path("data/live_smoke_20261120.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
