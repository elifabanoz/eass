import argparse
import os
import re
import secrets
from pathlib import Path

DOMAIN = re.compile(r"(?=.{4,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}")


def auth_key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MALWAREBAZAAR_AUTH_KEY="):
            value = line.partition("=")[2].strip()
            if value and not any(char.isspace() for char in value):
                return value
    raise ValueError("MalwareBazaar Auth-Key is missing from the local .env")


def prepare(domain: str, source: Path, output: Path) -> None:
    if not DOMAIN.fullmatch(domain):
        raise ValueError("a DNS domain is required")
    key = auth_key(source)
    values = {
        "EASS_DOMAIN": domain,
        "EASS_API_KEY": secrets.token_urlsafe(36),
        "EASS_DB_PASSWORD": secrets.token_hex(32),
        "MALWAREBAZAAR_AUTH_KEY": key,
        "EASS_VPS_REPORTS_ENABLED": "false",
        "EASS_VPS_OFFLINE_MODE": "true",
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            for name, value in values.items():
                target.write(f"{name}={value}\n")
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domain")
    parser.add_argument("--source", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=Path("vps.env"))
    arguments = parser.parse_args()
    prepare(arguments.domain, arguments.source, arguments.output)
    print(f"Created {arguments.output} with private API and database credentials")


if __name__ == "__main__":
    main()
