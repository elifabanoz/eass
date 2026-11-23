from datetime import UTC, datetime
from pathlib import Path

import pefile

SUSPICIOUS_IMPORTS = {
    "VirtualAllocEx",
    "WriteProcessMemory",
    "CreateRemoteThread",
    "OpenProcess",
    "VirtualProtectEx",
}


def analyze(path: Path) -> dict:
    pe = pefile.PE(str(path), fast_load=False)
    try:
        sections = []
        for section in pe.sections:
            sections.append(
                {
                    "name": section.Name.rstrip(b"\x00").decode("ascii", errors="replace"),
                    "entropy": round(section.get_entropy(), 2),
                    "raw_size": section.SizeOfRawData,
                }
            )
        imports = []
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            for item in entry.imports:
                if item.name:
                    name = item.name.decode("ascii", errors="replace")
                    if name in SUSPICIOUS_IMPORTS:
                        imports.append(name)
        timestamp = pe.FILE_HEADER.TimeDateStamp
        now = int(datetime.now(UTC).timestamp())
        return {
            "status": "complete",
            "machine": hex(pe.FILE_HEADER.Machine),
            "timestamp": timestamp,
            "timestamp_anomaly": timestamp == 0 or timestamp > now + 366 * 86400,
            "sections": sections,
            "suspicious_imports": sorted(set(imports)),
            "upx_sections": [s["name"] for s in sections if s["name"].upper().startswith("UPX")],
        }
    finally:
        pe.close()
