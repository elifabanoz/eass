import re
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["hash", "yara", "pe", "ml", "coverage"]
    detail: str = Field(max_length=180)


class ReportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["clean", "suspicious", "malicious"] | None
    score: int | None = Field(ge=0, le=100)
    file_type: str = Field(max_length=16)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: list[Evidence] = Field(max_length=40)


def _safe_name(value: object, limit: int = 64) -> str:
    return re.sub(r"[^A-Za-z0-9_.@?$-]", "", str(value))[:limit] or "unnamed"


def _valid_probability(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
        return round(float(value), 4)
    return None


def build_input(scan: SimpleNamespace) -> ReportInput:
    result = scan.result or {}
    layers = result.get("layers", {})
    evidence: list[Evidence] = []

    local = layers.get("hash", {})
    if local.get("status") == "complete" and local.get("matched") is True:
        source = local.get("source")
        detail = (
            "EICAR test hash matched." if source == "eicar-test" else "Local SHA-256 list matched."
        )
        evidence.append(Evidence(id="hash-local", kind="hash", detail=detail))
    bazaar = layers.get("malwarebazaar", {})
    if bazaar.get("status") == "complete" and bazaar.get("matched") is True:
        evidence.append(
            Evidence(id="hash-malwarebazaar", kind="hash", detail="MalwareBazaar hash matched.")
        )

    yara = layers.get("yara", {})
    if yara.get("status") == "complete":
        for index, match in enumerate(yara.get("matches", [])[:16]):
            rule = _safe_name(match.get("rule", ""))
            evidence.append(
                Evidence(id=f"yara-{index}", kind="yara", detail=f"YARA rule {rule} matched.")
            )

    pe = layers.get("pe", {})
    if pe.get("status") == "complete":
        for index, section in enumerate(pe.get("sections", [])[:16]):
            entropy = section.get("entropy")
            size = section.get("raw_size")
            if (
                isinstance(entropy, (int, float))
                and not isinstance(entropy, bool)
                and 7.2 < entropy <= 8
                and isinstance(size, int)
                and size >= 1024
            ):
                name = _safe_name(section.get("name", ""), 8)
                evidence.append(
                    Evidence(
                        id=f"pe-entropy-{index}",
                        kind="pe",
                        detail=f"PE section {name} has entropy {entropy:.2f}.",
                    )
                )
        imports = set(pe.get("suspicious_imports", [])[:16])
        chain = {"VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread"}
        if chain <= imports:
            evidence.append(
                Evidence(
                    id="pe-import-chain",
                    kind="pe",
                    detail="PE imports VirtualAllocEx, WriteProcessMemory and CreateRemoteThread.",
                )
            )
        for index, name in enumerate(pe.get("upx_sections", [])[:8]):
            evidence.append(
                Evidence(
                    id=f"pe-upx-{index}",
                    kind="pe",
                    detail=f"PE section {_safe_name(name, 8)} has a UPX-style name.",
                )
            )
        if pe.get("timestamp_anomaly") is True:
            evidence.append(
                Evidence(id="pe-timestamp", kind="pe", detail="PE timestamp is anomalous.")
            )

    ml = layers.get("ml", {})
    if ml.get("status") == "complete":
        probability = _valid_probability(ml.get("probability"))
        threshold = _valid_probability(ml.get("threshold"))
        if probability is not None and threshold is not None:
            model_name = {
                "ember_full_v2": "Full EMBER v2 model",
                "byte_histogram_entropy_strings_v1": "Byte, entropy and strings model",
            }.get(ml.get("feature_set"), "Byte-histogram model")
            evidence.append(
                Evidence(
                    id="ml-model",
                    kind="ml",
                    detail=(
                        f"{model_name} probability {probability:.4f}; threshold {threshold:.4f}."
                    ),
                )
            )

    if any(layer.get("status") == "error" for layer in layers.values() if isinstance(layer, dict)):
        evidence.append(
            Evidence(id="coverage-error", kind="coverage", detail="At least one scan layer failed.")
        )
    if not evidence:
        evidence.append(
            Evidence(id="no-indicators", kind="coverage", detail="No reported indicator matched.")
        )
    file_type = _safe_name(scan.file_type)[:16]
    return ReportInput(
        verdict=result.get("verdict"),
        score=result.get("score"),
        file_type=file_type,
        size=scan.size,
        sha256=scan.sha256.lower(),
        evidence=evidence[:40],
    )
