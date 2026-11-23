import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from eass.config import Settings
from eass.report.input import ReportInput
from eass.report.llm import ModelClient

logger = logging.getLogger(__name__)
ATTACK_MAP = json.loads((Path(__file__).resolve().parent / "attack_map.json").read_text())


class SummarySelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_ids: list[str] = Field(max_length=4)


class TechniqueSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technique_id: str
    evidence_id: str


class AttackSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    techniques: list[TechniqueSelection] = Field(max_length=8)


class ReportState(TypedDict, total=False):
    scan: ReportInput
    summary: dict
    attack: dict
    intel: dict
    report: dict


def _attempt(
    client: ModelClient,
    prompt: str,
    payload: dict,
    schema: type[BaseModel],
    valid=None,
):
    for attempt in range(2):
        try:
            result = client.complete(prompt, payload, schema)
            if valid is not None and not valid(result):
                raise ValueError("model selection is outside supplied evidence")
            return result
        except Exception as exc:
            if attempt:
                logger.warning("report %s unavailable: %s", prompt, type(exc).__name__)
    return None


def _safe_intel(value: object, limit: int = 64) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"[^A-Za-z0-9 _.,:/@+-]", "", value)[:limit].strip()
    return cleaned or None


def lookup_intel(sha256: str, settings: Settings) -> dict:
    if settings.offline_mode or not settings.malwarebazaar_key:
        return {"status": "skipped"}
    body = urllib.parse.urlencode({"query": "get_info", "hash": sha256}).encode("ascii")
    request = urllib.request.Request(
        "https://mb-api.abuse.ch/api/v1/",
        data=body,
        headers={
            "Auth-Key": settings.malwarebazaar_key,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            raw = response.read(32769)
        if len(raw) > 32768:
            raise ValueError("intel response too large")
        payload = json.loads(raw)
        if payload.get("query_status") == "hash_not_found":
            return {"status": "complete", "found": False}
        if payload.get("query_status") != "ok":
            raise ValueError("unexpected intel response")
        data = payload.get("data") or []
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ValueError("missing intel record")
        item = data[0]
        return {
            "status": "complete",
            "found": True,
            "family": _safe_intel(item.get("signature")),
            "first_seen": _safe_intel(item.get("first_seen"), 32),
            "tags": [tag for tag in (_safe_intel(x, 32) for x in item.get("tags", [])[:8]) if tag],
        }
    except Exception as exc:
        logger.warning("report intel unavailable: %s", type(exc).__name__)
        return {"status": "unavailable"}


def build_graph(settings: Settings, client: ModelClient | None = None):
    model = client or ModelClient(settings)

    def summarize(state: ReportState) -> dict:
        scan = state["scan"]
        known = {item.id: item for item in scan.evidence}
        selection = _attempt(
            model,
            "summary",
            scan.model_dump(),
            SummarySelection,
            valid=lambda output: all(item in known for item in output.fact_ids),
        )
        if selection is None:
            return {"summary": {"status": "unavailable", "fact_ids": []}}
        ids = list(dict.fromkeys(selection.fact_ids))[:4]
        return {"summary": {"status": "complete", "fact_ids": ids}}

    def attack_map(state: ReportState) -> dict:
        scan = state["scan"]
        candidates = [
            {"evidence_id": item.id, **ATTACK_MAP[item.id]}
            for item in scan.evidence
            if item.id in ATTACK_MAP
        ]
        if not candidates:
            return {"attack": {"status": "complete", "techniques": []}}
        selection = _attempt(model, "attack", {"candidates": candidates}, AttackSelection)
        if selection is None:
            return {"attack": {"status": "unavailable", "techniques": []}}
        allowed = {(item["evidence_id"], item["technique_id"]) for item in candidates}
        techniques = []
        for item in selection.techniques:
            if (item.evidence_id, item.technique_id) in allowed:
                techniques.append(
                    {
                        "technique_id": item.technique_id,
                        "name": ATTACK_MAP[item.evidence_id]["name"],
                        "evidence_id": item.evidence_id,
                        "assessment": "candidate",
                    }
                )
        return {"attack": {"status": "complete", "techniques": techniques}}

    def intel(state: ReportState) -> dict:
        return {"intel": lookup_intel(state["scan"].sha256, settings)}

    def write(state: ReportState) -> dict:
        scan = state["scan"]
        by_id = {item.id: item for item in scan.evidence}
        selected = state["summary"]["fact_ids"] or [item.id for item in scan.evidence[:3]]
        status = scan.verdict or "inconclusive"
        first = f"The recorded scan verdict is {status}."
        summary = " ".join([first] + [by_id[item].detail for item in selected if item in by_id])
        next_step = (
            "quarantine"
            if scan.verdict == "malicious"
            else "release"
            if scan.verdict == "clean"
            else "sandbox"
        )
        report = {
            "verdict": scan.verdict,
            "score": scan.score,
            "summary": summary,
            "evidence": [item.model_dump() for item in scan.evidence],
            "techniques": state["attack"]["techniques"],
            "intel": state["intel"],
            "recommended_next_step": next_step,
            "parts": {
                "summary": state["summary"]["status"],
                "attack_mapping": state["attack"]["status"],
                "intel": state["intel"]["status"],
            },
        }
        return {"report": report}

    graph = StateGraph(ReportState)
    graph.add_node("summary", summarize)
    graph.add_node("attack", attack_map)
    graph.add_node("intel", intel)
    graph.add_node("writer", write)
    graph.add_edge(START, "summary")
    graph.add_edge(START, "attack")
    graph.add_edge(START, "intel")
    graph.add_edge(["summary", "attack", "intel"], "writer")
    graph.add_edge("writer", END)
    return graph.compile()


def build_report(scan: ReportInput, settings: Settings, client: ModelClient | None = None) -> dict:
    return build_graph(settings, client).invoke({"scan": scan})["report"]
