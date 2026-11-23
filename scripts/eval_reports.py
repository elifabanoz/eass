import json
import os
from pathlib import Path
from types import SimpleNamespace

from eass.config import get_settings
from eass.report.graph import ATTACK_MAP, build_report
from eass.report.input import build_input

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "scans"


class FixedModel:
    def complete(self, prompt_name, payload, schema):
        if prompt_name == "summary":
            return schema.model_validate(
                {"fact_ids": [item["id"] for item in payload["evidence"][:4]]}
            )
        return schema.model_validate(
            {
                "techniques": [
                    {"technique_id": item["technique_id"], "evidence_id": item["evidence_id"]}
                    for item in payload["candidates"]
                ]
            }
        )


def main() -> None:
    settings = get_settings()
    live = os.getenv("REPORT_EVAL_LIVE", "false").lower() == "true"
    model = None if live else FixedModel()
    checked = 0
    mapped = 0
    for path in sorted(FIXTURES.glob("*.json")):
        scan = SimpleNamespace(**json.loads(path.read_text(encoding="utf-8")))
        report_input = build_input(scan)
        report = build_report(report_input, settings, model)
        assert report["verdict"] == report_input.verdict
        assert report["score"] == report_input.score
        evidence_ids = {item.id for item in report_input.evidence}
        assert {item["id"] for item in report["evidence"]} == evidence_ids
        for technique in report["techniques"]:
            evidence_id = technique["evidence_id"]
            assert evidence_id in evidence_ids
            assert ATTACK_MAP[evidence_id]["technique_id"] == technique["technique_id"]
            mapped += 1
        checked += 1
    print(
        json.dumps(
            {
                "mode": "live" if live else "fixed",
                "reports_checked": checked,
                "techniques_checked": mapped,
            }
        )
    )


if __name__ == "__main__":
    main()
