import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eass.report.graph import build_report, lookup_intel
from eass.report.input import Evidence, ReportInput


class FakeModel:
    def __init__(self):
        self.calls = []

    def complete(self, prompt_name, payload, schema):
        self.calls.append(prompt_name)
        if prompt_name == "summary":
            return schema.model_validate({"fact_ids": [payload["evidence"][0]["id"]]})
        return schema.model_validate(
            {
                "techniques": [
                    {"technique_id": "T1055", "evidence_id": "pe-import-chain"},
                    {"technique_id": "T9999", "evidence_id": "pe-import-chain"},
                ]
            }
        )


def settings():
    return SimpleNamespace(offline_mode=True, malwarebazaar_key="unused")


class ReportGraphTests(unittest.TestCase):
    def test_report_uses_only_cited_evidence(self):
        scan = ReportInput(
            verdict="suspicious",
            score=45,
            file_type="pe",
            size=200,
            sha256="a" * 64,
            evidence=[
                Evidence(
                    id="pe-import-chain",
                    kind="pe",
                    detail="PE imports VirtualAllocEx, WriteProcessMemory and CreateRemoteThread.",
                )
            ],
        )
        model = FakeModel()
        report = build_report(scan, settings(), model)
        self.assertEqual(report["verdict"], "suspicious")
        self.assertEqual(report["score"], 45)
        self.assertEqual(report["recommended_next_step"], "sandbox")
        self.assertEqual(report["techniques"][0]["technique_id"], "T1055")
        self.assertEqual(len(report["techniques"]), 1)
        self.assertEqual(report["techniques"][0]["assessment"], "candidate")
        self.assertEqual(set(model.calls), {"summary", "attack"})
        self.assertEqual(report["intel"]["status"], "skipped")

    def test_offline_mode_never_calls_network(self):
        with patch("eass.report.graph.urllib.request.urlopen") as network:
            self.assertEqual(lookup_intel("a" * 64, settings()), {"status": "skipped"})
        network.assert_not_called()

    def test_invalid_model_output_does_not_change_scan_verdict(self):
        class BrokenModel:
            calls = 0

            def complete(self, *_args):
                self.calls += 1
                raise ValueError("bad output")

        scan = ReportInput(
            verdict="malicious",
            score=100,
            file_type="text",
            size=68,
            sha256="a" * 64,
            evidence=[Evidence(id="hash-local", kind="hash", detail="EICAR test hash matched.")],
        )
        model = BrokenModel()
        report = build_report(scan, settings(), model)
        self.assertEqual(report["verdict"], "malicious")
        self.assertEqual(report["score"], 100)
        self.assertEqual(report["parts"]["summary"], "unavailable")
        self.assertEqual(model.calls, 2)


if __name__ == "__main__":
    unittest.main()
