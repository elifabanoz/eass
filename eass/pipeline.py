import logging
from pathlib import Path

from eass.config import get_settings
from eass.layers import hashrep, ml, pe_static, yara_scan
from eass.scoring import score_layers

logger = logging.getLogger(__name__)
RULE_PATH = Path(__file__).resolve().parent.parent / "rules" / "eass.yar"


def run_scan(path: Path, file_type: str, sha256: str) -> dict:
    layers = {}
    for name, action in (
        ("hash", lambda: hashrep.check_local(sha256, get_settings().hash_list_path)),
        (
            "malwarebazaar",
            lambda: hashrep.check_malwarebazaar(sha256, get_settings().malwarebazaar_key),
        ),
        ("yara", lambda: yara_scan.scan(path, RULE_PATH)),
        ("pe", lambda: pe_static.analyze(path) if file_type == "pe" else {"status": "skipped"}),
        (
            "ml",
            lambda: (
                ml.analyze(path, get_settings().model_dir)
                if file_type == "pe"
                else {"status": "skipped", "reason": "not a PE file"}
            ),
        ),
    ):
        try:
            layers[name] = action()
        except Exception as exc:
            logger.warning("%s layer failed: %s", name, type(exc).__name__)
            layers[name] = {"status": "error", "error": type(exc).__name__}
    return {"layers": layers, **score_layers(file_type, layers)}
