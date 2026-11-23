"""Evaluate a fixed BOS candidate on new, nonoverlapping SOREL test blocks."""

import hashlib
import json
from pathlib import Path

import lightgbm as lgb

from experiments.evaluate_bodmas import score_summary
from experiments.experiment_sorel_adapt import USED_TEST_STARTS, load_split

OUTPUT = Path("data/ermds_bos_20261120")


def run():
    metadata = json.loads(Path("models/ember_full_lgbm.json").read_text(encoding="utf-8"))
    validation = json.loads((OUTPUT / "bos_mixed_validation.json").read_text(encoding="utf-8"))
    selected = next(
        item for item in validation["validation"]["candidates"] if item["tree_count"] == 820
    )
    sources = [
        (Path("data/sorel_adapt_20261120/comparison.json"), ("design", "new_test")),
        (Path("data/sorel_adapt_20261120/stage_selection.json"), ("sorel_third_test_source",)),
        (Path("data/mixed_adapt_20261120/test_sorel_fourth.json"), ("sample",)),
    ]
    excluded = set(USED_TEST_STARTS)
    for path, keys in sources:
        document = json.loads(path.read_text(encoding="utf-8"))
        entry = document
        for key in keys:
            entry = entry[key]
        excluded.update(entry["starts"])
    x, y, sample = load_split("test", 32, 20261207, tuple(excluded))
    if any(
        start < old + 256 and old < start + 256 for start in sample["starts"] for old in excluded
    ):
        raise ValueError("test blocks overlap an earlier test sample")
    paths = {
        "primary": Path("models/ember_full_lgbm.txt"),
        "secondary": Path("models/ember_full_ablation_lgbm.txt"),
        "candidate": OUTPUT / "bos_mixed_primary_lgbm.txt",
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    if hashes["primary"] != metadata["model_sha256"]:
        raise ValueError("deployed primary changed")
    if hashes["secondary"] != metadata["ensemble"]["secondary_model_sha256"]:
        raise ValueError("deployed secondary changed")
    if hashes["candidate"] != validation["candidate_sha256"]:
        raise ValueError("candidate changed")
    scores = {
        name: lgb.Booster(model_file=str(path)).predict(x, num_threads=4)
        for name, path in paths.items()
    }
    baseline = (scores["primary"] + scores["secondary"]) / 2
    candidate = (scores["candidate"] + scores["secondary"]) / 2
    result = {
        "sample": sample,
        "excluded_prior_test_block_count": len(excluded),
        "model_sha256": hashes,
        "baseline": score_summary(y, baseline, metadata["threshold"]),
        "candidate": score_summary(y, candidate, selected["threshold"]),
        "limitations": [
            "SOREL source has been viewed in prior experiments; block rows are disjoint.",
            "Exact SHA overlap across EMBER, BODMAS, and SOREL is unknown.",
            "Only ML scores were measured.",
        ],
    }
    (OUTPUT / "bos_mixed_sorel_test.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
