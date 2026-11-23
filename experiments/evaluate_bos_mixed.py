"""Compare the fixed BOS candidate with the deployed ensemble on reserved data."""

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from experiments.evaluate_bodmas import metadata_dates, score_models, score_summary

OUTPUT = Path("data/ermds_bos_20261120")


def run():
    metadata = json.loads(Path("models/ember_full_lgbm.json").read_text(encoding="utf-8"))
    validation = json.loads((OUTPUT / "bos_mixed_validation.json").read_text(encoding="utf-8"))
    selected = next(
        item for item in validation["validation"]["candidates"] if item["tree_count"] == 820
    )
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
        raise ValueError("candidate changed since validation")

    models = {name: lgb.Booster(model_file=str(path)) for name, path in paths.items()}
    with np.load(OUTPUT / "bos_splits.npz") as archive:
        x = archive["X_test"]
        y = archive["y_test"]
    if len(y) != 3000 or np.bincount(y, minlength=2).tolist() != [1500, 1500]:
        raise ValueError("unexpected reserved BOS test split")
    scores = {name: model.predict(x, num_threads=4) for name, model in models.items()}
    baseline = (scores["primary"] + scores["secondary"]) / 2
    candidate = (scores["candidate"] + scores["secondary"]) / 2
    bos_result = {
        "baseline": score_summary(y, baseline, metadata["threshold"]),
        "candidate": score_summary(y, candidate, selected["threshold"]),
    }
    print(
        "BOS reserved test",
        {
            name: (round(value["recall"], 4), round(value["false_positive_rate"], 4))
            for name, value in bos_result.items()
        },
        flush=True,
    )
    del x, y, scores, baseline, candidate

    labels, scores = score_models(Path("data/bodmas/bodmas.npz"), paths)
    dates = metadata_dates(Path("data/bodmas/bodmas_metadata.csv"), len(labels))
    mask = np.asarray([date[:4] == "2020" for date in dates])
    if np.count_nonzero(mask) != 87284:
        raise ValueError("unexpected BODMAS 2020 count")
    y = labels[mask]
    baseline = (scores["primary"][mask] + scores["secondary"][mask]) / 2
    candidate = (scores["candidate"][mask] + scores["secondary"][mask]) / 2
    bodmas_result = {
        "baseline": score_summary(y, baseline, metadata["threshold"]),
        "candidate": score_summary(y, candidate, selected["threshold"]),
    }
    print(
        "BODMAS 2020",
        {
            name: (round(value["recall"], 4), round(value["false_positive_rate"], 4))
            for name, value in bodmas_result.items()
        },
        flush=True,
    )
    result = {
        "model_sha256": hashes,
        "candidate_tree_count": 820,
        "candidate_threshold_from_validation": selected["threshold"],
        "bos_reserved_test": bos_result,
        "bodmas_2020": bodmas_result,
        "limitations": [
            "BOS split is SHA-disjoint but may share parent binaries across obfuscation variants.",
            "BODMAS 2020 was viewed in earlier experiments.",
            "Exact cross-source SHA overlap with EMBER and SOREL is unknown.",
            "Only the ML layer was measured.",
        ],
    }
    (OUTPUT / "bos_mixed_test.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
