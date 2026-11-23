"""Test a BOS adaptation without changing the deployed model."""

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from experiments.calibrate_recent_model import threshold_for_fpr
from experiments.evaluate_bodmas import score_summary

OUTPUT = Path("data/ermds_bos_20261120")
STAGES = (720, 740, 760, 780, 800, 820)


def run():
    metadata = json.loads(Path("models/ember_full_lgbm.json").read_text(encoding="utf-8"))
    primary_path = Path("models/ember_full_lgbm.txt")
    secondary_path = Path("models/ember_full_ablation_lgbm.txt")
    if hashlib.sha256(primary_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("primary model differs from deployed metadata")
    if (
        hashlib.sha256(secondary_path.read_bytes()).hexdigest()
        != metadata["ensemble"]["secondary_model_sha256"]
    ):
        raise ValueError("secondary model differs from deployed metadata")

    primary = lgb.Booster(model_file=str(primary_path))
    secondary = lgb.Booster(model_file=str(secondary_path))
    if primary.num_trees() != 700:
        raise ValueError("unexpected primary tree count")
    with np.load("data/mixed_adapt_20261120/sample_cache.npz") as cached:
        normal_fit = cached["x_fit"]
        normal_labels = cached["y_fit"]
        validation = {
            "bodmas_2019": (cached["x_bod_val"], cached["y_bod_val"]),
            "sorel": (cached["x_sorel_val"], cached["y_sorel_val"]),
        }
    with np.load(OUTPUT / "bos_splits.npz") as bos:
        bos_fit = bos["X_fit"]
        bos_labels = bos["y_fit"]
        validation["bos"] = (bos["X_validation"], bos["y_validation"])
    x_fit = np.concatenate((normal_fit, bos_fit))
    y_fit = np.concatenate((normal_labels, bos_labels))
    del normal_fit, bos_fit
    if x_fit.shape != (len(y_fit), 2381) or not np.isfinite(x_fit).all():
        raise ValueError("invalid training features")
    print(f"Training on {len(y_fit)} rows", flush=True)
    train_set = lgb.Dataset(x_fit, label=y_fit, free_raw_data=True)
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.02,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "num_threads": 4,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=120,
        init_model=str(primary_path),
    )
    if model.num_trees() != 820:
        raise ValueError("unexpected candidate tree count")
    del x_fit, train_set
    print("Candidate trained; measuring validation sources", flush=True)

    old_threshold = metadata["threshold"]
    secondary_scores = {
        name: secondary.predict(x, num_threads=4) for name, (x, _) in validation.items()
    }
    baseline = {
        name: score_summary(
            y,
            (primary.predict(x, num_threads=4) + secondary_scores[name]) / 2,
            old_threshold,
        )
        for name, (x, y) in validation.items()
    }
    candidates = []
    for count in STAGES:
        scores = {
            name: (model.predict(x, num_iteration=count, num_threads=4) + secondary_scores[name])
            / 2
            for name, (x, _) in validation.items()
        }
        threshold = max(
            threshold_for_fpr(scores[name][y == 0], 0.01 if name == "bos" else 0.005)
            for name, (_, y) in validation.items()
        )
        candidates.append(
            {
                "tree_count": count,
                "threshold": threshold,
                "sources": {
                    name: score_summary(y, scores[name], threshold)
                    for name, (_, y) in validation.items()
                },
            }
        )
        print(
            count,
            "threshold",
            round(threshold, 4),
            "recall",
            {name: round(item["recall"], 4) for name, item in candidates[-1]["sources"].items()},
            flush=True,
        )
    path = OUTPUT / "bos_mixed_primary_lgbm.txt"
    model.save_model(str(path))
    result = {
        "training": {
            "normal_fit_count": len(normal_labels),
            "bos_fit_count": len(bos_labels),
            "total_count": len(y_fit),
        },
        "validation": {
            "source_counts": {name: len(y) for name, (_, y) in validation.items()},
            "baseline": baseline,
            "candidates": candidates,
        },
        "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "deployed_primary_sha256": metadata["model_sha256"],
        "limitations": [
            "BOS SHA split does not guarantee independence across variants of one original binary.",
            "BOS benign and malicious records show strong source-specific feature separation.",
            "All normal validation sources were viewed in earlier experiments.",
            "No test rows were read for this selection.",
            "Only the ML layer was measured.",
        ],
    }
    (OUTPUT / "bos_mixed_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
