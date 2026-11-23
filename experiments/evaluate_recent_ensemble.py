import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.calibrate_recent_model import threshold_for_fpr
from experiments.evaluate_bodmas import metadata_dates, score_models, score_summary


def blended(left, right):
    if left.shape != right.shape:
        raise ValueError("model score shapes differ")
    return (left.astype(np.float64) + right.astype(np.float64)) / 2


def evaluate(
    current_model: Path,
    current_metadata: Path,
    ablation_model: Path,
    ablation_metadata: Path,
    current_validation: Path,
    ablation_validation: Path,
    current_official: Path,
    ablation_official: Path,
    bodmas_npz: Path,
    bodmas_metadata: Path,
    output: Path,
):
    current_meta = json.loads(current_metadata.read_text(encoding="utf-8"))
    ablation_meta = json.loads(ablation_metadata.read_text(encoding="utf-8"))
    if hashlib.sha256(current_model.read_bytes()).hexdigest() != current_meta["model_sha256"]:
        raise ValueError("current model SHA-256 mismatch")
    if hashlib.sha256(ablation_model.read_bytes()).hexdigest() != ablation_meta["model_sha256"]:
        raise ValueError("ablation model SHA-256 mismatch")
    with np.load(current_validation) as current, np.load(ablation_validation) as ablated:
        ember_labels = current["ember_validation_labels"]
        bodmas_labels = current["bodmas_validation_labels"]
        if not np.array_equal(ember_labels, ablated["ember_labels"]):
            raise ValueError("EMBER validation labels differ")
        if not np.array_equal(bodmas_labels, ablated["bodmas_labels"]):
            raise ValueError("BODMAS validation labels differ")
        ember_scores = blended(current["ember_validation_scores"], ablated["ember_scores"])
        bodmas_scores = blended(current["bodmas_validation_scores"], ablated["bodmas_scores"])
    ember_required = threshold_for_fpr(ember_scores[ember_labels == 0], 0.01)
    bodmas_required = threshold_for_fpr(bodmas_scores[bodmas_labels == 0], 0.005)
    threshold = max(ember_required, bodmas_required)
    with np.load(current_official) as current, np.load(ablation_official) as ablated:
        official_labels = current["labels"]
        if not np.array_equal(official_labels, ablated["labels"]):
            raise ValueError("official EMBER test labels differ")
        official_scores = blended(current["scores"], ablated["scores"])
    external_labels, external_scores = score_models(
        bodmas_npz, {"current": current_model, "ablation": ablation_model}
    )
    dates = metadata_dates(bodmas_metadata, len(external_labels))
    year_2020 = np.asarray([date[:4] == "2020" for date in dates])
    external_blend = blended(external_scores["current"], external_scores["ablation"])
    result = {
        "model_sha256": {
            "current": current_meta["model_sha256"],
            "ablation": ablation_meta["model_sha256"],
        },
        "blend_weights": {"current": 0.5, "ablation": 0.5},
        "threshold_rule": "max(EMBER validation 1% FPR cutoff, BODMAS late-2019 validation 0.5% FPR cutoff)",
        "ember_required_threshold": ember_required,
        "bodmas_required_threshold": bodmas_required,
        "threshold": threshold,
        "ember_validation": score_summary(ember_labels, ember_scores, threshold),
        "bodmas_2019_validation": score_summary(bodmas_labels, bodmas_scores, threshold),
        "diagnostic_ember_official_test": score_summary(
            official_labels, official_scores, threshold
        ),
        "diagnostic_bodmas_2020": score_summary(
            external_labels[year_2020], external_blend[year_2020], threshold
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "current-model",
        "current-metadata",
        "ablation-model",
        "ablation-metadata",
        "current-validation",
        "ablation-validation",
        "current-official",
        "ablation-official",
        "bodmas-npz",
        "bodmas-metadata",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.current_model,
                args.current_metadata,
                args.ablation_model,
                args.ablation_metadata,
                args.current_validation,
                args.ablation_validation,
                args.current_official,
                args.ablation_official,
                args.bodmas_npz,
                args.bodmas_metadata,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
