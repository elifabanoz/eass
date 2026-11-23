import argparse
import hashlib
import json
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from experiments.evaluate_bodmas import metadata_dates, score_models, score_summary
from experiments.train_full_experiment import load_reference
from experiments.train_full_more import DATE_PATTERN, remote_records


def ember_validation(extractor, per_class: int = 100_000):
    indices = np.arange(per_class * 2)
    _, validation_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=indices >= per_class
    )
    positions = np.full(per_class * 2, -1, dtype=np.int32)
    positions[validation_indices] = np.arange(len(validation_indices))
    matrix = np.empty((len(validation_indices), extractor.dim), dtype=np.float32)
    filled = np.zeros(len(validation_indices), dtype=bool)
    counts = {0: 0, 1: 0}
    rng = random.Random(42)
    for record in remote_records("train"):
        label = record["label"]
        counts[label] += 1
        if counts[label] <= per_class:
            index = counts[label] - 1
        else:
            index = rng.randrange(counts[label])
            if index >= per_class:
                continue
        if not DATE_PATTERN.fullmatch(record.get("appeared", "")):
            raise ValueError("invalid EMBER appeared date")
        destination = positions[label * per_class + index]
        if destination >= 0:
            matrix[destination] = extractor.process_raw_features(record)
            filled[destination] = True
    if min(counts.values()) < per_class or not filled.all():
        raise ValueError("incomplete EMBER validation selection")
    return matrix, (validation_indices >= per_class).astype(np.uint8), counts


def threshold_for_fpr(scores, maximum_rate: float):
    if not 0 <= maximum_rate < 1 or len(scores) == 0:
        raise ValueError("invalid false-positive target or scores")
    sorted_scores = np.sort(np.asarray(scores, dtype=np.float64))
    if not np.isfinite(sorted_scores).all():
        raise ValueError("non-finite validation scores")
    allowed = int(np.floor(maximum_rate * len(sorted_scores)))
    return float(np.nextafter(sorted_scores[-allowed - 1], np.inf))


def calibrate(
    reference: Path,
    model_path: Path,
    current_metadata_path: Path,
    bodmas_npz: Path,
    bodmas_metadata: Path,
    official_scores_path: Path,
    output_dir: Path,
):
    current_metadata = json.loads(current_metadata_path.read_text(encoding="utf-8"))
    old_threshold = current_metadata["threshold"]
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_sha256 != current_metadata["model_sha256"]:
        raise ValueError("model and metadata SHA-256 mismatch")
    extractor = load_reference(reference)
    matrix, ember_labels, counts = ember_validation(extractor)
    model = lgb.Booster(model_file=str(model_path))
    ember_scores = model.predict(matrix, num_threads=4)
    del matrix
    bodmas_labels, bodmas_predictions = score_models(bodmas_npz, {"recent": model_path})
    bodmas_scores = bodmas_predictions["recent"]
    dates = metadata_dates(bodmas_metadata, len(bodmas_labels))
    recent_validation = np.asarray([date[:7] in ("2019-11", "2019-12") for date in dates])
    year_2020 = np.asarray([date[:4] == "2020" for date in dates])
    recent_labels = bodmas_labels[recent_validation]
    recent_scores = bodmas_scores[recent_validation]
    ember_required = threshold_for_fpr(ember_scores[ember_labels == 0], 0.01)
    recent_required = threshold_for_fpr(recent_scores[recent_labels == 0], 0.005)
    selected = max(old_threshold, ember_required, recent_required)
    with np.load(official_scores_path) as archive:
        official_labels = archive["labels"]
        official_scores = archive["scores"]
    if len(official_labels) != 200_000:
        raise ValueError("incomplete official EMBER test scores")
    prior_test = score_summary(official_labels, official_scores, old_threshold)
    if prior_test["confusion_matrix"] != current_metadata["official_test"]["confusion_matrix"]:
        raise ValueError("official test scores do not match current model metadata")
    result = {
        "model_sha256": model_sha256,
        "ember_train_source_counts": counts,
        "selection_rule": "max(previous threshold, EMBER validation 1% FPR cutoff, BODMAS late-2019 validation 0.5% FPR cutoff)",
        "previous_threshold": old_threshold,
        "ember_required_threshold": ember_required,
        "bodmas_required_threshold": recent_required,
        "selected_threshold": selected,
        "ember_2018_validation": score_summary(ember_labels, ember_scores, selected),
        "bodmas_late_2019_validation": score_summary(recent_labels, recent_scores, selected),
        "diagnostic_ember_2018_official_test": score_summary(
            official_labels, official_scores, selected
        ),
        "diagnostic_bodmas_2020": score_summary(
            bodmas_labels[year_2020], bodmas_scores[year_2020], selected
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "calibration_scores.npz",
        ember_validation_labels=ember_labels,
        ember_validation_scores=ember_scores,
        bodmas_validation_labels=recent_labels,
        bodmas_validation_scores=recent_scores,
    )
    (output_dir / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--current-metadata", type=Path, required=True)
    parser.add_argument("--bodmas-npz", type=Path, required=True)
    parser.add_argument("--bodmas-metadata", type=Path, required=True)
    parser.add_argument("--official-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            calibrate(
                args.reference,
                args.model,
                args.current_metadata,
                args.bodmas_npz,
                args.bodmas_metadata,
                args.official_scores,
                args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
