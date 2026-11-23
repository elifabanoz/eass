import argparse
import hashlib
import json
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np

from experiments.calibrate_recent_model import threshold_for_fpr
from experiments.evaluate_bodmas import score_summary
from experiments.experiment_sorel_adapt import (
    TARGET_FPR,
    USED_TEST_STARTS,
    load_split,
)

STAGES = (720, 740, 780, 820)
CALIBRATION_SIZE = 10_000
SPLIT_SEED = 20261127
THIRD_TEST_SEED = 20261126


def split_ember(scores_path: Path):
    with np.load(scores_path) as archive:
        data = {name: archive[name] for name in archive.files}
    total = len(data["labels"])
    if total != 20_000:
        raise ValueError("unexpected EMBER sample size")
    chosen = random.Random(SPLIT_SEED).sample(range(total), CALIBRATION_SIZE)
    calibration = np.zeros(total, dtype=bool)
    calibration[chosen] = True
    return data, calibration, ~calibration


def stage_scores(model: lgb.Booster, secondary: lgb.Booster, features: np.ndarray):
    partner = secondary.predict(features, num_threads=4)
    return {
        count: (model.predict(features, num_iteration=count, num_threads=4) + partner) / 2
        for count in STAGES
    }


def evaluate(model_dir: Path, candidate_dir: Path):
    deployed = json.loads((model_dir / "ember_full_lgbm.json").read_text(encoding="utf-8"))
    old_threshold = deployed["threshold"]
    primary = lgb.Booster(model_file=str(model_dir / "ember_full_lgbm.txt"))
    secondary = lgb.Booster(model_file=str(model_dir / "ember_full_ablation_lgbm.txt"))
    model_path = candidate_dir / "adapted_primary_lgbm.txt"
    adapted = lgb.Booster(model_file=str(model_path))
    if primary.num_trees() != 700 or adapted.num_trees() != 820:
        raise ValueError("unexpected model tree counts")

    ember, calibration, control = split_ember(candidate_dir / "ember_stage_scores.npz")
    ember_y = ember["labels"]
    ember_baseline = (ember["primary"].astype(np.float64) + ember["secondary"]) / 2
    baseline_calibration = score_summary(
        ember_y[calibration], ember_baseline[calibration], old_threshold
    )
    allowed_ember_fpr = baseline_calibration["false_positive_rate"]

    x_val, y_val, val_info = load_split("validation", 16, 20261123)
    baseline_val = (
        primary.predict(x_val, num_threads=4) + secondary.predict(x_val, num_threads=4)
    ) / 2
    stages_val = stage_scores(adapted, secondary, x_val)
    candidates = []
    for count in STAGES:
        ember_scores = (ember[f"stage_{count}"].astype(np.float64) + ember["secondary"]) / 2
        sorel_threshold = threshold_for_fpr(stages_val[count][y_val == 0], TARGET_FPR)
        ember_threshold = threshold_for_fpr(
            ember_scores[calibration & (ember_y == 0)], allowed_ember_fpr
        )
        threshold = max(sorel_threshold, ember_threshold)
        candidates.append(
            {
                "tree_count": count,
                "threshold": threshold,
                "sorel_validation": score_summary(y_val, stages_val[count], threshold),
                "ember_calibration": score_summary(
                    ember_y[calibration], ember_scores[calibration], threshold
                ),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["sorel_validation"]["recall"],
            item["ember_calibration"]["recall"],
        ),
        reverse=True,
    )
    chosen = candidates[0]
    count = chosen["tree_count"]
    threshold = chosen["threshold"]
    print(f"selected stage {count}, threshold {threshold:.6f}", flush=True)

    ember_chosen = (ember[f"stage_{count}"].astype(np.float64) + ember["secondary"]) / 2
    ember_baseline_control = score_summary(ember_y[control], ember_baseline[control], old_threshold)
    ember_candidate_control = score_summary(ember_y[control], ember_chosen[control], threshold)

    previous = json.loads((candidate_dir / "comparison.json").read_text(encoding="utf-8"))
    excluded = tuple(USED_TEST_STARTS) + tuple(previous["design"]["new_test"]["starts"])
    x_test, y_test, test_info = load_split("test", 32, THIRD_TEST_SEED, excluded)
    sorel_baseline_test = (
        primary.predict(x_test, num_threads=4) + secondary.predict(x_test, num_threads=4)
    ) / 2
    sorel_candidate_test = stage_scores(adapted, secondary, x_test)[count]
    sorel_baseline_result = score_summary(y_test, sorel_baseline_test, old_threshold)
    sorel_candidate_result = score_summary(y_test, sorel_candidate_test, threshold)

    accepted = (
        ember_candidate_control["false_positive_rate"]
        <= ember_baseline_control["false_positive_rate"]
        and sorel_candidate_result["recall"] > sorel_baseline_result["recall"]
        and sorel_candidate_result["false_positive_rate"] <= 0.01
    )
    partial_path = candidate_dir / "selected_stage_primary_lgbm.txt"
    adapted.save_model(str(partial_path), num_iteration=count)
    result = {
        "selection_rule": "maximize SOREL validation recall while SOREL validation FPR <= 0.5% and EMBER calibration FPR <= current model",
        "ember_split_seed": SPLIT_SEED,
        "ember_calibration_count": int(calibration.sum()),
        "ember_control_count": int(control.sum()),
        "sorel_validation": val_info,
        "sorel_third_test_source": test_info,
        "excluded_prior_test_starts": list(excluded),
        "baseline_sorel_validation": score_summary(y_val, baseline_val, old_threshold),
        "baseline_ember_calibration": baseline_calibration,
        "candidates_ranked": candidates,
        "selected_tree_count": count,
        "selected_threshold": threshold,
        "selected_model_sha256": hashlib.sha256(partial_path.read_bytes()).hexdigest(),
        "ember_control": {
            "baseline": ember_baseline_control,
            "candidate": ember_candidate_control,
        },
        "sorel_third_test": {
            "baseline": sorel_baseline_result,
            "candidate": sorel_candidate_result,
        },
        "passed_ember_sorel_gates_only": accepted,
        "limitations": [
            "The EMBER official test was viewed before this experiment; its split is an engineering diagnostic.",
            "SOREL test source was viewed earlier, although third-test rows exclude prior samples.",
            "Exact cross-dataset SHA overlap is unknown.",
            "Only the ML layer was measured.",
        ],
    }
    (candidate_dir / "stage_selection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--candidate-dir", type=Path, default=Path("data/sorel_adapt_20261120"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model_dir, args.candidate_dir), indent=2))


if __name__ == "__main__":
    main()
