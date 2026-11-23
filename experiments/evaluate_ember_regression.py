import argparse
import hashlib
import json
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np

from experiments.evaluate_bodmas import score_summary
from experiments.train_full_experiment import load_reference
from experiments.train_full_more import remote_records

SAMPLE_SIZE = 20_000
TEST_SIZE = 200_000
SEED = 20261125


def evaluate(model_dir: Path, candidate_dir: Path, reference: Path):
    selected = set(random.Random(SEED).sample(range(TEST_SIZE), SAMPLE_SIZE))
    deployed = json.loads((model_dir / "ember_full_lgbm.json").read_text(encoding="utf-8"))
    comparison = json.loads((candidate_dir / "comparison.json").read_text(encoding="utf-8"))
    model_paths = {
        "primary": model_dir / "ember_full_lgbm.txt",
        "secondary": model_dir / "ember_full_ablation_lgbm.txt",
        "adapted": candidate_dir / "adapted_primary_lgbm.txt",
    }
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in model_paths.items()
    }
    if hashes["primary"] != deployed["model_sha256"]:
        raise ValueError("primary model hash changed")
    if hashes["secondary"] != deployed["ensemble"]["secondary_model_sha256"]:
        raise ValueError("secondary model hash changed")
    if hashes["adapted"] != comparison["candidate"]["model_sha256"]:
        raise ValueError("candidate model hash changed")
    models = {name: lgb.Booster(model_file=str(path)) for name, path in model_paths.items()}
    if models["primary"].num_trees() != 700 or models["adapted"].num_trees() != 820:
        raise ValueError("unexpected model tree counts")
    extractor = load_reference(reference)
    labels = []
    stages = (720, 740, 780, 820)
    scores = {name: [] for name in (*models, *(f"stage_{count}" for count in stages))}
    batch = []
    batch_labels = []

    def flush():
        if not batch:
            return
        matrix = np.asarray(batch, dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("non-finite EMBER vector")
        labels.extend(batch_labels)
        for name, model in models.items():
            scores[name].extend(model.predict(matrix, num_threads=4))
        for count in stages[:-1]:
            scores[f"stage_{count}"].extend(
                models["adapted"].predict(matrix, num_iteration=count, num_threads=4)
            )
        scores["stage_820"].extend(scores["adapted"][-len(batch_labels) :])
        batch.clear()
        batch_labels.clear()

    count = 0
    for index, record in enumerate(remote_records("test")):
        count = index + 1
        if index in selected:
            batch.append(extractor.process_raw_features(record))
            batch_labels.append(record["label"])
            if len(batch) >= 256:
                flush()
        if count % 25_000 == 0:
            print(f"EMBER test: {count}/{TEST_SIZE} records", flush=True)
    flush()
    if count != TEST_SIZE or len(labels) != SAMPLE_SIZE:
        raise ValueError("incomplete EMBER test sample")
    y = np.asarray(labels, dtype=np.uint8)
    primary = np.asarray(scores["primary"], dtype=np.float64)
    secondary = np.asarray(scores["secondary"], dtype=np.float64)
    adapted = np.asarray(scores["adapted"], dtype=np.float64)
    baseline = (primary + secondary) / 2
    candidate = (adapted + secondary) / 2
    old_threshold = deployed["threshold"]
    new_threshold = comparison["candidate"]["validation_calibrated"]["threshold"]
    result = {
        "dataset": "official EMBER 2018 test, diagnostic sample",
        "sample_rule": f"{SAMPLE_SIZE} random indices from {TEST_SIZE}, seed {SEED}",
        "source_sha256": deployed["dataset_sha256"],
        "model_sha256": hashes,
        "baseline_fixed": score_summary(y, baseline, old_threshold),
        "candidate_fixed": score_summary(y, candidate, old_threshold),
        "candidate_sorel_calibrated": score_summary(y, candidate, new_threshold),
        "limitations": [
            "The EMBER official test was viewed in earlier experiments; this is a regression diagnostic.",
            "Only the ML layer was measured.",
        ],
    }
    output = candidate_dir / "cross_source_ember2018.json"
    np.savez_compressed(
        candidate_dir / "ember_stage_scores.npz",
        labels=y,
        positions=np.asarray(sorted(selected), dtype=np.int32),
        **{name: np.asarray(values, dtype=np.float32) for name, values in scores.items()},
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--candidate-dir", type=Path, default=Path("data/sorel_adapt_20261120"))
    parser.add_argument(
        "--reference", type=Path, default=Path("data/wheels/ember_features_reference.py")
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model_dir, args.candidate_dir, args.reference), indent=2))


if __name__ == "__main__":
    main()
