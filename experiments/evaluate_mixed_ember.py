import hashlib
import json
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np

from experiments.evaluate_bodmas import score_summary
from experiments.train_full_experiment import load_reference
from experiments.train_full_more import remote_records

TEST_ROWS = 200_000
SAMPLE_ROWS = 20_000
OLD_SEED = 20261125
NEW_SEED = 20261202
OUTPUT_DIR = Path("data/mixed_adapt_20261120")


def evaluate():
    old_positions = set(random.Random(OLD_SEED).sample(range(TEST_ROWS), SAMPLE_ROWS))
    population = [index for index in range(TEST_ROWS) if index not in old_positions]
    positions = set(random.Random(NEW_SEED).sample(population, SAMPLE_ROWS))
    metadata = json.loads(Path("models/ember_full_lgbm.json").read_text(encoding="utf-8"))
    candidate = json.loads((OUTPUT_DIR / "validation.json").read_text(encoding="utf-8"))
    selected = candidate["validation"]["selected"]
    paths = {
        "primary": Path("models/ember_full_lgbm.txt"),
        "secondary": Path("models/ember_full_ablation_lgbm.txt"),
        "mixed": OUTPUT_DIR / "mixed_selected_primary_lgbm.txt",
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    if hashes["primary"] != metadata["model_sha256"]:
        raise ValueError("primary model changed")
    if hashes["secondary"] != metadata["ensemble"]["secondary_model_sha256"]:
        raise ValueError("secondary model changed")
    if hashes["mixed"] != candidate["selected_model_sha256"]:
        raise ValueError("mixed model changed")
    models = {name: lgb.Booster(model_file=str(path)) for name, path in paths.items()}
    extractor = load_reference(Path("data/wheels/ember_features_reference.py"))
    labels = []
    scores = {name: [] for name in models}
    batch = []
    batch_labels = []

    def flush():
        if not batch:
            return
        matrix = np.asarray(batch, dtype=np.float32)
        labels.extend(batch_labels)
        for name, model in models.items():
            scores[name].extend(model.predict(matrix, num_threads=4))
        batch.clear()
        batch_labels.clear()

    count = 0
    for index, record in enumerate(remote_records("test")):
        count = index + 1
        if index in positions:
            batch.append(extractor.process_raw_features(record))
            batch_labels.append(record["label"])
            if len(batch) == 256:
                flush()
        if count % 25_000 == 0:
            print(f"EMBER test: {count}/{TEST_ROWS} records", flush=True)
    flush()
    if count != TEST_ROWS or len(labels) != SAMPLE_ROWS:
        raise ValueError("incomplete EMBER diagnostic sample")
    y = np.asarray(labels, dtype=np.uint8)
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in scores.items()}
    baseline = (arrays["primary"] + arrays["secondary"]) / 2
    mixed = (arrays["mixed"] + arrays["secondary"]) / 2
    np.savez_compressed(
        OUTPUT_DIR / "mixed_ember_scores.npz",
        labels=y,
        positions=np.asarray(sorted(positions), dtype=np.int32),
        baseline=baseline.astype(np.float32),
        candidate=mixed.astype(np.float32),
    )
    result = {
        "dataset": "official EMBER 2018 test, disjoint diagnostic sample",
        "sample_rule": f"{SAMPLE_ROWS} from rows excluded from prior sample, seed {NEW_SEED}",
        "source_sha256": metadata["dataset_sha256"],
        "model_sha256": hashes,
        "baseline": score_summary(y, baseline, metadata["threshold"]),
        "candidate": score_summary(y, mixed, selected["threshold"]),
        "limitations": [
            "The official EMBER test source was viewed in earlier experiments.",
            "Only the ML layer was measured.",
        ],
    }
    (OUTPUT_DIR / "test_ember2018.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
