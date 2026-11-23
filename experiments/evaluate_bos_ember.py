"""Measure the fixed BOS candidate on a new EMBER 2018 test sample."""

import hashlib
import json
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np

from experiments.evaluate_bodmas import score_summary
from experiments.train_full_experiment import load_reference
from experiments.train_full_more import remote_records

OUTPUT = Path("data/ermds_bos_20261120")
TEST_ROWS = 200_000
SAMPLE_ROWS = 10_000


def run():
    old = set(random.Random(20261125).sample(range(TEST_ROWS), 20_000))
    population = [index for index in range(TEST_ROWS) if index not in old]
    second = set(random.Random(20261202).sample(population, 20_000))
    population = [index for index in population if index not in second]
    selected_rows = set(random.Random(20261209).sample(population, SAMPLE_ROWS))
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
        raise ValueError("candidate changed")
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
        if index in selected_rows:
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
    candidate = (arrays["candidate"] + arrays["secondary"]) / 2
    result = {
        "sample_rule": "10000 rows disjoint from prior 20000 plus 20000 selections; seed 20261209",
        "source_sha256": metadata["dataset_sha256"],
        "model_sha256": hashes,
        "baseline": score_summary(y, baseline, metadata["threshold"]),
        "candidate": score_summary(y, candidate, selected["threshold"]),
        "limitations": [
            "EMBER test source was viewed previously; these selected rows were not.",
            "Sample rows may share families with prior fit or validation rows.",
            "Only ML scores were measured.",
        ],
    }
    (OUTPUT / "bos_mixed_ember_test.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
