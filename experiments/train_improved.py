import argparse
import json
import random
import tarfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from eass.layers.ml_features import FEATURE_SET, vector_from_record
from experiments.train import EMBER2018_SHA256, evaluate, verify_archive


def _records(archive_path: Path, kind: str):
    with tarfile.open(archive_path, mode="r|bz2") as archive:
        for member in archive:
            name = Path(member.name).name
            selected = (
                name.startswith("train_features_") and name.endswith(".jsonl")
                if kind == "train"
                else name == "test_features.jsonl"
            )
            if not member.isfile() or not selected:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            for line in stream:
                record = json.loads(line)
                if record.get("label") in (0, 1):
                    yield record
            print(f"Read {name}", flush=True)
            if kind == "test":
                return


def _sample_training(archive_path: Path, per_class: int):
    rng = random.Random(42)
    rows = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    for record in _records(archive_path, "train"):
        label = record["label"]
        seen[label] += 1
        group = rows[label]
        if len(group) < per_class:
            group.append(vector_from_record(record))
        else:
            index = rng.randrange(seen[label])
            if index < per_class:
                group[index] = vector_from_record(record)
    if min(map(len, rows.values())) < per_class:
        raise ValueError("not enough labeled EMBER training records")
    vectors = np.stack(rows[0] + rows[1])
    labels = np.array([0] * per_class + [1] * per_class, dtype=np.uint8)
    return vectors, labels, seen


def _test_scores(archive_path: Path, model: lgb.LGBMClassifier):
    labels = []
    scores = []
    vectors = []
    batch_labels = []
    for record in _records(archive_path, "test"):
        vectors.append(vector_from_record(record))
        batch_labels.append(record["label"])
        if len(vectors) >= 1024:
            scores.extend(model.predict_proba(np.stack(vectors))[:, 1])
            labels.extend(batch_labels)
            vectors.clear()
            batch_labels.clear()
    if vectors:
        scores.extend(model.predict_proba(np.stack(vectors))[:, 1])
        labels.extend(batch_labels)
    if len(labels) != 200_000:
        raise ValueError("official EMBER test split is incomplete")
    return np.asarray(labels, dtype=np.uint8), np.asarray(scores, dtype=np.float32)


def train(archive_path: Path, output_dir: Path, per_class: int) -> dict:
    verify_archive(archive_path)
    vectors, labels, seen = _sample_training(archive_path, per_class)
    x_fit, x_val, y_fit, y_val = train_test_split(
        vectors, labels, test_size=0.2, random_state=42, stratify=labels
    )
    del vectors, labels
    candidates = (
        {"n_estimators": 250, "num_leaves": 31, "learning_rate": 0.05},
        {"n_estimators": 400, "num_leaves": 63, "learning_rate": 0.05},
    )
    best = None
    trials = []
    for parameters in candidates:
        model = lgb.LGBMClassifier(
            **parameters,
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(x_fit, y_fit)
        validation_probabilities = model.predict_proba(x_val)[:, 1]
        threshold = float(np.quantile(validation_probabilities[y_val == 0], 0.99))
        validation = evaluate(y_val, validation_probabilities, threshold)
        trials.append({"parameters": parameters, "threshold": threshold, "validation": validation})
        print(
            json.dumps(
                {
                    "trial": parameters,
                    "validation_recall": validation["recall"],
                    "validation_fpr": validation["false_positive_rate"],
                }
            ),
            flush=True,
        )
        if best is None or (validation["recall"], validation["roc_auc"]) > (
            best["validation"]["recall"],
            best["validation"]["roc_auc"],
        ):
            best = {
                "model": model,
                "parameters": parameters,
                "threshold": threshold,
                "validation": validation,
            }
    y_test, test_probabilities = _test_scores(archive_path, best["model"])
    official_test = evaluate(y_test, test_probabilities, best["threshold"])
    metadata = {
        "dataset": "EMBER 2018 v2",
        "dataset_sha256": EMBER2018_SHA256,
        "feature_set": FEATURE_SET,
        "train_source_counts": seen,
        "train_selected": per_class * 2,
        "fit_count": len(y_fit),
        "validation_count": len(y_val),
        "official_test_count": len(y_test),
        "threshold": best["threshold"],
        "parameters": best["parameters"],
        "validation": best["validation"],
        "official_test": official_test,
        "trials": trials,
        "lightgbm_version": lgb.__version__,
        "model_type": "byte histogram, byte entropy and string features",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    best["model"].booster_.save_model(str(output_dir / "candidate_extended_lgbm.txt"))
    (output_dir / "candidate_extended_lgbm.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--per-class", type=int, default=40_000)
    args = parser.parse_args()
    print(json.dumps(train(args.archive, args.output_dir, args.per_class), indent=2))


if __name__ == "__main__":
    main()
