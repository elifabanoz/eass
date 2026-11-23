import argparse
import hashlib
import json
import random
import tarfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from eass.layers.ml import FEATURE_SET, vector_from_record

EMBER2018_SHA256 = "b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812"


def verify_archive(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        while chunk := archive.read(4 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != EMBER2018_SHA256:
        raise ValueError("EMBER 2018 v2 archive SHA-256 mismatch")


def load_dataset(archive_path: Path, per_class: int):
    rng = random.Random(42)
    train_rows = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    test_rows = []
    test_labels = []
    with tarfile.open(archive_path, mode="r|bz2") as archive:
        for member in archive:
            name = Path(member.name).name
            is_train = name.startswith("train_features_") and name.endswith(".jsonl")
            is_test = name == "test_features.jsonl"
            if not member.isfile() or not (is_train or is_test):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            for line in stream:
                record = json.loads(line)
                label = record.get("label")
                if label not in (0, 1):
                    continue
                vector = vector_from_record(record)
                if is_test:
                    test_rows.append(vector)
                    test_labels.append(label)
                else:
                    seen[label] += 1
                    rows = train_rows[label]
                    if len(rows) < per_class:
                        rows.append(vector)
                    else:
                        index = rng.randrange(seen[label])
                        if index < per_class:
                            rows[index] = vector
            print(f"Read {name}", flush=True)
    if min(len(train_rows[0]), len(train_rows[1])) < 1000 or len(test_rows) < 1000:
        raise ValueError("EMBER training or official test records are missing")
    x_train = np.stack(train_rows[0] + train_rows[1])
    y_train = np.array([0] * len(train_rows[0]) + [1] * len(train_rows[1]))
    x_test = np.stack(test_rows)
    y_test = np.array(test_labels)
    return x_train, y_train, x_test, y_test, seen


def evaluate(y_true, probabilities, threshold: float) -> dict:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train(archive: Path, model_dir: Path, per_class: int) -> dict:
    verify_archive(archive)
    x_train, y_train, x_test, y_test, seen = load_dataset(archive, per_class)
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(x_fit, y_fit)
    val_probabilities = model.predict_proba(x_val)[:, 1]
    threshold = max(0.5, float(np.quantile(val_probabilities[y_val == 0], 0.99)))
    test_probabilities = model.predict_proba(x_test)[:, 1]
    metadata = {
        "dataset": "EMBER 2018 v2",
        "dataset_sha256": EMBER2018_SHA256,
        "feature_set": FEATURE_SET,
        "train_source_counts": seen,
        "train_selected": len(y_train),
        "fit_count": len(y_fit),
        "validation_count": len(y_val),
        "official_test_count": len(y_test),
        "threshold": threshold,
        "validation": evaluate(y_val, val_probabilities, threshold),
        "official_test": evaluate(y_test, test_probabilities, threshold),
        "lightgbm_version": lgb.__version__,
        "model_type": "histogram-only baseline",
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_dir / "ember_histogram_lgbm.txt"))
    (model_dir / "ember_histogram_lgbm.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--per-class", type=int, default=30_000)
    args = parser.parse_args()
    print(json.dumps(train(args.archive, args.model_dir, args.per_class), indent=2))


if __name__ == "__main__":
    main()
