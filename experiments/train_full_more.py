import argparse
import gc
import hashlib
import json
import random
import re
import tarfile
from pathlib import Path
from urllib.request import urlopen

import lightgbm as lgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from experiments.train import EMBER2018_SHA256, evaluate
from experiments.train_full_experiment import load_reference

DATASET_URL = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"
DATE_PATTERN = re.compile(r"^20\d{2}-(0[1-9]|1[0-2])$")


class HashingReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        chunk = self.source.read(1024 * 1024 if size < 0 else size)
        self.digest.update(chunk)
        return chunk


def remote_records(kind: str):
    with urlopen(DATASET_URL, timeout=120) as response:
        stream = HashingReader(response)
        training_members_seen = 0
        with tarfile.open(fileobj=stream, mode="r|bz2") as archive:
            for member in archive:
                name = Path(member.name).name
                selected = (
                    name.startswith("train_features_") and name.endswith(".jsonl")
                    if kind == "train"
                    else name == "test_features.jsonl"
                )
                if not member.isfile() or not selected:
                    continue
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    continue
                for line in file_obj:
                    record = json.loads(line)
                    if record.get("label") in (0, 1):
                        yield record
                print(f"Read {name}", flush=True)
                if kind == "test":
                    break
                training_members_seen += 1
                if training_members_seen == 6:
                    break
        while stream.read(1024 * 1024):
            pass
        if stream.digest.hexdigest() != EMBER2018_SHA256:
            raise ValueError("EMBER archive SHA-256 mismatch")
        print("Verified EMBER archive SHA-256", flush=True)


def sample_training(extractor, per_class: int):
    rng = random.Random(42)
    matrix = np.empty((per_class * 2, extractor.dim), dtype=np.float32)
    dates = [""] * (per_class * 2)
    counts = {0: 0, 1: 0}
    for record in remote_records("train"):
        label = record["label"]
        counts[label] += 1
        if counts[label] <= per_class:
            index = counts[label] - 1
        else:
            index = rng.randrange(counts[label])
            if index >= per_class:
                continue
        position = label * per_class + index
        appeared = record.get("appeared", "")
        if not DATE_PATTERN.fullmatch(appeared):
            raise ValueError("invalid EMBER appeared date")
        dates[position] = appeared
        matrix[position] = extractor.process_raw_features(record)
    if min(counts.values()) < per_class:
        raise ValueError("not enough labeled EMBER training records")
    return matrix, dates, counts


def temporal_indices(dates: list[str], per_class: int):
    ordered = sorted(range(per_class * 2), key=lambda index: (dates[index], index))
    cutoff = int(len(ordered) * 0.8)
    return np.asarray(ordered[:cutoff]), np.asarray(ordered[cutoff:])


def test_scores(extractor, model):
    labels = []
    scores = []
    batch = []
    batch_labels = []
    for record in remote_records("test"):
        batch.append(extractor.process_raw_features(record))
        batch_labels.append(record["label"])
        if len(batch) >= 512:
            scores.extend(model.predict_proba(np.asarray(batch))[:, 1])
            labels.extend(batch_labels)
            batch.clear()
            batch_labels.clear()
    if batch:
        scores.extend(model.predict_proba(np.asarray(batch))[:, 1])
        labels.extend(batch_labels)
    if len(labels) != 200_000:
        raise ValueError("official EMBER test split is incomplete")
    return np.asarray(labels, dtype=np.uint8), np.asarray(scores, dtype=np.float32)


def train(
    reference: Path, output_dir: Path, per_class: int, split_strategy: str, skip_official_test: bool
):
    if per_class < 1000 or per_class > 300_000:
        raise ValueError("per_class outside supported range")
    extractor = load_reference(reference)
    matrix, dates, counts = sample_training(extractor, per_class)
    if split_strategy == "temporal":
        fit_indices, val_indices = temporal_indices(dates, per_class)
    elif split_strategy == "stratified_random":
        indices = np.arange(per_class * 2)
        labels = indices >= per_class
        fit_indices, val_indices = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labels
        )
    else:
        raise ValueError("unsupported split strategy")
    sparse = csr_matrix(matrix)
    del matrix
    gc.collect()
    x_fit = sparse[fit_indices]
    x_val = sparse[val_indices]
    del sparse
    gc.collect()
    y_fit = np.asarray(fit_indices >= per_class, dtype=np.uint8)
    y_val = np.asarray(val_indices >= per_class, dtype=np.uint8)
    if (y_val == 0).sum() < 1000 or (y_val == 1).sum() < 1000:
        raise ValueError("temporal validation split has too few examples of a class")
    model = lgb.LGBMClassifier(
        n_estimators=700,
        num_leaves=127,
        learning_rate=0.05,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(x_fit, y_fit)
    val_scores = model.predict_proba(x_val)[:, 1]
    threshold = float(np.quantile(val_scores[y_val == 0], 0.99))
    validation = evaluate(y_val, val_scores, threshold)
    print(
        json.dumps(
            {
                "validation_recall": validation["recall"],
                "validation_fpr": validation["false_positive_rate"],
            }
        ),
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(output_dir / "candidate_more_lgbm.txt"))
    train_dates = [dates[index] for index in fit_indices]
    validation_dates = [dates[index] for index in val_indices]
    del x_fit, x_val, y_fit, y_val, val_scores
    gc.collect()
    metadata = {
        "dataset": "EMBER 2018 v2",
        "dataset_sha256": EMBER2018_SHA256,
        "feature_set": "ember_full_v2_experiment",
        "feature_dim": extractor.dim,
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "train_source_counts": counts,
        "train_selected": per_class * 2,
        "fit_count": len(fit_indices),
        "validation_count": len(val_indices),
        "train_appeared_min": min(train_dates),
        "train_appeared_max": max(train_dates),
        "validation_appeared_min": min(validation_dates),
        "validation_appeared_max": max(validation_dates),
        "split_strategy": split_strategy,
        "threshold": threshold,
        "parameters": {"n_estimators": 700, "num_leaves": 127, "learning_rate": 0.05},
        "validation": validation,
        "lightgbm_version": lgb.__version__,
        "selection_note": "official test is diagnostic only and has been viewed by earlier experiments",
    }
    if not skip_official_test:
        y_test, test_probabilities = test_scores(extractor, model)
        metadata["official_test_count"] = len(y_test)
        metadata["official_test"] = evaluate(y_test, test_probabilities, threshold)
    (output_dir / "candidate_more_lgbm.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/full_more"))
    parser.add_argument("--per-class", type=int, default=80_000)
    parser.add_argument("--split", choices=("temporal", "stratified_random"), default="temporal")
    parser.add_argument("--skip-official-test", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.reference, args.output_dir, args.per_class, args.split, args.skip_official_test
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
