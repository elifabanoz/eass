import argparse
import hashlib
import importlib.util
import json
import random
import sys
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from experiments.train import EMBER2018_SHA256, evaluate, verify_archive
from experiments.train_improved import _records


def load_reference(source: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    lief_stub = types.ModuleType("lief")
    lief_stub.__version__ = "0.10.1"
    previous = sys.modules.get("lief")
    sys.modules["lief"] = lief_stub
    try:
        spec = importlib.util.spec_from_file_location("ember_features_reference", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("lief", None)
        else:
            sys.modules["lief"] = previous

    original_hasher = module.FeatureHasher

    class CompatibleFeatureHasher(original_hasher):
        def transform(self, raw_samples):
            if self.input_type == "string" and len(raw_samples) == 1:
                sample = raw_samples[0]
                if isinstance(sample, str):
                    raw_samples = [[sample]]
            return super().transform(raw_samples)

    module.FeatureHasher = CompatibleFeatureHasher
    extractor = module.PEFeatureExtractor(feature_version=2, print_feature_warning=False)
    if extractor.dim != 2381:
        raise ValueError(f"unexpected EMBER feature dimension {extractor.dim}")
    return extractor


def sample_training(
    archive: Path, extractor, per_class: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = random.Random(42)
    matrix = np.empty((per_class * 2, extractor.dim), dtype=np.float32)
    counts = {0: 0, 1: 0}
    for record in _records(archive, "train"):
        label = record["label"]
        counts[label] += 1
        if counts[label] <= per_class:
            index = counts[label] - 1
        else:
            index = rng.randrange(counts[label])
            if index >= per_class:
                continue
        matrix[label * per_class + index] = extractor.process_raw_features(record)
    if min(counts.values()) < per_class:
        raise ValueError("not enough labeled EMBER training records")
    labels = np.repeat(np.array([0, 1], dtype=np.uint8), per_class)
    return matrix, labels, counts


def test_scores(archive: Path, extractor, model: lgb.LGBMClassifier):
    labels = []
    scores = []
    vectors = []
    batch_labels = []
    for record in _records(archive, "test"):
        vectors.append(extractor.process_raw_features(record))
        batch_labels.append(record["label"])
        if len(vectors) >= 512:
            scores.extend(model.predict_proba(np.asarray(vectors))[:, 1])
            labels.extend(batch_labels)
            vectors.clear()
            batch_labels.clear()
    if vectors:
        scores.extend(model.predict_proba(np.asarray(vectors))[:, 1])
        labels.extend(batch_labels)
    if len(labels) != 200_000:
        raise ValueError("official EMBER test split is incomplete")
    return np.asarray(labels, dtype=np.uint8), np.asarray(scores, dtype=np.float32)


def train(archive: Path, reference: Path, output_dir: Path, per_class: int, single: bool) -> dict:
    verify_archive(archive)
    extractor = load_reference(reference)
    matrix, labels, counts = sample_training(archive, extractor, per_class)
    x_fit, x_val, y_fit, y_val = train_test_split(
        matrix, labels, test_size=0.2, random_state=42, stratify=labels
    )
    del matrix, labels
    candidates = (
        {"n_estimators": 400, "num_leaves": 63, "learning_rate": 0.05},
        {"n_estimators": 700, "num_leaves": 127, "learning_rate": 0.05},
    )
    if single:
        candidates = candidates[-1:]
    best = None
    trials = []
    for parameters in candidates:
        model = lgb.LGBMClassifier(**parameters, random_state=42, n_jobs=4, verbosity=-1)
        model.fit(x_fit, y_fit)
        val_scores = model.predict_proba(x_val)[:, 1]
        threshold = float(np.quantile(val_scores[y_val == 0], 0.99))
        validation = evaluate(y_val, val_scores, threshold)
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
    fit_count = len(y_fit)
    validation_count = len(y_val)
    del x_fit, x_val, y_fit, y_val
    y_test, test_probabilities = test_scores(archive, extractor, best["model"])
    official_test = evaluate(y_test, test_probabilities, best["threshold"])
    metadata = {
        "dataset": "EMBER 2018 v2",
        "dataset_sha256": EMBER2018_SHA256,
        "feature_set": "ember_full_v2_experiment",
        "feature_dim": extractor.dim,
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "train_source_counts": counts,
        "train_selected": per_class * 2,
        "fit_count": fit_count,
        "validation_count": validation_count,
        "official_test_count": len(y_test),
        "threshold": best["threshold"],
        "parameters": best["parameters"],
        "validation": best["validation"],
        "official_test": official_test,
        "trials": trials,
        "lightgbm_version": lgb.__version__,
        "runtime_compatible": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    best["model"].booster_.save_model(str(output_dir / "candidate_full_ember_lgbm.txt"))
    (output_dir / "candidate_full_ember_lgbm.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--per-class", type=int, default=20_000)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            train(args.archive, args.reference, args.output_dir, args.per_class, args.single),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
