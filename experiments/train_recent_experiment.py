import argparse
import gc
import hashlib
import json
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import train_test_split

from experiments.evaluate_bodmas import metadata_dates, read_header
from experiments.train import EMBER2018_SHA256, evaluate
from experiments.train_full_experiment import load_reference
from experiments.train_full_more import sample_training


def bodmas_2019(npz_path: Path, metadata_path: Path):
    with zipfile.ZipFile(npz_path) as archive:
        if set(archive.namelist()) != {"X.npy", "y.npy"}:
            raise ValueError("unexpected BODMAS archive members")
        with archive.open("y.npy") as stream:
            labels = np.load(stream, allow_pickle=False)
        dates = metadata_dates(metadata_path, len(labels))
        if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
            raise ValueError("invalid BODMAS labels")
        fit_indices = np.flatnonzero(["2019-08" <= date[:7] <= "2019-10" for date in dates])
        val_indices = np.flatnonzero(["2019-11" <= date[:7] <= "2019-12" for date in dates])
        selected = np.concatenate((fit_indices, val_indices))
        matrix = np.empty((len(selected), 2381), dtype=np.float32)
        locations = np.empty(len(labels), dtype=np.int32)
        locations.fill(-1)
        locations[selected] = np.arange(len(selected))
        with archive.open("X.npy") as stream:
            shape, fortran, dtype = read_header(stream)
            dtype = np.dtype(dtype)
            if shape != (len(labels), 2381) or fortran or dtype != np.dtype("float32"):
                raise ValueError("incompatible BODMAS features")
            for start in range(0, len(labels), 512):
                count = min(512, len(labels) - start)
                raw = stream.read(count * 2381 * 4)
                if len(raw) != count * 2381 * 4:
                    raise ValueError("truncated BODMAS features")
                positions = locations[start : start + count]
                keep = positions >= 0
                if keep.any():
                    batch = np.frombuffer(raw, dtype=np.float32).reshape(count, 2381)
                    matrix[positions[keep]] = batch[keep]
            if stream.read(1):
                raise ValueError("unexpected BODMAS feature bytes")
    if len(fit_indices) < 1000 or len(val_indices) < 1000:
        raise ValueError("insufficient BODMAS 2019 examples")
    y_fit = labels[fit_indices]
    y_val = labels[val_indices]
    if min(np.bincount(y_fit, minlength=2)) < 1000:
        raise ValueError("insufficient BODMAS fit examples by class")
    if min(np.bincount(y_val, minlength=2)) < 1000:
        raise ValueError("insufficient BODMAS validation examples by class")
    return matrix[: len(fit_indices)], y_fit, matrix[len(fit_indices) :], y_val


def train(
    reference: Path,
    npz_path: Path,
    metadata_path: Path,
    output_dir: Path,
    per_class: int = 100_000,
    dropped_features: tuple[int, ...] = (),
):
    if per_class < 1000 or per_class > 300_000:
        raise ValueError("per_class outside supported range")
    extractor = load_reference(reference)
    if len(set(dropped_features)) != len(dropped_features) or any(
        index < 0 or index >= extractor.dim for index in dropped_features
    ):
        raise ValueError("invalid dropped feature indices")
    ember_matrix, _, counts = sample_training(extractor, per_class)
    if dropped_features:
        ember_matrix[:, dropped_features] = 0
    indices = np.arange(per_class * 2)
    fit_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=indices >= per_class
    )
    ember_sparse = csr_matrix(ember_matrix)
    del ember_matrix
    gc.collect()
    x_ember_fit = ember_sparse[fit_indices]
    x_ember_val = ember_sparse[val_indices]
    del ember_sparse
    gc.collect()
    x_recent_fit, y_recent_fit, x_recent_val, y_recent_val = bodmas_2019(npz_path, metadata_path)
    if dropped_features:
        x_recent_fit[:, dropped_features] = 0
        x_recent_val[:, dropped_features] = 0
    recent_fit_count = len(y_recent_fit)
    recent_val_count = len(y_recent_val)
    x_fit = vstack((x_ember_fit, csr_matrix(x_recent_fit)), format="csr")
    y_fit = np.concatenate((fit_indices >= per_class, y_recent_fit)).astype(np.uint8)
    del x_ember_fit, x_recent_fit, y_recent_fit
    gc.collect()
    model = lgb.LGBMClassifier(
        n_estimators=700,
        num_leaves=127,
        learning_rate=0.05,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(x_fit, y_fit)
    del x_fit, y_fit
    gc.collect()
    recent_scores = model.predict_proba(x_recent_val)[:, 1]
    threshold = float(np.quantile(recent_scores[y_recent_val == 0], 0.99))
    recent_validation = evaluate(y_recent_val, recent_scores, threshold)
    ember_val_labels = (val_indices >= per_class).astype(np.uint8)
    ember_val_scores = model.predict_proba(x_ember_val)[:, 1]
    ember_validation = evaluate(ember_val_labels, ember_val_scores, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "validation_scores.npz",
        ember_labels=ember_val_labels,
        ember_scores=ember_val_scores,
        bodmas_labels=y_recent_val,
        bodmas_scores=recent_scores,
    )
    model_path = output_dir / "candidate_recent_lgbm.txt"
    model.booster_.save_model(str(model_path))
    result = {
        "dataset": "EMBER 2018 v2 and BODMAS 2019",
        "dataset_sha256": EMBER2018_SHA256,
        "bodmas_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "feature_set": "ember_full_v2_experiment",
        "feature_dim": extractor.dim,
        "dropped_feature_indices": list(dropped_features),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "train_source_counts": counts,
        "ember_selected": per_class * 2,
        "ember_fit_count": len(fit_indices),
        "ember_validation_count": len(val_indices),
        "bodmas_fit_count": recent_fit_count,
        "bodmas_validation_count": recent_val_count,
        "bodmas_fit_period": "2019-08 through 2019-10",
        "bodmas_validation_period": "2019-11 through 2019-12",
        "threshold": threshold,
        "parameters": {"n_estimators": 700, "num_leaves": 127, "learning_rate": 0.05},
        "bodmas_2019_validation": recent_validation,
        "ember_2018_validation": ember_validation,
        "lightgbm_version": lgb.__version__,
        "selection_note": "2020 BODMAS and official EMBER test were viewed in earlier experiments; diagnostic only",
    }
    (output_dir / "candidate_recent_lgbm.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=100_000)
    parser.add_argument("--drop-feature", type=int, action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.reference,
                args.npz,
                args.metadata,
                args.output_dir,
                args.per_class,
                tuple(args.drop_feature),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
