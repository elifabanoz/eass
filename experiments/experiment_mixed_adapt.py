import gc
import hashlib
import json
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np

from experiments.calibrate_recent_model import threshold_for_fpr
from experiments.evaluate_bodmas import metadata_dates, read_header, score_summary
from experiments.experiment_sorel_adapt import load_split
from experiments.train_full_experiment import load_reference
from experiments.train_full_more import sample_training

OUTPUT_DIR = Path("data/mixed_adapt_20261120")
STAGES = (720, 740, 760, 780, 800)


def load_bodmas():
    path = Path("data/bodmas/bodmas.npz")
    with zipfile.ZipFile(path) as archive, archive.open("y.npy") as stream:
        labels = np.load(stream, allow_pickle=False)
    dates = metadata_dates(Path("data/bodmas/bodmas_metadata.csv"), len(labels))
    fit_mask = np.asarray(["2019-08" <= date[:7] <= "2019-10" for date in dates])
    val_mask = np.asarray(["2019-11" <= date[:7] <= "2019-12" for date in dates])
    rng = np.random.default_rng(20261130)
    fit = np.concatenate(
        [
            rng.choice(np.flatnonzero(fit_mask & (labels == label)), 4000, replace=False)
            for label in (0, 1)
        ]
    )
    val = np.concatenate(
        [
            rng.choice(np.flatnonzero(val_mask & (labels == label)), 2000, replace=False)
            for label in (0, 1)
        ]
    )
    selected = np.concatenate([fit, val])
    positions = np.full(len(labels), -1, dtype=np.int32)
    positions[selected] = np.arange(len(selected))
    matrix = np.empty((len(selected), 2381), dtype=np.float32)
    with zipfile.ZipFile(path) as archive, archive.open("X.npy") as stream:
        shape, fortran, dtype = read_header(stream)
        if shape != (len(labels), 2381) or fortran or np.dtype(dtype) != np.dtype("float32"):
            raise ValueError("incompatible BODMAS features")
        for start in range(0, len(labels), 512):
            count = min(512, len(labels) - start)
            raw = stream.read(count * 2381 * 4)
            if len(raw) != count * 2381 * 4:
                raise ValueError("truncated BODMAS features")
            keep = positions[start : start + count] >= 0
            if keep.any():
                batch = np.frombuffer(raw, dtype=np.float32).reshape(count, 2381)
                matrix[positions[start : start + count][keep]] = batch[keep]
        if stream.read(1):
            raise ValueError("unexpected BODMAS trailing data")
    return matrix[: len(fit)], labels[fit], matrix[len(fit) :], labels[val]


def evaluate():
    deployed = json.loads(Path("models/ember_full_lgbm.json").read_text(encoding="utf-8"))
    primary_path = Path("models/ember_full_lgbm.txt")
    secondary_path = Path("models/ember_full_ablation_lgbm.txt")
    primary = lgb.Booster(model_file=str(primary_path))
    secondary = lgb.Booster(model_file=str(secondary_path))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT_DIR / "sample_cache.npz"
    cache_info_path = OUTPUT_DIR / "sample_cache.json"
    if cache_path.is_file() and cache_info_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            x_fit = cached["x_fit"]
            y_fit = cached["y_fit"]
            x_bod_val = cached["x_bod_val"]
            y_bod_val = cached["y_bod_val"]
            x_sorel_val = cached["x_sorel_val"]
            y_sorel_val = cached["y_sorel_val"]
        cache_info = json.loads(cache_info_path.read_text(encoding="utf-8"))
        print("mixed samples loaded from local cache", flush=True)
    else:
        reference = load_reference(Path("data/wheels/ember_features_reference.py"))
        x_ember, _, counts = sample_training(reference, 4000)
        y_ember = np.repeat(np.array([0, 1], dtype=np.uint8), 4000)
        print("EMBER training sample ready", flush=True)
        x_bod_fit, y_bod_fit, x_bod_val, y_bod_val = load_bodmas()
        print("BODMAS training and validation samples ready", flush=True)
        x_sorel_fit, y_sorel_fit, sorel_train_info = load_split("train", 32, 20261129)
        x_sorel_val, y_sorel_val, sorel_val_info = load_split("validation", 16, 20261123)
        x_fit = np.concatenate([x_ember, x_bod_fit, x_sorel_fit])
        y_fit = np.concatenate([y_ember, y_bod_fit, y_sorel_fit])
        del x_ember, x_bod_fit, x_sorel_fit
        gc.collect()
        cache_info = {
            "ember_records_seen_by_class": counts,
            "bodmas_2019_fit_count": len(y_bod_fit),
            "sorel_train": sorel_train_info,
            "sorel_validation": sorel_val_info,
        }
        np.savez_compressed(
            cache_path,
            x_fit=x_fit,
            y_fit=y_fit,
            x_bod_val=x_bod_val,
            y_bod_val=y_bod_val,
            x_sorel_val=x_sorel_val,
            y_sorel_val=y_sorel_val,
        )
        cache_info_path.write_text(json.dumps(cache_info, indent=2) + "\n", encoding="utf-8")
        print("mixed samples cached", flush=True)
    train_set = lgb.Dataset(x_fit, label=y_fit, free_raw_data=False)
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.02,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "num_threads": 4,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=100,
        init_model=str(primary_path),
    )
    del train_set, x_fit
    gc.collect()
    if model.num_trees() != 800:
        raise ValueError("mixed model tree count mismatch")
    print("mixed candidate trained", flush=True)
    old_threshold = deployed["threshold"]
    baseline_sorel = (
        primary.predict(x_sorel_val, num_threads=4) + secondary.predict(x_sorel_val, num_threads=4)
    ) / 2
    baseline_bodmas = (
        primary.predict(x_bod_val, num_threads=4) + secondary.predict(x_bod_val, num_threads=4)
    ) / 2
    baseline_sorel_result = score_summary(y_sorel_val, baseline_sorel, old_threshold)
    baseline_bodmas_result = score_summary(y_bod_val, baseline_bodmas, old_threshold)
    secondary_sorel = secondary.predict(x_sorel_val, num_threads=4)
    secondary_bodmas = secondary.predict(x_bod_val, num_threads=4)
    candidates = []
    for count in STAGES:
        sorel_scores = (
            model.predict(x_sorel_val, num_iteration=count, num_threads=4) + secondary_sorel
        ) / 2
        bodmas_scores = (
            model.predict(x_bod_val, num_iteration=count, num_threads=4) + secondary_bodmas
        ) / 2
        threshold = max(
            threshold_for_fpr(sorel_scores[y_sorel_val == 0], 0.005),
            threshold_for_fpr(bodmas_scores[y_bod_val == 0], 0.005),
        )
        candidates.append(
            {
                "tree_count": count,
                "threshold": threshold,
                "sorel_validation": score_summary(y_sorel_val, sorel_scores, threshold),
                "bodmas_2019_validation": score_summary(y_bod_val, bodmas_scores, threshold),
            }
        )
    eligible = [
        item
        for item in candidates
        if item["sorel_validation"]["recall"] > baseline_sorel_result["recall"]
        and item["bodmas_2019_validation"]["recall"] >= baseline_bodmas_result["recall"]
    ]
    selected = (
        max(eligible, key=lambda item: item["sorel_validation"]["recall"]) if eligible else None
    )
    model_path = OUTPUT_DIR / "mixed_adapted_primary_lgbm.txt"
    model.save_model(str(model_path), num_iteration=-1)
    result = {
        "training": {
            "ember_records_seen_by_class": cache_info["ember_records_seen_by_class"],
            "ember_selected_per_class": 4000,
            "bodmas_2019_fit_count": cache_info["bodmas_2019_fit_count"],
            "sorel_train": cache_info["sorel_train"],
            "train_count": len(y_fit),
        },
        "validation": {
            "sorel": cache_info["sorel_validation"],
            "bodmas_2019_count": len(y_bod_val),
            "baseline_sorel": baseline_sorel_result,
            "baseline_bodmas_2019": baseline_bodmas_result,
            "candidates": candidates,
            "selected": selected,
        },
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "limitations": [
            "EMBER training rows may overlap the deployed model training rows.",
            "BODMAS 2019 validation was used in earlier experiments.",
            "Only the ML layer is trained and evaluated.",
        ],
    }
    (OUTPUT_DIR / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
