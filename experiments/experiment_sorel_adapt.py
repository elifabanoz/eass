import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np

from experiments.calibrate_recent_model import threshold_for_fpr
from experiments.evaluate_bodmas import score_summary
from experiments.evaluate_sorel_sample import RangeFile, array_layout

SOURCE_BASE = "https://sorel-20m.s3.amazonaws.com/09-DEC-2020/lightGBM-features/"
MODEL_NAMES = ("ember_full_lgbm.txt", "ember_full_ablation_lgbm.txt")
TARGET_FPR = 0.005
BLOCK_ROWS = 256
TRAIN_BLOCKS = 64
VALIDATION_BLOCKS = 16
TEST_BLOCKS = 32
USED_TEST_STARTS = (
    116243,
    415523,
    660468,
    887911,
    1125741,
    1499383,
    1628957,
    2021819,
    2133976,
    2469155,
    2649723,
    3126246,
    3397871,
    3519721,
    3705768,
    4158350,
)


def block_starts(row_count: int, block_count: int, seed: int, exclude: tuple[int, ...] = ()):
    rng = random.Random(seed)
    starts = []
    for index in range(block_count):
        left = row_count * index // block_count
        right = row_count * (index + 1) // block_count
        if right - left < BLOCK_ROWS:
            raise ValueError("split too small")
        for _ in range(100):
            candidate = rng.randint(left, right - BLOCK_ROWS)
            if all(
                candidate + BLOCK_ROWS <= old or old + BLOCK_ROWS <= candidate for old in exclude
            ):
                starts.append(candidate)
                break
        else:
            raise ValueError("could not select a disjoint block")
    return starts


def load_split(name: str, block_count: int, seed: int, exclude: tuple[int, ...] = ()):
    remote = RangeFile(SOURCE_BASE + name + "-features.npz")
    with zipfile.ZipFile(remote) as archive:
        if set(archive.namelist()) != {"arr_0.npy", "arr_1.npy"}:
            raise ValueError("unexpected SOREL archive")
        x_start, x_shape, x_dtype = array_layout(remote, archive.getinfo("arr_0.npy"))
        y_start, y_shape, y_dtype = array_layout(remote, archive.getinfo("arr_1.npy"))
    if x_dtype != np.dtype("<f4") or x_shape[-1] != 2381 or len(x_shape) != 2:
        raise ValueError("incompatible SOREL feature matrix")
    if y_shape != (x_shape[0],) or y_dtype != np.dtype("<i8"):
        raise ValueError("incompatible SOREL labels")
    starts = block_starts(x_shape[0], block_count, seed, exclude)
    x = np.empty((block_count * BLOCK_ROWS, 2381), dtype=np.float32)
    y = np.empty(block_count * BLOCK_ROWS, dtype=np.uint8)
    for index, start in enumerate(starts):
        pos = slice(index * BLOCK_ROWS, (index + 1) * BLOCK_ROWS)
        x[pos] = np.frombuffer(
            remote.get_range(x_start + start * 2381 * 4, BLOCK_ROWS * 2381 * 4),
            dtype=x_dtype,
        ).reshape(BLOCK_ROWS, 2381)
        values = np.frombuffer(
            remote.get_range(y_start + start * y_dtype.itemsize, BLOCK_ROWS * y_dtype.itemsize),
            dtype=y_dtype,
        )
        if not np.isin(values, [0, 1]).all():
            raise ValueError("invalid SOREL labels")
        y[pos] = values
        if (index + 1) % 8 == 0:
            print(f"{name}: {index + 1}/{block_count} blocks", flush=True)
    if not np.isfinite(x).all() or min(np.bincount(y, minlength=2)) < 200:
        raise ValueError("invalid or inadequate SOREL sample")
    info = {
        "url": remote.url,
        "etag": remote.etag,
        "source_rows": x_shape[0],
        "starts": starts,
        "row_count": len(y),
        "benign_count": int(np.sum(y == 0)),
        "malware_count": int(np.sum(y == 1)),
        "sample_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
        "range_requests": remote.request_count,
        "bytes_fetched": remote.bytes_fetched,
    }
    return x, y, info


def calibrated_summary(labels: np.ndarray, scores: np.ndarray):
    threshold = threshold_for_fpr(scores[labels == 0], TARGET_FPR)
    return score_summary(labels, scores, threshold)


def evaluate(model_dir: Path, output_dir: Path):
    deployed = json.loads((model_dir / "ember_full_lgbm.json").read_text(encoding="utf-8"))
    if deployed["feature_set"] != "ember_full_v2" or deployed["feature_dim"] != 2381:
        raise ValueError("unexpected deployed feature layout")
    paths = [model_dir / name for name in MODEL_NAMES]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    if hashes != [deployed["model_sha256"], deployed["ensemble"]["secondary_model_sha256"]]:
        raise ValueError("deployed model hashes differ")
    primary, secondary = [lgb.Booster(model_file=str(path)) for path in paths]

    x_train, y_train, train_info = load_split("train", TRAIN_BLOCKS, 20261122)
    x_val, y_val, val_info = load_split("validation", VALIDATION_BLOCKS, 20261123)
    baseline_val = (
        primary.predict(x_val, num_threads=4) + secondary.predict(x_val, num_threads=4)
    ) / 2
    baseline_calibrated = calibrated_summary(y_val, baseline_val)
    baseline_fixed = score_summary(y_val, baseline_val, deployed["threshold"])

    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=True)
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set, free_raw_data=True)
    adapted = lgb.train(
        {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "num_threads": 4,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=120,
        valid_sets=[val_set],
        init_model=str(paths[0]),
        callbacks=[lgb.early_stopping(15, verbose=False)],
    )
    candidate_val = (
        adapted.predict(x_val, num_threads=4) + secondary.predict(x_val, num_threads=4)
    ) / 2
    candidate_calibrated = calibrated_summary(y_val, candidate_val)
    candidate_fixed = score_summary(y_val, candidate_val, deployed["threshold"])

    selected = (
        "candidate"
        if (
            candidate_calibrated["recall"] > baseline_calibrated["recall"]
            and candidate_calibrated["false_positive_rate"] <= TARGET_FPR
        )
        else "baseline"
    )
    print(f"validation selected: {selected}", flush=True)

    x_test, y_test, test_info = load_split("test", TEST_BLOCKS, 20261124, USED_TEST_STARTS)
    baseline_test = (
        primary.predict(x_test, num_threads=4) + secondary.predict(x_test, num_threads=4)
    ) / 2
    candidate_test = (
        adapted.predict(x_test, num_threads=4) + secondary.predict(x_test, num_threads=4)
    ) / 2

    result = {
        "design": {
            "target_validation_fpr": TARGET_FPR,
            "block_rows": BLOCK_ROWS,
            "train": train_info,
            "validation": val_info,
            "new_test": test_info,
            "old_test_block_starts_excluded": list(USED_TEST_STARTS),
            "deployed_model_sha256": dict(zip(MODEL_NAMES, hashes, strict=True)),
        },
        "baseline": {
            "validation_fixed": baseline_fixed,
            "validation_calibrated": baseline_calibrated,
            "new_test_fixed": score_summary(y_test, baseline_test, deployed["threshold"]),
            "new_test_calibrated": score_summary(
                y_test, baseline_test, baseline_calibrated["threshold"]
            ),
        },
        "candidate": {
            "best_iteration": adapted.best_iteration,
            "validation_fixed": candidate_fixed,
            "validation_calibrated": candidate_calibrated,
            "new_test_fixed": score_summary(y_test, candidate_test, deployed["threshold"]),
            "new_test_calibrated": score_summary(
                y_test, candidate_test, candidate_calibrated["threshold"]
            ),
        },
        "validation_selected": selected,
        "limitations": [
            "The earlier SOREL test sample was viewed; new test rows exclude its 16 blocks.",
            "Exact SHA overlap with EMBER and BODMAS is unknown.",
            "Only the ML layer was measured.",
            "SOREL sample training may trade performance on other data sources.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "adapted_primary_lgbm.txt"
    adapted.save_model(str(model_path))
    result["candidate"]["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/sorel_adapt_20261120"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
