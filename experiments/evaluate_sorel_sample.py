"""Measure the deployed EMBER ensemble on a fixed, range-read SOREL test sample."""

import argparse
import hashlib
import io
import json
import random
import struct
import urllib.request
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
from numpy.lib import format as npy_format
from sklearn.metrics import roc_auc_score

TEST_URL = "https://sorel-20m.s3.amazonaws.com/09-DEC-2020/lightGBM-features/test-features.npz"
SEED = 20261120
BLOCK_COUNT = 16
BLOCK_ROWS = 256


class RangeFile:
    def __init__(self, url: str):
        self.url = url
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=30
        ) as response:
            self.length = int(response.headers["Content-Length"])
            self.etag = response.headers["ETag"]
        self.position = 0
        self.bytes_fetched = 0
        self.request_count = 0

    def get_range(self, start: int, length: int) -> bytes:
        if length < 0 or start < 0 or start + length > self.length:
            raise ValueError("invalid remote byte range")
        if length == 0:
            return b""
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{start + length - 1}", "If-Match": self.etag},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            expected = f"bytes {start}-{start + length - 1}/{self.length}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("server did not return the requested byte range")
            data = response.read(length + 1)
        if len(data) != length:
            raise ValueError("truncated remote byte range")
        self.bytes_fetched += length
        self.request_count += 1
        return data

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        data = self.get_range(self.position, size)
        self.position += size
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        base = (0, self.position, self.length)[whence]
        new_position = base + offset
        if not 0 <= new_position <= self.length:
            raise ValueError("remote seek out of bounds")
        self.position = new_position
        return self.position

    def tell(self) -> int:
        return self.position


def array_layout(remote: RangeFile, member: zipfile.ZipInfo):
    if member.compress_type != zipfile.ZIP_STORED:
        raise ValueError("range sampling requires uncompressed NPZ members")
    local_header = remote.get_range(member.header_offset, 30)
    signature, *fields = struct.unpack("<IHHHHHIIIHH", local_header)
    if signature != 0x04034B50:
        raise ValueError("invalid ZIP local header")
    name_length, extra_length = fields[-2:]
    member_start = member.header_offset + 30 + name_length + extra_length
    header_bytes = remote.get_range(member_start, 4096)
    stream = io.BytesIO(header_bytes)
    version = npy_format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = npy_format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = npy_format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported NPY version {version}")
    if fortran:
        raise ValueError("Fortran-order arrays cannot be range sampled by row")
    dtype = np.dtype(dtype)
    data_start = member_start + stream.tell()
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    if member.file_size != stream.tell() + expected_bytes:
        raise ValueError("NPY array size does not match ZIP member")
    return data_start, shape, dtype


def selected_blocks(row_count: int):
    rng = random.Random(SEED)
    blocks = []
    for i in range(BLOCK_COUNT):
        left = row_count * i // BLOCK_COUNT
        right = row_count * (i + 1) // BLOCK_COUNT
        if right - left < BLOCK_ROWS:
            raise ValueError("test set is too small for the fixed sample design")
        blocks.append(rng.randint(left, right - BLOCK_ROWS))
    return blocks


def wilson(successes: int, trials: int) -> list[float]:
    if trials == 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [float(center - half_width), float(center + half_width)]


def evaluate(model_dir: Path, output: Path):
    metadata = json.loads((model_dir / "ember_full_lgbm.json").read_text(encoding="utf-8"))
    if metadata["feature_set"] != "ember_full_v2" or metadata["feature_dim"] != 2381:
        raise ValueError("unexpected deployed feature layout")
    ensemble = metadata["ensemble"]
    if (
        ensemble["method"] != "mean_two"
        or ensemble["secondary_model"] != "ember_full_ablation_lgbm.txt"
    ):
        raise ValueError("unexpected deployed ensemble")
    first_path = model_dir / "ember_full_lgbm.txt"
    second_path = model_dir / ensemble["secondary_model"]
    hashes = {
        "primary": hashlib.sha256(first_path.read_bytes()).hexdigest(),
        "secondary": hashlib.sha256(second_path.read_bytes()).hexdigest(),
    }
    if hashes != {
        "primary": metadata["model_sha256"],
        "secondary": ensemble["secondary_model_sha256"],
    }:
        raise ValueError("deployed model hash mismatch")
    first_model = lgb.Booster(model_file=str(first_path))
    second_model = lgb.Booster(model_file=str(second_path))
    if first_model.num_feature() != 2381 or second_model.num_feature() != 2381:
        raise ValueError("unexpected deployed model dimension")

    remote = RangeFile(TEST_URL)
    with zipfile.ZipFile(remote) as archive:
        names = set(archive.namelist())
        if names != {"arr_0.npy", "arr_1.npy"}:
            raise ValueError(f"unexpected NPZ members: {sorted(names)}")
        x_start, x_shape, x_dtype = array_layout(remote, archive.getinfo("arr_0.npy"))
        y_start, y_shape, y_dtype = array_layout(remote, archive.getinfo("arr_1.npy"))
    if x_shape[-1] != 2381 or x_dtype != np.dtype("<f4"):
        raise ValueError(f"incompatible SOREL features: {x_shape}, {x_dtype}")
    row_count = int(np.prod(x_shape[:-1]))
    if int(np.prod(y_shape)) != row_count or y_dtype.kind not in "biuf":
        raise ValueError(f"incompatible SOREL labels: {y_shape}, {y_dtype}")

    starts = selected_blocks(row_count)
    labels = []
    scores = []
    primary_scores = []
    secondary_scores = []
    for index, start in enumerate(starts, 1):
        feature_bytes = remote.get_range(x_start + start * 2381 * 4, BLOCK_ROWS * 2381 * 4)
        label_bytes = remote.get_range(
            y_start + start * y_dtype.itemsize, BLOCK_ROWS * y_dtype.itemsize
        )
        x = np.frombuffer(feature_bytes, dtype=x_dtype).reshape(BLOCK_ROWS, 2381)
        y = np.frombuffer(label_bytes, dtype=y_dtype)
        if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError(f"invalid data in block {index}")
        left = first_model.predict(x, num_threads=4)
        right = second_model.predict(x, num_threads=4)
        labels.append(y.astype(np.uint8))
        scores.append((left + right) / 2)
        primary_scores.append(left)
        secondary_scores.append(right)
        print(f"scored block {index}/{BLOCK_COUNT}", flush=True)

    y = np.concatenate(labels)
    p = np.concatenate(scores)
    primary = np.concatenate(primary_scores)
    secondary = np.concatenate(secondary_scores)
    flagged = p >= metadata["threshold"]
    tn = int(np.sum((y == 0) & ~flagged))
    fp = int(np.sum((y == 0) & flagged))
    fn = int(np.sum((y == 1) & ~flagged))
    tp = int(np.sum((y == 1) & flagged))
    misses = (y == 1) & ~flagged
    diagnostic = {
        "misses_with_blend_probability_below_0_1": int(np.sum(misses & (p < 0.1))),
        "misses_with_blend_probability_0_1_to_0_5": int(np.sum(misses & (p >= 0.1) & (p < 0.5))),
        "misses_with_blend_probability_0_5_to_threshold": int(np.sum(misses & (p >= 0.5))),
        "misses_flagged_by_primary_alone": int(np.sum(misses & (primary >= metadata["threshold"]))),
        "misses_flagged_by_secondary_alone": int(
            np.sum(misses & (secondary >= metadata["threshold"]))
        ),
        "malware_probability_quantiles": [
            float(value) for value in np.quantile(p[y == 1], [0.01, 0.05, 0.1, 0.5])
        ],
        "benign_probability_quantiles": [
            float(value) for value in np.quantile(p[y == 0], [0.5, 0.9, 0.95, 0.99])
        ],
        "threshold_0_5_diagnostic": {
            "recall": float(np.mean(p[y == 1] >= 0.5)),
            "false_positive_rate": float(np.mean(p[y == 0] >= 0.5)),
        },
    }
    result = {
        "dataset": "SOREL-20M official test, fixed multi-block sample",
        "source_url": TEST_URL,
        "source_etag": remote.etag,
        "source_size_bytes": remote.length,
        "source_feature_shape": list(x_shape),
        "source_label_shape": list(y_shape),
        "sample_rule": f"{BLOCK_COUNT} equal row strata, one {BLOCK_ROWS}-row block per stratum, seed {SEED}",
        "block_starts": starts,
        "sample_count": len(y),
        "sample_label_counts": {"benign": tn + fp, "malware": fn + tp},
        "model_sha256": hashes,
        "threshold": metadata["threshold"],
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "recall": tp / (tp + fn),
        "recall_wilson_95": wilson(tp, tp + fn),
        "false_positive_rate": fp / (fp + tn),
        "fpr_wilson_95": wilson(fp, fp + tn),
        "precision_in_sample": tp / (tp + fp) if tp + fp else None,
        "roc_auc_in_sample": float(roc_auc_score(y, p)),
        "post_hoc_diagnostic_do_not_use_to_select_threshold": diagnostic,
        "bytes_fetched": remote.bytes_fetched,
        "range_requests": remote.request_count,
        "limitations": [
            "Only the ML layer was measured; hash, YARA, PE and ICAP verdicts were not.",
            "The sample is not the full SOREL test set.",
            "Cross-dataset SHA-256 overlap with EMBER/BODMAS was not verified.",
            "Wilson intervals assume independent sampled rows; contiguous blocks may make them too narrow.",
            "Confidence intervals do not cover dataset shift or labeling error.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/sorel_sample_20261120/evaluation.json")
    )
    parser.add_argument("--accept-terms", action="store_true")
    args = parser.parse_args()
    if not args.accept_terms:
        parser.error("SOREL-20M terms must be accepted before accessing the dataset")
    print(json.dumps(evaluate(args.model_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
