import argparse
import csv
import json
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
from numpy.lib import format as npy_format
from sklearn.metrics import confusion_matrix, roc_auc_score


def read_header(stream):
    version = npy_format.read_magic(stream)
    if version == (1, 0):
        return npy_format.read_array_header_1_0(stream)
    if version == (2, 0):
        return npy_format.read_array_header_2_0(stream)
    raise ValueError(f"unsupported NPY version: {version}")


def score_models(npz_path: Path, models: dict[str, Path], batch_size: int = 512):
    predictors = {name: lgb.Booster(model_file=str(path)) for name, path in models.items()}
    with zipfile.ZipFile(npz_path) as archive:
        names = set(archive.namelist())
        if names != {"X.npy", "y.npy"}:
            raise ValueError(f"unexpected BODMAS archive members: {sorted(names)}")
        with archive.open("y.npy") as label_stream:
            labels = np.load(label_stream, allow_pickle=False)
        if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
            raise ValueError("invalid BODMAS labels")
        scores = {name: np.empty(len(labels), dtype=np.float32) for name in models}
        with archive.open("X.npy") as feature_stream:
            shape, fortran, dtype = read_header(feature_stream)
            dtype = np.dtype(dtype)
            if shape != (len(labels), 2381) or fortran or dtype.kind != "f":
                raise ValueError(f"incompatible BODMAS features: {shape}, {fortran}, {dtype}")
            row_bytes = shape[1] * dtype.itemsize
            for start in range(0, len(labels), batch_size):
                count = min(batch_size, len(labels) - start)
                raw = feature_stream.read(count * row_bytes)
                if len(raw) != count * row_bytes:
                    raise ValueError("truncated BODMAS feature array")
                batch = np.frombuffer(raw, dtype=dtype).reshape(count, shape[1])
                for name, predictor in predictors.items():
                    scores[name][start : start + count] = predictor.predict(batch, num_threads=4)
            if feature_stream.read(1):
                raise ValueError("unexpected trailing BODMAS feature bytes")
    return labels, scores


def score_summary(labels, scores, threshold):
    flagged = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, flagged, labels=[0, 1]).ravel()
    return {
        "count": len(labels),
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "recall": float(tp / (tp + fn)),
        "false_positive_rate": float(fp / (fp + tn)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def metadata_dates(path: Path, count: int):
    with path.open(newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames != ["sha", "timestamp", "family"]:
            raise ValueError("unexpected BODMAS metadata columns")
        dates = []
        hashes = set()
        for row in reader:
            dates.append(row["timestamp"])
            hashes.add(row["sha"])
    if len(dates) != count:
        raise ValueError("BODMAS metadata row count differs from features")
    if len(hashes) != count:
        raise ValueError("duplicate BODMAS hashes")
    return dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model", action="append", nargs=3, metavar=("NAME", "PATH", "META"), required=True
    )
    args = parser.parse_args()
    model_paths = {name: Path(path) for name, path, _ in args.model}
    labels, scores = score_models(args.npz, model_paths)
    dates = metadata_dates(args.metadata, len(labels))
    year_2019 = np.asarray([date[:4] == "2019" for date in dates])
    year_2020 = np.asarray([date[:4] == "2020" for date in dates])
    newer = year_2019 | year_2020
    result = {}
    for name, _, metadata_path in args.model:
        threshold = json.loads(Path(metadata_path).read_text(encoding="utf-8"))["threshold"]
        result[name] = {
            "all": score_summary(labels, scores[name], threshold),
            "2019_2020": score_summary(labels[newer], scores[name][newer], threshold),
            "2019": score_summary(labels[year_2019], scores[name][year_2019], threshold),
            "2020": score_summary(labels[year_2020], scores[name][year_2020], threshold),
        }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
