import csv
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from experiments.train_full_experiment import load_reference

SOURCE_SHA256 = "2ae3487c8084aa0872eafbe095cea89851fa7320761104adff8416113ef11fb9"
SOURCE_URL = "https://raw.githubusercontent.com/lcjia94/ERMDS/master/ERMDS-X/test.json"


def parse_records(path: Path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("ERMDS source file hash changed")
    content = data.decode("utf-8")
    decoder = json.JSONDecoder()
    offset = 0
    records = []
    while offset < len(content):
        while offset < len(content) and content[offset].isspace():
            offset += 1
        if offset == len(content):
            break
        record, offset = decoder.raw_decode(content, offset)
        records.append(record)
    if len(records) != 1000:
        raise ValueError("unexpected ERMDS record count")
    if any(record.get("label") != 1 or record.get("obfuscation") != "BOS" for record in records):
        raise ValueError("unexpected ERMDS sample labels")
    return records


def evaluate():
    records = parse_records(Path("data/ermds_official/test.jsonl"))
    hashes = [record["sha256"] for record in records]
    if any(len(value) != 64 for value in hashes):
        raise ValueError("invalid ERMDS SHA-256")
    reference = load_reference(Path("data/wheels/ember_features_reference.py"))
    features = np.stack([reference.process_raw_features(record) for record in records])
    if features.shape != (1000, 2381) or not np.isfinite(features).all():
        raise ValueError("invalid ERMDS vectorization")

    deployed = json.loads(Path("models/ember_full_lgbm.json").read_text(encoding="utf-8"))
    candidate = json.loads(
        Path("data/mixed_adapt_20261120/validation.json").read_text(encoding="utf-8")
    )
    paths = {
        "primary": Path("models/ember_full_lgbm.txt"),
        "secondary": Path("models/ember_full_ablation_lgbm.txt"),
        "mixed": Path("data/mixed_adapt_20261120/mixed_selected_primary_lgbm.txt"),
    }
    model_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
    }
    if model_hashes["primary"] != deployed["model_sha256"]:
        raise ValueError("deployed primary model changed")
    if model_hashes["secondary"] != deployed["ensemble"]["secondary_model_sha256"]:
        raise ValueError("deployed secondary model changed")
    if model_hashes["mixed"] != candidate["selected_model_sha256"]:
        raise ValueError("mixed candidate changed")
    models = {name: lgb.Booster(model_file=str(path)) for name, path in paths.items()}
    predictions = {name: model.predict(features, num_threads=4) for name, model in models.items()}
    baseline = (predictions["primary"] + predictions["secondary"]) / 2
    mixed = (predictions["mixed"] + predictions["secondary"]) / 2
    baseline_flag = baseline >= deployed["threshold"]
    mixed_flag = mixed >= candidate["validation"]["selected"]["threshold"]

    unique_indices = []
    seen = set()
    for index, value in enumerate(hashes):
        if value not in seen:
            unique_indices.append(index)
            seen.add(value)
    unique_indices = np.asarray(unique_indices, dtype=np.int32)
    with Path("data/bodmas/bodmas_metadata.csv").open(newline="", encoding="utf-8-sig") as stream:
        bodmas_hashes = {row["sha"] for row in csv.DictReader(stream)}
    result = {
        "source": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "record_count": len(records),
        "unique_sha256_count": len(unique_indices),
        "label": "malicious only",
        "obfuscation": "BOS",
        "bodmas_sha_overlap_count": len(seen & bodmas_hashes),
        "model_sha256": model_hashes,
        "baseline": {
            "threshold": deployed["threshold"],
            "row_recall": float(np.mean(baseline_flag)),
            "unique_sha_recall": float(np.mean(baseline_flag[unique_indices])),
            "unique_sha_detected": int(np.sum(baseline_flag[unique_indices])),
        },
        "mixed_candidate": {
            "threshold": candidate["validation"]["selected"]["threshold"],
            "row_recall": float(np.mean(mixed_flag)),
            "unique_sha_recall": float(np.mean(mixed_flag[unique_indices])),
            "unique_sha_detected": int(np.sum(mixed_flag[unique_indices])),
        },
        "unique_sha_rescued": int(
            np.sum(~baseline_flag[unique_indices] & mixed_flag[unique_indices])
        ),
        "unique_sha_lost": int(np.sum(baseline_flag[unique_indices] & ~mixed_flag[unique_indices])),
        "limitations": [
            "All records are malicious; false-positive rate cannot be measured.",
            "The 1000 rows contain repeated SHA-256 values; unique-hash recall is primary.",
            "Exact overlap with EMBER and SOREL is unknown.",
            "Source raw features were extracted using a different LIEF version; vectorization uses the project's EMBER v2 reference.",
            "Only the ML layer was measured.",
        ],
    }
    output = Path("data/ermds_official/evaluation.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
