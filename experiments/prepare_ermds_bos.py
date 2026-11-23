import csv
import hashlib
import json
import random
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from experiments.evaluate_ermds import parse_records
from experiments.train_full_experiment import load_reference

OUTPUT_DIR = Path("data/ermds_bos_20261120")
SOURCES = {
    0: ("1xnm3WZFK2HejK-lQaOadMny3G9n5p5VG", 121518011),
    1: ("1HtuHSk6HlkpZUlfH0lX6eFsItzJME3T9", 410263974),
}
CAPS = {"fit": 5000, "validation": 1500, "test": 1500}


class Reservoir:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.seen = 0
        self.vectors = np.empty((capacity, 2381), dtype=np.float32)
        self.hashes = [""] * capacity

    def add(self, sha: str, vector: np.ndarray):
        self.seen += 1
        index = self.seen - 1 if self.seen <= self.capacity else self.rng.randrange(self.seen)
        if index < self.capacity:
            self.vectors[index] = vector
            self.hashes[index] = sha

    def arrays(self):
        count = min(self.seen, self.capacity)
        return self.vectors[:count], np.asarray(self.hashes[:count], dtype="S64")


def split_for_sha(sha: str):
    value = hashlib.sha256(("eass-ermds-bos-v1:" + sha).encode("ascii")).digest()
    bucket = int.from_bytes(value[:8], "big") % 10
    return "fit" if bucket < 6 else "validation" if bucket < 8 else "test"


def iter_records(response, digest, byte_count):
    decoder = json.JSONDecoder()
    buffer = b""
    while chunk := response.read(1024 * 1024):
        digest.update(chunk)
        byte_count[0] += len(chunk)
        buffer += chunk
        lines = buffer.split(b"\n")
        buffer = lines.pop()
        for line in lines:
            text = line.decode("utf-8")
            position = 0
            while position < len(text):
                while position < len(text) and text[position].isspace():
                    position += 1
                if position == len(text):
                    break
                record, position = decoder.raw_decode(text, position)
                yield record
    if buffer.strip():
        text = buffer.decode("utf-8")
        position = 0
        while position < len(text):
            while position < len(text) and text[position].isspace():
                position += 1
            if position == len(text):
                break
            record, position = decoder.raw_decode(text, position)
            yield record


def prepare():
    known_test = {
        record["sha256"] for record in parse_records(Path("data/ermds_official/test.jsonl"))
    }
    with Path("data/bodmas/bodmas_metadata.csv").open(newline="", encoding="utf-8-sig") as stream:
        bodmas_hashes = {row["sha"] for row in csv.DictReader(stream)}
    reference = load_reference(Path("data/wheels/ember_features_reference.py"))
    reservoirs = {
        (split, label): Reservoir(capacity, 20261120 + 100 * label + index)
        for index, (split, capacity) in enumerate(CAPS.items())
        for label in (0, 1)
    }
    seen_sha = set()
    source_info = {}
    excluded_known_test = 0
    excluded_bodmas = 0
    duplicate_count = 0
    invalid_count = 0
    for label, (file_id, expected_size) in SOURCES.items():
        url = (
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        )
        digest = hashlib.sha256()
        byte_count = [0]
        source_count = 0
        with urlopen(url, timeout=120) as response:
            if response.status != 200 or int(response.headers["Content-Length"]) != expected_size:
                raise ValueError("unexpected ERMDS download response")
            for record in iter_records(response, digest, byte_count):
                source_count += 1
                if record.get("label") != label or record.get("obfuscation") != "BOS":
                    raise ValueError("unexpected ERMDS label or obfuscation")
                sha = record.get("sha256", "")
                if len(sha) != 64 or any(character not in "0123456789abcdef" for character in sha):
                    raise ValueError("invalid ERMDS SHA-256")
                if sha in seen_sha:
                    duplicate_count += 1
                    continue
                seen_sha.add(sha)
                if sha in known_test:
                    excluded_known_test += 1
                    continue
                if sha in bodmas_hashes:
                    excluded_bodmas += 1
                    continue
                try:
                    vector = reference.process_raw_features(record)
                except (KeyError, ValueError, TypeError):
                    invalid_count += 1
                    continue
                if vector.shape != (2381,) or not np.isfinite(vector).all():
                    invalid_count += 1
                    continue
                split = split_for_sha(sha)
                reservoirs[split, label].add(sha, vector)
                if source_count % 5000 == 0:
                    print(f"BOS label {label}: {source_count} records", flush=True)
        if byte_count[0] != expected_size:
            raise ValueError("truncated ERMDS source file")
        source_info[str(label)] = {
            "drive_file_id": file_id,
            "expected_size_bytes": expected_size,
            "sha256": digest.hexdigest(),
            "bytes_read": byte_count[0],
            "record_count": source_count,
        }
        print(f"BOS label {label}: completed {source_count} records", flush=True)

    arrays = {}
    counts = {}
    sampled_shas = set()
    for split in CAPS:
        parts = []
        labels = []
        hash_parts = []
        counts[split] = {}
        for label in (0, 1):
            reservoir = reservoirs[split, label]
            vectors, hashes = reservoir.arrays()
            counts[split][str(label)] = {"eligible": reservoir.seen, "selected": len(hashes)}
            parts.append(vectors)
            labels.append(np.full(len(hashes), label, dtype=np.uint8))
            hash_parts.append(hashes)
            for value in hashes.astype(str):
                if value in sampled_shas:
                    raise ValueError("SHA-256 leaked between splits")
                sampled_shas.add(value)
        arrays[f"X_{split}"] = np.concatenate(parts)
        arrays[f"y_{split}"] = np.concatenate(labels)
        arrays[f"sha_{split}"] = np.concatenate(hash_parts)
    if any(min(counts[split][str(label)]["selected"] for label in (0, 1)) < 500 for split in CAPS):
        raise ValueError("insufficient ERMDS samples per class")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_DIR / "bos_splits.npz", **arrays)
    result = {
        "source": "https://github.com/lcjia94/ERMDS",
        "source_files": source_info,
        "split_rule": "SHA-256 hash buckets 60/20/20, then seeded class-wise reservoir caps",
        "caps_per_class": CAPS,
        "counts": counts,
        "excluded_known_github_test_hashes": excluded_known_test,
        "excluded_bodmas_hashes": excluded_bodmas,
        "duplicate_records": duplicate_count,
        "invalid_vectors": invalid_count,
        "selected_unique_hashes": len(sampled_shas),
        "feature_dim": reference.dim,
        "limitations": [
            "EMBER and SOREL SHA overlap could not be checked from local feature-only files.",
            "Feature records were extracted using LIEF 0.9.0; local vectorization uses EMBER v2 reference code.",
            "Split caps are a sampled subset of the full BOS corpus.",
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
