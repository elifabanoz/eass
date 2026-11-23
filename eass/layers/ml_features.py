import re

import numpy as np

FEATURE_SET = "byte_histogram_entropy_strings_v1"
FEATURE_DIM = 256 + 256 + 104
_STRINGS = re.compile(rb"[\x20-\x7f]{5,}")
_PATHS = re.compile(rb"c:\\", re.IGNORECASE)
_URLS = re.compile(rb"https?://", re.IGNORECASE)
_REGISTRY = re.compile(rb"HKEY_")
_MZ = re.compile(rb"MZ")


def _normalized(values, length: int) -> np.ndarray:
    counts = np.asarray(values, dtype=np.float32)
    if counts.shape != (length,) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("invalid EMBER feature counts")
    total = counts.sum()
    if total <= 0:
        raise ValueError("empty EMBER feature counts")
    return counts / total


def _byte_entropy(data: bytes) -> np.ndarray:
    window = 2048
    step = 1024
    values = np.frombuffer(data, dtype=np.uint8)
    output = np.zeros((16, 16), dtype=np.float32)
    positions = (0,) if len(values) < window else range(0, len(values) - window + 1, step)
    for position in positions:
        block = values[position : position + window]
        counts = np.bincount(block >> 4, minlength=16)
        probabilities = counts[counts > 0].astype(np.float32) / window
        entropy = float(-np.sum(probabilities * np.log2(probabilities)) * 2)
        entropy_bin = min(15, int(entropy * 2))
        output[entropy_bin] += counts
    return _normalized(output.reshape(-1), 256)


def _string_vector(raw: dict) -> np.ndarray:
    histogram = np.asarray(raw["printabledist"], dtype=np.float32)
    if histogram.shape != (96,) or not np.isfinite(histogram).all() or (histogram < 0).any():
        raise ValueError("invalid printable character histogram")
    printables = float(raw["printables"])
    if printables < 0 or (printables == 0 and histogram.sum() != 0):
        raise ValueError("invalid printable character count")
    vector = np.concatenate(
        [
            np.asarray([raw["numstrings"], raw["avlength"], printables], dtype=np.float32),
            histogram / (printables if printables > 0 else 1),
            np.asarray(
                [
                    raw["entropy"],
                    raw["paths"],
                    raw["urls"],
                    raw["registry"],
                    raw["MZ"],
                ],
                dtype=np.float32,
            ),
        ]
    )
    if vector.shape != (104,) or not np.isfinite(vector).all():
        raise ValueError("invalid EMBER string features")
    return vector


def _strings_from_bytes(data: bytes) -> dict:
    found = _STRINGS.findall(data)
    if found:
        printable = np.frombuffer(b"".join(found), dtype=np.uint8) - 32
        counts = np.bincount(printable, minlength=96)
        length = int(counts.sum())
        probabilities = counts[counts > 0].astype(np.float32) / length
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        average_length = sum(map(len, found)) / len(found)
    else:
        counts = np.zeros(96, dtype=np.float32)
        length = 0
        entropy = 0.0
        average_length = 0.0
    return {
        "numstrings": len(found),
        "avlength": average_length,
        "printabledist": counts,
        "printables": length,
        "entropy": entropy,
        "paths": len(_PATHS.findall(data)),
        "urls": len(_URLS.findall(data)),
        "registry": len(_REGISTRY.findall(data)),
        "MZ": len(_MZ.findall(data)),
    }


def vector_from_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("empty file")
    histogram = _normalized(np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256), 256)
    vector = np.concatenate(
        [histogram, _byte_entropy(data), _string_vector(_strings_from_bytes(data))]
    )
    if vector.shape != (FEATURE_DIM,):
        raise ValueError("wrong feature dimension")
    return vector


def vector_from_record(record: dict) -> np.ndarray:
    vector = np.concatenate(
        [
            _normalized(record["histogram"], 256),
            _normalized(record["byteentropy"], 256),
            _string_vector(record["strings"]),
        ]
    )
    if vector.shape != (FEATURE_DIM,):
        raise ValueError("wrong feature dimension")
    return vector
