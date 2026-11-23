import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

FEATURE_SET = "normalized_byte_histogram_v1"
EXTENDED_FEATURE_SET = "byte_histogram_entropy_strings_v1"
EXTENDED_FEATURE_DIM = 616
FULL_FEATURE_SET = "ember_full_v2"
FULL_FEATURE_DIM = 2381


def vector_from_bytes(data: bytes):
    import numpy as np

    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float32)
    total = counts.sum()
    if total == 0:
        raise ValueError("empty file")
    return counts / total


def vector_from_record(record: dict):
    import numpy as np

    counts = np.asarray(record["histogram"], dtype=np.float32)
    if counts.shape != (256,) or counts.sum() <= 0:
        raise ValueError("invalid EMBER histogram")
    return counts / counts.sum()


@lru_cache(maxsize=3)
def load_model(model_path: Path, metadata_path: Path):
    import lightgbm as lgb

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_set = metadata.get("feature_set")
    if feature_set not in {FEATURE_SET, EXTENDED_FEATURE_SET, FULL_FEATURE_SET}:
        raise ValueError("model feature set mismatch")
    model = lgb.Booster(model_file=str(model_path))
    expected_features = {
        FEATURE_SET: 256,
        EXTENDED_FEATURE_SET: EXTENDED_FEATURE_DIM,
        FULL_FEATURE_SET: FULL_FEATURE_DIM,
    }[feature_set]
    if model.num_feature() != expected_features:
        raise ValueError("model feature count mismatch")
    return model, metadata


@lru_cache(maxsize=2)
def load_full_partner(model_path: Path, expected_sha256: str):
    import lightgbm as lgb

    if hashlib.sha256(model_path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("ensemble model SHA-256 mismatch")
    model = lgb.Booster(model_file=str(model_path))
    if model.num_feature() != FULL_FEATURE_DIM:
        raise ValueError("ensemble model feature count mismatch")
    return model


def _full_vector(data: bytes, url: str):
    import numpy as np

    request = Request(url, data=data, headers={"Content-Type": "application/octet-stream"})
    with urlopen(request, timeout=4) as response:
        body = response.read(FULL_FEATURE_DIM * 4 + 1)
    if len(body) != FULL_FEATURE_DIM * 4:
        raise ValueError("invalid full EMBER feature response")
    vector = np.frombuffer(body, dtype="<f4")
    if not np.isfinite(vector).all():
        raise ValueError("non-finite full EMBER features")
    return vector


def analyze(path: Path, model_dir: Path) -> dict:
    data = path.read_bytes()
    fallback_reason = None
    full_model = model_dir / "ember_full_lgbm.txt"
    full_metadata = model_dir / "ember_full_lgbm.json"
    extractor_url = os.getenv("EMBER_EXTRACTOR_URL")
    if full_model.exists() and full_metadata.exists() and extractor_url:
        model, metadata = load_model(full_model, full_metadata)
        try:
            vector = _full_vector(data, extractor_url)
        except (OSError, TimeoutError, URLError, ValueError):
            fallback_reason = "full EMBER extractor unavailable"
        else:
            features = vector.reshape(1, -1)
            probability = float(model.predict(features)[0])
            ensemble = metadata.get("ensemble")
            if ensemble is not None:
                if (
                    ensemble.get("method") != "mean_two"
                    or ensemble.get("secondary_model") != "ember_full_ablation_lgbm.txt"
                ):
                    raise ValueError("unsupported full EMBER ensemble")
                partner = load_full_partner(
                    model_dir / ensemble["secondary_model"],
                    ensemble["secondary_model_sha256"],
                )
                probability = (probability + float(partner.predict(features)[0])) / 2
            return {
                "status": "complete",
                "probability": round(probability, 6),
                "threshold": metadata["threshold"],
                "feature_set": FULL_FEATURE_SET,
                **({"model_variant": "recent_equal_v1"} if ensemble is not None else {}),
            }
    extended_model = model_dir / "ember_extended_lgbm.txt"
    extended_metadata = model_dir / "ember_extended_lgbm.json"
    if extended_model.exists() and extended_metadata.exists():
        model_path, metadata_path = extended_model, extended_metadata
    else:
        model_path = model_dir / "ember_histogram_lgbm.txt"
        metadata_path = model_dir / "ember_histogram_lgbm.json"
    if not model_path.exists() or not metadata_path.exists():
        return {"status": "skipped", "reason": "model is not trained"}
    model, metadata = load_model(model_path, metadata_path)
    feature_set = metadata["feature_set"]
    if feature_set == EXTENDED_FEATURE_SET:
        from eass.layers import ml_features

        vector = ml_features.vector_from_bytes(data)
    else:
        vector = vector_from_bytes(data)
    probability = float(model.predict(vector.reshape(1, -1))[0])
    return {
        "status": "complete",
        "probability": round(probability, 6),
        "threshold": metadata["threshold"],
        "feature_set": feature_set,
        **({"fallback_reason": fallback_reason} if fallback_reason else {}),
    }
