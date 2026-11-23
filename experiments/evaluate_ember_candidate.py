import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from experiments.train import evaluate
from experiments.train_full_experiment import load_reference
from experiments.train_full_more import remote_records


def score_test(reference: Path, model_path: Path):
    extractor = load_reference(reference)
    model = lgb.Booster(model_file=str(model_path))
    labels = []
    scores = []
    batch = []
    batch_labels = []
    for record in remote_records("test"):
        batch.append(extractor.process_raw_features(record))
        batch_labels.append(record["label"])
        if len(batch) == 512:
            scores.extend(model.predict(np.asarray(batch), num_threads=4))
            labels.extend(batch_labels)
            batch.clear()
            batch_labels.clear()
    if batch:
        scores.extend(model.predict(np.asarray(batch), num_threads=4))
        labels.extend(batch_labels)
    if len(labels) != 200_000:
        raise ValueError("official EMBER test split is incomplete")
    return np.asarray(labels, dtype=np.uint8), np.asarray(scores, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    labels, scores = score_test(args.reference, args.model)
    result = evaluate(labels, scores, args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "ember_test_scores.npz", labels=labels, scores=scores)
    (args.output_dir / "ember_test_comparison.json").write_text(
        json.dumps({"threshold": args.threshold, "official_test": result}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
