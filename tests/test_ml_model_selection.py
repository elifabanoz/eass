import tempfile
import unittest
from os import environ
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import URLError

import numpy as np

from eass.layers import ml
from eass.layers.ml_features import FEATURE_DIM


class ModelSelectionTests(unittest.TestCase):
    def test_full_model_is_used_with_extractor(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            sample = model_dir / "sample.exe"
            sample.write_bytes(b"MZ" + bytes(range(256)))
            (model_dir / "ember_full_lgbm.txt").touch()
            (model_dir / "ember_full_lgbm.json").touch()
            model = Mock()
            model.predict.return_value = np.array([0.95])
            with (
                patch.dict(environ, {"EMBER_EXTRACTOR_URL": "http://extractor/features"}),
                patch.object(ml, "_full_vector", return_value=np.zeros(ml.FULL_FEATURE_DIM)),
                patch.object(
                    ml,
                    "load_model",
                    return_value=(model, {"feature_set": ml.FULL_FEATURE_SET, "threshold": 0.9}),
                ),
            ):
                result = ml.analyze(sample, model_dir)
            self.assertEqual(result["feature_set"], ml.FULL_FEATURE_SET)
            self.assertEqual(model.predict.call_args.args[0].shape, (1, ml.FULL_FEATURE_DIM))

    def test_extended_model_is_used_when_extractor_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            sample = model_dir / "sample.exe"
            sample.write_bytes(b"MZ" + bytes(range(256)))
            for name in ("ember_full_lgbm", "ember_extended_lgbm"):
                (model_dir / f"{name}.txt").touch()
                (model_dir / f"{name}.json").touch()
            model = Mock()
            model.predict.return_value = np.array([0.2])

            def fake_load(model_path, _metadata_path):
                feature_set = (
                    ml.FULL_FEATURE_SET
                    if model_path.name == "ember_full_lgbm.txt"
                    else ml.EXTENDED_FEATURE_SET
                )
                return model, {"feature_set": feature_set, "threshold": 0.9}

            with (
                patch.dict(environ, {"EMBER_EXTRACTOR_URL": "http://extractor/features"}),
                patch.object(ml, "_full_vector", side_effect=URLError("unavailable")),
                patch.object(ml, "load_model", side_effect=fake_load),
            ):
                result = ml.analyze(sample, model_dir)
            self.assertEqual(result["feature_set"], ml.EXTENDED_FEATURE_SET)
            self.assertEqual(result["fallback_reason"], "full EMBER extractor unavailable")

    def test_full_ensemble_averages_two_model_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            sample = model_dir / "sample.exe"
            sample.write_bytes(b"MZ" + bytes(range(256)))
            (model_dir / "ember_full_lgbm.txt").touch()
            (model_dir / "ember_full_lgbm.json").touch()
            primary = Mock()
            primary.predict.return_value = np.array([0.9])
            partner = Mock()
            partner.predict.return_value = np.array([0.3])
            metadata = {
                "feature_set": ml.FULL_FEATURE_SET,
                "threshold": 0.7,
                "ensemble": {
                    "method": "mean_two",
                    "secondary_model": "ember_full_ablation_lgbm.txt",
                    "secondary_model_sha256": "a" * 64,
                },
            }
            with (
                patch.dict(environ, {"EMBER_EXTRACTOR_URL": "http://extractor/features"}),
                patch.object(ml, "_full_vector", return_value=np.zeros(ml.FULL_FEATURE_DIM)),
                patch.object(ml, "load_model", return_value=(primary, metadata)),
                patch.object(ml, "load_full_partner", return_value=partner) as load_partner,
            ):
                result = ml.analyze(sample, model_dir)
            self.assertEqual(result["probability"], 0.6)
            self.assertEqual(result["model_variant"], "recent_equal_v1")
            self.assertEqual(load_partner.call_args.args[0].name, "ember_full_ablation_lgbm.txt")

    def test_full_ensemble_rejects_invalid_partner_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partner.txt"
            path.write_bytes(b"unexpected model")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                ml.load_full_partner(path, "a" * 64)

    def test_extended_model_is_used_when_both_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            sample = model_dir / "sample.exe"
            sample.write_bytes(b"MZ" + bytes(range(256)) * 12)
            (model_dir / "ember_extended_lgbm.txt").touch()
            (model_dir / "ember_extended_lgbm.json").touch()
            model = Mock()
            model.predict.return_value = np.array([0.8])
            with patch.object(
                ml,
                "load_model",
                return_value=(model, {"feature_set": ml.EXTENDED_FEATURE_SET, "threshold": 0.7}),
            ) as load:
                result = ml.analyze(sample, model_dir)
            self.assertEqual(load.call_args.args[0].name, "ember_extended_lgbm.txt")
            self.assertEqual(model.predict.call_args.args[0].shape, (1, FEATURE_DIM))
            self.assertEqual(result["feature_set"], ml.EXTENDED_FEATURE_SET)

    def test_baseline_is_used_when_extended_artifacts_are_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            sample = model_dir / "sample.exe"
            sample.write_bytes(b"MZ" + bytes(range(256)))
            (model_dir / "ember_histogram_lgbm.txt").touch()
            (model_dir / "ember_histogram_lgbm.json").touch()
            (model_dir / "ember_extended_lgbm.txt").touch()
            model = Mock()
            model.predict.return_value = np.array([0.2])
            with patch.object(
                ml,
                "load_model",
                return_value=(model, {"feature_set": ml.FEATURE_SET, "threshold": 0.7}),
            ) as load:
                result = ml.analyze(sample, model_dir)
            self.assertEqual(load.call_args.args[0].name, "ember_histogram_lgbm.txt")
            self.assertEqual(model.predict.call_args.args[0].shape, (1, 256))
            self.assertEqual(result["feature_set"], ml.FEATURE_SET)


if __name__ == "__main__":
    unittest.main()
