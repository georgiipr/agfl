"""Dataset/split invariants protect scientific comparisons, not just shapes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from agfl.datasets import (SignalDataset, get_dataset_spec, get_split, list_datasets,
                           load_dataset, prepare_data, validate_split)
from agfl.datasets.base import bandpass_finite_spans, normalize_samples


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bundle = load_dataset("synthetic_eeg", {"subjects": 6, "samples_per_subject": 12})

    def test_registry_discovery_and_unknown_keys(self):
        self.assertTrue({"eeg", "ecg", "synthetic_eeg", "synthetic_ecg", "npz_eeg"} <= set(list_datasets()))
        self.assertEqual(get_dataset_spec("ecg").modality, "ecg")
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            load_dataset("typo")
        with self.assertRaisesRegex(ValueError, "Unknown synthetic_eeg data settings"):
            load_dataset("synthetic_eeg", {"typo": 1})

    def test_split_shared_by_models_and_persisted(self):
        config = {"dataset": "synthetic_eeg", "data": {"subjects": 6, "samples_per_subject": 12},
                  "split": {"protocol": "group"}, "split_dir": str(self.root)}
        _, first = prepare_data(dict(config, model="eegnet", attention="agfl"), 8)
        _, second = prepare_data(dict(config, model="eegnet", attention="performer"), 8)
        self.assertEqual(first, second)
        self.assertEqual(len(list(self.root.glob("*.json"))), 1)
        self.assertFalse(set(first["groups"]["train"]) & set(first["groups"]["test"]))
        self.assertEqual(set(first["train"] + first["validation"] + first["test"]), set(range(len(self.bundle))))
        third = get_split(self.bundle, {"protocol": "group"}, 9, self.root)
        self.assertNotEqual(first["split_id"], third["split_id"])

    def test_stratified_split_preserves_class_coverage(self):
        split = get_split(self.bundle, {"protocol": "stratified"}, 5, self.root)
        for name in ("train", "validation", "test"):
            self.assertEqual(set(self.bundle.y[split[name]]), set(range(4)))
        self.assertFalse(split["subject_independent"])

    def test_ecg_trial_split_requires_explicit_diagnostic_intent(self):
        ecg = load_dataset("synthetic_ecg")
        with self.assertRaisesRegex(ValueError, "patient identity"):
            get_split(ecg, {"protocol": "stratified"}, 0, self.root)

    def test_session_validation_rejects_swapping_t_and_e_trials(self):
        self.bundle.metadata['sample_sessions'] = ['T'] * 36 + ['E'] * 36
        split = get_split(self.bundle, {'protocol': 'session'}, 0, self.root)
        first, second = split['train'][0], split['test'][0]
        split['train'][0], split['test'][0] = second, first
        with self.assertRaisesRegex(ValueError, 'test only on E'):
            validate_split(self.bundle, split)

    def test_persisted_indices_cannot_silently_change(self):
        split = get_split(self.bundle, {}, 3, self.root)
        path = self.root / f"{split['split_id']}.json"
        split["train"][0] = split["test"][0]
        path.write_text(json.dumps(split))
        with self.assertRaisesRegex(ValueError, "overlap"):
            get_split(self.bundle, {}, 3, self.root)

    def test_content_and_preprocessing_change_fingerprint(self):
        modified = SignalDataset(self.bundle.x + 0.01, self.bundle.y, self.bundle.groups,
                                 self.bundle.sample_ids, deepcopy(self.bundle.metadata))
        self.assertNotEqual(self.bundle.fingerprint, modified.fingerprint)
        settings = load_dataset("synthetic_eeg", {"subjects": 6, "samples_per_subject": 12,
                                                "normalization": "none"})
        self.assertNotEqual(self.bundle.fingerprint, settings.fingerprint)
        split = get_split(self.bundle, {}, 0, self.root)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_split(modified, split)

    def test_nonfinite_fractional_label_and_duplicate_identity_rejected(self):
        for y, ids, message in [([0.5, 1], ["a", "b"], "integer"),
                                ([0, 1], ["a", "a"], "unique")]:
            with self.assertRaisesRegex(ValueError, message):
                SignalDataset(np.zeros((2, 1, 16)), np.asarray(y), np.array(["one", "two"]), ids,
                              {"modality": "ecg", "num_classes": 2})
        bad = np.zeros((2, 1, 16))
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            SignalDataset(bad, np.array([0, 1]), np.array(["one", "two"]), ["a", "b"],
                          {"modality": "ecg", "num_classes": 2})

    def test_filter_does_not_propagate_nan_between_runs(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 1200))
        x[:, 500:600] = np.nan
        filtered, valid = bandpass_finite_spans(x, 250, 2, 30, [600])
        self.assertTrue(valid[:500].all())
        self.assertFalse(valid[500:600].any())
        self.assertTrue(valid[600:].all())
        isolated, _ = bandpass_finite_spans(x[:, 600:], 250, 2, 30)
        np.testing.assert_array_equal(filtered[:, 600:], isolated)

    def test_train_channel_normalization_is_deferred(self):
        x = np.arange(32, dtype=np.float32).reshape(2, 1, 16)
        np.testing.assert_array_equal(normalize_samples(x, "train_channel"), x)
        normalized = normalize_samples(x, "per_sample")
        np.testing.assert_allclose(normalized.mean(axis=-1), 0, atol=1e-7)
        np.testing.assert_allclose(normalized.std(axis=-1), 1, atol=1e-7)

    def test_flat_ecg_is_not_treated_as_a_gdf_missing_run(self):
        flat = np.zeros((1, 1024))
        filtered, valid = bandpass_finite_spans(flat, 360, .5, 45)
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(filtered, flat)
        _, gdf_valid = bandpass_finite_spans(flat, 250, 2, 30, gdf_missing=True)
        self.assertFalse(gdf_valid.any())

    def test_eeg_defaults_match_individual_trial_evaluation(self):
        defaults = get_dataset_spec('eeg').defaults
        self.assertEqual(defaults['filter_scope'], 'trial')
        self.assertEqual(defaults['normalization'], 'train_channel')
        self.assertEqual(defaults['artifact_policy'], 'exclude')
        self.assertEqual(defaults['offset_seconds'], 0)
        self.assertEqual(defaults['window'], 1000)

    def test_npz_requires_explicit_subject_ids(self):
        path = self.root / "signals.npz"
        np.savez(path, x=self.bundle.x, y=self.bundle.y)
        with self.assertRaisesRegex(ValueError, "groups"):
            load_dataset("npz_eeg", {"path": str(path)})
        np.savez(path, x=self.bundle.x, y=self.bundle.y, groups=self.bundle.groups)
        actual = load_dataset("npz_eeg", {"path": str(path), "normalization": "none"})
        np.testing.assert_array_equal(actual.x, self.bundle.x)
        np.testing.assert_array_equal(actual.y, self.bundle.y)

    def test_ecg_keeps_same_patient_together_and_selects_named_lead(self):
        records = ["104", "114", "201", "202"]
        for record in records:
            for suffix in (".hea", ".dat", ".atr"):
                (self.root / f"{record}{suffix}").write_text(record)
        selections = {}
        def rdheader(stem):
            record = Path(stem).name
            names = ["V5", "V2"] if record == "104" else ["V5", "MLII"] if record == "114" else ["MLII", "V1"]
            return SimpleNamespace(sig_name=names)
        def rdsamp(stem, channels):
            selections[Path(stem).name] = channels
            t = np.arange(1024) / 360
            return np.sin(2 * np.pi * 5 * t)[:, None], {"fs": 360}
        fake = SimpleNamespace(rdheader=rdheader, rdsamp=rdsamp,
                               rdann=lambda *_: SimpleNamespace(sample=[300, 600], symbol=["N", "V"]))
        with patch.dict("sys.modules", {"wfdb": fake}):
            dataset = load_dataset("ecg", {"data_dir": str(self.root), "records": records})
        self.assertEqual(selections["114"], [1])
        self.assertEqual(dataset.metadata["excluded_records"][0]["record"], "104")
        self.assertEqual(len(set(dataset.groups[2:])), 1)
        self.assertEqual(dataset.x.shape, (6, 1, 256))
        self.assertEqual(dataset.y.tolist(), [0, 1, 0, 1, 0, 1])

    def test_training_cannot_omit_a_declared_class(self):
        metadata = deepcopy(self.bundle.metadata)
        metadata["num_classes"] = 5
        declared = SignalDataset(self.bundle.x, self.bundle.y, self.bundle.groups, self.bundle.sample_ids, metadata)
        with self.assertRaisesRegex(ValueError, "Training split lacks classes"):
            get_split(declared, {}, 1, self.root)


if __name__ == "__main__":
    unittest.main()
