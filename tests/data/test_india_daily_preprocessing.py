import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.india_daily_preprocessing import (
    IndiaFeatureScaler,
    load_indian_daily_dataset,
)


class IndiaDailyPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv_path = self.root / "validated.csv"
        self.manifest_path = self.root / "manifest.json"
        dates = pd.bdate_range("2023-01-02", periods=90)
        rows = []
        for position, date in enumerate(dates):
            for symbol_index, symbol in enumerate(("NIFTYBEES", "BANKBEES")):
                close = 100.0 + position * 0.2 + symbol_index
                rows.append({
                    "Date": date.strftime("%Y-%m-%d"),
                    "Symbol": symbol,
                    "Open": close - 0.5,
                    "High": close + 1.0,
                    "Low": close - 1.0,
                    "Close": close,
                    "Adjusted Close": close,
                    "Volume": 100000,
                })
        pd.DataFrame(rows).to_csv(self.csv_path, index=False)
        self.manifest_path.write_text(json.dumps({
            "training_ready": True,
            "symbols": ["NIFTYBEES", "BANKBEES"],
            "provider": "Test Provider",
            "retrieval_timestamp": "2024-01-01T00:00:00Z",
            "adjustment_methodology": "provider-adjusted-close",
            "license_permission_status": "approved",
        }), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        return load_indian_daily_dataset(
            self.csv_path,
            manifest_json=self.manifest_path,
            symbols=("NIFTYBEES", "BANKBEES"),
        )

    def test_loads_configurable_universe_and_common_dates(self):
        dataset = self.load()
        self.assertEqual(dataset.symbols, ("NIFTYBEES", "BANKBEES"))
        self.assertTrue(len(dataset.dates) > 21)
        self.assertTrue(dataset.features.index.is_monotonic_increasing)
        self.assertEqual(
            set(dataset.open_prices.columns),
            {"NIFTYBEES", "BANKBEES"},
        )

    def test_rejects_missing_required_schema_column(self):
        frame = pd.read_csv(self.csv_path).drop(columns=["Volume"])
        bad_path = self.root / "bad_schema.csv"
        frame.to_csv(bad_path, index=False)
        with self.assertRaisesRegex(ValueError, "missing required Phase 1"):
            load_indian_daily_dataset(
                bad_path,
                manifest_json=self.manifest_path,
                symbols=("NIFTYBEES", "BANKBEES"),
            )

    def test_rejects_duplicate_records_missing_values_and_bad_ohlc(self):
        frame = pd.read_csv(self.csv_path)
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        duplicate_path = self.root / "duplicates.csv"
        duplicate.to_csv(duplicate_path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_indian_daily_dataset(
                duplicate_path,
                manifest_json=self.manifest_path,
                symbols=("NIFTYBEES", "BANKBEES"),
            )

        missing = frame.copy()
        missing.loc[0, "Open"] = np.nan
        missing_path = self.root / "missing.csv"
        missing.to_csv(missing_path, index=False)
        with self.assertRaisesRegex(ValueError, "missing numeric"):
            load_indian_daily_dataset(
                missing_path,
                manifest_json=self.manifest_path,
                symbols=("NIFTYBEES", "BANKBEES"),
            )

        bad_ohlc = frame.copy()
        bad_ohlc.loc[0, "High"] = 0.1
        bad_path = self.root / "bad_ohlc.csv"
        bad_ohlc.to_csv(bad_path, index=False)
        with self.assertRaisesRegex(ValueError, "inconsistent OHLC"):
            load_indian_daily_dataset(
                bad_path,
                manifest_json=self.manifest_path,
                symbols=("NIFTYBEES", "BANKBEES"),
            )

    def test_chronological_splits_do_not_overlap(self):
        dataset = self.load()
        train_start, train_end = dataset.splits.train
        validation_start, validation_end = dataset.splits.validation
        test_start, test_end = dataset.splits.test
        self.assertLessEqual(train_start, train_end)
        self.assertLess(train_end, validation_start)
        self.assertLessEqual(validation_start, validation_end)
        self.assertLess(validation_end, test_start)
        self.assertLessEqual(test_start, test_end)

    def test_scaler_is_fit_only_on_training_rows(self):
        dataset = self.load()
        train = dataset.features.loc[
            (dataset.features.index >= dataset.splits.train[0])
            & (dataset.features.index <= dataset.splits.train[1])
        ]
        validation = dataset.features.loc[
            (dataset.features.index >= dataset.splits.validation[0])
            & (dataset.features.index <= dataset.splits.validation[1])
        ]
        scaler = IndiaFeatureScaler.fit(train)
        means = np.asarray(scaler.means)
        validation_transformed = scaler.transform(validation)
        self.assertEqual(len(means), len(dataset.features.columns))
        self.assertTrue(np.isfinite(validation_transformed).all())

    def test_missing_symbol_observation_removes_session_not_fills_it(self):
        frame = pd.read_csv(self.csv_path)
        missing = frame[
            ~(
                (frame["Symbol"] == "BANKBEES")
                & (frame["Date"] == "2023-01-03")
            )
        ]
        missing_path = self.root / "partial.csv"
        missing.to_csv(missing_path, index=False)
        dataset = load_indian_daily_dataset(
            missing_path,
            manifest_json=self.manifest_path,
            symbols=("NIFTYBEES", "BANKBEES"),
        )
        self.assertNotIn(
            pd.Timestamp("2023-01-03"),
            dataset.open_prices.index,
        )

    def test_unapproved_or_ambiguous_adjustment_is_rejected(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["training_ready"] = False
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not training-ready"):
            self.load()


if __name__ == "__main__":
    unittest.main()
