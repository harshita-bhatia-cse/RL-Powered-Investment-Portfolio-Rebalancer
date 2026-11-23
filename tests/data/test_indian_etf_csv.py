import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.indian_etf_csv import (
    IndianETFIngestionConfig,
    ingest_indian_etf_csv,
    validate_indian_etf_csv,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


class IndianETFCSVTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.input_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def valid_row(self, date="2024-01-02", symbol="NIFTYBEES", close=100.0):
        return {
            "Date": date,
            "Symbol": symbol,
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Adjusted Close": close,
            "Volume": 1000,
        }

    def test_valid_csv_is_training_ready_with_approved_provenance(self):
        csv_path = self.input_dir / "indian_etfs.csv"
        write_csv(csv_path, [
            self.valid_row("2024-01-02", "NIFTYBEES"),
            self.valid_row("2024-01-03", "NIFTYBEES"),
            self.valid_row("2024-01-02", "BANKBEES"),
            self.valid_row("2024-01-03", "BANKBEES"),
        ])
        config = IndianETFIngestionConfig(
            symbols=("NIFTYBEES", "BANKBEES"),
            provider="Test Provider",
            retrieval_timestamp="2024-01-04T00:00:00+00:00",
            adjustment_methodology="provider-adjusted-close",
            license_permission_status="approved",
        )

        report = ingest_indian_etf_csv(csv_path, config, output_dir=self.output_dir)

        self.assertTrue(report["valid"])
        self.assertTrue(report["training_ready"])
        self.assertEqual(report["common_sessions"]["count"], 2)
        self.assertTrue((self.output_dir / "raw" / "indian_etfs.csv").exists())
        self.assertTrue(
            (self.output_dir / "validated" / "indian_etf_validated.csv").exists()
        )
        manifest = self.output_dir / "indian_etf_ingestion_manifest.json"
        self.assertTrue(manifest.exists())

    def test_unknown_provenance_prevents_training_ready_status(self):
        csv_path = self.input_dir / "unknown.csv"
        write_csv(csv_path, [self.valid_row()])
        report = ingest_indian_etf_csv(
            csv_path,
            IndianETFIngestionConfig(symbols=("NIFTYBEES",)),
            output_dir=self.output_dir,
        )
        self.assertFalse(report["training_ready"])
        self.assertEqual(report["provider"], "unknown")
        self.assertEqual(report["license_permission_status"], "unverified")

    def test_missing_columns_are_rejected(self):
        csv_path = self.input_dir / "missing.csv"
        pd.DataFrame([{
            "Date": "2024-01-02",
            "Symbol": "NIFTYBEES",
            "Close": 100,
        }]).to_csv(csv_path, index=False)

        with self.assertRaisesRegex(ValueError, "missing required CSV columns"):
            validate_indian_etf_csv(
                csv_path,
                IndianETFIngestionConfig(symbols=("NIFTYBEES",)),
            )

    def test_duplicate_dates_invalid_prices_volume_and_ohlc_are_reported(self):
        csv_path = self.input_dir / "invalid.csv"
        write_csv(csv_path, [
            self.valid_row("2024-01-02", "NIFTYBEES"),
            self.valid_row("2024-01-02", "NIFTYBEES"),
            {
                "Date": "2024-01-03",
                "Symbol": "NIFTYBEES",
                "Open": 100,
                "High": 90,
                "Low": 95,
                "Close": 96,
                "Adjusted Close": -1,
                "Volume": -5,
            },
        ])

        report = validate_indian_etf_csv(
            csv_path,
            IndianETFIngestionConfig(symbols=("NIFTYBEES",)),
        )
        self.assertFalse(report["valid"])
        self.assertFalse(report["training_ready"])
        self.assertIn("duplicate dates detected per symbol", report["issues"])
        self.assertIn(
            "invalid or non-positive price values detected", report["issues"]
        )
        self.assertIn("negative or missing volume values detected", report["issues"])
        self.assertIn("inconsistent OHLC values detected", report["issues"])

    def test_missing_values_are_reported(self):
        csv_path = self.input_dir / "missing_values.csv"
        write_csv(csv_path, [{
            "Date": "",
            "Symbol": "NIFTYBEES",
            "Open": 99,
            "High": 102,
            "Low": 98,
            "Close": 100,
            "Adjusted Close": "",
            "Volume": 100,
        }])
        report = validate_indian_etf_csv(
            csv_path,
            IndianETFIngestionConfig(symbols=("NIFTYBEES",)),
        )
        self.assertIn("missing values detected", report["issues"])
        self.assertIn("invalid or missing dates detected", report["issues"])

    def test_chronology_and_common_sessions_are_reported_without_filling(self):
        csv_path = self.input_dir / "coverage.csv"
        write_csv(csv_path, [
            self.valid_row("2024-01-03", "NIFTYBEES"),
            self.valid_row("2024-01-02", "NIFTYBEES"),
            self.valid_row("2024-01-02", "BANKBEES", 50),
        ])

        report = validate_indian_etf_csv(
            csv_path,
            IndianETFIngestionConfig(symbols=("NIFTYBEES", "BANKBEES")),
        )
        self.assertIn("NIFTYBEES", report["chronological_violations"])
        self.assertIn(
            "2024-01-03",
            report["missing_dates_by_symbol"]["BANKBEES"],
        )
        self.assertEqual(report["common_sessions"]["count"], 1)

    def test_directory_ingestion_preserves_each_raw_csv(self):
        write_csv(self.input_dir / "nifty.csv", [self.valid_row()])
        write_csv(
            self.input_dir / "bank.csv",
            [self.valid_row("2024-01-02", "BANKBEES", 50)],
        )
        report = ingest_indian_etf_csv(
            self.input_dir,
            IndianETFIngestionConfig(
                symbols=("NIFTYBEES", "BANKBEES"),
            ),
            output_dir=self.output_dir,
        )
        preserved_names = {
            Path(item["preserved_file"]).name
            for item in report["preserved_raw_files"]
        }
        self.assertEqual(preserved_names, {"nifty.csv", "bank.csv"})

    def test_positive_and_negative_infinite_volume_are_rejected(self):
        csv_path = self.input_dir / "infinite_volume.csv"
        positive = self.valid_row("2024-01-02")
        positive["Volume"] = float("inf")
        negative = self.valid_row("2024-01-03")
        negative["Volume"] = float("-inf")
        write_csv(csv_path, [positive, negative])

        report = validate_indian_etf_csv(
            csv_path,
            IndianETFIngestionConfig(symbols=("NIFTYBEES",)),
        )

        self.assertIn("non-finite volume values detected", report["issues"])
        self.assertFalse(report["training_ready"])

    def test_invalid_and_unknown_retrieval_timestamps_block_training_ready(self):
        csv_path = self.input_dir / "timestamp.csv"
        write_csv(csv_path, [self.valid_row()])
        for timestamp in ("unknown", "2024-01-02", "not-a-timestamp", ""):
            with self.subTest(timestamp=timestamp):
                report = ingest_indian_etf_csv(
                    csv_path,
                    IndianETFIngestionConfig(
                        symbols=("NIFTYBEES",),
                        provider="Test Provider",
                        retrieval_timestamp=timestamp,
                        adjustment_methodology="provider-adjusted-close",
                        license_permission_status="approved",
                    ),
                    output_dir=self.output_dir,
                )
                self.assertFalse(report["training_ready"])
                self.assertFalse(report["retrieval_timestamp_valid"])
                self.assertIn(
                    "invalid, missing or unknown retrieval timestamp",
                    report["issues"],
                )

    def test_valid_timezone_timestamp_is_normalized_in_manifest(self):
        csv_path = self.input_dir / "timezone.csv"
        write_csv(csv_path, [self.valid_row()])
        report = ingest_indian_etf_csv(
            csv_path,
            IndianETFIngestionConfig(
                symbols=("NIFTYBEES",),
                provider="Test Provider",
                retrieval_timestamp="2024-01-04T05:30:00+05:30",
                adjustment_methodology="provider-adjusted-close",
                license_permission_status="approved",
            ),
            output_dir=self.output_dir,
        )
        self.assertTrue(report["retrieval_timestamp_valid"])
        self.assertEqual(
            report["retrieval_timestamp_normalized_utc"],
            "2024-01-04T00:00:00Z",
        )
        self.assertEqual(
            report["retrieval_timestamp"],
            "2024-01-04T05:30:00+05:30",
        )

    def test_manifest_contains_top_level_symbols_and_raw_validated_ranges(self):
        csv_path = self.input_dir / "ranges.csv"
        write_csv(csv_path, [
            self.valid_row("2024-01-02", "NIFTYBEES"),
            self.valid_row("2024-01-03", "NIFTYBEES"),
            self.valid_row("2024-01-04", "NIFTYBEES", 100),
            {
                "Date": "2024-01-05",
                "Symbol": "NIFTYBEES",
                "Open": 100,
                "High": 90,
                "Low": 95,
                "Close": 96,
                "Adjusted Close": 96,
                "Volume": 100,
            },
        ])
        report = ingest_indian_etf_csv(
            csv_path,
            IndianETFIngestionConfig(
                symbols=("NIFTYBEES",),
                provider="Test Provider",
                retrieval_timestamp="2024-01-04T00:00:00Z",
                adjustment_methodology="provider-adjusted-close",
                license_permission_status="approved",
            ),
            output_dir=self.output_dir,
        )
        self.assertEqual(report["symbols"], ["NIFTYBEES"])
        self.assertEqual(
            report["raw_observed_date_range"],
            {"start_date": "2024-01-02", "end_date": "2024-01-05"},
        )
        self.assertEqual(
            report["date_range"],
            {"start_date": "2024-01-02", "end_date": "2024-01-04"},
        )

    def test_missing_required_columns_preserve_raw_and_write_manifest(self):
        csv_path = self.input_dir / "missing_columns.csv"
        pd.DataFrame([{
            "Date": "2024-01-02",
            "Symbol": "NIFTYBEES",
            "Close": 100,
        }]).to_csv(csv_path, index=False)

        report = ingest_indian_etf_csv(
            csv_path,
            IndianETFIngestionConfig(symbols=("NIFTYBEES",)),
            output_dir=self.output_dir,
        )

        self.assertFalse(report["training_ready"])
        self.assertFalse(report["valid"])
        self.assertTrue((self.output_dir / "raw" / "missing_columns.csv").exists())
        self.assertTrue((self.output_dir / "indian_etf_ingestion_manifest.json").exists())
        self.assertTrue(report["missing_columns"])


if __name__ == "__main__":
    unittest.main()
