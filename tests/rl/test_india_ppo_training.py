import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.india_daily_preprocessing import load_indian_daily_dataset
from src.rl.india_ppo_training import (
    INDIA_ARTIFACT_ROOT,
    IndiaDataNotReadyError,
    IndiaHistoricalAuthorization,
    prepare_india_ppo_training,
    train_india_ppo,
)
from src.rl.ppo_config import PPOConfig, TrainingOptions


class IndiaPPOTrainingPreparationTests(unittest.TestCase):
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
                    "Open": close,
                    "High": close + 1,
                    "Low": close - 1,
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

    def test_preparation_builds_three_separate_environment_splits(self):
        preparation = prepare_india_ppo_training(
            validated_csv=self.csv_path,
            manifest_json=self.manifest_path,
            symbols=("NIFTYBEES", "BANKBEES"),
            seed=42,
            repository_root=self.root,
            output_dir=self.root / INDIA_ARTIFACT_ROOT,
        )
        self.assertEqual(
            preparation.manifest["source_type"],
            "india_historical_csv",
        )
        self.assertFalse(preparation.manifest["training_started"])
        self.assertTrue(preparation.output_dir == (
            self.root / INDIA_ARTIFACT_ROOT
        ).resolve())
        self.assertFalse(preparation.output_dir.exists())
        for env in (
            preparation.bundle.train_env,
            preparation.bundle.validation_env,
            preparation.bundle.test_env,
        ):
            observation, _ = env.reset(seed=42)
            self.assertEqual(observation.dtype, np.float32)
            self.assertTrue(env.observation_space.contains(observation))

    def test_unapproved_license_blocks_preparation(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["license_permission_status"] = "unverified"
        manifest["training_ready"] = False
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(IndiaDataNotReadyError):
            prepare_india_ppo_training(
                validated_csv=self.csv_path,
                manifest_json=self.manifest_path,
                symbols=("NIFTYBEES", "BANKBEES"),
                seed=42,
                repository_root=self.root,
            )

    def test_training_requires_explicit_authorization(self):
        preparation = prepare_india_ppo_training(
            validated_csv=self.csv_path,
            manifest_json=self.manifest_path,
            symbols=("NIFTYBEES", "BANKBEES"),
            seed=42,
            repository_root=self.root,
        )
        config = PPOConfig(seed=42, device="cpu")
        options = TrainingOptions(total_timesteps=16, seed=42, verbose=0)
        with self.assertRaises(IndiaDataNotReadyError):
            train_india_ppo(
                preparation,
                authorization=IndiaHistoricalAuthorization(),
                config=config,
                training_options=options,
            )
        self.assertFalse(preparation.output_dir.exists())

    def test_output_directory_cannot_use_legacy_synthetic_artifacts(self):
        with self.assertRaisesRegex(ValueError, "India PPO"):
            prepare_india_ppo_training(
                validated_csv=self.csv_path,
                manifest_json=self.manifest_path,
                symbols=("NIFTYBEES", "BANKBEES"),
                seed=42,
                output_dir=self.root / ".artifacts" / "ppo",
                repository_root=self.root,
            )

    def test_manifest_training_options_and_split_metadata(self):
        options = TrainingOptions(total_timesteps=100, seed=42, verbose=0)
        preparation = prepare_india_ppo_training(
            validated_csv=self.csv_path,
            manifest_json=self.manifest_path,
            symbols=("NIFTYBEES", "BANKBEES"),
            seed=42,
            training_options=options,
            repository_root=self.root,
        )
        self.assertEqual(
            preparation.manifest["training_options"]["total_timesteps"],
            100,
        )
        splits = preparation.manifest["splits"]
        self.assertIn("train", splits)
        self.assertIn("validation", splits)
        self.assertIn("test", splits)
        self.assertTrue(
            splits["train"][1] < splits["validation"][0]
        )
        self.assertTrue(
            splits["validation"][1] < splits["test"][0]
        )


if __name__ == "__main__":
    unittest.main()
