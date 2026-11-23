import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from src.data.india_daily_preprocessing import (
    IndiaFeatureScaler,
    load_indian_daily_dataset,
)
from src.rl.environment.india_daily_portfolio_env import IndiaDailyPortfolioEnv


class IndiaDailyPortfolioEnvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv_path = self.root / "validated.csv"
        self.manifest_path = self.root / "manifest.json"
        dates = pd.bdate_range("2023-01-02", periods=100)
        rows = []
        for position, date in enumerate(dates):
            for symbol_index, symbol in enumerate(("NIFTYBEES", "BANKBEES")):
                close = 100.0 + position * 0.5 + symbol_index
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
        self.dataset = load_indian_daily_dataset(
            self.csv_path,
            manifest_json=self.manifest_path,
            symbols=("NIFTYBEES", "BANKBEES"),
        )
        train_features = self.dataset.features.loc[
            (self.dataset.features.index >= self.dataset.splits.train[0])
            & (self.dataset.features.index <= self.dataset.splits.train[1])
        ]
        self.scaler = IndiaFeatureScaler.fit(train_features)

    def tearDown(self):
        self.temp.cleanup()

    def test_reset_observation_and_action_contract(self):
        env = IndiaDailyPortfolioEnv(
            self.dataset,
            partition="train",
            scaler=self.scaler,
        )
        observation, info = env.reset(seed=42)
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(env.action_space.shape, (3,))
        self.assertEqual(info["source_type"], "india_historical_csv")
        self.assertTrue(info["corporate_action_limitation"])

    def test_all_cash_action_has_valid_weights_and_no_asset_trades(self):
        env = IndiaDailyPortfolioEnv(
            self.dataset,
            partition="train",
            scaler=self.scaler,
            initial_capital=100000,
        )
        env.reset(seed=7)
        _, reward, _, truncated, info = env.step(
            np.asarray([-1.0, -1.0, 1.0], dtype=np.float32)
        )
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, 0.0)
        self.assertAlmostEqual(sum(info["applied_weights"].values()), 1.0)
        self.assertAlmostEqual(info["applied_weights"]["CASH"], 1.0)
        self.assertAlmostEqual(info["transaction_fees"], 0.0)
        self.assertAlmostEqual(info["slippage_cost"], 0.0)

    def test_transaction_costs_and_slippage_are_charged(self):
        env = IndiaDailyPortfolioEnv(
            self.dataset,
            partition="train",
            scaler=self.scaler,
            initial_capital=100000,
            transaction_cost_rate=0.01,
            slippage_bps=100,
        )
        env.reset(seed=7)
        _, _, _, _, info = env.step(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        )
        self.assertGreater(info["total_transaction_cost"], 0.0)
        self.assertGreaterEqual(info["transaction_fees"], 0.0)
        self.assertGreaterEqual(info["slippage_cost"], 0.0)
        self.assertLess(info["nav"], 100000)

    def test_weights_stay_long_only_and_sum_to_one(self):
        env = IndiaDailyPortfolioEnv(
            self.dataset,
            partition="train",
            scaler=self.scaler,
        )
        env.reset(seed=11)
        _, _, _, _, info = env.step(
            np.asarray([3.0, -2.0, 1.0], dtype=np.float32)
        )
        weights = np.asarray(list(info["applied_weights"].values()), dtype=np.float64)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(np.all(weights >= -1e-8))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_reset_and_step_reproduce_with_fixed_seed(self):
        env = IndiaDailyPortfolioEnv(
            self.dataset,
            partition="train",
            scaler=self.scaler,
            random_start=True,
        )
        first_observation, first_info = env.reset(seed=19)
        first_action = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        first_next, first_reward, _, _, first_step = env.step(first_action)
        second_observation, second_info = env.reset(seed=19)
        second_next, second_reward, _, _, second_step = env.step(first_action)
        np.testing.assert_array_equal(first_observation, second_observation)
        np.testing.assert_array_equal(first_next, second_next)
        self.assertEqual(first_info["date"], second_info["date"])
        self.assertEqual(first_reward, second_reward)
        self.assertEqual(first_step["nav"], second_step["nav"])

    def test_episode_terminates_at_partition_end(self):
        env = IndiaDailyPortfolioEnv(
            self.dataset,
            partition="train",
            scaler=self.scaler,
        )
        env.reset(seed=5)
        terminated = False
        for _ in range(len(env.partition_dates) + 1):
            if terminated:
                break
            _, _, terminated, truncated, _ = env.step(
                np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
            )
            self.assertFalse(truncated)
        self.assertTrue(terminated)

    def test_gymnasium_environment_checker(self):
        env = IndiaDailyPortfolioEnv(
            self.dataset,
            partition="train",
            scaler=self.scaler,
        )
        check_env(env, skip_render_check=True)


if __name__ == "__main__":
    unittest.main()
