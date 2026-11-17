import unittest

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from src.data.data_store import get_data_store
from src.rl.benchmarks import evaluate_benchmarks
from src.rl.environment import HistoricalPortfolioEnv, RewardConfig


class HistoricalPortfolioEnvTests(unittest.TestCase):
    def test_sqlite_environment_initializes_and_steps(self):
        env = HistoricalPortfolioEnv(
            start_date="2024-01-01",
            end_date="2025-12-31",
            reward_config=RewardConfig(lookback=20),
        )
        observation, info = env.reset(seed=11)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["date"], "2024-01-31")
        next_observation, reward, terminated, truncated, step_info = env.step(
            np.asarray([0.2, 0.2, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)
        )
        self.assertTrue(env.observation_space.contains(next_observation))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(truncated)
        self.assertAlmostEqual(float(step_info["weights"].sum()), 1.0)
        self.assertGreaterEqual(float(step_info["weights"].min()), -1e-8)
        self.assertIn("reward_components", step_info)
        self.assertTrue(np.isfinite(list(step_info["reward_components"].values())).all())

    def test_out_of_bounds_actions_are_clipped_without_short_positions(self):
        env = HistoricalPortfolioEnv(start_date="2024-01-01", end_date="2025-12-31")
        env.reset()
        _, _, _, _, info = env.step(np.asarray([-2, 2, 0, 0, 0, 0], dtype=np.float32))
        self.assertTrue(info["action_was_clipped"])
        self.assertGreaterEqual(float(info["weights"].min()), -1e-8)
        self.assertAlmostEqual(float(info["weights"].sum()), 1.0)

    def test_gymnasium_checker(self):
        env = HistoricalPortfolioEnv(start_date="2024-01-01", end_date="2025-12-31")
        check_env(env, skip_render_check=True)

    def test_real_sqlite_records_are_used(self):
        stats = get_data_store().get_storage_stats()
        self.assertGreaterEqual(stats["price_records"], 2510)

    def test_benchmarks_share_period_and_are_finite(self):
        raw = get_data_store().get_price_data(
            ["SPY", "QQQ", "GLD", "TLT", "BND"], "2024-01-01", "2025-12-31"
        )
        prices = raw.assign(datadate=pd.to_datetime(raw["datadate"])).pivot(
            index="datadate", columns="tic", values="adj_close"
        ).sort_index().dropna()
        result = evaluate_benchmarks(prices)
        self.assertEqual(len(result), 4)
        self.assertTrue(np.isfinite(result.to_numpy(dtype=float)).all())
        self.assertTrue((result["benchmark"].index if "benchmark" in result.columns else result.index).tolist())


if __name__ == "__main__":
    unittest.main()
