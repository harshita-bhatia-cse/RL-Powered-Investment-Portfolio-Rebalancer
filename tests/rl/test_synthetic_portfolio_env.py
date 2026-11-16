import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from src.rl.environment.synthetic_portfolio_env import (
    DEFAULT_INITIAL_CAPITAL,
    OBSERVATION_SIZE,
    OBSERVATION_WARMUP_SESSIONS,
)
from src.rl.environment import SyntheticPortfolioEnv
from src.rl.synthetic_data import generate_synthetic_splits


def make_market_data(sessions=67):
    assets = np.zeros((sessions, 6), dtype=np.float64)
    benchmark = np.zeros(sessions, dtype=np.float64)
    vix = np.full(sessions, 20.0, dtype=np.float64)
    return assets, benchmark, vix


def make_env(*, sessions=67, **kwargs):
    assets, benchmark, vix = make_market_data(sessions)
    return SyntheticPortfolioEnv(
        assets,
        benchmark,
        vix,
        data_metadata={
            "source_type": "synthetic",
            "synthetic": True,
            "seed": 7,
            "generator_version": "test",
        },
        **kwargs,
    )


def action_with_weights(weights):
    return np.asarray(weights, dtype=np.float32)


class SyntheticPortfolioEnvTests(unittest.TestCase):
    def test_initialization_and_reset_contract(self):
        env = make_env()
        observation, info = env.reset(seed=11)

        self.assertIsInstance(env, gym.Env)
        self.assertEqual(env.action_space.shape, (7,))
        self.assertEqual(env.observation_space.shape, (OBSERVATION_SIZE,))
        self.assertEqual(OBSERVATION_SIZE, 43)
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["nav"], DEFAULT_INITIAL_CAPITAL)
        self.assertEqual(info["session_index"], OBSERVATION_WARMUP_SESSIONS)
        self.assertTrue(info["synthetic"])
        self.assertEqual(info["weights"]["CASH"], 1.0)

    def test_environment_passes_gymnasium_checker(self):
        check_env(make_env(), skip_render_check=True)

    def test_observation_features_are_finite_and_use_trailing_values(self):
        assets, benchmark, vix = make_market_data()
        assets[-63:, 0] = 0.01
        benchmark[-63:] = 0.02
        vix[OBSERVATION_WARMUP_SESSIONS - 1] = 25.0
        env = SyntheticPortfolioEnv(
            assets,
            benchmark,
            vix,
            data_metadata={"source_type": "synthetic", "synthetic": True},
        )

        observation, _ = env.reset()

        self.assertTrue(np.isfinite(observation).all())
        self.assertTrue(env.observation_space.contains(observation))
        self.assertAlmostEqual(float(observation[0]), 0.01, places=7)
        self.assertAlmostEqual(float(observation[1]), (1.01**5) - 1.0, places=6)
        self.assertAlmostEqual(float(observation[30]), 0.02, places=7)
        self.assertAlmostEqual(float(observation[32]), 0.25, places=7)

    def test_action_is_projected_and_constraints_are_enforced(self):
        env = make_env()
        env.reset()
        action = np.asarray([1, 1, 1, 1, 1, 1, -1], dtype=np.float32)

        _, _, _, _, info = env.step(action)

        weights = np.asarray(list(info["applied_weights"].values()))
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
        self.assertTrue(all(weight <= 0.35 + 1e-10 for weight in weights[:6]))
        self.assertGreaterEqual(weights[-1], 0.02 - 1e-10)
        self.assertLessEqual(info["gross_turnover"], 1.0 + 1e-10)
        self.assertTrue(info["turnover_limited"])
        self.assertFalse(info["constraint_recovery"])

    def test_turnover_limit_allows_monotonic_recovery_from_market_drift(self):
        dataset = generate_synthetic_splits(seed=42).train
        env = SyntheticPortfolioEnv(**dataset.environment_kwargs())
        observation, _ = env.reset(seed=dataset.seed)
        action_rng = np.random.default_rng(42)
        recovery_observed = False

        for _ in range(500):
            action = action_rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
            observation, _, terminated, truncated, info = env.step(action)
            self.assertLessEqual(info["gross_turnover"], 1.0 + 1e-10)
            if info["constraint_recovery"]:
                recovery_observed = True
                weights = np.asarray(list(info["applied_weights"].values()))
                self.assertTrue(np.isfinite(weights).all())
                self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
            if terminated or truncated:
                observation, _ = env.reset(seed=dataset.seed)

        self.assertTrue(env.observation_space.contains(observation))
        self.assertTrue(recovery_observed)

    def test_finite_out_of_bounds_action_is_clipped_and_reported(self):
        env = make_env()
        env.reset()
        action = np.asarray([4, -4, 0, 0, 0, 0, 0], dtype=np.float32)

        _, _, _, _, info = env.step(action)

        self.assertTrue(info["action_clipped"])
        self.assertTrue(
            all(weight >= 0.0 for weight in info["applied_weights"].values())
        )

    def test_malformed_and_non_finite_actions_are_rejected(self):
        env = make_env()
        env.reset()
        with self.assertRaisesRegex(ValueError, "shape"):
            env.step(np.zeros(6, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "finite"):
            env.step(np.asarray([0, 0, 0, 0, 0, np.nan, 0], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "finite"):
            env.step(np.asarray([0, 0, 0, 0, 0, np.inf, 0], dtype=np.float32))

    def test_zero_return_cash_action_preserves_nav_and_has_zero_reward(self):
        env = make_env()
        env.reset()

        observation, reward, terminated, truncated, info = env.step(
            action_with_weights([-1, -1, -1, -1, -1, -1, 1])
        )

        self.assertAlmostEqual(info["nav"], DEFAULT_INITIAL_CAPITAL)
        self.assertEqual(reward, 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(env.observation_space.contains(observation))

    def test_accounting_costs_turnover_and_log_reward_match_hand_calculation(self):
        assets, benchmark, vix = make_market_data()
        assets[63:, 0] = 0.10
        env = SyntheticPortfolioEnv(
            assets,
            benchmark,
            vix,
            data_metadata={"source_type": "synthetic", "synthetic": True},
            period_sessions=2,
            max_asset_weight=1.0,
            min_cash_weight=0.0,
            max_turnover=2.0,
            transaction_cost_rate=0.001,
            slippage_bps=5.0,
        )
        env.reset()
        action = action_with_weights([0.5, 0, 0, 0, 0, 0, 0.5])

        _, reward, _, _, info = env.step(action)

        # Half the portfolio moves from cash to SPY: gross turnover = 1.0.
        self.assertAlmostEqual(info["gross_turnover"], 1.0, places=8)
        self.assertAlmostEqual(info["one_way_turnover"], 0.5, places=8)
        self.assertAlmostEqual(info["traded_notional"], 100_000.0, places=6)
        self.assertAlmostEqual(info["transaction_fees"], 100.0, places=6)
        self.assertAlmostEqual(info["slippage_cost"], 50.0, places=6)
        self.assertAlmostEqual(info["post_cost_nav"], 99_850.0, places=6)
        expected_nav = 99_850.0 * (0.5 * (1.1**2) + 0.5)
        self.assertAlmostEqual(info["nav"], expected_nav, places=6)
        self.assertAlmostEqual(reward, np.log(expected_nav / 100_000.0), places=12)
        self.assertAlmostEqual(info["reward_log_nav_return"], reward, places=12)

    def test_last_partial_interval_terminates_episode(self):
        env = make_env(sessions=66, period_sessions=21)
        env.reset()

        _, _, terminated, truncated, info = env.step(
            action_with_weights([-1, -1, -1, -1, -1, -1, 1])
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["sessions_in_interval"], 3)
        with self.assertRaisesRegex(RuntimeError, "after episode termination"):
            env.step(action_with_weights([-1, -1, -1, -1, -1, -1, 1]))

    def test_same_seed_reset_reproduces_observation(self):
        env = make_env()
        first_observation, first_info = env.reset(seed=23)
        env.step(np.zeros(7, dtype=np.float32))
        second_observation, second_info = env.reset(seed=23)

        np.testing.assert_array_equal(first_observation, second_observation)
        self.assertEqual(first_info["nav"], second_info["nav"])
        self.assertEqual(first_info["weights"], second_info["weights"])

    def test_rejects_non_synthetic_metadata_and_invalid_returns(self):
        assets, benchmark, vix = make_market_data()
        with self.assertRaisesRegex(ValueError, "synthetic data only"):
            SyntheticPortfolioEnv(
                assets,
                benchmark,
                vix,
                data_metadata={"source_type": "historical", "synthetic": False},
            )
        assets[0, 0] = -1.01
        with self.assertRaisesRegex(ValueError, "returns"):
            SyntheticPortfolioEnv(
                assets,
                benchmark,
                vix,
                data_metadata={"source_type": "synthetic", "synthetic": True},
            )

    def test_rejects_invalid_constraints_and_unaligned_inputs(self):
        assets, benchmark, vix = make_market_data()
        metadata = {"source_type": "synthetic", "synthetic": True}
        with self.assertRaisesRegex(ValueError, "cannot exceed 100%"):
            SyntheticPortfolioEnv(
                assets,
                benchmark,
                vix,
                data_metadata=metadata,
                min_cash_weight=1.1,
            )
        with self.assertRaisesRegex(ValueError, "must align"):
            SyntheticPortfolioEnv(
                assets,
                benchmark[:-1],
                vix,
                data_metadata=metadata,
            )

    def test_render_modes(self):
        env = make_env(render_mode="ansi")
        env.reset()
        rendered = env.render()
        self.assertIsInstance(rendered, str)
        self.assertIn("Synthetic portfolio", rendered)
        env.close()


if __name__ == "__main__":
    unittest.main()
