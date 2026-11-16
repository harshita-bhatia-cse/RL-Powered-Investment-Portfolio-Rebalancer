import unittest
from dataclasses import replace

import gymnasium
import numpy as np
from gymnasium.utils.env_checker import check_env

from src.rl.environment.synthetic_portfolio_env import SyntheticPortfolioEnv
from src.rl.synthetic_data import (
    GENERATED_SERIES,
    SyntheticMarketConfig,
    generate_synthetic_market,
    generate_synthetic_splits,
)
from src.rl.synthetic_pipeline import (
    build_synthetic_pipeline,
    run_deterministic_actions,
)


ALL_CASH_ACTION = np.asarray([-1, -1, -1, -1, -1, -1, 1], dtype=np.float32)
BALANCED_ACTION = np.asarray([0.5, 0, 0, 0, 0, 0, 0.5], dtype=np.float32)


def zero_volatility_config(sessions):
    base = SyntheticMarketConfig(sessions=sessions)
    regimes = tuple(
        replace(
            regime,
            annualized_log_returns=(0.0,) * len(GENERATED_SERIES),
            annualized_volatilities=(0.0,) * len(GENERATED_SERIES),
            vix_annualized_log_volatility=0.0,
        )
        for regime in base.regimes
    )
    return replace(base, regimes=regimes)


def deterministic_accounting_dataset():
    dataset = generate_synthetic_market(zero_volatility_config(65), seed=12)
    asset_returns = np.zeros((65, 6), dtype=np.float64)
    asset_returns[63:65, 0] = 0.10
    benchmark_returns = np.zeros(65, dtype=np.float64)
    vix_levels = np.full(65, 20.0, dtype=np.float64)
    for array in (asset_returns, benchmark_returns, vix_levels):
        array.setflags(write=False)
    return replace(
        dataset,
        asset_returns=asset_returns,
        benchmark_returns=benchmark_returns,
        vix_levels=vix_levels,
    )


class SyntheticPipelineIntegrationTests(unittest.TestCase):
    def test_builds_and_checks_three_independent_environments(self):
        pipeline = build_synthetic_pipeline(
            SyntheticMarketConfig(sessions=110), seed=314
        )

        self.assertEqual(set(pipeline.environments), {"train", "validation", "test"})
        self.assertEqual(
            set(pipeline.datasets.train.asset_names),
            {"SPY", "QQQ", "IEF", "GLD", "XLU", "XLV"},
        )
        for name in ("train", "validation", "test"):
            dataset = getattr(pipeline.datasets, name)
            environment = pipeline.environments[name]
            self.assertIsInstance(environment, gymnasium.Env)
            self.assertEqual(dataset.seed, dataset.metadata["seed"])
            self.assertEqual(dataset.metadata["scenario_id"], f"{name}-{dataset.seed}")
            self.assertTrue(environment.data_metadata["synthetic"])
            observation, info = environment.reset(seed=dataset.seed)
            self.assertEqual(observation.shape, (43,))
            self.assertEqual(observation.dtype, np.float32)
            self.assertTrue(np.isfinite(observation).all())
            self.assertTrue(environment.observation_space.contains(observation))
            self.assertEqual(info["data_metadata"]["seed"], dataset.seed)
            check_env(environment, skip_render_check=True)

    def test_deterministic_actions_collect_observations_rewards_nav_weights_and_done(self):
        pipeline = build_synthetic_pipeline(
            SyntheticMarketConfig(sessions=110), seed=10
        )
        actions = [ALL_CASH_ACTION, BALANCED_ACTION, ALL_CASH_ACTION]

        run = run_deterministic_actions(pipeline.datasets.train, actions)

        self.assertEqual(run.seed, pipeline.datasets.train.seed)
        self.assertEqual(run.scenario_id, pipeline.datasets.train.metadata["scenario_id"])
        self.assertEqual(run.initial_observation.shape, (43,))
        self.assertTrue(np.isfinite(run.initial_observation).all())
        self.assertLessEqual(len(run.steps), len(actions))
        for step in run.steps:
            self.assertEqual(step.observation.shape, (43,))
            self.assertEqual(step.observation.dtype, np.float32)
            self.assertTrue(np.isfinite(step.observation).all())
            self.assertTrue(np.isfinite(step.reward))
            self.assertIn("nav", step.info)
            self.assertIn("weights", step.info)
            self.assertEqual(step.terminated, step.info["session_end"] >= 110)
            self.assertFalse(step.truncated)

    def test_observation_at_decision_is_independent_of_future_interval_returns(self):
        original = generate_synthetic_market(
            SyntheticMarketConfig(sessions=90), seed=77
        )
        changed_assets = original.asset_returns.copy()
        changed_benchmark = original.benchmark_returns.copy()
        changed_vix = original.vix_levels.copy()
        changed_assets[63:, :] = 0.25
        changed_benchmark[63:] = -0.5
        changed_vix[63:] = 80.0
        changed_dataset = replace(
            original,
            asset_returns=changed_assets,
            benchmark_returns=changed_benchmark,
            vix_levels=changed_vix,
        )
        first_env = SyntheticPortfolioEnv(**original.environment_kwargs())
        second_env = SyntheticPortfolioEnv(**changed_dataset.environment_kwargs())

        before_first, _ = first_env.reset(seed=3)
        before_second, _ = second_env.reset(seed=3)
        np.testing.assert_array_equal(before_first, before_second)
        after_first, _, _, _, _ = first_env.step(ALL_CASH_ACTION)
        after_second, _, _, _, _ = second_env.step(ALL_CASH_ACTION)
        self.assertFalse(np.array_equal(after_first, after_second))

    def test_hand_calculated_costs_turnover_compounding_nav_weights_and_reward(self):
        dataset = deterministic_accounting_dataset()
        environment = SyntheticPortfolioEnv(
            **dataset.environment_kwargs(),
            period_sessions=2,
            max_asset_weight=1.0,
            min_cash_weight=0.0,
            max_turnover=2.0,
            transaction_cost_rate=0.001,
            slippage_bps=5.0,
        )
        observation, reset_info = environment.reset(seed=1)
        self.assertEqual(reset_info["nav"], 100_000.0)
        self.assertEqual(reset_info["weights"]["CASH"], 1.0)
        self.assertEqual(observation.shape, (43,))
        self.assertTrue(environment.observation_space.contains(observation))

        next_observation, reward, terminated, truncated, info = environment.step(
            BALANCED_ACTION
        )

        expected_post_cost_nav = 100_000.0 - 100_000.0 * (0.001 + 5.0 / 10_000)
        expected_growth = 0.5 * (1.1**2) + 0.5
        expected_nav = expected_post_cost_nav * expected_growth
        expected_spy_value = expected_post_cost_nav * 0.5 * (1.1**2)
        expected_cash_value = expected_post_cost_nav * 0.5
        self.assertAlmostEqual(info["pre_trade_nav"], 100_000.0, places=8)
        self.assertAlmostEqual(info["requested_weights"]["SPY"], 0.5, places=7)
        self.assertAlmostEqual(info["applied_weights"]["SPY"], 0.5, places=7)
        self.assertAlmostEqual(info["applied_weights"]["CASH"], 0.5, places=7)
        self.assertAlmostEqual(info["gross_turnover"], 1.0, places=7)
        self.assertAlmostEqual(info["one_way_turnover"], 0.5, places=7)
        self.assertAlmostEqual(info["transaction_fees"], 100.0, places=7)
        self.assertAlmostEqual(info["slippage_cost"], 50.0, places=7)
        self.assertAlmostEqual(info["transaction_cost"], 150.0, places=7)
        self.assertAlmostEqual(info["post_cost_nav"], expected_post_cost_nav, places=7)
        self.assertAlmostEqual(info["nav"], expected_nav, places=7)
        self.assertAlmostEqual(
            info["weights"]["SPY"], expected_spy_value / expected_nav, places=10
        )
        self.assertAlmostEqual(
            info["weights"]["CASH"], expected_cash_value / expected_nav, places=10
        )
        self.assertAlmostEqual(reward, np.log(expected_nav / 100_000.0), places=12)
        self.assertAlmostEqual(info["reward_log_nav_return"], reward, places=12)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(environment.observation_space.contains(next_observation))

    def test_train_validation_test_arrays_are_isolated_and_not_preprocessed(self):
        splits = generate_synthetic_splits(
            SyntheticMarketConfig(sessions=100), seed=901
        )
        datasets = (splits.train, splits.validation, splits.test)
        for first_index, first in enumerate(datasets):
            for second in datasets[first_index + 1 :]:
                self.assertFalse(np.shares_memory(first.asset_returns, second.asset_returns))
                self.assertFalse(
                    np.shares_memory(first.benchmark_returns, second.benchmark_returns)
                )
                self.assertFalse(np.shares_memory(first.vix_levels, second.vix_levels))
                self.assertNotEqual(first.seed, second.seed)
                self.assertEqual(
                    first.metadata["configuration_hash"],
                    second.metadata["configuration_hash"],
                )
                with self.assertRaises(ValueError):
                    first.asset_returns[0, 0] = 99.0
        self.assertFalse(hasattr(splits, "scaler"))
        self.assertFalse(hasattr(splits, "preprocessor"))

    def test_environment_owns_copy_of_mutable_inputs(self):
        dataset = generate_synthetic_market(
            SyntheticMarketConfig(sessions=80), seed=100
        )
        caller_assets = dataset.asset_returns.copy()
        caller_benchmark = dataset.benchmark_returns.copy()
        caller_vix = dataset.vix_levels.copy()
        environment = SyntheticPortfolioEnv(
            caller_assets,
            caller_benchmark,
            caller_vix,
            data_metadata=dataset.metadata,
        )
        expected = environment.asset_returns[0, 0]
        caller_assets[0, 0] = expected + 10.0
        self.assertEqual(environment.asset_returns[0, 0], expected)
        self.assertFalse(np.shares_memory(environment.asset_returns, caller_assets))

    def test_returned_metadata_is_deeply_isolated_between_calls(self):
        dataset = generate_synthetic_market(
            SyntheticMarketConfig(sessions=80), seed=101
        )
        environment = SyntheticPortfolioEnv(**dataset.environment_kwargs())
        _, reset_info = environment.reset(seed=1)
        reset_info["data_metadata"]["generation_config"]["sessions"] = -1
        self.assertEqual(environment.data_metadata["generation_config"]["sessions"], 80)

        _, _, _, _, step_info = environment.step(ALL_CASH_ACTION)
        step_info["data_metadata"]["generation_config"]["sessions"] = -2
        self.assertEqual(environment.data_metadata["generation_config"]["sessions"], 80)

    def test_same_splits_and_actions_reproduce_all_environment_outputs(self):
        config = SyntheticMarketConfig(sessions=110)
        first = generate_synthetic_splits(config, seed=2026)
        second = generate_synthetic_splits(config, seed=2026)
        actions = [BALANCED_ACTION, ALL_CASH_ACTION, BALANCED_ACTION]

        for split_name in ("train", "validation", "test"):
            first_data = getattr(first, split_name)
            second_data = getattr(second, split_name)
            np.testing.assert_array_equal(first_data.asset_returns, second_data.asset_returns)
            np.testing.assert_array_equal(
                first_data.benchmark_returns, second_data.benchmark_returns
            )
            np.testing.assert_array_equal(first_data.vix_levels, second_data.vix_levels)
            self.assertEqual(first_data.regime_labels, second_data.regime_labels)
            self.assertEqual(first_data.seed, second_data.seed)
            self.assertEqual(
                first_data.metadata["configuration_hash"],
                second_data.metadata["configuration_hash"],
            )
            self.assertNotEqual(
                first_data.metadata["generated_at_utc"],
                second_data.metadata["generated_at_utc"],
            )

            first_run = run_deterministic_actions(first_data, actions)
            second_run = run_deterministic_actions(second_data, actions)
            np.testing.assert_array_equal(
                first_run.initial_observation, second_run.initial_observation
            )
            self.assertEqual(first_run.seed, second_run.seed)
            self.assertEqual(len(first_run.steps), len(second_run.steps))
            for first_step, second_step in zip(first_run.steps, second_run.steps):
                np.testing.assert_array_equal(
                    first_step.observation, second_step.observation
                )
                self.assertEqual(first_step.reward, second_step.reward)
                self.assertEqual(first_step.info["nav"], second_step.info["nav"])
                self.assertEqual(first_step.info["weights"], second_step.info["weights"])
                self.assertEqual(first_step.terminated, second_step.terminated)
                self.assertEqual(first_step.truncated, second_step.truncated)


if __name__ == "__main__":
    unittest.main()
