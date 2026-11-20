import unittest

import numpy as np
import pandas as pd

from src.rl.environment.portfolio_env import (
    ASSET_UNIVERSE,
    BENCHMARK_SYMBOL,
    MARKET_SYMBOLS,
    OBSERVATION_SIZE,
    PARTITION_BOUNDS,
    TEST_END,
    PortfolioTradingEnv,
    fit_training_scaler,
    project_to_simplex,
)
from src.strategies.allocation import (
    CorporateActionEvent,
    CorporateActionType,
    PlanStatus,
    RebalanceConfig,
)


FIXTURE_DATES = pd.bdate_range("2017-01-03", "2026-11-11")


def build_market_fixture():
    """Create tiny controlled price fixtures; these are not market history."""
    data = {}
    for symbol in MARKET_SYMBOLS:
        close = np.full(len(FIXTURE_DATES), 100.0)
        frame = pd.DataFrame(
            {
                "open": close.copy(),
                "high": close.copy(),
                "low": close.copy(),
                "close": close.copy(),
                "adj_close": close.copy(),
                "volume": np.full(len(FIXTURE_DATES), 1000.0),
            },
            index=FIXTURE_DATES,
        )
        frame.index.name = "date"
        data[symbol] = frame
    return data


def action_for_target(weights, cash_weight, *, assets=6):
    """Create a vector whose simplex projection yields the requested weights."""
    desired = np.asarray([*weights, cash_weight], dtype=np.float64)
    if desired.shape != (assets + 1,):
        raise ValueError("unexpected test target size")
    return desired


class PortfolioEnvTests(unittest.TestCase):
    def setUp(self):
        self.market = build_market_fixture()
        self.scaler = fit_training_scaler(self.market)

    def make_env(self, **kwargs):
        return PortfolioTradingEnv(
            self.market,
            scaler=self.scaler,
            **kwargs,
        )

    def test_initialization_and_observation_contract(self):
        env = self.make_env(partition="train")
        observation, info = env.reset(seed=7)

        self.assertEqual(env.action_space.shape, (len(ASSET_UNIVERSE) + 1,))
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(tuple(info["holdings"]), ())
        self.assertFalse(info["corporate_actions_verified"])
        self.assertIn("not verified", env.metadata["return_accuracy"])
        self.assertIn("not verified", info["return_accuracy"])
        self.assertIn("reset-date close", info["starting_state_assumption"])

    def test_action_bounds_and_simplex_projection(self):
        env = self.make_env(partition="train")
        action = np.asarray([-1, -1, -1, -1, -1, -1, 1], dtype=np.float32)

        weights = project_to_simplex(action)

        self.assertTrue(env.action_space.contains(action))
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(float(weights[-1]), 1.0)

    def test_simplex_projection_is_deterministic_for_extreme_finite_actions(self):
        action = np.asarray([10.0, -50.0, 3.0, 0.0, -2.0, 1.0, 4.0])

        first = project_to_simplex(action)
        second = project_to_simplex(action)

        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first >= 0.0))
        self.assertAlmostEqual(float(first.sum()), 1.0)

    def test_cash_allocation_creates_no_trades(self):
        env = self.make_env(partition="train", initial_cash=1000.0)
        env.reset(seed=1)
        action = np.asarray([-1, -1, -1, -1, -1, -1, 1], dtype=np.float32)

        _, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(info["trades"], ())
        self.assertEqual(info["settled_cash"], 1000.0)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_zero_cash_is_allowed_when_initial_holdings_have_positive_nav(self):
        env = self.make_env(
            partition="train",
            initial_cash=0.0,
            initial_holdings={"SPY": 10.0},
        )

        observation, info = env.reset()

        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["settled_cash"], 0.0)
        self.assertEqual(info["nav"], 1000.0)

    def test_zero_cash_and_no_holdings_rejects_non_positive_initial_nav(self):
        env = self.make_env(partition="train", initial_cash=0.0)

        with self.assertRaisesRegex(ValueError, "Initial portfolio NAV"):
            env.reset()

    def test_reset_with_seed_reproduces_random_start(self):
        env = self.make_env(partition="train", random_start=True)
        first_observation, first_info = env.reset(seed=29)
        second_observation, second_info = env.reset(seed=29)

        np.testing.assert_array_equal(first_observation, second_observation)
        self.assertEqual(first_info["date"], second_info["date"])

    def test_execution_occurs_at_next_session_open_and_marks_at_close(self):
        start_day = pd.Timestamp("2021-12-30")
        next_day = self.market["SPY"].index[
            self.market["SPY"].index.get_loc(start_day) + 1
        ]
        self.market["SPY"].loc[next_day, "open"] = 100.0
        self.market["SPY"].loc[next_day, "high"] = 200.0
        self.market["SPY"].loc[next_day, "close"] = 200.0
        env = self.make_env(partition="train", initial_cash=1000.0)
        env.reset(options={"start_date": start_day})
        action = action_for_target([1, 0, 0, 0, 0, 0], 0)

        _, _, _, _, info = env.step(action)

        self.assertEqual(info["previous_date"], "2021-12-30")
        self.assertEqual(info["date"], next_day.date().isoformat())
        self.assertEqual(info["trades"][0].reference_price, 100.0)
        self.assertEqual(info["trades"][0].execution_price, 100.0)
        self.assertEqual(info["holdings"]["SPY"], 10.0)
        self.assertEqual(info["nav"], 2000.0)

    def test_allocator_uses_separate_execution_and_mark_prices(self):
        next_day = self.market["SPY"].index[
            self.market["SPY"].index.get_loc(pd.Timestamp("2018-01-03")) + 1
        ]
        self.market["SPY"].loc[next_day, ["open", "high", "close"]] = [
            100.0,
            110.0,
            110.0,
        ]
        env = self.make_env(partition="train", initial_cash=1000.0)
        env.reset(options={"start_date": "2018-01-03"})
        action = action_for_target([0.5, 0, 0, 0, 0, 0], 0.5)

        _, _, _, _, info = env.step(action)

        self.assertEqual(info["trades"][0].reference_price, 100.0)
        self.assertEqual(info["trades"][0].execution_price, 100.0)
        self.assertAlmostEqual(info["nav"], 1050.0)
        self.assertAlmostEqual(info["holdings"]["SPY"], 5.0)

    def test_transaction_costs_and_slippage_are_applied_once_in_nav(self):
        next_day = self.market["SPY"].index[
            self.market["SPY"].index.get_loc(pd.Timestamp("2018-01-03")) + 1
        ]
        self.market["SPY"].loc[next_day, ["open", "high", "close"]] = [
            100.0,
            101.0,
            100.0,
        ]
        env = self.make_env(
            partition="train",
            initial_cash=1000.0,
            allocation_config=RebalanceConfig(
                transaction_cost_rate=0.01,
                slippage_bps=100,
            ),
        )
        env.reset(options={"start_date": "2018-01-03"})
        action = action_for_target([0.5, 0, 0, 0, 0, 0], 0.5)

        _, reward, _, _, info = env.step(action)

        self.assertAlmostEqual(info["transaction_fees"], 5.05)
        self.assertAlmostEqual(info["slippage_cost"], 5.0)
        self.assertAlmostEqual(info["nav"], 989.95)
        self.assertAlmostEqual(reward, np.log(989.95 / 1000.0))

    def test_reward_is_logarithmic_nav_change(self):
        next_day = self.market["SPY"].index[
            self.market["SPY"].index.get_loc(pd.Timestamp("2018-01-03")) + 1
        ]
        self.market["SPY"].loc[next_day, ["open", "high", "close"]] = [
            100.0,
            120.0,
            120.0,
        ]
        env = self.make_env(partition="train", initial_cash=1000.0)
        env.reset(options={"start_date": "2018-01-03"})
        action = action_for_target([0.5, 0, 0, 0, 0, 0], 0.5)

        _, reward, _, _, info = env.step(action)

        self.assertAlmostEqual(reward, np.log(info["nav"] / 1000.0))

    def test_split_is_applied_before_rebalance_and_nav_is_continuous(self):
        event = CorporateActionEvent(
            symbol="SPY",
            event_type=CorporateActionType.SPLIT,
            source_id="fixture-split",
            effective_date="2018-01-04",
            split_ratio=2.0,
        )
        previous_day = pd.Timestamp("2018-01-03")
        next_day = pd.Timestamp("2018-01-04")
        self.market["SPY"].loc[previous_day, ["open", "high", "low", "close", "adj_close"]] = [
            100,
            100,
            100,
            100,
            100,
        ]
        self.market["SPY"].loc[next_day, ["open", "high", "low", "close", "adj_close"]] = [
            50,
            50,
            50,
            50,
            50,
        ]
        env = self.make_env(
            partition="train",
            initial_cash=0.0,
            initial_holdings={"SPY": 2.0},
            corporate_action_events=[event],
            allocation_config=RebalanceConfig(max_turnover=0.0),
        )
        env.reset(options={"start_date": previous_day})
        action = action_for_target([1, 0, 0, 0, 0, 0], 0)

        _, _, _, _, info = env.step(action)

        self.assertEqual(info["holdings"]["SPY"], 4.0)
        self.assertAlmostEqual(info["nav"], 200.0)
        self.assertEqual(info["corporate_actions_verified"], False)

    def test_step_hold_marks_existing_holdings_at_moved_market_prices(self):
        previous_day = pd.Timestamp("2018-01-03")
        next_day = pd.Timestamp("2018-01-04")
        self.market["SPY"].loc[
            next_day, ["open", "high", "low", "close", "adj_close"]
        ] = [100.0, 120.0, 100.0, 120.0, 120.0]
        env = self.make_env(
            partition="train",
            initial_cash=0.0,
            initial_holdings={"SPY": 10.0},
        )
        _, initial_info = env.reset(options={"start_date": previous_day})
        previous_nav = initial_info["nav"]
        previous_quantity = initial_info["holdings"]["SPY"]

        _, reward, _, _, info = env.step_hold()

        self.assertEqual(info["holdings"]["SPY"], previous_quantity)
        self.assertEqual(info["nav"], 10.0 * 120.0)
        self.assertGreater(info["nav"], previous_nav)
        self.assertAlmostEqual(reward, np.log(info["nav"] / previous_nav))
        self.assertEqual(info["trades"], ())
        self.assertEqual(info["turnover_ratio"], 0.0)
        self.assertEqual(info["transaction_fees"], 0.0)
        self.assertEqual(info["slippage_cost"], 0.0)

    def test_step_hold_applies_split_without_voluntary_rebalance(self):
        event = CorporateActionEvent(
            symbol="SPY",
            event_type=CorporateActionType.SPLIT,
            source_id="fixture-hold-split",
            effective_date="2018-01-04",
            split_ratio=2.0,
        )
        previous_day = pd.Timestamp("2018-01-03")
        next_day = pd.Timestamp("2018-01-04")
        self.market["SPY"].loc[
            next_day, ["open", "high", "low", "close", "adj_close"]
        ] = [50.0, 50.0, 50.0, 50.0, 50.0]
        env = self.make_env(
            partition="train",
            initial_cash=0.0,
            initial_holdings={"SPY": 2.0},
            corporate_action_events=[event],
        )
        _, initial_info = env.reset(options={"start_date": previous_day})
        previous_nav = initial_info["nav"]

        _, reward, _, _, info = env.step_hold()

        self.assertEqual(info["holdings"]["SPY"], 4.0)
        self.assertAlmostEqual(info["nav"], previous_nav)
        self.assertAlmostEqual(reward, np.log(info["nav"] / previous_nav))
        self.assertEqual(info["trades"], ())
        self.assertEqual(info["turnover_ratio"], 0.0)
        self.assertEqual(info["transaction_fees"], 0.0)
        self.assertEqual(info["slippage_cost"], 0.0)

    def test_step_hold_records_dividend_receivable_in_nav_without_trades(self):
        event = CorporateActionEvent(
            symbol="SPY",
            event_type=CorporateActionType.CASH_DIVIDEND,
            source_id="fixture-hold-dividend",
            ex_date="2018-01-04",
            payment_date="2018-01-10",
            distribution_per_share=2.0,
        )
        previous_day = pd.Timestamp("2018-01-03")
        next_day = pd.Timestamp("2018-01-04")
        self.market["SPY"].loc[
            next_day, ["open", "high", "low", "close", "adj_close"]
        ] = [100.0, 100.0, 98.0, 98.0, 98.0]
        env = self.make_env(
            partition="train",
            initial_cash=0.0,
            initial_holdings={"SPY": 10.0},
            corporate_action_events=[event],
        )
        _, initial_info = env.reset(options={"start_date": previous_day})

        _, _, _, _, info = env.step_hold()

        self.assertEqual(
            info["dividend_receivables"], {"SPY:fixture-hold-dividend": 20.0}
        )
        self.assertEqual(info["nav"], 10.0 * 98.0 + 20.0)
        self.assertAlmostEqual(info["nav"], initial_info["nav"])
        self.assertEqual(info["trades"], ())
        self.assertEqual(info["turnover_ratio"], 0.0)
        self.assertEqual(info["transaction_fees"], 0.0)
        self.assertEqual(info["slippage_cost"], 0.0)

    def test_dividend_receivable_is_in_nav_but_not_available_to_allocator(self):
        event = CorporateActionEvent(
            symbol="SPY",
            event_type=CorporateActionType.CASH_DIVIDEND,
            source_id="fixture-dividend",
            ex_date="2018-01-04",
            payment_date="2018-01-10",
            distribution_per_share=2.0,
        )
        env = self.make_env(
            partition="train",
            initial_cash=0.0,
            initial_holdings={"SPY": 10.0},
            corporate_action_events=[event],
            allocation_config=RebalanceConfig(max_turnover=0.0),
        )
        env.reset(options={"start_date": "2018-01-03"})
        action = action_for_target([0, 1, 0, 0, 0, 0], 0)

        _, _, _, _, info = env.step(action)

        self.assertEqual(info["settled_cash"], 0.0)
        self.assertEqual(info["allocator_available_cash"], 0.0)
        self.assertEqual(info["dividend_receivables"], {"SPY:fixture-dividend": 20.0})
        self.assertEqual(info["nav"], 1020.0)
        self.assertEqual(info["allocation_status"], PlanStatus.PARTIALLY_CONSTRAINED.value)

    def test_rejected_plan_is_reported_and_not_applied(self):
        env = self.make_env(
            partition="train",
            initial_cash=0.0,
            initial_holdings={"SPY": 10.0},
            allocation_config=RebalanceConfig(
                cash_reserve_pct=0.5,
                max_turnover=0.0,
            ),
        )
        env.reset(options={"start_date": "2018-01-03"})
        action = action_for_target([0, 0, 0, 0, 0, 0], 1)

        _, _, _, _, info = env.step(action)

        self.assertTrue(info["allocation_rejected"])
        self.assertEqual(info["allocation_status"], PlanStatus.REJECTED.value)
        self.assertEqual(info["trades"], ())
        self.assertEqual(info["holdings"]["SPY"], 10.0)

    def test_invalid_actions_are_rejected_or_clipped_explicitly(self):
        env = self.make_env(partition="train")
        env.reset(seed=1)
        with self.assertRaisesRegex(ValueError, "shape"):
            env.step(np.zeros(2, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "finite"):
            env.step(np.full(len(ASSET_UNIVERSE) + 1, np.nan, dtype=np.float32))

        out_of_bounds = np.full(len(ASSET_UNIVERSE) + 1, 2.0, dtype=np.float32)
        _, _, _, _, info = env.step(out_of_bounds)
        self.assertTrue(info["action_was_clipped"])
        self.assertAlmostEqual(
            sum(info["target_weights"].values()) + info["target_cash_weight"],
            1.0,
        )

    def test_step_terminates_at_partition_end_without_crossing_boundary(self):
        env = self.make_env(partition="train")
        last_start = env.partition_dates[-2]
        env.reset(options={"start_date": last_start})
        action = action_for_target([0, 0, 0, 0, 0, 0], 1)

        _, _, terminated, truncated, info = env.step(action)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertLessEqual(pd.Timestamp(info["date"]), PARTITION_BOUNDS["train"][1])
        self.assertEqual(info["date"], env.partition_dates[-1].date().isoformat())
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step(action)

    def test_step_hold_marks_existing_holdings_without_rebalancing(self):
        env = self.make_env(
            partition="train",
            initial_cash=1_000.0,
            initial_holdings={"SPY": 10.0},
        )
        _, initial_info = env.reset(seed=19)

        observation, reward, terminated, truncated, info = env.step_hold()

        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["allocation_status"], "held")
        self.assertFalse(info["allocation_rejected"])
        self.assertEqual(info["turnover_ratio"], 0.0)
        self.assertEqual(info["transaction_fees"], 0.0)
        self.assertEqual(info["slippage_cost"], 0.0)
        self.assertEqual(info["trades"], ())
        self.assertEqual(info["turnover_ratio"], 0.0)
        self.assertEqual(info["holdings"], initial_info["holdings"])
        self.assertEqual(info["nav"], initial_info["nav"])

    def test_validation_and_test_require_training_scaler(self):
        for partition in ("validation", "test"):
            with self.subTest(partition=partition), self.assertRaisesRegex(
                ValueError, "training-fitted scaler"
            ):
                PortfolioTradingEnv(self.market, partition=partition)

    def test_feature_scaler_uses_only_training_rows(self):
        baseline_means = np.asarray(self.scaler.means)
        self.market["SPY"].loc[TEST_END, "adj_close"] = 1_000_000.0

        scaler = fit_training_scaler(self.market)
        env = PortfolioTradingEnv(self.market, partition="test", scaler=scaler)

        self.assertEqual(scaler.fitted_start, "2018-01-03")
        self.assertEqual(scaler.fitted_end, "2021-12-31")
        self.assertEqual(env.partition_dates[-1], TEST_END)
        np.testing.assert_array_equal(np.asarray(scaler.means), baseline_means)
        self.assertEqual(
            float(self.market["SPY"].loc[TEST_END, "adj_close"]), 1_000_000.0
        )

    def test_future_price_change_does_not_change_prior_features(self):
        baseline = build_market_fixture()
        changed = build_market_fixture()
        decision_date = pd.Timestamp("2019-06-03")
        future_date = decision_date + pd.offsets.BDay(1)
        changed["SPY"].loc[future_date, "adj_close"] = 500.0

        from src.rl.environment.portfolio_env import _build_market_features, _normalize_market_data

        baseline_frames, baseline_dates = _normalize_market_data(baseline)
        changed_frames, changed_dates = _normalize_market_data(changed)
        baseline_features = _build_market_features(baseline_frames, baseline_dates)
        changed_features = _build_market_features(changed_frames, changed_dates)

        np.testing.assert_array_equal(
            baseline_features.loc[decision_date].to_numpy(),
            changed_features.loc[decision_date].to_numpy(),
        )

    def test_partition_definitions_are_chronological_and_non_overlapping(self):
        bounds = [PARTITION_BOUNDS[name] for name in ("train", "validation", "test")]
        self.assertLess(bounds[0][1], bounds[1][0])
        self.assertLess(bounds[1][1], bounds[2][0])
        self.assertEqual(bounds[2][1], TEST_END)

    def test_missing_market_rows_are_rejected_not_forward_filled(self):
        incomplete = build_market_fixture()
        incomplete["XLV"] = incomplete["XLV"].drop(pd.Timestamp("2019-05-01"))

        with self.assertRaisesRegex(ValueError, "Missing executable/feature rows"):
            PortfolioTradingEnv(
                incomplete,
                partition="train",
                scaler=self.scaler,
            )

    def test_verified_event_metadata_requires_explicit_source(self):
        with self.assertRaisesRegex(ValueError, "corporate_action_source"):
            self.make_env(
                partition="train",
                corporate_actions_verified=True,
            )


if __name__ == "__main__":
    unittest.main()
