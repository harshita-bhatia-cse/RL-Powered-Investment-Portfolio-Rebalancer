import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtest.historical_metrics import calculate_historical_metrics
from src.backtest.historical_service import (
    HistoricalBacktestService,
    HistoricalDataUnavailableError,
    UnsupportedStrategyError,
)
from src.rl.environment.portfolio_env import (
    MARKET_SYMBOLS,
    PortfolioTradingEnv,
    fit_training_scaler,
)

FIXTURE_DATES = pd.bdate_range("2017-01-03", "2026-11-11")


def build_market_fixture():
    """Create deterministic data that satisfies the fixed environment contract."""
    data = {}
    positions = np.arange(len(FIXTURE_DATES), dtype=np.float64)
    for symbol_index, symbol in enumerate(MARKET_SYMBOLS):
        baseline = 50.0 + symbol_index * 20.0
        close = (
            baseline
            + positions * (0.008 + symbol_index * 0.0005)
            + np.sin(positions / (13.0 + symbol_index)) * (0.4 + symbol_index * 0.02)
        )
        open_prices = close * (1.0 + 0.001 * np.sin(positions / 5.0))
        frame = pd.DataFrame(
            {
                "open": open_prices,
                "high": np.maximum(open_prices, close) * 1.002,
                "low": np.minimum(open_prices, close) * 0.998,
                "close": close,
                "adj_close": close * (1.0 + 0.0001 * np.cos(positions / 17.0)),
                "volume": np.full(len(positions), 1_000_000.0),
            },
            index=FIXTURE_DATES,
        )
        frame.index.name = "date"
        data[symbol] = frame
    return data


class HistoricalMetricsTests(unittest.TestCase):
    def test_metrics_use_nav_and_actual_cost_inputs(self):
        nav = pd.Series(
            [100.0, 110.0, 99.0, 108.9],
            index=pd.date_range("2024-01-02", periods=4, freq="B"),
        )

        metrics, daily_returns = calculate_historical_metrics(
            nav,
            turnover=0.75,
            transaction_fees=1.5,
            slippage_cost=0.5,
        )

        self.assertAlmostEqual(metrics["initial_portfolio_value"], 100.0)
        self.assertAlmostEqual(metrics["final_portfolio_value"], 108.9)
        self.assertAlmostEqual(metrics["total_return"], 0.089)
        self.assertAlmostEqual(metrics["maximum_drawdown"], -0.1)
        self.assertAlmostEqual(metrics["transaction_fees"], 1.5)
        self.assertAlmostEqual(metrics["slippage_cost"], 0.5)
        self.assertAlmostEqual(metrics["total_transaction_cost"], 2.0)
        self.assertEqual(len(daily_returns), 3)
        self.assertTrue(np.isfinite(daily_returns.to_numpy()).all())

    def test_zero_volatility_and_insufficient_observations_are_safe(self):
        constant_nav = pd.Series(
            [100.0, 100.0, 100.0],
            index=pd.date_range("2024-01-02", periods=3, freq="B"),
        )
        metrics, _ = calculate_historical_metrics(
            constant_nav,
            turnover=0.0,
            transaction_fees=0.0,
            slippage_cost=0.0,
        )

        self.assertEqual(metrics["annualized_volatility"], 0.0)
        self.assertIsNone(metrics["sharpe_ratio"])
        self.assertIsNone(metrics["sortino_ratio"])
        self.assertIsNone(metrics["calmar_ratio"])

        one_mark = constant_nav.iloc[:1]
        insufficient_metrics, returns = calculate_historical_metrics(
            one_mark,
            turnover=0.0,
            transaction_fees=0.0,
            slippage_cost=0.0,
        )
        self.assertEqual(len(returns), 0)
        self.assertIsNone(insufficient_metrics["annualized_volatility"])
        self.assertIsNone(insufficient_metrics["sharpe_ratio"])


class HistoricalBacktestServiceTests(unittest.TestCase):
    def setUp(self):
        self.market = build_market_fixture()
        self.service = HistoricalBacktestService(data_dir="unused-in-tests")

    def test_valid_backtest_returns_finite_nav_and_corporate_action_warning(self):
        with patch(
            "src.backtest.historical_service.load_market_data",
            return_value=self.market,
        ):
            result = self.service.run(
                strategy_name="equal_weight",
                partition="validation",
                initial_capital=25_000.0,
            )

        self.assertEqual(result["strategy"], "equal_weight")
        self.assertEqual(result["partition"], "validation")
        self.assertEqual(result["start_date"], "2022-01-03")
        self.assertEqual(result["end_date"], "2022-12-30")
        self.assertGreater(result["number_of_trading_sessions"], 2)
        self.assertEqual(
            len(result["nav_history"]), result["number_of_trading_sessions"]
        )
        self.assertEqual(
            len(result["daily_returns"]), result["number_of_trading_sessions"] - 1
        )
        self.assertTrue(
            np.isfinite([point["nav"] for point in result["nav_history"]]).all()
        )
        self.assertTrue(
            np.isfinite([point["value"] for point in result["daily_returns"]]).all()
        )
        self.assertFalse(result["corporate_actions_verified"])
        self.assertTrue(result["warnings"])
        self.assertIn("defaults transaction fees", " ".join(result["warnings"]))
        self.assertEqual(result["metrics"]["initial_portfolio_value"], 25_000.0)

    def test_invalid_strategy_and_partition_fail_before_loading_data(self):
        with patch(
            "src.backtest.historical_service.load_market_data",
            side_effect=AssertionError("data should not be loaded"),
        ):
            with self.assertRaises(UnsupportedStrategyError):
                self.service.run(strategy_name="unknown", partition="test")
            with self.assertRaisesRegex(ValueError, "partition"):
                self.service.run(strategy_name="equal_weight", partition="invalid")

    def test_missing_market_data_is_reported_as_unavailable(self):
        with patch(
            "src.backtest.historical_service.load_market_data",
            return_value={},
        ):
            with self.assertRaises(HistoricalDataUnavailableError):
                self.service.run(strategy_name="equal_weight", partition="test")

    def test_scaler_is_training_fitted_and_shared_for_comparison(self):
        changed_market = {symbol: frame.copy() for symbol, frame in self.market.items()}
        changed_market["SPY"].loc[
            changed_market["SPY"].index >= pd.Timestamp("2023-01-03"), "adj_close"
        ] *= 10.0
        baseline_scaler = fit_training_scaler(self.market)
        changed_scaler = fit_training_scaler(changed_market)
        self.assertEqual(changed_scaler.fitted_start, "2018-01-03")
        self.assertEqual(changed_scaler.fitted_end, "2021-12-31")
        np.testing.assert_array_equal(baseline_scaler.means, changed_scaler.means)
        np.testing.assert_array_equal(baseline_scaler.scales, changed_scaler.scales)

        captured_scalers = []
        original_env = PortfolioTradingEnv

        def capture_env(*args, **kwargs):
            captured_scalers.append(kwargs["scaler"])
            return original_env(*args, **kwargs)

        with patch(
            "src.backtest.historical_service.load_market_data",
            return_value=self.market,
        ), patch(
            "src.backtest.historical_service.PortfolioTradingEnv",
            side_effect=capture_env,
        ):
            result = self.service.compare(
                strategy_names=["equal_weight", "buy_and_hold"],
                partition="test",
                initial_capital=10_000.0,
            )

        self.assertEqual(len(captured_scalers), 2)
        self.assertIs(captured_scalers[0], captured_scalers[1])
        self.assertEqual(
            [item["strategy"] for item in result["results"]],
            ["equal_weight", "buy_and_hold"],
        )
        self.assertEqual(
            {item["initial_nav"] for item in result["results"]},
            {10_000.0},
        )
        self.assertTrue(
            all(
                item["corporate_actions_verified"] is False
                for item in result["results"]
            )
        )
        self.assertEqual(
            result["results"][0]["number_of_trading_sessions"],
            result["results"][1]["number_of_trading_sessions"],
        )

    def test_invalid_strategy_parameters_are_rejected(self):
        with self.assertRaises(UnsupportedStrategyError):
            self.service.run(
                strategy_name="equal_weight",
                partition="test",
                parameters={"lookback": 20},
            )


if __name__ == "__main__":
    unittest.main()
