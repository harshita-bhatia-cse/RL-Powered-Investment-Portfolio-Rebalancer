import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.analytics_api import app
from src.rl.environment.portfolio_env import MARKET_SYMBOLS

FIXTURE_DATES = pd.bdate_range("2017-01-03", "2026-11-11")


def build_market_fixture():
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


class HistoricalBacktestAPITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.market = build_market_fixture()

    def test_historical_endpoint_uses_schema_and_does_not_access_sqlite(self):
        with patch(
            "src.backtest.historical_service.load_market_data",
            return_value=self.market,
        ), patch(
            "src.api.analytics_api.get_data_store",
            side_effect=AssertionError("historical CSV route must not use SQLite"),
        ):
            response = self.client.post(
                "/api/backtest/historical",
                json={
                    "strategy": "buy_and_hold",
                    "partition": "validation",
                    "initial_capital": 50_000.0,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["strategy"], "buy_and_hold")
        self.assertEqual(body["number_of_trading_sessions"], len(body["nav_history"]))
        self.assertEqual(
            body["number_of_trading_sessions"] - 1, len(body["daily_returns"])
        )
        self.assertEqual(body["corporate_actions_verified"], False)
        self.assertIn("total-return accurate", " ".join(body["warnings"]))
        self.assertIn("defaults transaction fees", " ".join(body["warnings"]))
        self.assertTrue(
            np.isfinite([point["nav"] for point in body["nav_history"]]).all()
        )
        self.assertIn("annualized_volatility", body["metrics"])
        self.assertIn("transaction_fees", body["transaction_cost_summary"])

    def test_invalid_strategy_and_partition_return_client_errors(self):
        unknown_strategy = self.client.post(
            "/api/backtest/historical",
            json={"strategy": "min_variance", "partition": "test"},
        )
        invalid_partition = self.client.post(
            "/api/backtest/historical",
            json={"strategy": "equal_weight", "partition": "holdout"},
        )

        self.assertEqual(unknown_strategy.status_code, 422)
        self.assertIn("Unsupported strategy", unknown_strategy.json()["detail"])
        self.assertEqual(invalid_partition.status_code, 422)

    def test_missing_or_invalid_dataset_returns_service_unavailable(self):
        with patch(
            "src.backtest.historical_service.load_market_data",
            return_value={},
        ):
            response = self.client.post(
                "/api/backtest/historical",
                json={"strategy": "equal_weight", "partition": "test"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("Historical CSV data", response.json()["detail"])

    def test_comparison_and_health_endpoints(self):
        with patch(
            "src.backtest.historical_service.load_market_data",
            return_value=self.market,
        ):
            comparison = self.client.post(
                "/api/backtest/compare",
                json={
                    "strategies": ["equal_weight", "buy_and_hold"],
                    "partition": "validation",
                    "initial_capital": 12_500.0,
                },
            )
            health = self.client.get("/api/backtest/health")

        self.assertEqual(comparison.status_code, 200, comparison.text)
        comparison_body = comparison.json()
        self.assertEqual(comparison_body["initial_capital"], 12_500.0)
        self.assertEqual(len(comparison_body["results"]), 2)
        self.assertEqual(
            {result["initial_nav"] for result in comparison_body["results"]},
            {12_500.0},
        )
        self.assertEqual(health.status_code, 200, health.text)
        self.assertTrue(health.json()["available"])
        self.assertTrue(health.json()["dataset_validation"]["valid"])

    def test_health_returns_structured_503_for_unexpected_loader_error(self):
        client = TestClient(app, raise_server_exceptions=False)
        with self.assertLogs("src.api.analytics_api", level="ERROR") as captured:
            with patch(
                "src.backtest.historical_service.load_market_data",
                side_effect=RuntimeError("unexpected loader failure"),
            ):
                response = client.get("/api/backtest/health")

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.headers["content-type"], "application/json")
        body = response.json()
        self.assertIsInstance(body, dict)
        self.assertEqual(body["detail"], "Historical backtest health check failed")
        self.assertTrue(
            any("unexpected loader failure" in message for message in captured.output)
        )

    def test_compare_rejects_duplicate_strategy_names(self):
        response = self.client.post(
            "/api/backtest/compare",
            json={
                "strategies": ["equal_weight", "equal_weight"],
                "partition": "validation",
            },
        )

        self.assertEqual(response.status_code, 422, response.text)
        self.assertTrue(response.json()["detail"])

    def test_compare_rejects_parameters_for_unrequested_strategy(self):
        response = self.client.post(
            "/api/backtest/compare",
            json={
                "strategies": ["equal_weight"],
                "partition": "validation",
                "parameters": {"buy_and_hold": {"unused_parameter": 1.0}},
            },
        )

        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("Parameters supplied for unrequested strategies", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
