import unittest

import pandas as pd

from src.strategies.run_adaptive_rotation_strategy import _simulate_daily_equity


class DailyEquitySimulationTests(unittest.TestCase):
    def test_orders_execute_next_session_and_include_turnover_costs(self):
        dates = pd.to_datetime(
            ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"]
        )
        prices = pd.DataFrame(
            {
                "A": [100.0, 100.0, 110.0, 110.0, 110.0],
                "B": [100.0, 100.0, 100.0, 100.0, 110.0],
                "SPY": [100.0, 100.0, 101.0, 102.0, 103.0],
            },
            index=dates,
        )
        events = [
            {"date": pd.Timestamp("2024-01-05"), "weights": {"A": 1.0}},
            {"date": pd.Timestamp("2024-01-09"), "weights": {"B": 1.0}},
        ]

        equity = _simulate_daily_equity(prices, events, transaction_cost_bps=10.0)

        self.assertAlmostEqual(equity.loc["2024-01-08", "portfolio"], 0.999)
        self.assertAlmostEqual(equity.loc["2024-01-09", "portfolio"], 1.0989)
        self.assertAlmostEqual(equity.loc["2024-01-10", "portfolio"], 1.0989 * 0.998)
        self.assertAlmostEqual(equity.loc["2024-01-11", "portfolio"], 1.0989 * 0.998 * 1.1)
        self.assertAlmostEqual(equity.loc["2024-01-08", "turnover"], 1.0)
        self.assertAlmostEqual(equity.loc["2024-01-10", "turnover"], 2.0)

    def test_negative_transaction_cost_is_rejected(self):
        prices = pd.DataFrame(
            {"A": [100.0, 101.0]},
            index=pd.to_datetime(["2024-01-05", "2024-01-08"]),
        )
        with self.assertRaises(ValueError):
            _simulate_daily_equity(prices, [], transaction_cost_bps=-1.0)


if __name__ == "__main__":
    unittest.main()
