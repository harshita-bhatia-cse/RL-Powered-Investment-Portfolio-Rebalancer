import unittest

import numpy as np
import pandas as pd

from src.analytics import (
    analyze_correlations,
    calculate_portfolio_metrics,
    calculate_risk_metrics,
    validate_weights,
)
from src.data.data_quality import calculate_price_quality


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2024-01-01", periods=5, freq="D")
        self.returns = pd.DataFrame(
            {"AAA": [0.01, 0.0, -0.01, 0.02, 0.01], "BBB": [0.0, 0.01, 0.0, -0.01, 0.02]},
            index=index,
        )

    def test_portfolio_value_and_return(self):
        result = calculate_portfolio_metrics(self.returns, {"AAA": 0.5, "BBB": 0.5}, 1000)
        expected = 1000 * (1 + self.returns.mean(axis=1)).prod()
        self.assertAlmostEqual(result["current_value"], expected)
        self.assertAlmostEqual(result["total_return"], expected / 1000 - 1)

    def test_risk_metrics_are_deterministic(self):
        result = calculate_risk_metrics(self.returns["AAA"])
        self.assertAlmostEqual(result["maximum_drawdown"], -0.01)
        self.assertLess(result["var"], 0.01)
        self.assertTrue(np.isfinite(result["annualized_volatility"]))

    def test_invalid_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_weights({"AAA": 0.6, "BBB": 0.5})

    def test_correlation_matrix(self):
        result = analyze_correlations(self.returns)
        self.assertEqual(result["matrix"].shape, (2, 2))
        self.assertEqual(result["matrix"].loc["AAA", "AAA"], 1.0)

    def test_quality_requires_real_rows(self):
        with self.assertRaises(ValueError):
            calculate_price_quality(pd.DataFrame())

    def test_quality_scores_use_price_rows(self):
        data = pd.DataFrame({
            "ticker": ["AAA", "AAA"],
            "date": ["2024-01-01", "2024-01-02"],
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "adj_close": [10.5, 11.5],
            "volume": [100, 120],
        })
        result = calculate_price_quality(data, pd.Timestamp("2024-01-03").to_pydatetime())
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["completeness"], 100.0)
        self.assertEqual(result["accuracy"], 100.0)


if __name__ == "__main__":
    unittest.main()
