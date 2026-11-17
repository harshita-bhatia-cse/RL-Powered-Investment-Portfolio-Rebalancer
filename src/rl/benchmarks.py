"""Deterministic historical benchmark strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.risk_metrics import calculate_risk_metrics


def evaluate_benchmarks(
    prices: pd.DataFrame,
    initial_capital: float = 100_000.0,
    transaction_cost_rate: float = 0.001,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Evaluate buy-and-hold, equal-weight, monthly, and threshold strategies."""
    prices = prices.sort_index().astype(float).dropna()
    if prices.shape[1] < 2 or len(prices) < 2:
        raise ValueError("At least two assets and two dates are required")
    returns = prices.pct_change().dropna()
    n_assets = prices.shape[1]
    equal = np.full(n_assets, 1.0 / n_assets)
    schedules = {
        "SPY buy-and-hold": lambda i, weights: np.eye(1, n_assets, 0).reshape(-1),
        "Equal-weight": lambda i, weights: equal,
        "Monthly rebalancing": lambda i, weights: equal if prices.index[i].month != prices.index[i - 1].month else weights,
        "Threshold rebalancing": lambda i, weights: equal if np.max(np.abs(weights - equal)) >= threshold else weights,
    }
    results = []
    for name, target_fn in schedules.items():
        weights = np.zeros(n_assets)
        weights[0] = 1.0
        values = [initial_capital]
        daily = []
        for index, (_, row) in enumerate(returns.iterrows(), start=1):
            target = target_fn(index, weights)
            turnover = float(np.abs(target - weights).sum())
            cost = turnover * transaction_cost_rate
            net = float(np.dot(weights, row.to_numpy()) - cost)
            daily.append(net)
            values.append(values[-1] * (1.0 + net))
            weights = target
        series = pd.Series(daily, index=returns.index)
        risk = calculate_risk_metrics(series)
        years = max(len(series) / 252.0, 1 / 252.0)
        total = values[-1] / initial_capital - 1.0
        results.append({
            "benchmark": name,
            "total_return": total,
            "annualized_return": (1.0 + total) ** (1.0 / years) - 1.0,
            "annualized_volatility": risk["annualized_volatility"],
            "sharpe_ratio": risk["sharpe_ratio"],
            "maximum_drawdown": risk["maximum_drawdown"],
        })
    return pd.DataFrame(results).set_index("benchmark")
