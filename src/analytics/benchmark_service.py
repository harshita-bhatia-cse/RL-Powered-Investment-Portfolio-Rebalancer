"""Benchmark comparison using aligned historical return series."""

from __future__ import annotations

import pandas as pd

from .portfolio_metrics import calculate_portfolio_metrics
from .risk_metrics import calculate_risk_metrics


def compare_benchmarks(
    portfolio_returns: pd.Series,
    benchmark_returns: dict[str, pd.Series],
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compare portfolio and benchmark return/risk statistics."""
    rows = []
    series = {"Portfolio": portfolio_returns, **benchmark_returns}
    for name, values in series.items():
        values = values.dropna()
        if values.empty:
            continue
        wealth = (1.0 + values).cumprod()
        total_return = float(wealth.iloc[-1] - 1.0)
        annualized = float((1.0 + total_return) ** (periods_per_year / len(values)) - 1.0)
        risk = calculate_risk_metrics(values, periods_per_year=periods_per_year)
        rows.append({
            "benchmark": name,
            "total_return": total_return,
            "annualized_return": annualized,
            "annualized_volatility": risk["annualized_volatility"],
            "sharpe_ratio": risk["sharpe_ratio"],
            "maximum_drawdown": risk["maximum_drawdown"],
        })
    return pd.DataFrame(rows).set_index("benchmark")
