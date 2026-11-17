"""Portfolio performance calculations from historical adjusted prices."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a date-indexed positive price frame into daily simple returns."""
    if prices.empty:
        raise ValueError("Historical prices are required")
    prices = prices.sort_index().astype(float)
    if prices.isna().all().any() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite and strictly positive")
    return prices.pct_change().dropna(how="all").replace([np.inf, -np.inf], np.nan).dropna(how="all")


def validate_returns(returns: pd.Series) -> pd.Series:
    """Validate and normalize a daily return series."""
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty or not np.isfinite(values).all():
        raise ValueError("At least one finite return is required")
    return values


def calculate_portfolio_metrics(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    initial_investment: float = 100000.0,
    periods_per_year: int = 252,
) -> dict:
    """Calculate value and return series for a fixed-weight portfolio."""
    if initial_investment <= 0:
        raise ValueError("Initial investment must be positive")
    if returns.empty:
        raise ValueError("Historical returns are required")
    selected = returns.loc[:, [asset for asset in weights if asset in returns.columns]]
    if selected.empty:
        raise ValueError("No requested assets have historical returns")
    weight_series = pd.Series(weights, dtype=float).reindex(selected.columns).fillna(0.0)
    if (weight_series < 0).any() or not np.isclose(weight_series.sum(), 1.0):
        raise ValueError("Portfolio weights must be non-negative and sum to 1")
    daily_returns = selected.mul(weight_series, axis=1).sum(axis=1)
    cumulative_returns = (1.0 + daily_returns).cumprod() - 1.0
    values = initial_investment * (1.0 + daily_returns).cumprod()
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1 / 365.25)
    total_return = float(values.iloc[-1] / initial_investment - 1.0)
    annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    return {
        "initial_investment": float(initial_investment),
        "current_value": float(values.iloc[-1]),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "average_daily_return": float(daily_returns.mean()),
        "daily_returns": daily_returns,
        "cumulative_returns": cumulative_returns,
        "portfolio_values": values,
    }
