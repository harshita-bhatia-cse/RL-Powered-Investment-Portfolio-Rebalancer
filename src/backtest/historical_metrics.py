"""Performance metrics calculated from realized environment NAV and costs."""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
import pandas as pd

TRADING_SESSIONS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.25


class HistoricalPerformanceMetrics(TypedDict):
    """Finite JSON-ready metrics; undefined ratios use ``None``."""

    initial_portfolio_value: float
    final_portfolio_value: float
    total_return: float
    annualized_return: float | None
    annualized_volatility: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    maximum_drawdown: float
    calmar_ratio: float | None
    turnover: float
    transaction_fees: float
    slippage_cost: float
    total_transaction_cost: float


def calculate_historical_metrics(
    nav_history: pd.Series,
    *,
    turnover: float,
    transaction_fees: float,
    slippage_cost: float,
) -> tuple[HistoricalPerformanceMetrics, pd.Series]:
    """Calculate performance from NAV marks and environment-reported costs.

    Annualized volatility, Sharpe, and Sortino use 252 trading sessions per
    year and a zero risk-free rate. CAGR uses the elapsed calendar days and
    365.25 days per year. Sortino downside deviation is the annualized RMS of
    negative daily returns relative to zero. Undefined ratios are returned as
    ``None`` rather than a fabricated zero.
    """
    if not isinstance(nav_history.index, pd.DatetimeIndex):
        raise ValueError("NAV history must have a DatetimeIndex")
    if nav_history.empty:
        raise ValueError("NAV history is required")
    if (
        nav_history.index.has_duplicates
        or not nav_history.index.is_monotonic_increasing
    ):
        raise ValueError("NAV dates must be unique and increasing")

    nav = pd.to_numeric(nav_history, errors="coerce").astype(float)
    values = nav.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("NAV values must be finite and positive")

    cost_values = (turnover, transaction_fees, slippage_cost)
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0 for value in cost_values
    ):
        raise ValueError(
            "Turnover and transaction costs must be finite and non-negative"
        )

    daily_returns = nav.pct_change(fill_method=None).iloc[1:]
    returns = daily_returns.to_numpy(dtype=np.float64)
    if not np.isfinite(returns).all():
        raise ValueError("NAV history produced non-finite daily returns")

    initial_value = float(values[0])
    final_value = float(values[-1])
    total_return = final_value / initial_value - 1.0
    if not math.isfinite(total_return):
        raise ValueError("Total return cannot be represented as a finite number")

    elapsed_days = (nav.index[-1] - nav.index[0]).total_seconds() / 86_400.0
    annualized_return: float | None = None
    if elapsed_days > 0.0:
        annualized_log_growth = (
            (math.log(final_value) - math.log(initial_value))
            * CALENDAR_DAYS_PER_YEAR
            / elapsed_days
        )
        if annualized_log_growth <= math.log(np.finfo(np.float64).max):
            annualized_return = float(math.expm1(annualized_log_growth))
            if not math.isfinite(annualized_return):
                annualized_return = None

    annualized_volatility: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    if len(returns) >= 2:
        daily_volatility = float(np.std(returns, ddof=1))
        annualized_volatility = daily_volatility * math.sqrt(TRADING_SESSIONS_PER_YEAR)
        if daily_volatility > 0.0:
            sharpe_ratio = (
                float(np.mean(returns))
                / daily_volatility
                * math.sqrt(TRADING_SESSIONS_PER_YEAR)
            )

        downside = np.minimum(returns, 0.0)
        downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
        if downside_deviation > 0.0:
            sortino_ratio = (
                float(np.mean(returns))
                / downside_deviation
                * math.sqrt(TRADING_SESSIONS_PER_YEAR)
            )

    drawdowns = values / np.maximum.accumulate(values) - 1.0
    maximum_drawdown = float(np.min(drawdowns))
    calmar_ratio = None
    if annualized_return is not None and maximum_drawdown < 0.0:
        candidate = annualized_return / abs(maximum_drawdown)
        if math.isfinite(candidate):
            calmar_ratio = float(candidate)

    total_cost = float(transaction_fees) + float(slippage_cost)
    metrics: HistoricalPerformanceMetrics = {
        "initial_portfolio_value": initial_value,
        "final_portfolio_value": final_value,
        "total_return": float(total_return),
        "annualized_return": annualized_return,
        "annualized_volatility": _finite_or_none(annualized_volatility),
        "sharpe_ratio": _finite_or_none(sharpe_ratio),
        "sortino_ratio": _finite_or_none(sortino_ratio),
        "maximum_drawdown": maximum_drawdown,
        "calmar_ratio": calmar_ratio,
        "turnover": float(turnover),
        "transaction_fees": float(transaction_fees),
        "slippage_cost": float(slippage_cost),
        "total_transaction_cost": total_cost,
    }
    return metrics, daily_returns


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)
