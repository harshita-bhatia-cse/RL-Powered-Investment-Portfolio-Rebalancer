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


def evaluate_synthetic_benchmarks(
    asset_returns: np.ndarray,
    *,
    initial_capital: float = 100_000.0,
    transaction_cost_rate: float = 0.001,
    slippage_bps: float = 5.0,
    period_sessions: int = 21,
    warmup_sessions: int = 63,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Evaluate four benchmark strategies over the synthetic evaluation window.

    The window starts after the environment warm-up. Rebalancing is applied at
    interval boundaries only, before returns in that interval are consumed.
    """
    returns = np.asarray(asset_returns, dtype=np.float64)
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError("asset_returns must be a two-dimensional asset return matrix")
    if returns.shape[0] <= warmup_sessions:
        raise ValueError("insufficient synthetic data after the warm-up period")
    if not np.isfinite(returns).all() or np.any(returns <= -1.0):
        raise ValueError("asset_returns must contain finite simple returns")
    if (
        isinstance(initial_capital, bool)
        or not np.isfinite(initial_capital)
        or initial_capital <= 0
    ):
        raise ValueError("initial_capital must be positive and finite")
    if (
        isinstance(transaction_cost_rate, bool)
        or not np.isfinite(transaction_cost_rate)
        or transaction_cost_rate < 0
        or transaction_cost_rate >= 1.0
    ):
        raise ValueError("transaction_cost_rate must be in [0, 1)")
    if (
        isinstance(slippage_bps, bool)
        or not np.isfinite(slippage_bps)
        or slippage_bps < 0
        or slippage_bps >= 10_000
    ):
        raise ValueError("slippage_bps must be in [0, 10000)")
    if (
        isinstance(period_sessions, bool)
        or not isinstance(period_sessions, int)
        or period_sessions <= 0
    ):
        raise ValueError("period_sessions must be a positive integer")
    if (
        isinstance(warmup_sessions, bool)
        or not isinstance(warmup_sessions, int)
        or warmup_sessions < 0
    ):
        raise ValueError("warmup_sessions must be a non-negative integer")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")

    evaluation_returns = returns[warmup_sessions:]
    n_assets = returns.shape[1]
    equal = np.append(
        np.full(n_assets, 1.0 / n_assets, dtype=np.float64),
        0.0,
    )
    spy_buy_and_hold = np.zeros(n_assets + 1, dtype=np.float64)
    spy_buy_and_hold[0] = 1.0
    cost_rate = float(transaction_cost_rate) + float(slippage_bps) / 10_000.0
    strategies = (
        ("SPY buy-and-hold", "buy_hold"),
        ("Equal-weight", "equal"),
        ("Monthly rebalancing", "monthly"),
        ("Threshold rebalancing", "threshold"),
    )

    results: list[dict[str, float | str]] = []
    for name, strategy in strategies:
        weights = np.zeros(n_assets + 1, dtype=np.float64)
        weights[-1] = 1.0
        nav = float(initial_capital)
        nav_history = [nav]
        total_turnover = 0.0
        total_cost = 0.0

        for day_index, daily_asset_returns in enumerate(evaluation_returns):
            if strategy == "buy_hold":
                target = spy_buy_and_hold if day_index == 0 else weights
            elif strategy == "equal":
                target = equal
            elif strategy == "monthly" and day_index % period_sessions == 0:
                target = equal
            elif (
                strategy == "threshold"
                and np.max(np.abs(weights - equal)) >= threshold
            ):
                target = equal
            else:
                target = weights
            gross_turnover = float(np.abs(target - weights).sum())
            cost = nav * gross_turnover * cost_rate
            nav -= cost
            total_turnover += gross_turnover / 2.0
            total_cost += cost
            weights = target.copy()

            daily_returns_with_cash = np.append(daily_asset_returns, 0.0)
            asset_growth = weights * (1.0 + daily_returns_with_cash)
            daily_growth = float(asset_growth.sum())
            if not np.isfinite(daily_growth) or daily_growth <= 0.0:
                raise FloatingPointError("benchmark returns produced invalid NAV")
            nav *= daily_growth
            weights = asset_growth / daily_growth
            nav_history.append(float(nav))

        metrics = calculate_daily_nav_metrics(
            nav_history,
            initial_capital=float(initial_capital),
            total_transaction_costs=total_cost,
            portfolio_turnover=total_turnover,
        )
        results.append({"benchmark": name, **metrics})
    return pd.DataFrame(results).set_index("benchmark")


def calculate_daily_nav_metrics(
    nav_history: list[float] | np.ndarray,
    *,
    initial_capital: float,
    total_transaction_costs: float = 0.0,
    portfolio_turnover: float = 0.0,
) -> dict[str, float]:
    """Calculate daily metrics; Sharpe/Sortino assume a zero risk-free rate."""
    nav = np.asarray(nav_history, dtype=np.float64).reshape(-1)
    if (
        len(nav) == 0
        or not np.isfinite(nav).all()
        or np.any(nav <= 0.0)
        or isinstance(initial_capital, bool)
        or not np.isfinite(initial_capital)
        or initial_capital <= 0.0
    ):
        raise ValueError("nav_history and initial_capital must be positive and finite")
    if (
        not np.isfinite(total_transaction_costs)
        or total_transaction_costs < 0.0
        or not np.isfinite(portfolio_turnover)
        or portfolio_turnover < 0.0
    ):
        raise ValueError("transaction costs and portfolio turnover must be finite and non-negative")

    full_nav = np.concatenate(([float(initial_capital)], nav))
    returns = full_nav[1:] / full_nav[:-1] - 1.0
    cumulative_return = float(full_nav[-1] / float(initial_capital) - 1.0)
    annualized_return = (1.0 + cumulative_return) ** (
        TRADING_DAYS_PER_YEAR / len(returns)
    ) - 1.0
    daily_volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = (
        float(np.mean(returns) / daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR))
        if daily_volatility > 0.0
        else 0.0
    )
    downside_returns = np.minimum(returns, 0.0)
    downside_deviation = float(
        np.sqrt(np.mean(np.square(downside_returns))) * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    sortino_ratio = (
        float(np.mean(returns) * TRADING_DAYS_PER_YEAR / downside_deviation)
        if downside_deviation > 0.0
        else 0.0
    )
    running_peak = np.maximum.accumulate(full_nav)
    maximum_drawdown = float(np.min(full_nav / running_peak - 1.0))

    return {
        "initial_portfolio_value": float(initial_capital),
        "final_portfolio_value": float(full_nav[-1]),
        "cumulative_return": cumulative_return,
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "maximum_drawdown": maximum_drawdown,
        "total_transaction_costs": float(total_transaction_costs),
        "portfolio_turnover": float(portfolio_turnover),
    }


TRADING_DAYS_PER_YEAR = 252
