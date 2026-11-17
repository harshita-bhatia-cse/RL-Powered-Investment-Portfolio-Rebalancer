"""Portfolio risk metrics using daily simple returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .portfolio_metrics import validate_returns


def calculate_risk_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    var_confidence: float = 0.95,
) -> dict:
    """Calculate volatility, ratios, drawdown, tail risk, beta, and alpha."""
    if not 0 < var_confidence < 1:
        raise ValueError("VaR confidence must be between 0 and 1")
    values = validate_returns(returns)
    excess = values - risk_free_rate / periods_per_year
    volatility = float(values.std(ddof=1) * np.sqrt(periods_per_year))
    downside = values[values < risk_free_rate / periods_per_year] - risk_free_rate / periods_per_year
    downside_deviation = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    sharpe = float(excess.mean() * periods_per_year / volatility) if volatility > 0 else 0.0
    sortino = float(excess.mean() * periods_per_year / downside_deviation) if downside_deviation > 0 else 0.0
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    threshold = float(values.quantile(1.0 - var_confidence))
    tail = values[values <= threshold]
    result = {
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": float(drawdown.min()),
        "var": threshold,
        "cvar": float(tail.mean()) if not tail.empty else threshold,
        "downside_deviation": downside_deviation,
    }
    if benchmark_returns is not None:
        aligned = pd.concat([values.rename("portfolio"), benchmark_returns.rename("benchmark")], axis=1).dropna()
        if len(aligned) >= 2 and aligned["benchmark"].var(ddof=1) > 0:
            covariance = aligned["portfolio"].cov(aligned["benchmark"])
            beta = float(covariance / aligned["benchmark"].var(ddof=1))
            alpha = float((aligned["portfolio"].mean() - risk_free_rate / periods_per_year)
                          - beta * (aligned["benchmark"].mean() - risk_free_rate / periods_per_year))
            result.update({"beta": beta, "alpha_daily": alpha, "alpha_annualized": alpha * periods_per_year})
        else:
            result.update({"beta": None, "alpha_daily": None, "alpha_annualized": None})
    return result
