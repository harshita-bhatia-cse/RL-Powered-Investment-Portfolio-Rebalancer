"""Per-asset performance analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_assets(returns: pd.DataFrame, weights: dict[str, float], periods_per_year: int = 252) -> dict:
    """Return per-asset metrics and portfolio contribution estimates."""
    if returns.empty:
        raise ValueError("Historical returns are required")
    rows = []
    for asset in returns.columns:
        series = returns[asset].dropna()
        if series.empty:
            continue
        wealth = (1.0 + series).cumprod()
        total_return = float(wealth.iloc[-1] - 1.0)
        annualized = float((1.0 + total_return) ** (periods_per_year / len(series)) - 1.0)
        volatility = float(series.std(ddof=1) * np.sqrt(periods_per_year))
        sharpe = float(series.mean() * periods_per_year / volatility) if volatility > 0 else 0.0
        drawdown = wealth / wealth.cummax() - 1.0
        weight = float(weights.get(asset, 0.0))
        rows.append({
            "asset": asset,
            "average_daily_return": float(series.mean()),
            "annualized_return": annualized,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "maximum_drawdown": float(drawdown.min()),
            "portfolio_return_contribution": weight * total_return,
        })
    if not rows:
        raise ValueError("No valid asset returns are available")
    frame = pd.DataFrame(rows).set_index("asset")
    return {
        "assets": frame,
        "best_performing_asset": str(frame["annualized_return"].idxmax()),
        "worst_performing_asset": str(frame["annualized_return"].idxmin()),
    }
