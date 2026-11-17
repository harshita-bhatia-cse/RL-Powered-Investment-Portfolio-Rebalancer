"""Read-only API endpoints for historical portfolio analytics."""

from __future__ import annotations

from datetime import date
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.analytics import (
    analyze_assets,
    analyze_correlations,
    calculate_portfolio_metrics,
    calculate_risk_metrics,
    compare_benchmarks,
    prices_to_returns,
    validate_weights,
)
from src.data.data_store import get_data_store

app = FastAPI(title="Portfolio Analytics API", version="1.0.0")


class AnalyticsRequest(BaseModel):
    tickers: List[str] = Field(min_length=1)
    weights: Dict[str, float]
    start_date: date
    end_date: date
    initial_investment: float = Field(default=100000.0, gt=0)
    benchmark: str | None = None

    @model_validator(mode="after")
    def validate_request(self):
        self.tickers = [ticker.upper().strip() for ticker in self.tickers if ticker.strip()]
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        self.weights = validate_weights(self.weights)
        if set(self.weights) != set(self.tickers):
            raise ValueError("tickers and weights must contain the same assets")
        return self


def _load(request: AnalyticsRequest, extra_tickers: List[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = request.tickers + [ticker for ticker in (extra_tickers or []) if ticker not in request.tickers]
    data = get_data_store().get_price_data(
        tickers, request.start_date.isoformat(), request.end_date.isoformat()
    )
    if data.empty:
        raise HTTPException(status_code=404, detail="No historical data found for the requested assets and dates")
    prices = data.assign(datadate=pd.to_datetime(data["datadate"])).pivot(
        index="datadate", columns="tic", values="adj_close"
    ).sort_index()
    returns = prices_to_returns(prices).dropna()
    if len(returns) < 2:
        raise HTTPException(status_code=422, detail="Insufficient historical observations")
    return prices, returns


def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        return value.reset_index().to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.rename("value").reset_index().to_dict(orient="records")
    return value


@app.post("/portfolio/performance")
def portfolio_performance(request: AnalyticsRequest):
    _, returns = _load(request)
    result = calculate_portfolio_metrics(returns, request.weights, request.initial_investment)
    return {key: _jsonable(value) for key, value in result.items()}


@app.post("/portfolio/risk")
def portfolio_risk(request: AnalyticsRequest):
    _, returns = _load(request)
    portfolio = calculate_portfolio_metrics(returns, request.weights)
    return calculate_risk_metrics(portfolio["daily_returns"])


@app.post("/portfolio/assets")
def portfolio_assets(request: AnalyticsRequest):
    _, returns = _load(request)
    return {
        "metrics": _jsonable(analyze_assets(returns, request.weights)["assets"]),
        "best_performing_asset": analyze_assets(returns, request.weights)["best_performing_asset"],
        "worst_performing_asset": analyze_assets(returns, request.weights)["worst_performing_asset"],
    }


@app.post("/portfolio/correlation")
def portfolio_correlation(request: AnalyticsRequest):
    _, returns = _load(request)
    result = analyze_correlations(returns)
    return {key: _jsonable(value) for key, value in result.items()}


@app.post("/portfolio/allocation")
def portfolio_allocation(request: AnalyticsRequest):
    return {
        "target_weights": request.weights,
        "total_weight": sum(request.weights.values()),
        "largest_weight": max(request.weights.values()),
        "herfindahl_index": sum(weight ** 2 for weight in request.weights.values()),
    }


@app.post("/portfolio/benchmarks")
def portfolio_benchmarks(request: AnalyticsRequest):
    benchmark_names = [request.benchmark] if request.benchmark else ["SPY"]
    prices, returns = _load(request, benchmark_names)
    portfolio = calculate_portfolio_metrics(returns, request.weights)
    benchmarks = {
        name: prices[name].pct_change().dropna()
        for name in benchmark_names
        if name in prices.columns
    }
    if not benchmarks:
        raise HTTPException(status_code=404, detail="No requested benchmark data is available")
    return _jsonable(compare_benchmarks(portfolio["daily_returns"], benchmarks))
