"""Read-only API endpoints for historical portfolio analytics."""

from __future__ import annotations
import json
import logging
import os

from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status as http_status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import OperationalError
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
from src.api.backtest_schemas import (
    HistoricalBacktestHealthResponse,
    HistoricalBacktestRequest,
    HistoricalBacktestResponse,
    HistoricalComparisonRequest,
    HistoricalComparisonResponse,
)
from src.api.account_api import router as account_router
from src.api.phase8_research_api import router as phase8_research_router
from src.backtest.historical_service import (
    HistoricalBacktestService,
    HistoricalDataUnavailableError,
    UnsupportedStrategyError,
)
from src.data.data_store import get_data_store
from src.database.migrations import initialize_database

app = FastAPI(title="Portfolio Analytics API", version="1.0.0")
app.state.account_database_ready = False
app.state.account_database_error = None
app.include_router(account_router)
app.include_router(phase8_research_router)


def initialize_account_database() -> None:
    """Initialize account storage without blocking the research application."""
    try:
        initialize_database()
    except Exception as error:
        app.state.account_database_ready = False
        app.state.account_database_error = type(error).__name__
        logging.getLogger(__name__).exception(
            "Account database initialization failed; account routes are unavailable"
        )
        return
    app.state.account_database_ready = True
    app.state.account_database_error = None


app.router.add_event_handler("startup", initialize_account_database)


@app.middleware("http")
async def account_database_unavailable_response(request, call_next):
    account_path = request.url.path.startswith(
        ("/api/auth/", "/api/portfolios", "/api/research/phase7-2")
    )
    try:
        return await call_next(request)
    except OperationalError:
        if not account_path:
            raise
        logging.getLogger(__name__).exception(
            "Account database request failed; returning service unavailable"
        )
        return JSONResponse(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Account persistence is unavailable"},
        )

_STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"


@app.get("/api/status", tags=["system"])
def status():
    """Return a small, dependency-free health response for the web dashboard."""
    return {"status": "ok", "mode": "research", "trading": "paper-only"}


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


@app.get("/api/dashboard/stats", tags=["dashboard"])
def dashboard_stats():
    """Return the existing local data-store statistics for the sidebar."""
    return get_data_store().get_storage_stats()


@app.post(
    "/api/backtest/historical",
    response_model=HistoricalBacktestResponse,
    tags=["backtest"],
)
def historical_backtest(
    request: HistoricalBacktestRequest,
) -> HistoricalBacktestResponse:
    """Run a broker-independent strategy simulation on historical CSV data."""
    service = HistoricalBacktestService()
    try:
        return service.run(
            strategy_name=request.strategy,
            partition=request.partition,
            initial_capital=request.initial_capital,
            parameters=request.parameters,
        )
    except UnsupportedStrategyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HistoricalDataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/api/backtest/compare",
    response_model=HistoricalComparisonResponse,
    tags=["backtest"],
)
def compare_historical_backtests(
    request: HistoricalComparisonRequest,
) -> HistoricalComparisonResponse:
    """Compare supported strategies using identical historical inputs."""
    service = HistoricalBacktestService()
    try:
        return service.compare(
            strategy_names=request.strategies,
            partition=request.partition,
            initial_capital=request.initial_capital,
            parameters=request.parameters,
        )
    except UnsupportedStrategyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HistoricalDataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/api/backtest/health",
    response_model=HistoricalBacktestHealthResponse,
    tags=["backtest"],
)
def historical_backtest_health() -> HistoricalBacktestHealthResponse:
    """Report dataset validation and historical-backtest availability."""
    try:
        return HistoricalBacktestService().health()
    except Exception as exc:
        logging.getLogger(__name__).exception(
            "Historical backtest health check failed"
        )
        raise HTTPException(
            status_code=503,
            detail="Historical backtest health check failed",
        ) from exc


@app.get("/api/synthetic/preview", tags=["synthetic"])
def synthetic_preview(
    seed: int = Query(default=42, ge=0, le=2_147_483_647),
    sessions: int = Query(default=252, ge=20, le=2_520),
):
    """Generate an in-memory, reproducible synthetic market summary.

    This does not train a model, save files, call a market-data provider, or
    place orders. It lets the plain-JavaScript dashboard use the project's real
    synthetic generator before a user explicitly starts an offline experiment.
    """
    from src.rl.synthetic_data import SyntheticMarketConfig, generate_synthetic_market

    dataset = generate_synthetic_market(SyntheticMarketConfig(sessions=sessions), seed=seed)
    regime_counts = {
        name: dataset.regime_labels.count(name)
        for name in ("bull", "bear", "sideways")
    }
    final_growth = np.cumprod(1.0 + dataset.benchmark_returns)[-1]
    return {
        "seed": dataset.seed,
        "sessions": sessions,
        "scenario_id": dataset.metadata["scenario_id"],
        "assets": list(dataset.asset_names),
        "regime_counts": regime_counts,
        "vix_start": round(float(dataset.vix_levels[0]), 2),
        "vix_end": round(float(dataset.vix_levels[-1]), 2),
        "benchmark_growth": round(float(final_growth), 4),
        "sample_vix": [round(float(value), 2) for value in dataset.vix_levels[:20]],
    }


def _get_ppo_validation_result(artifact_dir: Path | None = None):
    """Evaluate the saved synthetic PPO model on its reproducible validation scenario."""
    if artifact_dir is None:
        artifact_dir = Path(
            os.environ.get(
                "PPO_ARTIFACT_DIR",
                str(Path(__file__).resolve().parents[2] / ".artifacts" / "ppo"),
            )
        ).expanduser()

    model_path = next(
        (
            candidate
            for candidate in (
                artifact_dir / "trained_synthetic_ppo_seed42.zip",
                artifact_dir / "model.zip",
            )
            if candidate.is_file()
        ),
        artifact_dir / "trained_synthetic_ppo_seed42.zip",
    )
    report_path = artifact_dir / "validation_results.json"

    if not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Synthetic PPO model not found in {artifact_dir}. "
                "Run `python -m scripts.train_synthetic_ppo_dashboard` or set "
                "PPO_ARTIFACT_DIR to a folder containing a compatible model."
            ),
        )

    try:
        report = None
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("source_type") != "synthetic" or report.get("synthetic") is not True:
                raise ValueError("Validation report is not marked as synthetic.")
            if report.get("training_validation_seeds_distinct") is not True:
                raise ValueError("Training and validation seeds are not verified as distinct.")
            if report.get("validation_was_used_for_training_or_model_selection") is not False:
                raise ValueError("Validation data usage is not verified.")

        from src.rl.ppo_agent import (
            build_ppo_environments,
            load_ppo_model,
        )
        from src.rl.phase4_evaluation import evaluate_phase4
        from src.rl.ppo_config import PPOConfig

        bundle = build_ppo_environments(seed=42)

        model = load_ppo_model(
            model_path,
            env=bundle.validation_env,
            expected_config=PPOConfig(seed=42, device="cpu"),
        )

        evaluation = evaluate_phase4(
            model,
            bundle.validation_env,
            initial_capital=(
                float(report["starting_nav"])
                if report is not None
                else None
            ),
        )
        fresh_ppo = evaluation["ppo"]

        if report is None:
            return {
                "source_type": "synthetic",
                "synthetic": True,
                "scenario_id": fresh_ppo["scenario_id"],
                "scenario_seed": str(fresh_ppo["seed"]),
                "steps": fresh_ppo["steps"],
                "starting_nav": float(fresh_ppo["initial_portfolio_value"]),
                "ppo": {
                    "ending_nav": float(fresh_ppo["final_portfolio_value"]),
                    "episode_reward": float(fresh_ppo["episode_reward"]),
                    "gross_turnover_sum": float(fresh_ppo["gross_turnover_sum"]),
                    "one_way_turnover_sum": float(fresh_ppo["one_way_turnover_sum"]),
                    "transaction_fees_sum": float(fresh_ppo["transaction_fees_sum"]),
                    "slippage_cost_sum": float(fresh_ppo["slippage_cost_sum"]),
                    "transaction_cost_sum": float(fresh_ppo["transaction_cost_sum"]),
                    "ending_weights": fresh_ppo["ending_weights"],
                    "terminated": fresh_ppo["terminated"],
                    "truncated": fresh_ppo["truncated"],
                },
                "baselines": {},
                "evaluation": {
                    "evaluation_type": evaluation["evaluation_type"],
                    "period": evaluation["period"],
                    "cost_assumptions": evaluation["cost_assumptions"],
                    "strategies": evaluation["strategies"],
                },
                "evaluation_source": "fresh_deterministic_model_evaluation",
                "baseline_source": "shared_evaluation_engine",
                "note": (
                    "No saved validation report was found. All displayed strategy "
                    "metrics were calculated by the shared evaluation engine."
                ),
            }

        scenario_id = report["scenario_id"]
        scenario_seed = report["scenario_seed"]
        saved_ppo = report["ppo"]
        baselines = report["baselines"]

        if (
            fresh_ppo["scenario_id"] != scenario_id
            or str(fresh_ppo["seed"]) != str(scenario_seed)
        ):
            raise ValueError("Fresh PPO evaluation does not match the saved validation scenario.")

        if (
            saved_ppo["scenario_id"] != scenario_id
            or str(saved_ppo["scenario_seed"]) != str(scenario_seed)
        ):
            raise ValueError("Saved PPO metrics refer to a different validation scenario.")

        if fresh_ppo["steps"] != saved_ppo["steps"]:
            raise ValueError("Fresh PPO evaluation has a different number of steps.")

        # Confirm that fresh deterministic evaluation reproduces the saved PPO metrics.
        checks = {
            "ending_nav": fresh_ppo["final_portfolio_value"],
            "episode_reward": fresh_ppo["episode_reward"],
            "gross_turnover_sum": fresh_ppo["gross_turnover_sum"],
            "one_way_turnover_sum": fresh_ppo["one_way_turnover_sum"],
            "transaction_fees_sum": fresh_ppo["transaction_fees_sum"],
            "slippage_cost_sum": fresh_ppo["slippage_cost_sum"],
            "transaction_cost_sum": fresh_ppo["transaction_cost_sum"],
        }

        for key, fresh_value in checks.items():
            if not np.isclose(
                float(fresh_value),
                float(saved_ppo[key]),
                rtol=1e-9,
                atol=1e-7,
            ):
                raise ValueError(f"Fresh PPO evaluation does not reproduce saved metric: {key}")

        baseline_policies = {
            "equal_weight_rebalanced",
            "buy_and_hold_after_initial_equal_target",
        }
        baseline_results = {}

        for baseline in baselines:
            policy = baseline["policy"]

            if policy not in baseline_policies or policy in baseline_results:
                raise ValueError("Unexpected or duplicate baseline policy.")

            if (
                baseline["scenario_id"] != scenario_id
                or baseline["scenario_seed"] != scenario_seed
                or baseline["steps"] != fresh_ppo["steps"]
            ):
                raise ValueError("A baseline refers to a different validation scenario.")

            baseline_results[policy] = {
                "ending_nav": float(baseline["ending_nav"]),
                "episode_reward": float(baseline["episode_reward"]),
                "gross_turnover_sum": float(baseline["gross_turnover_sum"]),
                "one_way_turnover_sum": float(baseline["one_way_turnover_sum"]),
                "transaction_fees_sum": float(baseline["transaction_fees_sum"]),
                "slippage_cost_sum": float(baseline["slippage_cost_sum"]),
                "transaction_cost_sum": float(baseline["transaction_cost_sum"]),
                "ending_weights": {
                    key: float(value)
                    for key, value in baseline["ending_weights"].items()
                },
            }

        if set(baseline_results) != baseline_policies:
            raise ValueError("The validation report does not contain both expected baselines.")

        return {
            "source_type": "synthetic",
            "synthetic": True,
            "scenario_id": fresh_ppo["scenario_id"],
            "scenario_seed": str(fresh_ppo["seed"]),
            "steps": fresh_ppo["steps"],
            "starting_nav": float(report["starting_nav"]),
            "ppo": {
                "ending_nav": float(fresh_ppo["final_portfolio_value"]),
                "episode_reward": float(checks["episode_reward"]),
                "gross_turnover_sum": float(checks["gross_turnover_sum"]),
                "one_way_turnover_sum": float(checks["one_way_turnover_sum"]),
                "transaction_fees_sum": float(checks["transaction_fees_sum"]),
                "slippage_cost_sum": float(checks["slippage_cost_sum"]),
                "transaction_cost_sum": float(checks["transaction_cost_sum"]),
                "ending_weights": fresh_ppo["ending_weights"],
                "terminated": fresh_ppo["terminated"],
                "truncated": fresh_ppo["truncated"],
            },
            "baselines": baseline_results,
            "evaluation": {
                "evaluation_type": evaluation["evaluation_type"],
                "period": evaluation["period"],
                "cost_assumptions": evaluation["cost_assumptions"],
                "strategies": evaluation["strategies"],
            },
            "evaluation_source": "fresh_deterministic_model_evaluation",
            "baseline_source": "saved_validation_report",
            "note": report["notes"],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"PPO validation results are unavailable: {exc}",
        ) from exc


@app.get("/api/ppo/validation", tags=["synthetic"])
def ppo_validation():
    """Return a reproducible, read-only synthetic PPO validation comparison."""
    return _get_ppo_validation_result()


@app.post("/api/ppo/runs", status_code=http_status.HTTP_202_ACCEPTED, tags=["synthetic"])
def start_synthetic_ppo_training():
    """Start one isolated, synthetic-only PPO run for the dashboard."""
    from src.api.synthetic_training_jobs import start_training

    try:
        return start_training()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=503, detail=f"could not start PPO training: {exc}") from exc


@app.get("/api/ppo/runs/{run_id}", tags=["synthetic"])
def synthetic_ppo_training_status(run_id: str):
    """Return status for an isolated synthetic PPO dashboard run."""
    from src.api.synthetic_training_jobs import status

    try:
        return status(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="training run not found") from exc


@app.get("/api/ppo/runs/{run_id}/validation", tags=["synthetic"])
def synthetic_ppo_run_validation(run_id: str):
    """Evaluate a completed dashboard run without changing its artifacts."""
    from src.api.synthetic_training_jobs import run_directory, status

    try:
        current = status(run_id)
        if current["state"] != "completed":
            raise HTTPException(status_code=409, detail="training run has not completed")
        return _get_ppo_validation_result(run_directory(run_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="training run not found") from exc


# Mount the dashboard last: API routes above retain priority over static files.
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the dependency-free research dashboard."""
    return FileResponse(_STATIC_DIR / "index.html")
