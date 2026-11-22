"""Synthetic PPO and benchmark evaluation metrics for Phase 4."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.rl.benchmarks import (
    calculate_daily_nav_metrics,
    evaluate_synthetic_benchmarks,
)
from src.rl.environment.portfolio_env import ASSET_UNIVERSE
from src.rl.environment.synthetic_portfolio_env import (
    OBSERVATION_WARMUP_SESSIONS,
    OBSERVATION_SCHEMA_WITH_TIME,
    SyntheticPortfolioEnv,
    observation_size_for_schema,
)
from src.rl.ppo_agent import EvaluationResult, evaluate_ppo

TRADING_DAYS_PER_YEAR = 252
_ASSET_OBSERVATION_FEATURES = (
    "return_1",
    "return_5",
    "return_21",
    "volatility_21",
    "volatility_63",
)


def calculate_portfolio_metrics(
    nav_history: list[float] | np.ndarray,
    *,
    initial_capital: float,
    total_transaction_costs: float = 0.0,
    portfolio_turnover: float = 0.0,
) -> dict[str, float]:
    """Calculate daily-frequency return and risk metrics from NAV observations."""
    return calculate_daily_nav_metrics(
        nav_history,
        initial_capital=initial_capital,
        total_transaction_costs=total_transaction_costs,
        portfolio_turnover=portfolio_turnover,
    )


def evaluate_ppo_synthetic(
    model: Any,
    env: Any,
    *,
    deterministic: bool = True,
    max_steps: int | None = None,
    initial_capital: float | None = None,
    observation_diagnostics_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate a PPO model and return the Phase 4 synthetic metrics."""
    base = getattr(env, "unwrapped", env)
    if (
        getattr(base, "metadata", {}).get("synthetic") is not True
        or getattr(base, "data_metadata", {}).get("synthetic") is not True
        or getattr(base, "data_metadata", {}).get("source_type") != "synthetic"
    ):
        raise ValueError("PPO Phase 4 evaluation requires an explicitly synthetic environment")

    capital = float(base.initial_capital if initial_capital is None else initial_capital)
    if isinstance(initial_capital, bool) or not np.isfinite(capital) or capital <= 0.0:
        raise ValueError("initial_capital must be positive and finite")

    evaluation_env = SyntheticPortfolioEnv(
        base.asset_returns,
        base.benchmark_returns,
        base.vix_levels,
        data_metadata=base.data_metadata,
        initial_capital=capital,
        period_sessions=base.period_sessions,
        max_asset_weight=base.max_asset_weights,
        min_cash_weight=base.min_cash_weight,
        max_turnover=base.max_turnover,
        transaction_cost_rate=base.transaction_cost_rate,
        slippage_bps=base.slippage_bps,
        vix_level_scale=base.vix_level_scale,
        observation_schema=base.observation_schema,
    )
    result: EvaluationResult = evaluate_ppo(
        model,
        evaluation_env,
        deterministic=deterministic,
        max_steps=max_steps,
    )
    if observation_diagnostics_callback is not None:
        observation_diagnostics_callback(
            build_observation_action_diagnostics(
                result,
                vix_level_scale=evaluation_env.vix_level_scale,
            )
        )
    nav_history = [
        nav
        for step in result.steps
        for nav in step.daily_navs
    ]
    total_cost = sum(step.transaction_cost for step in result.steps)
    turnover = sum(step.one_way_turnover for step in result.steps)
    metrics = calculate_portfolio_metrics(
        nav_history,
        initial_capital=capital,
        total_transaction_costs=total_cost,
        portfolio_turnover=turnover,
    )
    metrics.update({
        "strategy": "Synthetic PPO",
        "synthetic": True,
        "source_type": "synthetic",
        "scenario_id": result.scenario_id,
        "seed": result.seed,
        "episode_reward": float(sum(step.reward for step in result.steps)),
        "gross_turnover_sum": float(sum(step.gross_turnover for step in result.steps)),
        "one_way_turnover_sum": float(turnover),
        "transaction_fees_sum": float(sum(step.transaction_fees for step in result.steps)),
        "slippage_cost_sum": float(sum(step.slippage_cost for step in result.steps)),
        "transaction_cost_sum": float(total_cost),
        "ending_weights": dict(result.steps[-1].weights),
        "steps": len(result.steps),
        "terminated": bool(result.steps[-1].terminated),
        "truncated": bool(result.steps[-1].truncated),
        "allocation_diagnostics": summarize_allocation_diagnostics(result.steps),
    })
    return metrics


def summarize_allocation_diagnostics(
    steps: Sequence[EvaluationStep],
) -> dict[str, Any]:
    """Summarize applied rebalance weights and retain each decision record."""
    if not steps:
        raise ValueError("at least one evaluation step is required")

    weight_names = (*ASSET_UNIVERSE, "CASH")
    applied_matrix: list[list[float]] = []
    records: list[dict[str, Any]] = []

    for decision_number, step in enumerate(steps, start=1):
        if step.requested_weights is None or step.applied_weights is None:
            raise ValueError("evaluation step is missing allocation diagnostics")
        requested = _validated_diagnostic_weights(
            step.requested_weights, weight_names, "requested target"
        )
        applied = _validated_diagnostic_weights(
            step.applied_weights, weight_names, "applied portfolio"
        )
        raw_action = np.asarray(step.action, dtype=np.float64)
        if raw_action.shape != (len(weight_names),) or not np.isfinite(raw_action).all():
            raise ValueError("evaluation step has an invalid raw action")
        if not np.isfinite(
            [
                step.nav,
                step.gross_turnover,
                step.one_way_turnover,
                step.transaction_fees,
                step.slippage_cost,
                step.transaction_cost,
            ]
        ).all():
            raise ValueError("evaluation step has non-finite diagnostic values")
        if (
            step.nav <= 0.0
            or step.gross_turnover < 0.0
            or step.one_way_turnover < 0.0
            or step.transaction_fees < 0.0
            or step.slippage_cost < 0.0
        ):
            raise ValueError(
                "evaluation NAV, turnover, and costs must be positive or non-negative"
            )
        if not np.isclose(
            step.one_way_turnover,
            step.gross_turnover / 2.0,
            rtol=1e-10,
            atol=1e-8,
        ):
            raise ValueError("one-way turnover must be half of gross turnover")
        if not np.isclose(
            step.transaction_cost,
            step.transaction_fees + step.slippage_cost,
            rtol=1e-10,
            atol=1e-8,
        ):
            raise ValueError("total transaction cost must equal fees plus slippage")

        applied_matrix.append([applied[name] for name in weight_names])
        records.append({
            "decision_number": decision_number,
            "raw_action": [float(value) for value in raw_action],
            "requested_target_weights": requested,
            "applied_portfolio_weights": applied,
            "cash_allocation": applied["CASH"],
            "portfolio_nav": float(step.nav),
            "gross_turnover": float(step.gross_turnover),
            "one_way_turnover": float(step.one_way_turnover),
            "transaction_fees": float(step.transaction_fees),
            "slippage_costs": float(step.slippage_cost),
            "total_transaction_costs": float(step.transaction_cost),
        })

    weights = np.asarray(applied_matrix, dtype=np.float64)
    cash = weights[:, -1]
    risky = weights[:, :-1].sum(axis=1)
    summary = {
        "decision_count": len(steps),
        "average_cash_allocation": float(np.mean(cash)),
        "minimum_cash_allocation": float(np.min(cash)),
        "maximum_cash_allocation": float(np.max(cash)),
        "average_risky_asset_allocation": float(np.mean(risky)),
        "asset_allocation_distribution": {
            name: {
                "average": float(np.mean(weights[:, index])),
                "minimum": float(np.min(weights[:, index])),
                "maximum": float(np.max(weights[:, index])),
            }
            for index, name in enumerate(weight_names)
        },
        "average_gross_turnover": float(
            np.mean([step.gross_turnover for step in steps])
        ),
        "average_one_way_turnover": float(
            np.mean([step.one_way_turnover for step in steps])
        ),
        "total_transaction_fees": float(sum(step.transaction_fees for step in steps)),
        "total_slippage_costs": float(sum(step.slippage_cost for step in steps)),
        "total_transaction_costs": float(sum(step.transaction_cost for step in steps)),
    }
    return {"summary": summary, "rebalance_history": records}


def _validated_diagnostic_weights(
    weights: Mapping[str, float],
    expected_names: Sequence[str],
    label: str,
    *,
    sum_tolerance: float = 1e-8,
) -> dict[str, float]:
    if set(weights) != set(expected_names):
        raise ValueError(f"{label} weights do not match the environment assets")
    values = {name: float(weights[name]) for name in expected_names}
    array = np.asarray(list(values.values()), dtype=np.float64)
    if (
        not np.isfinite(array).all()
        or np.any(array < -1e-10)
        or not np.isclose(
            float(array.sum()), 1.0, rtol=0.0, atol=sum_tolerance
        )
    ):
        raise ValueError(f"{label} weights must be finite, long-only, and sum to one")
    return values


def build_observation_action_diagnostics(
    result: EvaluationResult,
    *,
    vix_level_scale: float,
) -> dict[str, Any]:
    """Pair each pre-action PPO observation with its action and transition."""
    if not math.isfinite(vix_level_scale) or vix_level_scale <= 0.0:
        raise ValueError("vix_level_scale must be positive and finite")

    weight_names = (*ASSET_UNIVERSE, "CASH")
    observation_schema = result.observation_schema
    observation_size = observation_size_for_schema(observation_schema)
    has_time_remaining = observation_schema == OBSERVATION_SCHEMA_WITH_TIME
    decisions: list[dict[str, Any]] = []
    for decision_number, step in enumerate(result.steps, start=1):
        if step.decision_observation is None:
            raise ValueError("evaluation step is missing its decision observation")
        observation = np.asarray(step.decision_observation, dtype=np.float64)
        if observation.shape != (observation_size,) or not np.isfinite(observation).all():
            raise ValueError(
                f"decision observation must contain {observation_size} finite values"
            )
        if step.requested_weights is None or step.applied_weights is None:
            raise ValueError("evaluation step is missing allocation diagnostics")

        raw_action = np.asarray(step.action, dtype=np.float64)
        if raw_action.shape != (len(weight_names),) or not np.isfinite(raw_action).all():
            raise ValueError("evaluation step has an invalid raw action")

        observation_index = 0
        asset_features: dict[str, dict[str, float]] = {}
        for symbol in ASSET_UNIVERSE:
            asset_features[symbol] = {
                feature_name: float(observation[observation_index + offset])
                for offset, feature_name in enumerate(_ASSET_OBSERVATION_FEATURES)
            }
            observation_index += len(_ASSET_OBSERVATION_FEATURES)

        benchmark_features = {
            "return_1": float(observation[observation_index]),
            "volatility_21": float(observation[observation_index + 1]),
        }
        vix_input = float(observation[observation_index + 2])
        state_start = observation_index + 3
        current_weights = {
            name: float(observation[state_start + index])
            for index, name in enumerate(weight_names)
        }
        current_weights = _validated_diagnostic_weights(
            current_weights,
            weight_names,
            "observed current portfolio",
            sum_tolerance=1e-6,
        )
        nav_ratio = float(observation[state_start + len(weight_names)])
        drawdown = float(observation[state_start + len(weight_names) + 1])
        observation_features = {
            "asset_features": asset_features,
            "benchmark_features": benchmark_features,
            "vix_level": vix_input * vix_level_scale,
            "vix_model_input": vix_input,
            "vix_level_scale": float(vix_level_scale),
            "current_asset_weights": {
                symbol: current_weights[symbol] for symbol in ASSET_UNIVERSE
            },
            "cash_weight": current_weights["CASH"],
            "current_portfolio_weights": current_weights,
            "portfolio_value_ratio": nav_ratio,
            "drawdown": drawdown,
        }
        if has_time_remaining:
            time_remaining = float(observation[state_start + len(weight_names) + 2])
            if not 0.0 <= time_remaining <= 1.0:
                raise ValueError("decision observation contains invalid time feature")
            observation_features["time_remaining_fraction"] = time_remaining
        if (
            nav_ratio <= 0.0
            or vix_input <= 0.0
        ):
            raise ValueError("decision observation contains invalid state features")

        decisions.append({
            "decision_number": decision_number,
            "raw_observation": [float(value) for value in observation],
            "observation_features": observation_features,
            "raw_action": [float(value) for value in raw_action],
            "requested_target_weights": _validated_diagnostic_weights(
                step.requested_weights, weight_names, "requested target"
            ),
            "applied_portfolio_weights": _validated_diagnostic_weights(
                step.applied_weights, weight_names, "applied portfolio"
            ),
            "portfolio_nav_after_interval": float(step.nav),
            "reward": float(step.reward),
        })

    return {
        "format": "ppo-observation-action-diagnostics-v1",
        "source_type": result.source_type,
        "synthetic": result.synthetic,
        "scenario_id": result.scenario_id,
        "seed": result.seed,
        "observation_schema": observation_schema,
        "observation_size": observation_size,
        "action_size": len(weight_names),
        "decision_count": len(decisions),
        "decisions": decisions,
    }


def evaluate_phase4(
    model: Any,
    env: Any,
    *,
    initial_capital: float | None = None,
    observation_diagnostics_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compare synthetic PPO against the four existing benchmark strategies."""
    base = getattr(env, "unwrapped", env)
    if (
        getattr(base, "metadata", {}).get("synthetic") is not True
        or getattr(base, "data_metadata", {}).get("synthetic") is not True
    ):
        raise ValueError("Phase 4 evaluation requires an explicitly synthetic environment")

    if isinstance(initial_capital, bool):
        raise ValueError("initial_capital must be positive and finite")
    capital = float(base.initial_capital if initial_capital is None else initial_capital)
    if capital <= 0 or not np.isfinite(capital):
        raise ValueError("initial_capital must be positive and finite")
    ppo = evaluate_ppo_synthetic(
        model,
        env,
        initial_capital=capital,
        observation_diagnostics_callback=observation_diagnostics_callback,
    )
    benchmark_frame = evaluate_synthetic_benchmarks(
        base.asset_returns,
        initial_capital=capital,
        transaction_cost_rate=base.transaction_cost_rate,
        slippage_bps=base.slippage_bps,
        period_sessions=base.period_sessions,
        warmup_sessions=OBSERVATION_WARMUP_SESSIONS,
    )

    benchmark_results: dict[str, dict[str, float]] = {}
    for name, row in benchmark_frame.to_dict("index").items():
        benchmark_results[str(name)] = {
            key: float(value) for key, value in row.items()
        }
    return {
        "evaluation_type": "synthetic-only",
        "period": {
            "warmup_sessions": OBSERVATION_WARMUP_SESSIONS,
            "period_sessions": int(base.period_sessions),
            "evaluation_sessions": int(
                len(base.asset_returns) - OBSERVATION_WARMUP_SESSIONS
            ),
        },
        "cost_assumptions": {
            "initial_capital": capital,
            "transaction_cost_rate": float(base.transaction_cost_rate),
            "slippage_bps": float(base.slippage_bps),
        },
        "ppo": ppo,
        "benchmarks": benchmark_results,
        "strategies": {
            "Synthetic PPO": ppo,
            **benchmark_results,
        },
    }


def save_phase4_results(
    output_dir: str | Path,
    *,
    experiment: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Save structured experiment metadata and evaluation results."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    experiment_path = directory / "experiment.json"
    evaluation_path = directory / "evaluation.json"
    experiment_path.write_text(
        json.dumps(experiment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluation_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return experiment_path, evaluation_path


def save_phase4_diagnostics(
    output_dir: str | Path,
    evaluation: Mapping[str, Any],
) -> Path:
    """Save PPO allocation diagnostics separately from existing evaluation files."""
    ppo = evaluation["ppo"]
    diagnostics = ppo["allocation_diagnostics"]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ppo_diagnostics.json"
    payload = {
        "source_type": ppo["source_type"],
        "synthetic": ppo["synthetic"],
        "scenario_id": ppo["scenario_id"],
        "seed": ppo["seed"],
        **diagnostics,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def save_phase4_observation_action_diagnostics(
    output_dir: str | Path,
    diagnostics: Mapping[str, Any],
) -> Path:
    """Save observation/action traces without changing existing PPO artifacts."""
    decisions = diagnostics.get("decisions")
    if (
        not isinstance(decisions, list)
        or diagnostics.get("decision_count") != len(decisions)
    ):
        raise ValueError("diagnostics decision_count must match its decisions")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ppo_observation_action_diagnostics.json"
    path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path
def save_phase4_csv(output_dir: str | Path, evaluation: Mapping[str, Any]) -> Path:
    """Save one row per PPO/benchmark strategy as CSV."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "evaluation.csv"
    strategies: list[dict[str, Any]] = []
    ppo = dict(evaluation["ppo"])
    ppo_row = {"strategy": "Synthetic PPO"}
    ppo_row.update({key: value for key, value in ppo.items() if key not in ("strategy",)})
    strategies.append(ppo_row)
    for name, metrics in evaluation["benchmarks"].items():
        strategies.append({"strategy": name, **dict(metrics)})

    fieldnames = list(dict.fromkeys(key for row in strategies for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(strategies)
    return path
