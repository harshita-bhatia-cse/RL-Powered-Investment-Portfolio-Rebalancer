"""CSV-backed, broker-independent historical portfolio backtesting service."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.backtest.historical_metrics import calculate_historical_metrics
from src.backtest.historical_strategies import (
    SUPPORTED_STRATEGIES,
    HistoricalStrategy,
    create_strategy,
)
from src.rl.environment.portfolio_env import (
    PARTITION_BOUNDS,
    PortfolioTradingEnv,
    TrainOnlyFeatureScaler,
    fit_training_scaler,
    load_market_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICAL_DATA_DIR = PROJECT_ROOT / "data" / "historical_market_csv"


class HistoricalDataUnavailableError(RuntimeError):
    """Raised when the CSV dataset cannot satisfy the environment data contract."""


class UnsupportedStrategyError(ValueError):
    """Raised for unknown strategies or unsupported strategy parameters."""


class HistoricalBacktestService:
    """Load historical CSV data and run deterministic environment strategies."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = (
            Path(data_dir) if data_dir is not None else DEFAULT_HISTORICAL_DATA_DIR
        )

    def run(
        self,
        *,
        strategy_name: str,
        partition: str,
        initial_capital: float = 100_000.0,
        parameters: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run one strategy on a fixed environment partition."""
        self._validate_capital(initial_capital)
        self._validate_partition(partition)
        strategy = self._make_strategy(strategy_name, parameters)
        market_data, scaler = self._load_context()
        return self._run_with_context(
            strategy,
            market_data=market_data,
            scaler=scaler,
            partition=partition,
            initial_capital=initial_capital,
        )

    def compare(
        self,
        *,
        strategy_names: list[str],
        partition: str,
        initial_capital: float = 100_000.0,
        parameters: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, Any]:
        """Compare strategies with the same partition, capital, data, and scaler."""
        self._validate_capital(initial_capital)
        self._validate_partition(partition)
        if not strategy_names:
            raise ValueError("At least one strategy is required")
        if len(set(strategy_names)) != len(strategy_names):
            raise ValueError("Strategy names must be unique")
        strategy_parameters = parameters or {}
        strategies = [
            self._make_strategy(name, strategy_parameters.get(name))
            for name in strategy_names
        ]
        unknown_parameter_names = set(strategy_parameters) - set(strategy_names)
        if unknown_parameter_names:
            raise UnsupportedStrategyError(
                "Parameters supplied for unrequested strategies: "
                + ", ".join(sorted(unknown_parameter_names))
            )

        market_data, scaler = self._load_context()
        results = [
            self._run_with_context(
                strategy,
                market_data=market_data,
                scaler=scaler,
                partition=partition,
                initial_capital=initial_capital,
            )
            for strategy in strategies
        ]
        return {
            "partition": partition,
            "initial_capital": float(initial_capital),
            "results": results,
        }

    def health(self) -> dict[str, Any]:
        """Validate all partition boundaries and finite environment observations."""
        try:
            market_data, scaler = self._load_context()
            partitions: dict[str, dict[str, Any]] = {}
            for name in PARTITION_BOUNDS:
                env = self._build_env(
                    market_data, scaler, partition=name, initial_capital=100_000.0
                )
                observation, initial_info = env.reset(seed=42)
                if not np.isfinite(observation).all():
                    raise HistoricalDataUnavailableError(
                        f"{name} partition produced a non-finite observation"
                    )
                partitions[name] = {
                    "start_date": env.partition_dates[0].date().isoformat(),
                    "end_date": env.partition_dates[-1].date().isoformat(),
                    "trading_sessions": len(env.partition_dates),
                    "initial_nav": float(initial_info["nav"]),
                    "scaler_fit_start": scaler.fitted_start,
                    "scaler_fit_end": scaler.fitted_end,
                }
            return {
                "available": True,
                "dataset_validation": {"valid": True, "partitions": partitions},
                "supported_strategies": list(SUPPORTED_STRATEGIES),
                "corporate_actions_verified": False,
                "limitation": (
                    "Historical corporate-action coverage is unverified; results "
                    "are not guaranteed to be total-return accurate."
                ),
            }
        except HistoricalDataUnavailableError as exc:
            return {
                "available": False,
                "dataset_validation": {"valid": False, "error": str(exc)},
                "supported_strategies": list(SUPPORTED_STRATEGIES),
                "corporate_actions_verified": False,
                "limitation": (
                    "Historical corporate-action coverage is unverified; results "
                    "are not guaranteed to be total-return accurate."
                ),
            }

    def _load_context(self) -> tuple[dict[str, pd.DataFrame], TrainOnlyFeatureScaler]:
        try:
            market_data = load_market_data(self.data_dir)
            scaler = fit_training_scaler(market_data)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise HistoricalDataUnavailableError(
                f"Historical CSV data is unavailable or invalid: {exc}"
            ) from exc
        return market_data, scaler

    def _build_env(
        self,
        market_data: dict[str, pd.DataFrame],
        scaler: TrainOnlyFeatureScaler,
        *,
        partition: str,
        initial_capital: float,
    ) -> PortfolioTradingEnv:
        if partition not in PARTITION_BOUNDS:
            raise ValueError(f"partition must be one of {tuple(PARTITION_BOUNDS)}")
        try:
            env = PortfolioTradingEnv(
                market_data,
                partition=partition,
                initial_cash=initial_capital,
                scaler=scaler,
                random_start=False,
            )
        except ValueError as exc:
            raise HistoricalDataUnavailableError(
                f"Historical data cannot initialize the {partition} partition: {exc}"
            ) from exc

        expected_start, expected_end = PARTITION_BOUNDS[partition]
        if (
            env.partition_dates[0] != expected_start
            or env.partition_dates[-1] != expected_end
        ):
            raise HistoricalDataUnavailableError(
                f"Historical data does not cover the full {partition} interval "
                f"({expected_start.date()} through {expected_end.date()})"
            )
        return env

    def _run_with_context(
        self,
        strategy: HistoricalStrategy,
        *,
        market_data: dict[str, pd.DataFrame],
        scaler: TrainOnlyFeatureScaler,
        partition: str,
        initial_capital: float,
    ) -> dict[str, Any]:
        env = self._build_env(
            market_data,
            scaler,
            partition=partition,
            initial_capital=initial_capital,
        )
        observation, initial_info = env.reset(seed=42)
        dates = [pd.Timestamp(initial_info["date"])]
        nav_values = [float(initial_info["nav"])]
        turnover = 0.0
        transaction_fees = 0.0
        slippage_cost = 0.0
        warnings: list[str] = []
        corporate_action_limitation = initial_info.get("corporate_action_limitation")
        if corporate_action_limitation:
            warnings.append(str(corporate_action_limitation))
        if not initial_info["corporate_actions_verified"]:
            warnings.append(
                "NAV is not verified as corporate-action-complete or total-return accurate."
            )
        if (
            env.allocation_config.transaction_cost_rate == 0.0
            and env.allocation_config.slippage_bps == 0.0
        ):
            warnings.append(
                "The existing allocation configuration defaults transaction fees "
                "and slippage to zero; reported costs reflect those settings."
            )

        for step_index in range(len(env.partition_dates) - 1):
            action = strategy.action(observation, initial_step=step_index == 0)
            if action is None:
                observation, reward, terminated, truncated, info = env.step_hold()
            else:
                observation, reward, terminated, truncated, info = env.step(action)
            if truncated:
                raise RuntimeError(
                    "Historical portfolio environment truncated unexpectedly"
                )
            if not np.isfinite(observation).all() or not math.isfinite(float(reward)):
                raise RuntimeError(
                    "Historical backtest produced non-finite state values"
                )
            dates.append(pd.Timestamp(info["date"]))
            nav_values.append(float(info["nav"]))
            turnover += float(info["turnover_ratio"])
            transaction_fees += float(info["transaction_fees"])
            slippage_cost += float(info["slippage_cost"])
            if info["allocation_rejected"]:
                warning = "An allocation was rejected by the existing environment constraints."
                if warning not in warnings:
                    warnings.append(warning)
            if info["action_was_clipped"]:
                warning = "At least one strategy action was clipped to the environment bounds."
                if warning not in warnings:
                    warnings.append(warning)
            if terminated:
                break

        nav_history = pd.Series(
            nav_values,
            index=pd.DatetimeIndex(dates, name="date"),
            dtype=np.float64,
        )
        metrics, daily_returns = calculate_historical_metrics(
            nav_history,
            turnover=turnover,
            transaction_fees=transaction_fees,
            slippage_cost=slippage_cost,
        )
        return {
            "strategy": strategy.name,
            "partition": partition,
            "start_date": dates[0].date().isoformat(),
            "end_date": dates[-1].date().isoformat(),
            "number_of_trading_sessions": len(env.partition_dates),
            "initial_nav": nav_values[0],
            "final_nav": nav_values[-1],
            "metrics": metrics,
            "nav_history": [
                {"date": day.date().isoformat(), "nav": float(nav)}
                for day, nav in zip(dates, nav_values)
            ],
            "daily_returns": [
                {"date": day.date().isoformat(), "value": float(value)}
                for day, value in daily_returns.items()
            ],
            "transaction_cost_summary": {
                "turnover": turnover,
                "transaction_fees": transaction_fees,
                "slippage_cost": slippage_cost,
                "total_transaction_cost": transaction_fees + slippage_cost,
            },
            "corporate_actions_verified": bool(
                initial_info["corporate_actions_verified"]
            ),
            "corporate_action_limitation": corporate_action_limitation,
            "return_accuracy": initial_info["return_accuracy"],
            "warnings": warnings,
        }

    @staticmethod
    def _make_strategy(
        name: str, parameters: Mapping[str, float] | None
    ) -> HistoricalStrategy:
        if parameters:
            raise UnsupportedStrategyError(
                f"Strategy {name!r} does not accept parameters"
            )
        try:
            return create_strategy(name)
        except ValueError as exc:
            raise UnsupportedStrategyError(str(exc)) from exc

    @staticmethod
    def _validate_capital(initial_capital: float) -> None:
        if (
            isinstance(initial_capital, bool)
            or not math.isfinite(float(initial_capital))
            or initial_capital <= 0.0
        ):
            raise ValueError("initial_capital must be positive and finite")

    @staticmethod
    def _validate_partition(partition: str) -> None:
        if partition not in PARTITION_BOUNDS:
            raise ValueError(f"partition must be one of {tuple(PARTITION_BOUNDS)}")
