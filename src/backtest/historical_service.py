"""CSV-backed, broker-independent historical portfolio backtesting service."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date
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
from src.data.corporate_action_provider import (
    CorporateActionValidationError,
    load_normalized_events,
    validate_events,
)
from src.rl.environment.portfolio_env import (
    BENCHMARK_SYMBOL,
    PARTITION_BOUNDS,
    PortfolioTradingEnv,
    TrainOnlyFeatureScaler,
    _normalize_market_data,
    fit_training_scaler,
    load_market_data,
)
from src.strategies.allocation import CorporateActionEvent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICAL_DATA_DIR = PROJECT_ROOT / "data" / "historical_market_csv"
HISTORICAL_VERIFIED_CORPORATE_ACTIONS_ENV = "HISTORICAL_VERIFIED_CORPORATE_ACTIONS_CSV"
ISSUER_VERIFIED_EVENTS_FILENAME = "verified_normalized_events.csv"
ISSUER_VERIFICATION_MANIFEST_FILENAME = "manifest.json"
ISSUER_VERIFICATION_PROVIDER = (
    "State Street Global Advisors; Yahoo candidates used only for comparison"
)
ISSUER_VERIFICATION_SOURCE_WORKBOOK_URL = (
    "https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/"
    "etfs/us/spdr-etf-historical-distributions.xlsx"
)


class HistoricalDataUnavailableError(RuntimeError):
    """Raised when the CSV dataset cannot satisfy the environment data contract."""


class UnsupportedStrategyError(ValueError):
    """Raised for unknown strategies or unsupported strategy parameters."""


class CorporateActionSourceValidationError(ValueError):
    """Raised when a configured issuer-verified event artifact cannot be validated."""


def validate_issuer_verified_corporate_action_source(
    events_path: str | Path,
) -> dict[str, Any]:
    """Validate issuer-workflow provenance and artifact integrity without loading events.

    The manifest checksum is provenance evidence only; it is not a digital
    signature and does not establish cryptographic authenticity.
    """
    path = Path(events_path)
    if path.name != ISSUER_VERIFIED_EVENTS_FILENAME:
        raise CorporateActionSourceValidationError(
            "Expected the issuer workflow artifact "
            f"{ISSUER_VERIFIED_EVENTS_FILENAME!r}; Yahoo candidates, verification "
            "templates, and arbitrary normalized CSVs are not verified sources."
        )
    if not path.is_file():
        raise CorporateActionSourceValidationError(
            f"Configured issuer-verified event CSV does not exist: {path}"
        )

    manifest_path = path.parent / ISSUER_VERIFICATION_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise CorporateActionSourceValidationError(
            f"Required adjacent issuer verification manifest is missing: "
            f"{manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorporateActionSourceValidationError(
            f"Issuer verification manifest is unreadable or malformed: "
            f"{manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise CorporateActionSourceValidationError(
            "Issuer verification manifest must contain a JSON object."
        )
    if manifest.get("provider") != ISSUER_VERIFICATION_PROVIDER:
        raise CorporateActionSourceValidationError(
            "Unsupported issuer verification provider/workflow identity in "
            f"{manifest_path}."
        )
    if manifest.get("source_workbook_url") != ISSUER_VERIFICATION_SOURCE_WORKBOOK_URL:
        raise CorporateActionSourceValidationError(
            "Issuer verification manifest has an unsupported source_workbook_url."
        )

    date_range = manifest.get("date_range")
    if not isinstance(date_range, dict):
        raise CorporateActionSourceValidationError(
            "Issuer verification manifest must include a date_range object."
        )
    date_values: dict[str, date] = {}
    for field in ("start", "end"):
        value = date_range.get(field)
        if not isinstance(value, str):
            raise CorporateActionSourceValidationError(
                f"Issuer verification manifest date_range.{field} must be an "
                "ISO-format date (YYYY-MM-DD)."
            )
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError as exc:
            raise CorporateActionSourceValidationError(
                f"Issuer verification manifest date_range.{field} must be a "
                "valid ISO-format date (YYYY-MM-DD)."
            ) from exc
        if parsed_date.isoformat() != value:
            raise CorporateActionSourceValidationError(
                f"Issuer verification manifest date_range.{field} must use "
                "ISO format YYYY-MM-DD."
            )
        date_values[field] = parsed_date
    if date_values["start"] > date_values["end"]:
        raise CorporateActionSourceValidationError(
            "Issuer verification manifest date_range.start must not be after end."
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise CorporateActionSourceValidationError(
            "Issuer verification manifest must include an artifacts object."
        )
    artifact = artifacts.get(ISSUER_VERIFIED_EVENTS_FILENAME)
    if not isinstance(artifact, dict):
        raise CorporateActionSourceValidationError(
            "Issuer verification manifest does not declare "
            f"{ISSUER_VERIFIED_EVENTS_FILENAME!r} in artifacts."
        )
    declared_bytes = artifact.get("bytes")
    if (
        isinstance(declared_bytes, bool)
        or not isinstance(declared_bytes, int)
        or declared_bytes < 0
    ):
        raise CorporateActionSourceValidationError(
            "Issuer verification artifact bytes must be a non-negative integer."
        )
    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise CorporateActionSourceValidationError(
            f"Unable to inspect configured issuer-verified event CSV: {path}"
        ) from exc
    if actual_bytes != declared_bytes:
        raise CorporateActionSourceValidationError(
            f"Byte-size mismatch for {path.name}: manifest declares "
            f"{declared_bytes} bytes but file contains {actual_bytes} bytes."
        )

    declared_sha256 = artifact.get("sha256")
    if (
        not isinstance(declared_sha256, str)
        or len(declared_sha256) != 64
        or any(character not in "0123456789abcdef" for character in declared_sha256)
    ):
        raise CorporateActionSourceValidationError(
            "Issuer verification artifact sha256 must be a 64-character "
            "lowercase hexadecimal digest."
        )
    try:
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CorporateActionSourceValidationError(
            f"Unable to read configured issuer-verified event CSV: {path}"
        ) from exc
    if actual_sha256 != declared_sha256:
        raise CorporateActionSourceValidationError(
            f"SHA-256 checksum mismatch for {path.name}; regenerate or restore "
            "the issuer verification artifact and adjacent manifest together."
        )

    return manifest


class HistoricalBacktestService:
    """Load historical CSV data and run deterministic environment strategies."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        corporate_action_events_path: str | Path | None = None,
    ) -> None:
        self.data_dir = (
            Path(data_dir) if data_dir is not None else DEFAULT_HISTORICAL_DATA_DIR
        )
        configured_events_path = (
            corporate_action_events_path
            if corporate_action_events_path is not None
            else os.environ.get(HISTORICAL_VERIFIED_CORPORATE_ACTIONS_ENV)
        )
        self.corporate_action_events_path = (
            Path(configured_events_path) if configured_events_path else None
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
        market_data, scaler, corporate_action_events, source_metadata = (
            self._load_context()
        )
        return self._run_with_context(
            strategy,
            market_data=market_data,
            scaler=scaler,
            corporate_action_events=corporate_action_events,
            corporate_action_source_metadata=source_metadata,
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

        market_data, scaler, corporate_action_events, source_metadata = (
            self._load_context()
        )
        results = [
            self._run_with_context(
                strategy,
                market_data=market_data,
                scaler=scaler,
                corporate_action_events=corporate_action_events,
                corporate_action_source_metadata=source_metadata,
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
        verified_source: tuple[Path, dict[str, Any]] | None = None
        source_provenance: dict[str, Any] | None = None
        events_load_succeeded = False
        corporate_action_events: tuple[CorporateActionEvent, ...] = ()
        partition_event_counts: dict[str, int] = {}
        try:
            verified_source = self._validate_configured_corporate_action_source()
            if verified_source is not None:
                source_provenance = self._corporate_action_source_metadata(
                    verified_source[1]
                )
            (
                market_data,
                scaler,
                corporate_action_events,
                source_metadata,
            ) = self._load_context(verified_source=verified_source)
            events_load_succeeded = source_metadata is not None
            partitions: dict[str, dict[str, Any]] = {}
            for name in PARTITION_BOUNDS:
                env, partition_events = self._build_env_with_events(
                    market_data,
                    scaler,
                    partition=name,
                    initial_capital=100_000.0,
                    corporate_action_events=corporate_action_events,
                    corporate_action_source_metadata=source_metadata,
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
                partition_event_counts[name] = len(partition_events)
            return {
                "available": True,
                "dataset_validation": {"valid": True, "partitions": partitions},
                "supported_strategies": list(SUPPORTED_STRATEGIES),
                **self._corporate_action_health_metadata(
                    source_configured=self.corporate_action_events_path is not None,
                    source_provenance_validated=verified_source is not None,
                    events_load_succeeded=events_load_succeeded,
                    source_metadata=source_metadata,
                    events_loaded=len(corporate_action_events),
                    partition_event_counts=partition_event_counts,
                ),
            }
        except (
            CorporateActionSourceValidationError,
            CorporateActionValidationError,
            HistoricalDataUnavailableError,
            OSError,
        ) as exc:
            return {
                "available": False,
                "dataset_validation": {"valid": False, "error": str(exc)},
                "supported_strategies": list(SUPPORTED_STRATEGIES),
                **self._corporate_action_health_metadata(
                    source_configured=self.corporate_action_events_path is not None,
                    source_provenance_validated=verified_source is not None,
                    events_load_succeeded=events_load_succeeded,
                    source_metadata=source_provenance,
                    events_loaded=len(corporate_action_events),
                    partition_event_counts=partition_event_counts,
                ),
            }

    def _load_context(
        self,
        *,
        verified_source: tuple[Path, dict[str, Any]] | None = None,
    ) -> tuple[
        dict[str, pd.DataFrame],
        TrainOnlyFeatureScaler,
        tuple[CorporateActionEvent, ...],
        dict[str, Any] | None,
    ]:
        if verified_source is None:
            verified_source = self._validate_configured_corporate_action_source()
        try:
            market_data = load_market_data(self.data_dir)
            scaler = fit_training_scaler(market_data)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise HistoricalDataUnavailableError(
                f"Historical CSV data is unavailable or invalid: {exc}"
            ) from exc
        corporate_action_events = (
            self._load_corporate_action_events(market_data, *verified_source)
            if verified_source is not None
            else ()
        )
        source_metadata = (
            self._corporate_action_source_metadata(verified_source[1])
            if verified_source is not None
            else None
        )
        return market_data, scaler, corporate_action_events, source_metadata

    def _validate_configured_corporate_action_source(
        self,
    ) -> tuple[Path, dict[str, Any]] | None:
        if self.corporate_action_events_path is None:
            return None
        manifest = validate_issuer_verified_corporate_action_source(
            self.corporate_action_events_path
        )
        return self.corporate_action_events_path, manifest

    @staticmethod
    def _corporate_action_source_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
        artifact = manifest["artifacts"][ISSUER_VERIFIED_EVENTS_FILENAME]
        return {
            "provider": manifest["provider"],
            "source_workbook_url": manifest["source_workbook_url"],
            "artifact_filename": ISSUER_VERIFIED_EVENTS_FILENAME,
            "sha256": artifact["sha256"],
            "bytes": artifact["bytes"],
            "date_range": dict(manifest["date_range"]),
            "limitations": list(manifest.get("limitations", [])),
        }

    @staticmethod
    def _corporate_action_health_metadata(
        *,
        source_configured: bool,
        source_provenance_validated: bool,
        events_load_succeeded: bool,
        source_metadata: dict[str, Any] | None,
        events_loaded: int,
        partition_event_counts: dict[str, int],
    ) -> dict[str, Any]:
        actions_verified = source_configured and events_load_succeeded
        if actions_verified:
            limitation = (
                "The configured source and normalized events were validated, but "
                "event coverage is limited to the supplied records; completeness "
                "and total-return accuracy are not guaranteed. Manifest checksums "
                "are provenance evidence, not digital signatures."
            )
        else:
            limitation = (
                "Historical corporate-action coverage is unverified; results "
                "are not guaranteed to be total-return accurate."
            )
        return {
            "corporate_actions_verified": actions_verified,
            "corporate_action_source_configured": source_configured,
            "corporate_action_source_validated": source_provenance_validated,
            "corporate_action_events_load_succeeded": events_load_succeeded,
            "corporate_action_source": source_metadata,
            "corporate_action_events_loaded": events_loaded,
            "corporate_action_partition_event_counts": partition_event_counts,
            "limitation": limitation,
        }

    def _load_corporate_action_events(
        self,
        market_data: dict[str, pd.DataFrame],
        events_path: Path,
        manifest: dict[str, Any],
    ) -> tuple[CorporateActionEvent, ...]:
        """Load validated events against manifest bounds and the full market calendar."""
        date_range = manifest["date_range"]
        common_start = max(frame.index.min() for frame in market_data.values())
        common_end = min(frame.index.max() for frame in market_data.values())
        benchmark_dates = pd.DatetimeIndex(market_data[BENCHMARK_SYMBOL].index)
        market_sessions = benchmark_dates[
            (benchmark_dates >= common_start) & (benchmark_dates <= common_end)
        ]
        return load_normalized_events(
            events_path,
            start_date=date_range["start"],
            end_date=date_range["end"],
            market_sessions=market_sessions,
        )

    def _build_env(
        self,
        market_data: dict[str, pd.DataFrame],
        scaler: TrainOnlyFeatureScaler,
        *,
        partition: str,
        initial_capital: float,
        corporate_action_events: tuple[CorporateActionEvent, ...] = (),
        corporate_action_source_metadata: dict[str, Any] | None = None,
    ) -> PortfolioTradingEnv:
        env, _ = self._build_env_with_events(
            market_data,
            scaler,
            partition=partition,
            initial_capital=initial_capital,
            corporate_action_events=corporate_action_events,
            corporate_action_source_metadata=corporate_action_source_metadata,
        )
        return env

    def _build_env_with_events(
        self,
        market_data: dict[str, pd.DataFrame],
        scaler: TrainOnlyFeatureScaler,
        *,
        partition: str,
        initial_capital: float,
        corporate_action_events: tuple[CorporateActionEvent, ...] = (),
        corporate_action_source_metadata: dict[str, Any] | None = None,
    ) -> tuple[PortfolioTradingEnv, tuple[CorporateActionEvent, ...]]:
        if partition not in PARTITION_BOUNDS:
            raise ValueError(f"partition must be one of {tuple(PARTITION_BOUNDS)}")
        partition_events = self._partition_corporate_action_events(
            market_data, partition, corporate_action_events
        )
        try:
            env = PortfolioTradingEnv(
                market_data,
                partition=partition,
                initial_cash=initial_capital,
                corporate_action_events=partition_events,
                corporate_actions_verified=corporate_action_source_metadata is not None,
                corporate_action_source=(
                    corporate_action_source_metadata["provider"]
                    if corporate_action_source_metadata is not None
                    else None
                ),
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
        return env, partition_events

    @staticmethod
    def _partition_corporate_action_events(
        market_data: dict[str, pd.DataFrame],
        partition: str,
        corporate_action_events: tuple[CorporateActionEvent, ...],
    ) -> tuple[CorporateActionEvent, ...]:
        """Select and revalidate only actions occurring in a partition."""
        if not corporate_action_events:
            return ()

        _, all_sessions = _normalize_market_data(market_data)
        start, end = PARTITION_BOUNDS[partition]
        partition_sessions = all_sessions[
            (all_sessions >= start) & (all_sessions <= end)
        ]
        if partition_sessions.empty:
            raise HistoricalDataUnavailableError(
                f"Historical data has no sessions in the {partition} partition"
            )

        partition_start = partition_sessions[0].date()
        partition_end = partition_sessions[-1].date()
        selected_events = tuple(
            event
            for event in corporate_action_events
            if partition_start <= event.action_date <= partition_end
        )
        return validate_events(
            selected_events,
            start_date=partition_start,
            end_date=partition_end,
            market_sessions=partition_sessions,
        )

    def _run_with_context(
        self,
        strategy: HistoricalStrategy,
        *,
        market_data: dict[str, pd.DataFrame],
        scaler: TrainOnlyFeatureScaler,
        corporate_action_events: tuple[CorporateActionEvent, ...],
        corporate_action_source_metadata: dict[str, Any] | None,
        partition: str,
        initial_capital: float,
    ) -> dict[str, Any]:
        env, partition_events = self._build_env_with_events(
            market_data,
            scaler,
            partition=partition,
            initial_capital=initial_capital,
            corporate_action_events=corporate_action_events,
            corporate_action_source_metadata=corporate_action_source_metadata,
        )
        observation, initial_info = env.reset(seed=42)
        dates = [pd.Timestamp(initial_info["date"])]
        nav_values = [float(initial_info["nav"])]
        turnover = 0.0
        transaction_fees = 0.0
        slippage_cost = 0.0
        warnings: list[str] = []
        corporate_action_limitation = initial_info.get("corporate_action_limitation")
        if corporate_action_source_metadata is not None:
            corporate_action_limitation = (
                "The configured issuer-workflow source and normalized events were "
                "validated, but event coverage is limited to supplied records; "
                "completeness and total-return accuracy are not guaranteed. "
                "Manifest checksums are provenance evidence, not digital signatures."
            )
            warnings.append(corporate_action_limitation)
            warnings.extend(
                f"Source limitation: {limitation}"
                for limitation in corporate_action_source_metadata["limitations"]
            )
        if corporate_action_limitation:
            if corporate_action_source_metadata is None:
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
                corporate_action_source_metadata is not None
            ),
            "corporate_action_limitation": corporate_action_limitation,
            "return_accuracy": initial_info["return_accuracy"],
            "corporate_action_source_configured": (
                self.corporate_action_events_path is not None
            ),
            "corporate_action_source_validated": (
                corporate_action_source_metadata is not None
            ),
            "corporate_action_source": corporate_action_source_metadata,
            "corporate_action_events_loaded": len(corporate_action_events),
            "corporate_action_events_for_partition": len(partition_events),
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
