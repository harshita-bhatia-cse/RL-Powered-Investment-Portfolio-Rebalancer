"""Pydantic contracts for historical backtest requests and responses."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

BacktestPartition = Literal["train", "validation", "test"]


class HistoricalBacktestRequest(BaseModel):
    """Request a supported deterministic strategy on one fixed partition."""

    strategy: str
    partition: BacktestPartition
    initial_capital: float = Field(default=100_000.0, gt=0.0, allow_inf_nan=False)
    parameters: dict[str, float] = Field(default_factory=dict)


class HistoricalComparisonRequest(BaseModel):
    """Compare supported strategies under a shared capital and partition."""

    strategies: list[str] = Field(min_length=1)
    partition: BacktestPartition
    initial_capital: float = Field(default=100_000.0, gt=0.0, allow_inf_nan=False)
    parameters: dict[str, dict[str, float]] = Field(default_factory=dict)

    @field_validator("strategies")
    @classmethod
    def strategy_names_must_be_unique(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("strategies must not contain duplicates")
        return values


class HistoricalPerformanceMetricsResponse(BaseModel):
    """Summary metrics derived from realized NAV and environment costs."""

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


class NAVPoint(BaseModel):
    date: date
    nav: float


class DailyReturnPoint(BaseModel):
    date: date
    value: float


class TransactionCostSummary(BaseModel):
    turnover: float
    transaction_fees: float
    slippage_cost: float
    total_transaction_cost: float


class CorporateActionSourceMetadata(BaseModel):
    """Manifest provenance for the normalized event artifact used in a run."""

    provider: str
    source_workbook_url: str
    artifact_filename: str
    sha256: str
    bytes: int
    date_range: dict[str, str]
    limitations: list[str] = Field(default_factory=list)


class HistoricalBacktestResponse(BaseModel):
    strategy: str
    partition: BacktestPartition
    start_date: date
    end_date: date
    number_of_trading_sessions: int
    initial_nav: float
    final_nav: float
    metrics: HistoricalPerformanceMetricsResponse
    nav_history: list[NAVPoint]
    daily_returns: list[DailyReturnPoint]
    transaction_cost_summary: TransactionCostSummary
    corporate_actions_verified: bool
    corporate_action_limitation: str | None
    return_accuracy: str
    warnings: list[str]
    corporate_action_source_configured: bool = False
    corporate_action_source_validated: bool = False
    corporate_action_source: CorporateActionSourceMetadata | None = None
    corporate_action_events_loaded: int = Field(
        default=0,
        description="Total normalized events loaded from the validated artifact.",
    )
    corporate_action_events_for_partition: int = Field(
        default=0,
        description="Partition-filtered events passed to this environment.",
    )


class HistoricalComparisonResponse(BaseModel):
    partition: BacktestPartition
    initial_capital: float
    results: list[HistoricalBacktestResponse]


class HistoricalDatasetValidation(BaseModel):
    valid: bool
    partitions: dict[str, dict[str, float | int | str]] | None = None
    error: str | None = None


class HistoricalBacktestHealthResponse(BaseModel):
    available: bool
    dataset_validation: HistoricalDatasetValidation
    supported_strategies: list[str]
    corporate_actions_verified: bool
    limitation: str
    corporate_action_source_configured: bool = False
    corporate_action_source_validated: bool = False
    corporate_action_events_load_succeeded: bool = False
    corporate_action_source: CorporateActionSourceMetadata | None = None
    corporate_action_events_loaded: int = 0
    corporate_action_partition_event_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Partition-filtered events passed to each environment.",
    )
