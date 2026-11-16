"""Input and output models for deterministic portfolio rebalancing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Mapping, Optional


class PlanStatus(str, Enum):
    """Outcome of a rebalance plan."""

    COMPLETED = "completed"
    PARTIALLY_CONSTRAINED = "partially_constrained"
    REJECTED = "rejected"


@dataclass(frozen=True)
class RebalanceConfig:
    """Constraints and execution assumptions for a simulated rebalance.

    Rates are fractions unless named ``*_bps``. For example, a 0.001
    transaction cost is 10 basis points.
    """

    max_weight_per_asset: float = 1.0
    per_asset_max_weights: Mapping[str, float] = field(default_factory=dict)
    cash_reserve_pct: float = 0.0
    transaction_cost_rate: float = 0.0
    slippage_bps: float = 0.0
    allow_fractional_shares: bool = False
    fractional_share_precision: int = 6
    max_turnover: float = 1.0
    weight_sum_tolerance: float = 1e-8

    def validate(self) -> None:
        """Raise ``ValueError`` for an invalid configuration."""
        _require_finite("max_weight_per_asset", self.max_weight_per_asset)
        if not 0.0 <= self.max_weight_per_asset <= 1.0:
            raise ValueError("max_weight_per_asset must be between 0 and 1")

        if not isinstance(self.per_asset_max_weights, Mapping):
            raise ValueError("per_asset_max_weights must be a mapping")
        for symbol, weight in self.per_asset_max_weights.items():
            _require_finite(f"per_asset_max_weights[{symbol!r}]", weight)
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Per-asset cap symbols must be non-empty")
            if not 0.0 <= weight <= 1.0:
                raise ValueError(
                    f"Per-asset cap for {symbol!r} must be between 0 and 1"
                )

        _require_finite("cash_reserve_pct", self.cash_reserve_pct)
        if not 0.0 <= self.cash_reserve_pct <= 1.0:
            raise ValueError("cash_reserve_pct must be between 0 and 1")

        _require_finite("transaction_cost_rate", self.transaction_cost_rate)
        if not 0.0 <= self.transaction_cost_rate < 1.0:
            raise ValueError("transaction_cost_rate must be at least 0 and less than 1")

        _require_finite("slippage_bps", self.slippage_bps)
        if not 0.0 <= self.slippage_bps < 10_000.0:
            raise ValueError("slippage_bps must be at least 0 and less than 10000")

        if not isinstance(self.allow_fractional_shares, bool):
            raise ValueError("allow_fractional_shares must be a bool")
        if (
            not isinstance(self.fractional_share_precision, int)
            or isinstance(self.fractional_share_precision, bool)
            or self.fractional_share_precision < 0
        ):
            raise ValueError("fractional_share_precision must be a non-negative integer")

        _require_finite("max_turnover", self.max_turnover)
        if self.max_turnover < 0.0:
            raise ValueError("max_turnover must be non-negative")

        _require_finite("weight_sum_tolerance", self.weight_sum_tolerance)
        if not 0.0 <= self.weight_sum_tolerance < 1.0:
            raise ValueError("weight_sum_tolerance must be at least 0 and less than 1")

    def cap_for(self, symbol: str) -> float:
        """Return the effective per-asset cap."""
        return self.per_asset_max_weights.get(symbol, self.max_weight_per_asset)


@dataclass(frozen=True)
class RebalanceRequest:
    """A portfolio snapshot and desired long-only target allocation.

    Target weights are relative to the investable sleeve comprising holdings
    and settled cash. Dividend receivables are not spendable and must remain
    in the separate whole-account corporate-action ledger.

    Target weights may omit ``target_cash_weight``; in that case cash is the
    residual to 100%. A held symbol omitted from ``target_weights`` has a
    target weight of zero and is eligible for liquidation.
    """

    current_holdings: Mapping[str, float]
    available_cash: float
    prices: Mapping[str, float]
    target_weights: Mapping[str, float]
    config: RebalanceConfig = field(default_factory=RebalanceConfig)
    target_cash_weight: Optional[float] = None
    execution_prices: Optional[Mapping[str, float]] = None
    mark_prices: Optional[Mapping[str, float]] = None
    initial_mark_prices: Optional[Mapping[str, float]] = None


@dataclass(frozen=True)
class AdjustmentEvent:
    """Auditable record of a target or trade constraint adjustment."""

    code: str
    message: str
    symbol: Optional[str] = None
    requested_value: Optional[float] = None
    applied_value: Optional[float] = None


@dataclass(frozen=True)
class Trade:
    """A simulated trade; it is a report only and is never submitted."""

    symbol: str
    side: str
    quantity: float
    reference_price: float
    execution_price: float
    reference_notional: float
    execution_notional: float
    fee: float
    slippage_cost: float
    reason: str


@dataclass(frozen=True)
class RebalanceResult:
    """Complete simulated rebalance result and audit trail."""

    status: PlanStatus
    initial_nav: float
    final_nav: float
    adjusted_target_weights: Mapping[str, float]
    cash_target_weight: float
    target_values: Mapping[str, float]
    cash_target_value: float
    trades: tuple[Trade, ...]
    final_holdings: Mapping[str, float]
    final_cash: float
    total_fees: float
    total_slippage_cost: float
    total_transaction_cost: float
    actual_weights: Mapping[str, float]
    actual_cash_weight: float
    gross_turnover: float
    turnover_ratio: float
    audit_events: tuple[AdjustmentEvent, ...]
    rejection_reasons: tuple[str, ...] = ()


def _require_finite(name: str, value: float) -> None:
    """Validate a scalar numeric configuration value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    if not float("-inf") < float(value) < float("inf"):
        raise ValueError(f"{name} must be a finite number")
