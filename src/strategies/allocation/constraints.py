"""Target-weight validation and deterministic constraint projection."""

from __future__ import annotations

import math
from numbers import Real
from typing import Mapping

from .models import AdjustmentEvent, RebalanceConfig


def validate_and_adjust_weights(
    target_weights: Mapping[str, float],
    target_cash_weight: float | None,
    config: RebalanceConfig,
) -> tuple[dict[str, float], float, list[AdjustmentEvent]]:
    """Validate target weights, enforce cash reserve and asset caps.

    A target sum outside the configured tolerance is rejected rather than
    silently normalized. Small floating-point drift is normalized and logged.
    Excess weight caused by caps is redistributed proportionally among
    positively weighted, uncapped assets; any unallocatable remainder becomes
    cash.
    """
    config.validate()
    events: list[AdjustmentEvent] = []
    assets: dict[str, float] = {}

    for symbol, raw_weight in target_weights.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Target weight symbols must be non-empty strings")
        weight = _finite_number(f"target_weights[{symbol!r}]", raw_weight)
        if weight < 0.0:
            raise ValueError(f"Target weight for {symbol!r} must be non-negative")
        assets[symbol] = weight

    asset_total = math.fsum(assets.values())
    if target_cash_weight is None:
        if asset_total > 1.0 + config.weight_sum_tolerance:
            raise ValueError("Asset target weights exceed 100% and cash was omitted")
        cash_weight = max(0.0, 1.0 - asset_total)
    else:
        cash_weight = _finite_number("target_cash_weight", target_cash_weight)
        if cash_weight < 0.0:
            raise ValueError("target_cash_weight must be non-negative")

    total = math.fsum((*assets.values(), cash_weight))
    if total <= 0.0:
        raise ValueError("Target weights must have a positive total")
    if abs(total - 1.0) > config.weight_sum_tolerance:
        raise ValueError(
            f"Target asset and cash weights must sum to 1.0; received {total:.12g}"
        )
    if total != 1.0:
        assets = {symbol: weight / total for symbol, weight in assets.items()}
        cash_weight /= total
        events.append(
            AdjustmentEvent(
                code="weight_normalization",
                message="Normalized target weights to correct floating-point drift",
                requested_value=total,
                applied_value=1.0,
            )
        )

    initial_assets = dict(assets)
    if cash_weight < config.cash_reserve_pct:
        old_cash = cash_weight
        investable_weight = 1.0 - config.cash_reserve_pct
        asset_total = math.fsum(assets.values())
        assets = (
            {
                symbol: weight * investable_weight / asset_total
                for symbol, weight in assets.items()
            }
            if asset_total > 0.0
            else {symbol: 0.0 for symbol in assets}
        )
        cash_weight = config.cash_reserve_pct
        events.append(
            AdjustmentEvent(
                code="cash_reserve",
                message="Reduced asset targets to satisfy the configured cash reserve",
                requested_value=old_cash,
                applied_value=cash_weight,
            )
        )

    assets, capped_symbols = _apply_asset_caps(assets, config)
    for symbol in capped_symbols:
        events.append(
            AdjustmentEvent(
                code="asset_cap",
                message="Applied the configured maximum allocation and redistributed excess",
                symbol=symbol,
                requested_value=initial_assets.get(symbol, 0.0),
                applied_value=assets[symbol],
            )
        )

    cash_weight = max(
        config.cash_reserve_pct,
        1.0 - math.fsum(assets.values()),
    )
    # Correct only numerical drift; the cap projection never allocates over 1.
    total = math.fsum((*assets.values(), cash_weight))
    if total > 1.0:
        excess = total - 1.0
        cash_weight -= excess
    elif total < 1.0:
        cash_weight += 1.0 - total

    return assets, cash_weight, events


def _apply_asset_caps(
    weights: Mapping[str, float],
    config: RebalanceConfig,
) -> tuple[dict[str, float], list[str]]:
    """Project weights onto per-asset upper bounds deterministically."""
    projected = dict(weights)
    capped: list[str] = []
    for symbol in sorted(projected):
        cap = config.cap_for(symbol)
        if projected[symbol] > cap:
            projected[symbol] = cap
            capped.append(symbol)

    excess = max(0.0, math.fsum(weights.values()) - math.fsum(projected.values()))
    while excess > 1e-14:
        eligible = [
            symbol
            for symbol in sorted(projected)
            if projected[symbol] > 0.0
            and projected[symbol] < config.cap_for(symbol) - 1e-14
        ]
        if not eligible:
            break
        eligible_total = math.fsum(projected[symbol] for symbol in eligible)
        if eligible_total <= 0.0:
            break

        distributed = 0.0
        for symbol in eligible:
            share = excess * projected[symbol] / eligible_total
            capacity = config.cap_for(symbol) - projected[symbol]
            addition = min(share, capacity)
            projected[symbol] += addition
            distributed += addition
        if distributed <= 1e-14:
            break
        excess -= distributed

    return projected, capped


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number
