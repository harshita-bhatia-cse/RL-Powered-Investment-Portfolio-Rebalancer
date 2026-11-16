"""Deterministic, broker-independent portfolio rebalancing."""

from __future__ import annotations

import math
from numbers import Real
from typing import Mapping

from .constraints import validate_and_adjust_weights
from .models import (
    AdjustmentEvent,
    PlanStatus,
    RebalanceConfig,
    RebalanceRequest,
    RebalanceResult,
    Trade,
)


def rebalance(request: RebalanceRequest) -> RebalanceResult:
    """Create a simulated target rebalance without placing brokerage orders.

    ``prices`` remains the backwards-compatible price input. Optional
    ``execution_prices`` and ``mark_prices`` override it for fills and final
    valuation respectively. ``initial_mark_prices`` can independently value
    the investable portfolio before execution; it defaults to ``mark_prices``.
    Target share quantities and turnover use execution prices. Simulated
    execution prices include adverse slippage.

    Invalid input raises ``ValueError``. A valid request that cannot preserve
    the cash reserve under its constraints returns a rejected, no-trade result.
    """
    holdings = _validated_mapping("current_holdings", request.current_holdings)
    legacy_prices = _validated_mapping("prices", request.prices)
    execution_prices = _validated_mapping(
        "execution_prices",
        request.execution_prices
        if request.execution_prices is not None
        else legacy_prices,
    )
    mark_prices = _validated_mapping(
        "mark_prices",
        request.mark_prices if request.mark_prices is not None else legacy_prices,
    )
    initial_mark_prices = _validated_mapping(
        "initial_mark_prices",
        request.initial_mark_prices
        if request.initial_mark_prices is not None
        else mark_prices,
    )
    targets = _validated_mapping("target_weights", request.target_weights)
    cash = _finite_number("available_cash", request.available_cash)
    if cash < 0.0:
        raise ValueError("available_cash must be non-negative")

    for symbol, quantity in holdings.items():
        if quantity < 0.0:
            raise ValueError(f"Current holdings for {symbol!r} must be non-negative")
    for symbol, price in execution_prices.items():
        if price <= 0.0:
            raise ValueError(f"Execution price for {symbol!r} must be positive")
    for symbol, price in mark_prices.items():
        if price <= 0.0:
            raise ValueError(f"Mark price for {symbol!r} must be positive")
    for symbol, price in initial_mark_prices.items():
        if price <= 0.0:
            raise ValueError(f"Initial mark price for {symbol!r} must be positive")

    needed_symbols = set(holdings) | set(targets)
    missing_execution = sorted(
        symbol for symbol in needed_symbols if symbol not in execution_prices
    )
    if missing_execution:
        raise ValueError(
            "Missing positive execution prices for symbols: "
            + ", ".join(missing_execution)
        )
    missing_marks = sorted(symbol for symbol in needed_symbols if symbol not in mark_prices)
    if missing_marks:
        raise ValueError(
            "Missing positive mark prices for symbols: " + ", ".join(missing_marks)
        )
    missing_initial_marks = sorted(
        symbol for symbol in needed_symbols if symbol not in initial_mark_prices
    )
    if missing_initial_marks:
        raise ValueError(
            "Missing positive initial mark prices for symbols: "
            + ", ".join(missing_initial_marks)
        )

    config = request.config
    if not isinstance(config, RebalanceConfig):
        raise ValueError("config must be a RebalanceConfig")
    config.validate()
    initial_nav = cash + math.fsum(
        quantity * initial_mark_prices[symbol]
        for symbol, quantity in holdings.items()
    )
    if not math.isfinite(initial_nav) or initial_nav <= 0.0:
        raise ValueError("Initial portfolio NAV must be positive and finite")

    adjusted_targets, target_cash_weight, audit = validate_and_adjust_weights(
        targets,
        request.target_cash_weight,
        config,
    )
    all_symbols = sorted(set(holdings) | set(adjusted_targets))
    target_values = {
        symbol: initial_nav * adjusted_targets.get(symbol, 0.0)
        for symbol in all_symbols
    }
    cash_target_value = initial_nav * target_cash_weight
    cash_reserve = initial_nav * config.cash_reserve_pct

    ideal_deltas = {
        symbol: target_values[symbol] / execution_prices[symbol]
        - holdings.get(symbol, 0.0)
        for symbol in all_symbols
    }
    desired_turnover = math.fsum(
        abs(delta) * execution_prices[symbol]
        for symbol, delta in ideal_deltas.items()
    )
    desired_turnover_ratio = desired_turnover / initial_nav
    turnover_scale = (
        min(1.0, config.max_turnover / desired_turnover_ratio)
        if desired_turnover_ratio > 0.0
        else 1.0
    )
    if turnover_scale < 1.0:
        audit.append(
            AdjustmentEvent(
                code="turnover_limit",
                message="Scaled target trades to meet the maximum gross turnover",
                requested_value=desired_turnover_ratio,
                applied_value=config.max_turnover,
            )
        )

    deltas = {
        symbol: delta * turnover_scale for symbol, delta in ideal_deltas.items()
    }
    rounded_deltas = {
        symbol: _round_toward_zero(delta, config)
        for symbol, delta in deltas.items()
    }
    for symbol in all_symbols:
        if not _close(rounded_deltas[symbol], ideal_deltas[symbol]):
            code = (
                "share_rounding"
                if not _close(rounded_deltas[symbol], deltas[symbol])
                else "turnover_limit"
            )
            audit.append(
                AdjustmentEvent(
                    code=code,
                    message=(
                        "Rounded the trade quantity toward zero to respect share "
                        "precision and execution limits"
                    ),
                    symbol=symbol,
                    requested_value=ideal_deltas[symbol],
                    applied_value=rounded_deltas[symbol],
                )
            )

    trades: list[Trade] = []
    final_holdings = {
        symbol: holdings.get(symbol, 0.0) for symbol in all_symbols
    }
    final_cash = cash
    total_fees = 0.0
    total_slippage = 0.0
    slippage_fraction = config.slippage_bps / 10_000.0

    # Process sells first in stable symbol order.
    for symbol in all_symbols:
        delta = rounded_deltas[symbol]
        if delta >= 0.0:
            continue
        quantity = min(-delta, final_holdings[symbol])
        quantity = _round_toward_zero(quantity, config)
        if quantity <= 0.0:
            continue
        reference_price = execution_prices[symbol]
        execution_price = reference_price * (1.0 - slippage_fraction)
        execution_notional = quantity * execution_price
        fee = execution_notional * config.transaction_cost_rate
        slippage_cost = quantity * (reference_price - execution_price)
        final_cash += execution_notional - fee
        final_holdings[symbol] -= quantity
        total_fees += fee
        total_slippage += slippage_cost
        trades.append(
            Trade(
                symbol=symbol,
                side="sell",
                quantity=quantity,
                reference_price=reference_price,
                execution_price=execution_price,
                reference_notional=quantity * reference_price,
                execution_notional=execution_notional,
                fee=fee,
                slippage_cost=slippage_cost,
                reason="Move holdings toward the constrained target allocation",
            )
        )

    if final_cash < cash_reserve - 1e-9:
        return _rejected_result(
            initial_nav=initial_nav,
            holdings=holdings,
            cash=cash,
            mark_prices=mark_prices,
            targets=adjusted_targets,
            target_cash_weight=target_cash_weight,
            target_values=target_values,
            cash_target_value=cash_target_value,
            audit=audit
            + [
                AdjustmentEvent(
                    code="cash_reserve_infeasible",
                    message=(
                        "Sells permitted by the turnover constraint cannot raise "
                        "cash to the required reserve"
                    ),
                    requested_value=cash_reserve,
                    applied_value=final_cash,
                )
            ],
            reason="Required cash reserve cannot be maintained",
        )

    # Proportionally reduce buys if desired executions would spend the reserve.
    buy_quantities = {
        symbol: max(0.0, rounded_deltas[symbol]) for symbol in all_symbols
    }
    available_for_buys = max(0.0, final_cash - cash_reserve)
    desired_buy_cost = math.fsum(
        quantity
        * execution_prices[symbol]
        * (1.0 + slippage_fraction)
        * (1.0 + config.transaction_cost_rate)
        for symbol, quantity in buy_quantities.items()
    )
    buy_scale = (
        min(1.0, available_for_buys / desired_buy_cost)
        if desired_buy_cost > 0.0
        else 1.0
    )
    if buy_scale < 1.0:
        audit.append(
            AdjustmentEvent(
                code="cash_affordability",
                message="Reduced buy quantities to preserve the minimum cash reserve",
                requested_value=desired_buy_cost,
                applied_value=available_for_buys,
            )
        )
        scaled_buy_quantities = {}
        for symbol, quantity in buy_quantities.items():
            scaled_quantity = _round_toward_zero(quantity * buy_scale, config)
            scaled_buy_quantities[symbol] = scaled_quantity
            if not _close(scaled_quantity, quantity):
                audit.append(
                    AdjustmentEvent(
                        code="cash_affordability",
                        message="Reduced this buy to preserve available cash and reserve",
                        symbol=symbol,
                        requested_value=quantity,
                        applied_value=scaled_quantity,
                    )
                )
        buy_quantities = scaled_buy_quantities

    for symbol in all_symbols:
        quantity = buy_quantities[symbol]
        if quantity <= 0.0:
            continue
        reference_price = execution_prices[symbol]
        execution_price = reference_price * (1.0 + slippage_fraction)
        execution_notional = quantity * execution_price
        fee = execution_notional * config.transaction_cost_rate
        total_cost = execution_notional + fee

        # Guard against floating point boundary error while retaining stable
        # rounding behavior for both whole and fractional-share configurations.
        while total_cost > final_cash - cash_reserve + 1e-10:
            quantity = _round_toward_zero(
                quantity - _quantity_step(config), config
            )
            if quantity <= 0.0:
                break
            execution_notional = quantity * execution_price
            fee = execution_notional * config.transaction_cost_rate
            total_cost = execution_notional + fee
        if quantity <= 0.0:
            continue

        slippage_cost = quantity * (execution_price - reference_price)
        final_cash -= total_cost
        final_holdings[symbol] += quantity
        total_fees += fee
        total_slippage += slippage_cost
        trades.append(
            Trade(
                symbol=symbol,
                side="buy",
                quantity=quantity,
                reference_price=reference_price,
                execution_price=execution_price,
                reference_notional=quantity * reference_price,
                execution_notional=execution_notional,
                fee=fee,
                slippage_cost=slippage_cost,
                reason="Move holdings toward the constrained target allocation",
            )
        )

    # Remove tiny numerical residues, but never mask an actual negative balance.
    if -1e-9 <= final_cash < 0.0:
        final_cash = 0.0
    if final_cash < cash_reserve - 1e-8:
        return _rejected_result(
            initial_nav=initial_nav,
            holdings=holdings,
            cash=cash,
            mark_prices=mark_prices,
            targets=adjusted_targets,
            target_cash_weight=target_cash_weight,
            target_values=target_values,
            cash_target_value=cash_target_value,
            audit=audit,
            reason="Final cash would breach the required reserve",
        )

    gross_turnover = math.fsum(
        trade.reference_notional for trade in trades
    )
    turnover_ratio = gross_turnover / initial_nav
    if turnover_ratio > config.max_turnover + 1e-10:
        return _rejected_result(
            initial_nav=initial_nav,
            holdings=holdings,
            cash=cash,
            mark_prices=mark_prices,
            targets=adjusted_targets,
            target_cash_weight=target_cash_weight,
            target_values=target_values,
            cash_target_value=cash_target_value,
            audit=audit,
            reason="Rounded trades exceed the maximum gross turnover",
        )

    final_nav = final_cash + math.fsum(
        quantity * mark_prices[symbol]
        for symbol, quantity in final_holdings.items()
    )
    if final_nav <= 0.0 or not math.isfinite(final_nav):
        return _rejected_result(
            initial_nav=initial_nav,
            holdings=holdings,
            cash=cash,
            mark_prices=mark_prices,
            targets=adjusted_targets,
            target_cash_weight=target_cash_weight,
            target_values=target_values,
            cash_target_value=cash_target_value,
            audit=audit,
            reason="Simulated trades result in non-positive final NAV",
        )

    actual_weights = {
        symbol: quantity * mark_prices[symbol] / final_nav
        for symbol, quantity in final_holdings.items()
    }
    actual_cash_weight = final_cash / final_nav
    fees = math.fsum(trade.fee for trade in trades)
    slippage = math.fsum(trade.slippage_cost for trade in trades)
    if not _targets_match(
        adjusted_targets, target_cash_weight, actual_weights, actual_cash_weight
    ):
        audit.append(
            AdjustmentEvent(
                code="realized_allocation",
                message=(
                    "Final weights differ from targets because of costs, "
                    "share rounding, turnover, or cash constraints"
                ),
            )
        )

    status = (
        PlanStatus.PARTIALLY_CONSTRAINED
        if audit
        else PlanStatus.COMPLETED
    )
    return RebalanceResult(
        status=status,
        initial_nav=initial_nav,
        final_nav=final_nav,
        adjusted_target_weights=dict(adjusted_targets),
        cash_target_weight=target_cash_weight,
        target_values=target_values,
        cash_target_value=cash_target_value,
        trades=tuple(trades),
        final_holdings=final_holdings,
        final_cash=final_cash,
        total_fees=fees,
        total_slippage_cost=slippage,
        total_transaction_cost=fees + slippage,
        actual_weights=actual_weights,
        actual_cash_weight=actual_cash_weight,
        gross_turnover=gross_turnover,
        turnover_ratio=turnover_ratio,
        audit_events=tuple(audit),
    )


def _rejected_result(
    *,
    initial_nav: float,
    holdings: Mapping[str, float],
    cash: float,
    mark_prices: Mapping[str, float],
    targets: Mapping[str, float],
    target_cash_weight: float,
    target_values: Mapping[str, float],
    cash_target_value: float,
    audit: list[AdjustmentEvent],
    reason: str,
) -> RebalanceResult:
    """Return a rejected result with no simulated trades applied."""
    actual_nav = cash + math.fsum(
        quantity * mark_prices[symbol] for symbol, quantity in holdings.items()
    )
    actual_weights = {
        symbol: quantity * mark_prices[symbol] / actual_nav
        for symbol, quantity in holdings.items()
    }
    return RebalanceResult(
        status=PlanStatus.REJECTED,
        initial_nav=initial_nav,
        final_nav=actual_nav,
        adjusted_target_weights=dict(targets),
        cash_target_weight=target_cash_weight,
        target_values=dict(target_values),
        cash_target_value=cash_target_value,
        trades=(),
        final_holdings=dict(holdings),
        final_cash=cash,
        total_fees=0.0,
        total_slippage_cost=0.0,
        total_transaction_cost=0.0,
        actual_weights=actual_weights,
        actual_cash_weight=cash / actual_nav,
        gross_turnover=0.0,
        turnover_ratio=0.0,
        audit_events=tuple(audit),
        rejection_reasons=(reason,),
    )


def _validated_mapping(name: str, values: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for symbol, raw_value in values.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"{name} keys must be non-empty strings")
        result[symbol] = _finite_number(f"{name}[{symbol!r}]", raw_value)
    return result


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _round_toward_zero(quantity: float, config) -> float:
    if quantity == 0.0:
        return 0.0
    sign = -1.0 if quantity < 0.0 else 1.0
    magnitude = abs(quantity)
    if not config.allow_fractional_shares:
        return sign * float(math.floor(magnitude + 1e-12))
    factor = 10**config.fractional_share_precision
    return sign * math.floor(magnitude * factor + 1e-10) / factor


def _quantity_step(config) -> float:
    return 1.0 if not config.allow_fractional_shares else 10 ** (
        -config.fractional_share_precision
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


def _targets_match(
    targets: Mapping[str, float],
    target_cash_weight: float,
    actual_weights: Mapping[str, float],
    actual_cash_weight: float,
) -> bool:
    symbols = set(targets) | set(actual_weights)
    return all(
        _close(targets.get(symbol, 0.0), actual_weights.get(symbol, 0.0))
        for symbol in symbols
    ) and _close(target_cash_weight, actual_cash_weight)
