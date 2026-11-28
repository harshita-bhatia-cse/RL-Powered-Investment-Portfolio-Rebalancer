"""Portfolio-aware, read-only rebalance previews."""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP, localcontext
from math import isfinite
from typing import Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import Holding, Portfolio
from src.market_data.service import value_portfolio
from src.strategies.allocation import (
    RebalanceConfig,
    RebalanceRequest,
    rebalance,
)
from src.strategies.allocation.constraints import validate_and_adjust_weights

_LISTING_RE = re.compile(r"^(NSE|BSE):([A-Z0-9][A-Z0-9&.-]{0,31})$")
_INR_CENT = Decimal("0.01")


def _inr_amount(value: Decimal | float | int) -> Decimal:
    """Round an estimated INR amount to paise at the API boundary."""
    amount = Decimal(str(value))
    with localcontext() as context:
        context.prec = max(
            28,
            len(amount.as_tuple().digits) + max(0, amount.as_tuple().exponent) + 2,
        )
        return amount.quantize(_INR_CENT, rounding=ROUND_HALF_UP)


def _normalize_listing(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Listing identifiers must be strings such as NSE:TCS")
    listing = value.strip().upper()
    if not _LISTING_RE.fullmatch(listing):
        raise ValueError(
            f"Invalid listing {value!r}; use an exchange-qualified NSE:SYMBOL or BSE:SYMBOL"
        )
    return listing


def _weight_as_float(listing: str, weight: Decimal | float) -> float:
    if isinstance(weight, bool):
        raise ValueError(f"Target weight for {listing!r} must be finite")
    try:
        converted = float(weight)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Target weight for {listing!r} must be finite") from error
    if not isfinite(converted):
        raise ValueError(f"Target weight for {listing!r} must be finite")
    return converted


def _jsonable_price(holding: Mapping) -> dict:
    return {
        "listing": f"{holding['exchange']}:{holding['symbol']}",
        "exchange": holding["exchange"],
        "symbol": holding["symbol"],
        "close": _inr_amount(holding["close"]),
        "market_date": holding["market_date"],
        "provider": holding["provider"],
        "source": holding["source"],
        "fetched_at": holding["fetched_at"],
    }


def _adjustment_event_payload(event) -> dict:
    return {
        "code": event.code,
        "message": event.message,
        "listing": event.symbol,
        "requested_value": event.requested_value,
        "applied_value": event.applied_value,
    }


def _sum_displayed_amounts(values: Sequence[Decimal]) -> Decimal:
    return _inr_amount(sum(values, Decimal("0.00")))


def _unavailable_recommendations(
    holdings: Sequence[Holding],
    requested_weights: Mapping[str, float],
    warnings: Sequence[Mapping],
    *,
    prices: Mapping[str, float] | None = None,
    rejection_reasons: Sequence[str] = (),
) -> list[dict]:
    warning_by_listing = {
        f"{warning['exchange']}:{warning['symbol']}": warning
        for warning in warnings
    }
    recommendations = []
    for holding in holdings:
        listing = f"{holding.exchange}:{holding.symbol}"
        warning = warning_by_listing.get(listing)
        if rejection_reasons:
            reasons = [
                {
                    "code": "plan_rejected",
                    "message": message,
                }
                for message in rejection_reasons
            ]
        elif warning:
            code = {
                "missing_prices": "missing_price",
                "stale_prices": "stale_price",
                "unavailable_prices": "unpermitted_price",
            }.get(warning["category"], "price_unavailable")
            reasons = [
                {
                    "code": code,
                    "message": warning.get("reason")
                    or (
                        f"Price is not usable because it is {warning['category']}."
                    ),
                }
            ]
        else:
            reasons = [
                {
                    "code": "valuation_incomplete",
                    "message": "A complete permitted valuation is required before recommendations can be made.",
                }
            ]
        price = (prices or {}).get(listing)
        current_value = (
            _inr_amount(float(holding.quantity) * price)
            if price is not None
            else None
        )
        recommendations.append(
            {
                "listing": listing,
                "exchange": holding.exchange,
                "symbol": holding.symbol,
                "current_quantity": float(holding.quantity),
                "current_market_value": current_value,
                "requested_target_weight": requested_weights.get(listing),
                "target_weight": None,
                "target_value": None,
                "simulated_final_quantity": None,
                "simulated_final_value": None,
                "action": "UNAVAILABLE",
                "execution_status": "NOT_APPLICABLE",
                "reason_codes": [reason["code"] for reason in reasons],
                "explanations": reasons,
                "estimated_fees": _inr_amount(0),
                "estimated_slippage": _inr_amount(0),
            }
        )
    return recommendations


def _holding_recommendations(
    holdings: Sequence[Holding],
    prices: Mapping[str, float],
    requested_weights: Mapping[str, float],
    result,
    config: RebalanceConfig,
    allocation_adjustment_events: Sequence,
    displayed_trades: Sequence[Mapping],
) -> list[dict]:
    """Explain intended per-listing actions and what the engine simulated."""
    if result.status.value == "rejected":
        recommendations = []
        for holding in holdings:
            listing = f"{holding.exchange}:{holding.symbol}"
            price = prices[listing]
            quantity = float(holding.quantity)
            explanations = [
                {"code": "plan_rejected", "message": message}
                for message in result.rejection_reasons
            ]
            recommendations.append(
                {
                    "listing": listing,
                    "exchange": holding.exchange,
                    "symbol": holding.symbol,
                    "current_quantity": quantity,
                    "current_market_value": _inr_amount(quantity * price),
                    "requested_target_weight": requested_weights.get(listing),
                    "target_weight": result.adjusted_target_weights.get(listing, 0.0),
                    "target_value": _inr_amount(result.target_values.get(listing, 0.0)),
                    "simulated_final_quantity": quantity,
                    "simulated_final_value": _inr_amount(quantity * price),
                    "action": "UNAVAILABLE",
                    "execution_status": "NOT_APPLICABLE",
                    "reason_codes": [reason["code"] for reason in explanations],
                    "explanations": explanations,
                    "estimated_fees": _inr_amount(0),
                    "estimated_slippage": _inr_amount(0),
                }
            )
        return recommendations

    trades_by_listing = {trade["listing"]: trade for trade in displayed_trades}
    tolerance = (
        max(1e-10, 0.5 * 10 ** -config.fractional_share_precision)
        if config.allow_fractional_shares
        else 1e-10
    )
    recommendations = []
    for holding in holdings:
        listing = f"{holding.exchange}:{holding.symbol}"
        price = prices[listing]
        current_quantity = float(holding.quantity)
        current_value = current_quantity * price
        target_weight = result.adjusted_target_weights.get(listing, 0.0)
        target_value = result.target_values.get(listing, 0.0)
        target_quantity = target_value / price
        intended_delta = target_quantity - current_quantity
        if abs(intended_delta) <= tolerance:
            action = "HOLD"
            action_message = (
                "Current quantity already matches the constrained target within "
                f"{tolerance:g} shares."
            )
        elif intended_delta > 0:
            action = "BUY"
            action_message = (
                "Increase the position to move toward its constrained target allocation."
            )
        else:
            action = "SELL"
            action_message = (
                "Reduce the position to move toward its constrained target allocation."
            )

        trade = trades_by_listing.get(listing)
        simulated_quantity = result.final_holdings.get(listing, current_quantity)
        simulated_delta = simulated_quantity - current_quantity
        is_partially_executed = (
            abs(simulated_delta - intended_delta) > tolerance
            and abs(intended_delta) > tolerance
        )
        listing_events = [
            event
            for event in (
                *allocation_adjustment_events,
                *result.audit_events,
            )
            if event.symbol == listing
        ]
        affected_global_events = [
            event
            for event in result.audit_events
            if event.symbol is None
            and abs(intended_delta) > tolerance
            and event.code == "turnover_limit"
        ]
        if action == "HOLD":
            execution_status = "NOT_APPLICABLE"
            explanations = [
                {
                    "code": "target_already_matched",
                    "message": action_message,
                }
            ]
        elif trade is None:
            execution_status = "NOT_EXECUTED"
            explanations = [
                {
                    "code": f"target_{action.lower()}",
                    "message": action_message,
                },
            ]
            if not listing_events and not affected_global_events:
                explanations.append(
                    {
                        "code": "no_simulated_trade",
                        "message": (
                            "The intended position change did not produce a simulated trade."
                        ),
                    }
                )
        else:
            execution_status = (
                "EXECUTED"
                if abs(simulated_delta - intended_delta) <= tolerance
                else "PARTIALLY_EXECUTED"
            )
            explanations = [
                {
                    "code": f"target_{action.lower()}",
                    "message": action_message,
                },
                {
                    "code": "simulated_trade",
                    "message": trade["reason"],
                },
            ]

        if action != "HOLD":
            explanations.extend(
                _adjustment_event_payload(event)
                for event in listing_events
            )
        if action != "HOLD" and is_partially_executed:
            for event in affected_global_events:
                if not any(
                    explanation["code"] == event.code
                    for explanation in explanations
                ):
                    explanations.append(_adjustment_event_payload(event))

        recommendations.append(
            {
                "listing": listing,
                "exchange": holding.exchange,
                "symbol": holding.symbol,
                "current_quantity": current_quantity,
                "current_market_value": _inr_amount(current_value),
                "requested_target_weight": requested_weights.get(listing),
                "target_weight": target_weight,
                "target_value": _inr_amount(target_value),
                "simulated_final_quantity": simulated_quantity,
                "simulated_final_value": _inr_amount(
                    simulated_quantity * price
                ),
                "action": action,
                "execution_status": execution_status,
                "reason_codes": [item["code"] for item in explanations],
                "explanations": explanations,
                "estimated_fees": (
                    trade["fee"] if trade else _inr_amount(0)
                ),
                "estimated_slippage": _inr_amount(
                    trade["slippage_cost"] if trade else 0
                ),
            }
        )
    return recommendations


def preview_portfolio_rebalance(
    account_db: Session,
    market_db: Session,
    user_id: str,
    portfolio_id: str,
    *,
    target_weights: Mapping[str, Decimal | float] | None = None,
    target_cash_weight: Decimal | float | None = None,
    selected_holdings: Sequence[str] | None = None,
    additional_investment: Decimal = Decimal("0.00"),
    cash_reserve_pct: Decimal | float = Decimal("0.00"),
    max_weight_per_asset: Decimal | float = Decimal("1.00"),
    transaction_cost_rate: Decimal | float = Decimal("0.00"),
    slippage_bps: Decimal | float = Decimal("0.00"),
    allow_fractional_shares: bool = False,
    fractional_share_precision: int = 6,
    max_turnover: Decimal | float = Decimal("1.00"),
    today=None,
    max_age_days: int | None = None,
) -> dict:
    """Build a simulated rebalance result without changing account data."""
    portfolio = account_db.scalar(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user_id,
        )
    )
    if portfolio is None:
        raise ValueError("Portfolio not found")
    holdings = account_db.scalars(
        select(Holding)
        .where(Holding.portfolio_id == portfolio_id)
        .order_by(Holding.exchange, Holding.symbol)
    ).all()
    actual_listings = {
        f"{holding.exchange}:{holding.symbol}": holding for holding in holdings
    }

    custom_weights: dict[str, float] | None = None
    if target_weights is not None:
        custom_weights = {}
        for raw_listing, weight in target_weights.items():
            listing = _normalize_listing(raw_listing)
            if listing not in actual_listings:
                raise ValueError(
                    f"Target listing {listing} is not a current portfolio holding"
                )
            custom_weights[listing] = _weight_as_float(listing, weight)

    selected: list[str] | None = None
    if selected_holdings is not None:
        selected = [_normalize_listing(item) for item in selected_holdings]
        if len(set(selected)) != len(selected):
            raise ValueError("selected_holdings must not contain duplicate listings")
        unknown = sorted(set(selected) - set(actual_listings))
        if unknown:
            raise ValueError(
                "Selected listings are not current portfolio holdings: "
                + ", ".join(unknown)
            )
        if custom_weights is not None and not set(custom_weights).issubset(selected):
            raise ValueError(
                "Custom target weights must refer only to selected_holdings"
            )

    if additional_investment < 0 or not additional_investment.is_finite():
        raise ValueError("additional_investment must be a finite non-negative amount")

    config = RebalanceConfig(
        max_weight_per_asset=float(max_weight_per_asset),
        cash_reserve_pct=float(cash_reserve_pct),
        transaction_cost_rate=float(transaction_cost_rate),
        slippage_bps=float(slippage_bps),
        allow_fractional_shares=allow_fractional_shares,
        fractional_share_precision=fractional_share_precision,
        max_turnover=float(max_turnover),
    )
    config.validate()

    valuation = value_portfolio(
        account_db,
        market_db,
        user_id,
        portfolio_id,
        today=today,
        max_age_days=max_age_days,
    )
    price_by_listing = {
        f"{item['exchange']}:{item['symbol']}": item
        for item in valuation["valued_holdings"]
    }
    price_warnings = []
    for category in ("missing_prices", "stale_prices", "unavailable_prices"):
        for item in valuation[category]:
            warning = {"category": category, **item}
            if warning.get("close") is not None:
                warning["close"] = _inr_amount(warning["close"])
            price_warnings.append(warning)
    prices = {
        listing: float(item["close"])
        for listing, item in price_by_listing.items()
    }

    selected_universe = selected if selected is not None else sorted(actual_listings)
    if custom_weights is None:
        preserved_listings = sorted(set(actual_listings) - set(selected_universe))
    else:
        # A selected listing without an explicit custom target is also kept at
        # its current weight; omission from a custom mapping is not liquidation.
        preserved_listings = sorted(set(actual_listings) - set(custom_weights))
    excluded_weights: dict[str, float] = {}
    if valuation["valuation_complete"] and preserved_listings:
        nav_before_investment = portfolio.cash_balance + sum(
            (holding.quantity * price_by_listing[listing]["close"]
             for listing, holding in actual_listings.items()),
            Decimal("0"),
        )
        if nav_before_investment <= 0:
            excluded_weights = {listing: 0.0 for listing in preserved_listings}
        else:
            preview_nav = nav_before_investment + additional_investment
            excluded_weights = {
                listing: float(
                    actual_listings[listing].quantity
                    * price_by_listing[listing]["close"]
                    / preview_nav
                )
                for listing in preserved_listings
            }

    excluded_total = sum(excluded_weights.values())
    reserve = float(cash_reserve_pct)
    available_asset_weight = 1.0 - reserve - excluded_total
    if available_asset_weight < -config.weight_sum_tolerance:
        raise ValueError(
            "Cannot preserve excluded holdings and satisfy the configured cash reserve"
        )

    if custom_weights is None:
        source = "equal_weight_baseline"
        if selected_universe:
            equal_weight = max(0.0, available_asset_weight) / len(selected_universe)
            selected_weights = {
                listing: equal_weight for listing in selected_universe
            }
        else:
            selected_weights = {}
        requested_weights = {**excluded_weights, **selected_weights}
        requested_cash_weight = max(
            reserve,
            1.0 - sum(requested_weights.values()),
        )
    else:
        source = "custom"
        requested_weights = {**excluded_weights, **custom_weights}
        if target_cash_weight is None:
            requested_cash_weight = 1.0 - sum(requested_weights.values())
        else:
            requested_cash_weight = float(target_cash_weight)
        if requested_cash_weight < reserve - config.weight_sum_tolerance:
            raise ValueError(
                "Custom target weights leave insufficient cash to preserve excluded "
                "holdings and satisfy the configured cash reserve"
            )
        if sum(custom_weights.values()) + excluded_total > 1.0 + config.weight_sum_tolerance:
            raise ValueError(
                "Custom target weights exceed the allocation remaining after "
                "preserving excluded holdings"
            )

    if requested_cash_weight < -config.weight_sum_tolerance:
        raise ValueError(
            "Requested allocation exceeds 100% after preserving excluded holdings"
        )

    # Excluded listings are fixed at current weights. A cap below one of those
    # fixed weights makes the request infeasible rather than authorizing a sale.
    fixed_caps = dict(config.per_asset_max_weights)
    for listing, weight in excluded_weights.items():
        if weight > config.cap_for(listing) + config.weight_sum_tolerance:
            raise ValueError(
                f"Cannot preserve excluded holding {listing} within its asset cap"
            )
        fixed_caps[listing] = weight
    config = RebalanceConfig(
        max_weight_per_asset=config.max_weight_per_asset,
        per_asset_max_weights=fixed_caps,
        cash_reserve_pct=config.cash_reserve_pct,
        transaction_cost_rate=config.transaction_cost_rate,
        slippage_bps=config.slippage_bps,
        allow_fractional_shares=config.allow_fractional_shares,
        fractional_share_precision=config.fractional_share_precision,
        max_turnover=config.max_turnover,
        weight_sum_tolerance=config.weight_sum_tolerance,
    )

    # Validate with the shared policy before invoking the engine. If a
    # constraint would move a fixed excluded weight, report infeasibility.
    requested_weights_for_explanation = dict(requested_weights)
    requested_cash_for_explanation = requested_cash_weight
    adjusted_weights, adjusted_cash, allocation_adjustment_events = validate_and_adjust_weights(
        requested_weights,
        requested_cash_weight,
        config,
    )
    if any(
        abs(adjusted_weights.get(listing, 0.0) - weight)
        > config.weight_sum_tolerance
        for listing, weight in excluded_weights.items()
    ):
        raise ValueError(
            "Allocation constraints cannot be satisfied while preserving excluded holdings"
        )
    requested_weights = adjusted_weights
    requested_cash_weight = adjusted_cash

    common = {
        "portfolio_id": portfolio_id,
        "currency": "INR",
        "preview_available": False,
        "allocation_source": source,
        "selected_holdings": (
            selected
            if selected is not None
            else sorted(actual_listings)
        ),
        "user_target_weights": custom_weights,
        "user_target_cash_weight": (
            float(target_cash_weight)
            if target_cash_weight is not None
            else None
        ),
        "allocation_status": "COMPLETE",
        "allocation_unavailable_reason": None,
        "requested_target_weights": requested_weights_for_explanation,
        "requested_target_cash_weight": requested_cash_for_explanation,
        "adjusted_target_weights": dict(adjusted_weights),
        "adjusted_target_cash_weight": adjusted_cash,
        "target_cash_weight": adjusted_cash,
        "portfolio_cash_balance": _inr_amount(portfolio.cash_balance),
        "additional_investment": _inr_amount(additional_investment),
        "cash_available_for_preview": _inr_amount(
            portfolio.cash_balance + additional_investment
        ),
        "initial_nav": None,
        "final_nav": None,
        "status": None,
        "rejection_reasons": [],
        "market_prices": [
            _jsonable_price(item)
            for _, item in sorted(price_by_listing.items())
        ],
        "price_warnings": price_warnings,
        "proposed_trades": [],
        "recommendations": [],
        "estimated_transaction_costs": {
            "fees": 0.0,
            "slippage": 0.0,
            "total": 0.0,
        },
        "final_simulated_holdings": {},
        "final_simulated_weights": {},
        "final_simulated_cash_weight": None,
        "final_simulated_cash_balance": None,
        "constraint_audit_events": [],
        "portfolio_constraint_events": [],
        "allocation_adjustment_events": [
            _adjustment_event_payload(event)
            for event in allocation_adjustment_events
        ],
        "gross_turnover": None,
        "turnover_ratio": None,
        "freshness_policy": valuation["freshness_policy"],
    }
    if not valuation["valuation_complete"]:
        common.update(
            {
                "allocation_status": "UNAVAILABLE",
                "allocation_unavailable_reason": (
                    "A complete permitted valuation is required to calculate "
                    "preserved holding weights and a complete target allocation."
                ),
                "requested_target_weights": None,
                "requested_target_cash_weight": None,
                "adjusted_target_weights": None,
                "adjusted_target_cash_weight": None,
                "target_cash_weight": None,
                "final_simulated_weights": None,
                "final_simulated_cash_weight": None,
                "allocation_adjustment_events": [],
            }
        )
        common["recommendations"] = _unavailable_recommendations(
            holdings,
            custom_weights or {},
            price_warnings,
            prices=prices,
        )
        return common

    if portfolio.cash_balance + additional_investment == 0 and not holdings:
        common["status"] = "unavailable"
        common["rejection_reasons"] = [
            "Preview unavailable because total NAV is zero"
        ]
        return common

    current_holdings = {
        f"{holding.exchange}:{holding.symbol}": float(holding.quantity)
        for holding in holdings
    }
    if set(current_holdings) - set(prices):
        raise RuntimeError("Complete valuation did not include every current holding")

    if custom_weights is not None:
        absent_target_prices = sorted(set(requested_weights) - set(prices))
        if absent_target_prices:
            raise ValueError(
                "No permitted fresh price is available for target listings: "
                + ", ".join(absent_target_prices)
            )

    preview_cash = portfolio.cash_balance + additional_investment
    result = rebalance(
        RebalanceRequest(
            current_holdings=current_holdings,
            available_cash=float(preview_cash),
            prices=prices,
            target_weights=requested_weights,
            target_cash_weight=requested_cash_weight,
            config=config,
        )
    )

    displayed_trades = [
        {
            "listing": trade.symbol,
            "exchange": trade.symbol.split(":", 1)[0],
            "symbol": trade.symbol.split(":", 1)[1],
            "side": trade.side,
            "quantity": trade.quantity,
            "reference_price": _inr_amount(trade.reference_price),
            "execution_price": _inr_amount(trade.execution_price),
            "reference_notional": _inr_amount(trade.reference_notional),
            "execution_notional": _inr_amount(trade.execution_notional),
            "fee": _inr_amount(trade.fee),
            "slippage_cost": _inr_amount(trade.slippage_cost),
            "reason": trade.reason,
        }
        for trade in result.trades
    ]
    displayed_fees = _sum_displayed_amounts(
        [trade["fee"] for trade in displayed_trades]
    )
    displayed_slippage = _sum_displayed_amounts(
        [trade["slippage_cost"] for trade in displayed_trades]
    )
    displayed_turnover = _sum_displayed_amounts(
        [trade["reference_notional"] for trade in displayed_trades]
    )
    displayed_initial_nav = _inr_amount(result.initial_nav)
    displayed_turnover_ratio = (
        float(displayed_turnover / displayed_initial_nav)
        if displayed_initial_nav > 0
        else 0.0
    )

    common.update(
        {
            "preview_available": True,
            "adjusted_target_weights": dict(result.adjusted_target_weights),
            "adjusted_target_cash_weight": result.cash_target_weight,
            "target_cash_weight": result.cash_target_weight,
            "final_simulated_weights": dict(result.actual_weights),
            "final_simulated_cash_weight": result.actual_cash_weight,
            "initial_nav": _inr_amount(result.initial_nav),
            "final_nav": _inr_amount(result.final_nav),
            "status": result.status.value,
            "rejection_reasons": list(result.rejection_reasons),
            "proposed_trades": displayed_trades,
            "estimated_transaction_costs": {
                "fees": displayed_fees,
                "slippage": displayed_slippage,
                "total": _sum_displayed_amounts(
                    [displayed_fees, displayed_slippage]
                ),
            },
            "final_simulated_holdings": dict(result.final_holdings),
            "final_simulated_cash_balance": _inr_amount(result.final_cash),
            "constraint_audit_events": [
                _adjustment_event_payload(event)
                for event in (
                    *allocation_adjustment_events,
                    *result.audit_events,
                )
                if event.symbol is not None
            ],
            "portfolio_constraint_events": [
                _adjustment_event_payload(event)
                for event in (
                    *allocation_adjustment_events,
                    *result.audit_events,
                )
                if event.symbol is None
            ],
            "allocation_adjustment_events": [
                _adjustment_event_payload(event)
                for event in allocation_adjustment_events
            ],
            "recommendations": _holding_recommendations(
                holdings,
                prices,
                requested_weights_for_explanation,
                result,
                config,
                allocation_adjustment_events,
                displayed_trades,
            ),
            "gross_turnover": displayed_turnover,
            "turnover_ratio": displayed_turnover_ratio,
        }
    )
    return common
