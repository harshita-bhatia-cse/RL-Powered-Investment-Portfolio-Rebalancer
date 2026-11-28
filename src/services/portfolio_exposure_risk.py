"""Deterministic exposure-risk assessment for a complete portfolio valuation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
from typing import Any


EXPOSURE_RISK_POLICY_VERSION = "exposure-risk-v1"
EXPOSURE_RISK_THRESHOLDS = {
    "single_stock_exposure": Decimal("0.25"),
    "top_five_concentration": Decimal("0.70"),
    "minimum_cash_allocation": Decimal("0.05"),
    "equity_hhi": Decimal("0.18"),
}
_PERCENT = Decimal("100")


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{field} must be provided as an exact decimal value")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a valid decimal value") from error
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def _money(value: Any, field: str) -> Decimal:
    result = _decimal(value, field)
    numerator, denominator = result.as_integer_ratio()
    if numerator * 100 % denominator:
        raise ValueError(f"{field} must already be normalized to paise")
    return result


def _minor_units(value: Decimal) -> int:
    numerator, denominator = value.as_integer_ratio()
    return numerator * 100 // denominator


def _from_minor_units(value: int) -> Decimal:
    digits = tuple(int(digit) for digit in str(abs(value)))
    return Decimal((1 if value < 0 else 0, digits, -2))


def _ratio_greater(numerator: int, denominator: int, threshold: Decimal) -> bool:
    threshold_numerator, threshold_denominator = threshold.as_integer_ratio()
    return numerator * threshold_denominator > denominator * threshold_numerator


def _ratio_less(numerator: int, denominator: int, threshold: Decimal) -> bool:
    threshold_numerator, threshold_denominator = threshold.as_integer_ratio()
    return numerator * threshold_denominator < denominator * threshold_numerator


def _decimal_string(value: Decimal) -> str:
    return format(value, "f")


def _unavailable_result(
    *,
    session_date: date,
    evaluated_at: datetime,
    reason_codes: list[str],
    unavailable_listings: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "assessment_status": "NOT_EVALUATED",
        "policy_version": EXPOSURE_RISK_POLICY_VERSION,
        "evaluated_at": evaluated_at.isoformat(),
        "session_date": session_date.isoformat(),
        "portfolio_nav": None,
        "total_equity_market_value": None,
        "cash_balance": None,
        "listing_exposures": [],
        "top_five_weight_pct": None,
        "cash_weight_pct": None,
        "equity_hhi": None,
        "alerts": [],
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "unavailable_listings": unavailable_listings,
    }


def assess_portfolio_exposure(
    *,
    session_date: date,
    valuation_complete: bool,
    portfolio_nav: Decimal | str | int | None,
    cash_balance: Decimal | str | int,
    holdings: Sequence[Mapping[str, Any]],
    unavailable_listings: Sequence[Mapping[str, Any]] = (),
    data_quality_warnings: Sequence[Mapping[str, Any]] = (),
    evaluated_at: datetime | None = None,
) -> dict[str, Any]:
    """Assess exposure from a single already-validated valuation snapshot.

    The caller must pass the same valuation snapshot used by portfolio
    monitoring. This service performs no database or market-data access.
    Monetary inputs must be exact decimal-compatible values, never floats.
    """
    if not isinstance(session_date, date) or isinstance(session_date, datetime):
        raise ValueError("session_date must be a calendar date")
    assessment_time = evaluated_at or datetime.now(timezone.utc)
    if assessment_time.tzinfo is None:
        raise ValueError("evaluated_at must be timezone-aware")
    cash = _money(cash_balance, "cash_balance")
    if cash < 0:
        raise ValueError("cash_balance cannot be negative")
    nav = None if portfolio_nav is None else _money(portfolio_nav, "portfolio_nav")
    if nav is not None and nav < 0:
        raise ValueError("portfolio_nav cannot be negative")

    normalized_holdings: list[dict[str, Any]] = []
    seen_listings: set[tuple[str, str]] = set()
    for row in holdings:
        if not isinstance(row, Mapping):
            raise ValueError("each holding must be a mapping")
        raw_exchange = row.get("exchange")
        raw_symbol = row.get("symbol")
        if not isinstance(raw_exchange, str) or not isinstance(raw_symbol, str):
            raise ValueError("each holding must have a string NSE/BSE listing")
        exchange = raw_exchange.strip().upper()
        symbol = raw_symbol.strip().upper()
        if (
            exchange not in {"NSE", "BSE"}
            or not symbol
            or len(symbol) > 32
            or not symbol[0].isascii()
            or not symbol[0].isalnum()
            or any(
                not character.isascii()
                or not character.isalnum() and character not in "&.-"
                for character in symbol[1:]
            )
        ):
            raise ValueError("each holding must have an NSE/BSE listing")
        listing_key = (exchange, symbol)
        if listing_key in seen_listings:
            raise ValueError(f"duplicate holding listing {exchange}:{symbol}")
        seen_listings.add(listing_key)
        quantity = _decimal(row.get("quantity"), "holding quantity")
        if quantity <= 0:
            raise ValueError("holding quantity must be positive")

        valuation_status = row.get("valuation_status")
        if valuation_status == "UNAVAILABLE":
            if row.get("market_value") is not None:
                raise ValueError("an unavailable holding cannot have market_value")
            if valuation_complete:
                raise ValueError(
                    "complete valuation contains an unavailable holding"
                )
            continue
        if valuation_status != "VALUED":
            raise ValueError("holding valuation_status must be VALUED or UNAVAILABLE")

        market_date = row.get("market_date")
        if isinstance(market_date, str):
            try:
                market_date = date.fromisoformat(market_date)
            except ValueError as error:
                raise ValueError("holding market_date must be an ISO date") from error
        if (
            not isinstance(market_date, date)
            or isinstance(market_date, datetime)
            or market_date > session_date
        ):
            raise ValueError("holding price must be on or before session_date")

        market_value = _money(row.get("market_value"), "holding market_value")
        if market_value < 0:
            raise ValueError("holding market_value cannot be negative")
        normalized_holdings.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "listing": f"{exchange}:{symbol}",
                "quantity": quantity,
                "market_value": market_value,
            }
        )

    missing_rows = []
    for item in unavailable_listings:
        exchange = str(item.get("exchange", ""))
        symbol = str(item.get("symbol", ""))
        reason = str(item.get("reason") or "required valuation is unavailable")
        missing_rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "listing": f"{exchange}:{symbol}",
                "reason": reason,
            }
        )

    warning_codes = {
        str(item.get("code"))
        for item in data_quality_warnings
        if item.get("code")
    }
    if not valuation_complete:
        reasons = ["valuation_incomplete"]
        reasons.extend(
            code
            for code in (
                "missing_price",
                "stale_price",
                "unpermitted_price",
                "price_after_session",
            )
            if code in warning_codes
        )
        return _unavailable_result(
            session_date=session_date,
            evaluated_at=assessment_time,
            reason_codes=reasons,
            unavailable_listings=missing_rows,
        )

    if nav is None:
        return _unavailable_result(
            session_date=session_date,
            evaluated_at=assessment_time,
            reason_codes=["portfolio_nav_unavailable"],
            unavailable_listings=[],
        )
    if nav <= 0:
        return _unavailable_result(
            session_date=session_date,
            evaluated_at=assessment_time,
            reason_codes=["portfolio_nav_not_positive"],
            unavailable_listings=[],
        )

    normalized_holdings.sort(
        key=lambda item: (-item["market_value"], item["exchange"], item["symbol"])
    )
    cash_units = _minor_units(cash)
    nav_units = _minor_units(nav)
    holding_units = [_minor_units(item["market_value"]) for item in normalized_holdings]
    equity_units = sum(holding_units)
    equity_market_value = _from_minor_units(equity_units)
    if cash_units + equity_units != nav_units:
        return _unavailable_result(
            session_date=session_date,
            evaluated_at=assessment_time,
            reason_codes=["valuation_nav_mismatch"],
            unavailable_listings=[],
        )

    with localcontext() as context:
        context.prec = 50
        listing_exposures = []
        listing_weights = []
        for item in normalized_holdings:
            weight = item["market_value"] / nav
            listing_weights.append(weight)
            listing_exposures.append(
                {
                    "exchange": item["exchange"],
                    "symbol": item["symbol"],
                    "listing": item["listing"],
                    "quantity": _decimal_string(item["quantity"]),
                    "market_value": _decimal_string(item["market_value"]),
                    "portfolio_weight_pct": _decimal_string(
                        weight * _PERCENT
                    ),
                }
            )

        top_five_value = sum(holding_units[:5])
        top_five_weight = Decimal(top_five_value) / Decimal(nav_units)
        cash_weight = Decimal(cash_units) / Decimal(nav_units)
        hhi = (
            sum(
                (
                    (Decimal(value) / Decimal(equity_units)) ** 2
                    for value in holding_units
                ),
                Decimal("0"),
            )
            if equity_units > 0
            else None
        )
        top_five_weight_pct = top_five_weight * _PERCENT
        cash_weight_pct = cash_weight * _PERCENT

    alerts = []

    def add_alert(
        code: str,
        metric_name: str,
        observed: Decimal,
        threshold: Decimal,
        *,
        listing: str | None = None,
        percent: bool = False,
    ) -> None:
        with localcontext() as context:
            context.prec = 50
            shown_observed = observed * _PERCENT if percent else observed
            shown_threshold = threshold * _PERCENT if percent else threshold
        rendered_observed = _decimal_string(shown_observed)
        rendered_threshold = _decimal_string(shown_threshold)
        description = (
            f"{listing} represents {rendered_observed}% of total portfolio NAV, "
            f"exceeding the configured {rendered_threshold}% exposure threshold."
            if code == "SINGLE_STOCK_EXPOSURE_HIGH"
            else (
                f"The five largest listings represent {rendered_observed}% of "
                f"total portfolio NAV, exceeding the configured "
                f"{rendered_threshold}% concentration threshold."
                if code == "TOP_FIVE_CONCENTRATION_HIGH"
                else (
                    f"Cash represents {rendered_observed}% of total portfolio "
                    f"NAV, below the configured {rendered_threshold}% minimum."
                    if code == "CASH_ALLOCATION_LOW"
                    else (
                        f"Equity HHI is {rendered_observed}, exceeding the "
                        f"configured {rendered_threshold} concentration threshold."
                    )
                )
            )
        )
        alerts.append(
            {
                "code": code,
                "severity": "MEDIUM",
                "metric_name": metric_name,
                "affected_listing": listing,
                "observed_value": rendered_observed,
                "threshold": rendered_threshold,
                "policy_version": EXPOSURE_RISK_POLICY_VERSION,
                "explanation": description,
            }
        )

    single_stock_threshold = EXPOSURE_RISK_THRESHOLDS["single_stock_exposure"]
    for item, value_units, weight in zip(
        normalized_holdings, holding_units, listing_weights
    ):
        if _ratio_greater(
            value_units, nav_units, single_stock_threshold
        ):
            add_alert(
                "SINGLE_STOCK_EXPOSURE_HIGH",
                "single_stock_exposure",
                weight,
                single_stock_threshold,
                listing=item["listing"],
                percent=True,
            )

    top_five_threshold = EXPOSURE_RISK_THRESHOLDS["top_five_concentration"]
    if _ratio_greater(top_five_value, nav_units, top_five_threshold):
        add_alert(
            "TOP_FIVE_CONCENTRATION_HIGH",
            "top_five_concentration",
            top_five_weight,
            top_five_threshold,
            percent=True,
        )

    cash_threshold = EXPOSURE_RISK_THRESHOLDS["minimum_cash_allocation"]
    if _ratio_less(cash_units, nav_units, cash_threshold):
        add_alert(
            "CASH_ALLOCATION_LOW",
            "cash_allocation",
            cash_weight,
            cash_threshold,
            percent=True,
        )

    hhi_threshold = EXPOSURE_RISK_THRESHOLDS["equity_hhi"]
    if hhi is not None and _ratio_greater(
        sum(value * value for value in holding_units),
        equity_units * equity_units,
        hhi_threshold,
    ):
        add_alert(
            "EQUITY_HHI_HIGH",
            "equity_hhi",
            hhi,
            hhi_threshold,
        )

    return {
        "assessment_status": "EVALUATED",
        "policy_version": EXPOSURE_RISK_POLICY_VERSION,
        "evaluated_at": assessment_time.isoformat(),
        "session_date": session_date.isoformat(),
        "portfolio_nav": _decimal_string(nav),
        "total_equity_market_value": _decimal_string(equity_market_value),
        "cash_balance": _decimal_string(cash),
        "listing_exposures": listing_exposures,
        "top_five_weight_pct": _decimal_string(top_five_weight_pct),
        "cash_weight_pct": _decimal_string(cash_weight_pct),
        "equity_hhi": _decimal_string(hhi) if hhi is not None else None,
        "alerts": alerts,
        "reason_codes": [],
        "unavailable_listings": [],
    }
