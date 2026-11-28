"""Validation, persistence, retrieval, freshness, and paper valuation services."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Sequence
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from src.database.models import Holding, Portfolio
from src.market_data.models import EODPrice
from src.market_data.providers import (
    EODMarketDataProvider,
    EODPriceRecord,
    InvalidMarketDataError,
    MarketDataUnavailableError,
)

_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9&.-]{0,31}$")
_CENT = Decimal("0.01")
_ZERO = Decimal("0")
_INDIA_TZ = ZoneInfo("Asia/Kolkata")


class MarketDataConflictError(ValueError):
    """A price duplicates a previously stored listing/session."""


class MarketDataStorageUnavailableError(RuntimeError):
    """The dedicated EOD price database cannot serve a read."""


class PortfolioNotFoundError(LookupError):
    """The requested owned portfolio does not exist."""


def _validate_record(record: EODPriceRecord, today: date) -> EODPriceRecord:
    if not isinstance(record, EODPriceRecord):
        raise InvalidMarketDataError("provider returned a non-EOD price record")
    if not isinstance(record.exchange, str) or not isinstance(record.symbol, str):
        raise InvalidMarketDataError("exchange and symbol must be strings")
    exchange = record.exchange.strip().upper()
    symbol = record.symbol.strip().upper()
    if exchange not in {"NSE", "BSE"}:
        raise InvalidMarketDataError("exchange must be NSE or BSE")
    if not _SYMBOL.fullmatch(symbol):
        raise InvalidMarketDataError("symbol contains unsupported characters")
    if not isinstance(record.market_date, date) or isinstance(record.market_date, datetime):
        raise InvalidMarketDataError("market_date must be a date")
    if record.market_date > today:
        raise InvalidMarketDataError("future market dates are not accepted")
    if not isinstance(record.currency, str) or record.currency.strip().upper() != "INR":
        raise InvalidMarketDataError("only INR prices are supported")
    if (
        not isinstance(record.provider, str)
        or not record.provider.strip()
        or not isinstance(record.source, str)
        or not record.source.strip()
    ):
        raise InvalidMarketDataError("provider and source metadata are required")
    if len(record.provider.strip()) > 100 or len(record.source.strip()) > 200:
        raise InvalidMarketDataError("provider or source metadata exceeds storage limits")
    if not isinstance(record.fetched_at, datetime) or record.fetched_at.tzinfo is None:
        raise InvalidMarketDataError("fetched_at must include a timezone")
    if record.fetched_at.astimezone(timezone.utc) > datetime.now(timezone.utc):
        raise InvalidMarketDataError("fetched_at cannot be in the future")
    prices = (record.open, record.high, record.low, record.close)
    for value in prices:
        try:
            has_supported_precision = (
                isinstance(value, Decimal)
                and value.quantize(Decimal("0.000001")) == value
            )
        except InvalidOperation:
            has_supported_precision = False
        if (
            not isinstance(value, Decimal)
            or not value.is_finite()
            or value <= 0
            or value >= Decimal("100000000000000")
            or not has_supported_precision
        ):
            raise InvalidMarketDataError(
                "OHLC prices must be positive finite INR values with at most "
                "six decimal places"
            )
    if (
        record.high < record.low
        or record.high < record.open
        or record.high < record.close
        or record.low > record.open
        or record.low > record.close
    ):
        raise InvalidMarketDataError("OHLC values do not satisfy high/low relationships")
    if record.volume is not None and (
        isinstance(record.volume, bool)
        or not isinstance(record.volume, int)
        or record.volume < 0
        or record.volume > 9_223_372_036_854_775_807
    ):
        raise InvalidMarketDataError("volume must be a non-negative integer or null")
    if not isinstance(record.valuation_permitted, bool):
        raise InvalidMarketDataError("valuation_permitted must be a boolean")
    return EODPriceRecord(
        exchange=exchange,
        symbol=symbol,
        market_date=record.market_date,
        open=record.open,
        high=record.high,
        low=record.low,
        close=record.close,
        currency="INR",
        provider=record.provider.strip(),
        source=record.source.strip(),
        fetched_at=record.fetched_at.astimezone(timezone.utc),
        volume=record.volume,
        valuation_permitted=record.valuation_permitted,
    )


def _today_india() -> date:
    return datetime.now(_INDIA_TZ).date()


def _parse_max_age_days() -> int:
    raw = os.environ.get("MARKET_DATA_MAX_AGE_DAYS", "4")
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError("MARKET_DATA_MAX_AGE_DAYS must be a positive integer") from error
    if value < 1:
        raise RuntimeError("MARKET_DATA_MAX_AGE_DAYS must be a positive integer")
    return value


def ingest_history(
    db: Session,
    provider: EODMarketDataProvider,
    exchange: str,
    symbol: str,
    start_date: date,
    end_date: date,
) -> list[EODPrice]:
    if start_date > end_date:
        raise InvalidMarketDataError("start_date must not be after end_date")
    records = provider.fetch_history(exchange, symbol, start_date, end_date)
    if not records:
        raise MarketDataUnavailableError(
            f"Provider returned no EOD data for {exchange}:{symbol}"
        )
    validated = [_validate_record(record, _today_india()) for record in records]
    if any(
        record.exchange != exchange.strip().upper()
        or record.symbol != symbol.strip().upper()
        or not start_date <= record.market_date <= end_date
        for record in validated
    ):
        raise InvalidMarketDataError(
            "provider returned a listing or market date outside the request"
        )
    keys = [(record.exchange, record.symbol, record.market_date) for record in validated]
    if len(set(keys)) != len(keys):
        raise MarketDataConflictError("provider response contains duplicate listing dates")
    existing = db.scalars(
        select(EODPrice).where(
            EODPrice.exchange == exchange.strip().upper(),
            EODPrice.symbol == symbol.strip().upper(),
            EODPrice.market_date >= start_date,
            EODPrice.market_date <= end_date,
        )
    ).all()
    existing_keys = {
        (record.exchange, record.symbol, record.market_date) for record in existing
    }
    if existing_keys.intersection(keys):
        raise MarketDataConflictError("one or more EOD listing dates already exist")

    rows = [
        EODPrice(
            exchange=record.exchange,
            symbol=record.symbol,
            market_date=record.market_date,
            open=record.open,
            high=record.high,
            low=record.low,
            close=record.close,
            volume=record.volume,
            currency=record.currency,
            provider=record.provider,
            source=record.source,
            valuation_permitted=record.valuation_permitted,
            fetched_at=record.fetched_at,
        )
        for record in validated
    ]
    db.add_all(rows)
    try:
        db.flush()
    except IntegrityError as error:
        db.rollback()
        raise MarketDataConflictError(
            "one or more EOD listing dates already exist"
        ) from error
    return rows


def ingest_latest(
    db: Session,
    provider: EODMarketDataProvider,
    exchange: str,
    symbol: str,
) -> EODPrice:
    record = provider.fetch_latest(exchange, symbol)
    validated = _validate_record(record, _today_india())
    if (
        validated.exchange != exchange.strip().upper()
        or validated.symbol != symbol.strip().upper()
    ):
        raise InvalidMarketDataError("provider returned a different listing")
    if db.scalar(
        select(EODPrice.id).where(
            EODPrice.exchange == validated.exchange,
            EODPrice.symbol == validated.symbol,
            EODPrice.market_date == validated.market_date,
        )
    ) is not None:
        raise MarketDataConflictError("EOD listing date already exists")
    row = EODPrice(
        exchange=validated.exchange,
        symbol=validated.symbol,
        market_date=validated.market_date,
        open=validated.open,
        high=validated.high,
        low=validated.low,
        close=validated.close,
        volume=validated.volume,
        currency=validated.currency,
        provider=validated.provider,
        source=validated.source,
        valuation_permitted=validated.valuation_permitted,
        fetched_at=validated.fetched_at,
    )
    db.add(row)
    try:
        db.flush()
    except IntegrityError as error:
        db.rollback()
        raise MarketDataConflictError("EOD listing date already exists") from error
    return row


def get_history(
    db: Session,
    exchange: str,
    symbol: str,
    start_date: date,
    end_date: date,
) -> list[EODPrice]:
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    return list(
        db.scalars(
            select(EODPrice)
            .where(
                EODPrice.exchange == exchange.strip().upper(),
                EODPrice.symbol == symbol.strip().upper(),
                EODPrice.market_date >= start_date,
                EODPrice.market_date <= end_date,
            )
            .order_by(EODPrice.market_date)
        ).all()
    )


def _latest_permitted(
    db: Session, exchange: str, symbol: str
) -> EODPrice | None:
    try:
        return db.scalar(
            select(EODPrice)
            .where(
                EODPrice.exchange == exchange,
                EODPrice.symbol == symbol,
                EODPrice.valuation_permitted.is_(True),
            )
            .order_by(EODPrice.market_date.desc(), EODPrice.fetched_at.desc())
            .limit(1)
        )
    except OperationalError as error:
        raise MarketDataStorageUnavailableError(
            "Market-data storage is unavailable"
        ) from error


def _stale(price: EODPrice, today: date, max_age_days: int) -> bool:
    return (today - price.market_date).days > max_age_days


def value_portfolio(
    account_db: Session,
    market_db: Session,
    user_id: str,
    portfolio_id: str,
    *,
    today: date | None = None,
    max_age_days: int | None = None,
) -> dict:
    portfolio = account_db.scalar(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user_id,
        )
    )
    if portfolio is None:
        raise PortfolioNotFoundError("Portfolio not found")
    holdings = account_db.scalars(
        select(Holding)
        .where(Holding.portfolio_id == portfolio_id)
        .order_by(Holding.exchange, Holding.symbol)
    ).all()
    valuation_date = today or _today_india()
    permitted_age = max_age_days if max_age_days is not None else _parse_max_age_days()
    if permitted_age < 1:
        raise ValueError("max_age_days must be a positive integer")

    total_cost = _ZERO
    partial_market_value = _ZERO
    partial_unrealized = _ZERO
    valued_holdings = []
    stale_prices = []
    missing_prices = []
    unavailable_prices = []

    for holding in holdings:
        cost_basis = (holding.quantity * holding.average_purchase_price).quantize(
            _CENT, rounding=ROUND_HALF_UP
        )
        total_cost += cost_basis
        price = _latest_permitted(market_db, holding.exchange, holding.symbol)
        if price is None:
            try:
                any_price = market_db.scalar(
                    select(EODPrice.id)
                    .where(
                        EODPrice.exchange == holding.exchange,
                        EODPrice.symbol == holding.symbol,
                    )
                    .limit(1)
                )
            except OperationalError as error:
                raise MarketDataStorageUnavailableError(
                    "Market-data storage is unavailable"
                ) from error
            item = {
                "exchange": holding.exchange,
                "symbol": holding.symbol,
                "reason": (
                    "no permitted EOD price is stored"
                    if any_price is None
                    else "stored price source is not permitted for valuation"
                ),
            }
            if any_price is None:
                missing_prices.append(item)
            else:
                unavailable_prices.append(item)
            continue
        if _stale(price, valuation_date, permitted_age):
            stale_prices.append(
                {
                    "exchange": holding.exchange,
                    "symbol": holding.symbol,
                    "market_date": price.market_date,
                    "close": price.close,
                    "provider": price.provider,
                    "source": price.source,
                    "fetched_at": price.fetched_at,
                    "age_calendar_days": (valuation_date - price.market_date).days,
                }
            )
            continue

        market_value = (holding.quantity * price.close).quantize(
            _CENT, rounding=ROUND_HALF_UP
        )
        unrealized = (market_value - cost_basis).quantize(
            _CENT, rounding=ROUND_HALF_UP
        )
        partial_market_value += market_value
        partial_unrealized += unrealized
        valued_holdings.append(
            {
                "exchange": holding.exchange,
                "symbol": holding.symbol,
                "quantity": holding.quantity,
                "cost_basis": cost_basis,
                "market_date": price.market_date,
                "close": price.close,
                "market_value": market_value,
                "unrealized_profit_loss": unrealized,
                "provider": price.provider,
                "source": price.source,
                "fetched_at": price.fetched_at,
            }
        )

    complete = not stale_prices and not missing_prices and not unavailable_prices
    partial_market_value = partial_market_value.quantize(
        _CENT, rounding=ROUND_HALF_UP
    )
    partial_unrealized = partial_unrealized.quantize(
        _CENT, rounding=ROUND_HALF_UP
    )
    total_cost = total_cost.quantize(_CENT, rounding=ROUND_HALF_UP)
    cash = portfolio.cash_balance
    return {
        "portfolio_id": portfolio.id,
        "currency": "INR",
        "valuation_date": valuation_date,
        "valuation_status": "complete" if complete else "partial",
        "valuation_complete": complete,
        "invested_cost_basis": total_cost,
        "market_value": partial_market_value,
        "market_value_is_partial": not complete,
        "unrealized_profit_loss": partial_unrealized,
        "unrealized_profit_loss_is_partial": not complete,
        "cash_balance": cash,
        "total_portfolio_value": (
            (cash + partial_market_value).quantize(
                _CENT, rounding=ROUND_HALF_UP
            )
            if complete
            else None
        ),
        "valued_holdings": valued_holdings,
        "stale_prices": stale_prices,
        "missing_prices": missing_prices,
        "unavailable_prices": unavailable_prices,
        "freshness_policy": {
            "max_age_calendar_days": permitted_age,
            "basis": (
                "Elapsed calendar days from market_date; conservative freshness "
                "threshold, not an exchange trading-session calendar."
            ),
        },
    }
