"""Provider-independent EOD price contracts and a deterministic test provider."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Protocol, Sequence


class MarketDataProviderError(RuntimeError):
    """Base class for provider failures."""


class MarketDataUnavailableError(MarketDataProviderError):
    """The provider cannot supply the requested data."""


class InvalidMarketDataError(MarketDataProviderError):
    """The provider response is malformed or fails data validation."""


@dataclass(frozen=True)
class EODPriceRecord:
    exchange: str
    symbol: str
    market_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    provider: str
    source: str
    fetched_at: datetime
    currency: str = "INR"
    volume: int | None = None
    valuation_permitted: bool = False


class EODMarketDataProvider(Protocol):
    """Provider contract; responses contain unadjusted EOD OHLC by default."""

    def fetch_latest(self, exchange: str, symbol: str) -> EODPriceRecord:
        """Return the latest completed EOD bar or raise an explicit provider error."""
        ...

    def fetch_history(
        self, exchange: str, symbol: str, start_date: date, end_date: date
    ) -> Sequence[EODPriceRecord]:
        """Return completed EOD bars in the inclusive requested range."""
        ...


class MockEODProvider:
    """In-memory provider for deterministic tests; never performs network access."""

    def __init__(self, records: Sequence[EODPriceRecord] = ()) -> None:
        self._records = tuple(records)

    def fetch_latest(self, exchange: str, symbol: str) -> EODPriceRecord:
        records = self._matching(exchange, symbol)
        if not records:
            raise MarketDataUnavailableError(
                f"No EOD data for {exchange}:{symbol}"
            )
        return max(records, key=lambda record: record.market_date)

    def fetch_history(
        self, exchange: str, symbol: str, start_date: date, end_date: date
    ) -> Sequence[EODPriceRecord]:
        return tuple(
            record
            for record in self._matching(exchange, symbol)
            if start_date <= record.market_date <= end_date
        )

    def _matching(self, exchange: str, symbol: str) -> tuple[EODPriceRecord, ...]:
        normalized_exchange = exchange.strip().upper()
        normalized_symbol = symbol.strip().upper()
        return tuple(
            record
            for record in self._records
            if record.exchange == normalized_exchange
            and record.symbol == normalized_symbol
        )


def utc_now() -> datetime:
    """Current UTC timestamp for provider/test adapters."""
    return datetime.now(timezone.utc)
