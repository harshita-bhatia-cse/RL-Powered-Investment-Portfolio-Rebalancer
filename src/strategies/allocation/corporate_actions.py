"""Normalized corporate-action accounting, independent of any data provider."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from numbers import Real
from typing import Iterable, Mapping, Optional


class CorporateActionType(str, Enum):
    """Supported normalized event types."""

    SPLIT = "split"
    CASH_DIVIDEND = "cash_dividend"
    ETF_DISTRIBUTION = "etf_distribution"


@dataclass(frozen=True)
class CorporateActionEvent:
    """A provider-neutral split or per-share cash distribution record.

    ``source_id`` must uniquely identify the event for a symbol. A split ratio
    means new shares per old share; for example, 2.0 is a 2-for-1 split.
    """

    symbol: str
    event_type: CorporateActionType
    source_id: str
    currency: str = "USD"
    announcement_date: Optional[date | str] = None
    ex_date: Optional[date | str] = None
    effective_date: Optional[date | str] = None
    payment_date: Optional[date | str] = None
    split_ratio: Optional[float] = None
    distribution_per_share: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(self.event_type, CorporateActionType):
            raise ValueError("event_type must be a CorporateActionType")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if (
            not isinstance(self.currency, str)
            or len(self.currency.strip()) != 3
            or not self.currency.strip().isalpha()
        ):
            raise ValueError("currency must be a three-letter currency code")
        object.__setattr__(self, "currency", self.currency.strip().upper())

        for field_name in ("announcement_date", "ex_date", "effective_date", "payment_date"):
            raw_date = getattr(self, field_name)
            if raw_date is not None:
                object.__setattr__(self, field_name, _as_date(field_name, raw_date))

        if self.event_type is CorporateActionType.SPLIT:
            if self.effective_date is None:
                raise ValueError("split events require effective_date")
            if self.split_ratio is None or _positive("split_ratio", self.split_ratio) <= 0:
                raise ValueError("split events require a positive split_ratio")
            if self.ex_date is not None or self.payment_date is not None:
                raise ValueError("split events cannot have ex_date or payment_date")
            if self.distribution_per_share is not None:
                raise ValueError("split events cannot have distribution_per_share")
        else:
            if self.ex_date is None:
                raise ValueError("distribution events require ex_date")
            if self.distribution_per_share is None:
                raise ValueError("distribution events require distribution_per_share")
            _positive("distribution_per_share", self.distribution_per_share)
            if self.effective_date is not None or self.split_ratio is not None:
                raise ValueError("distribution events cannot have split fields")
            if self.payment_date is not None and self.payment_date < self.ex_date:
                raise ValueError("payment_date cannot precede ex_date")

    @property
    def event_id(self) -> str:
        """Return a stable event identity scoped to its symbol."""
        return f"{self.symbol}:{self.source_id}"

    @property
    def action_date(self) -> date:
        """Return the date on which the split or entitlement takes effect."""
        action_date = (
            self.effective_date
            if self.event_type is CorporateActionType.SPLIT
            else self.ex_date
        )
        if action_date is None:
            raise ValueError("Corporate action is missing its effective date")
        return action_date


@dataclass(frozen=True)
class PortfolioAccountingSnapshot:
    """Whole-account balances at one mark date."""

    asset_quantities: Mapping[str, float]
    settled_cash: float
    dividend_receivables: Mapping[str, float]
    total_nav: float


class CorporateActionLedger:
    """Apply normalized actions once while keeping receivables out of cash.

    This ledger does not fetch or infer corporate-action records. Events must
    come from a verified external source. Only one currency is supported; no
    FX conversion, tax, withholding, or cash-in-lieu accounting is performed.
    Fractional shares created by splits are retained.
    """

    def __init__(
        self,
        holdings: Mapping[str, float],
        settled_cash: float,
        *,
        currency: str = "USD",
        reinvest_distributions: bool = False,
        allow_fractional_shares: bool = True,
    ) -> None:
        if (
            not isinstance(currency, str)
            or len(currency.strip()) != 3
            or not currency.strip().isalpha()
        ):
            raise ValueError("currency must be a three-letter currency code")
        if not isinstance(reinvest_distributions, bool):
            raise ValueError("reinvest_distributions must be a bool")
        if not isinstance(allow_fractional_shares, bool):
            raise ValueError("allow_fractional_shares must be a bool")

        self.currency = currency.strip().upper()
        self.reinvest_distributions = reinvest_distributions
        self.allow_fractional_shares = allow_fractional_shares
        self._holdings = _validated_balances("holdings", holdings)
        self._settled_cash = _non_negative("settled_cash", settled_cash)
        self._events: dict[str, CorporateActionEvent] = {}
        self._applied_action_ids: set[str] = set()
        self._settled_payment_ids: set[str] = set()
        self._receivables: dict[str, float] = {}
        self._last_processed_date: Optional[date] = None

    @property
    def holdings(self) -> dict[str, float]:
        """Return a copy of current asset quantities."""
        return dict(self._holdings)

    @property
    def settled_cash(self) -> float:
        """Return cash available to the allocator."""
        return self._settled_cash

    @property
    def dividend_receivables(self) -> dict[str, float]:
        """Return unsettled per-event distribution balances."""
        return dict(self._receivables)

    @property
    def applied_event_ids(self) -> frozenset[str]:
        """Return action identities already applied to share quantities."""
        return frozenset(self._applied_action_ids)

    def process_date(
        self,
        as_of_date: date | str,
        events: Iterable[CorporateActionEvent],
        *,
        reinvestment_prices: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Apply actions/ex-date entitlements and settle payments through a date.

        Supply the full normalized event history on each call (or at least all
        events not yet encountered). Calls must move forward chronologically.
        For same-day events, distribution entitlements are calculated before
        split share adjustments. Repeating an event is idempotent.
        """
        day = _as_date("as_of_date", as_of_date)
        if self._last_processed_date is not None and day < self._last_processed_date:
            raise ValueError("as_of_date cannot move backwards")

        for event in events:
            if not isinstance(event, CorporateActionEvent):
                raise ValueError("events must contain CorporateActionEvent records")
            if event.currency != self.currency:
                raise ValueError(
                    f"Event {event.event_id!r} currency does not match ledger currency"
                )
            previous = self._events.get(event.event_id)
            if previous is not None and previous != event:
                raise ValueError(
                    f"Conflicting event records share identity {event.event_id!r}"
                )
            if (
                previous is None
                and self._last_processed_date is not None
                and event.action_date <= self._last_processed_date
            ):
                raise ValueError(
                    f"Late event {event.event_id!r} predates the processed ledger"
                )
            self._events[event.event_id] = event

        for event in self._events.values():
            if event.action_date < day and event.event_id not in self._applied_action_ids:
                raise ValueError(
                    f"Missed action date for event {event.event_id!r}; "
                    "process the ledger on each event date"
                )
            if (
                event.payment_date is not None
                and event.payment_date < day
                and event.event_id in self._receivables
                and event.event_id not in self._settled_payment_ids
            ):
                raise ValueError(
                    f"Missed payment date for event {event.event_id!r}; "
                    "process the ledger on each payment date"
                )

        due_actions = sorted(
            (
                event
                for event in self._events.values()
                if event.action_date == day
                and event.event_id not in self._applied_action_ids
            ),
            key=lambda event: (
                1 if event.event_type is CorporateActionType.SPLIT else 0,
                event.symbol,
                event.event_id,
            ),
        )
        due_payment_ids = set(self._receivables)
        due_payment_ids.update(
            event.event_id
            for event in due_actions
            if event.event_type is not CorporateActionType.SPLIT
            and event.payment_date == day
            and self._holdings.get(event.symbol, 0.0)
            * float(event.distribution_per_share)
            > 0.0
        )
        due_payments = sorted(
            (
                event
                for event in self._events.values()
                if event.payment_date == day
                and event.event_id in due_payment_ids
                and event.event_id not in self._settled_payment_ids
            ),
            key=lambda event: event.event_id,
        )

        prices = _validated_balances(
            "reinvestment_prices", reinvestment_prices or {}, allow_zero=False
        )
        if self.reinvest_distributions:
            missing_prices = sorted(
                {
                    event.symbol
                    for event in due_payments
                    if event.symbol not in prices
                }
            )
            if missing_prices:
                raise ValueError(
                    "Missing reinvestment prices for: " + ", ".join(missing_prices)
                )

        holdings = dict(self._holdings)
        settled_cash = self._settled_cash
        receivables = dict(self._receivables)
        applied_ids = set(self._applied_action_ids)
        settled_ids = set(self._settled_payment_ids)

        for event in due_actions:
            if event.event_type is CorporateActionType.SPLIT:
                ratio = float(event.split_ratio)
                holdings[event.symbol] = holdings.get(event.symbol, 0.0) * ratio
            else:
                amount = (
                    holdings.get(event.symbol, 0.0)
                    * float(event.distribution_per_share)
                )
                if amount:
                    receivables[event.event_id] = amount
            applied_ids.add(event.event_id)

        for event in due_payments:
            amount = receivables.pop(event.event_id, 0.0)
            if self.reinvest_distributions:
                price = prices[event.symbol]
                quantity = amount / price
                if not self.allow_fractional_shares:
                    quantity = math.floor(quantity)
                holdings[event.symbol] = holdings.get(event.symbol, 0.0) + quantity
                settled_cash += amount - quantity * price
            else:
                settled_cash += amount
            settled_ids.add(event.event_id)

        self._holdings = holdings
        self._settled_cash = settled_cash
        self._receivables = receivables
        self._applied_action_ids = applied_ids
        self._settled_payment_ids = settled_ids
        self._last_processed_date = day

    def update_investable_balances(
        self, holdings: Mapping[str, float], settled_cash: float
    ) -> None:
        """Synchronize balances from a non-broker simulated allocation result.

        Pass the allocator's final holdings and settled cash after accepting its
        plan. Receivables and corporate-action event state are left unchanged.
        """
        updated_holdings = _validated_balances("holdings", holdings)
        updated_cash = _non_negative("settled_cash", settled_cash)
        self._holdings = updated_holdings
        self._settled_cash = updated_cash

    def snapshot(self, mark_prices: Mapping[str, float]) -> PortfolioAccountingSnapshot:
        """Value holdings plus settled cash and receivables at supplied marks."""
        marks = _validated_balances("mark_prices", mark_prices, allow_zero=False)
        missing = sorted(symbol for symbol in self._holdings if symbol not in marks)
        if missing:
            raise ValueError("Missing mark prices for: " + ", ".join(missing))
        nav = (
            self._settled_cash
            + math.fsum(self._receivables.values())
            + math.fsum(
                quantity * marks[symbol]
                for symbol, quantity in self._holdings.items()
            )
        )
        if not math.isfinite(nav):
            raise ValueError("Total portfolio NAV must be finite")
        return PortfolioAccountingSnapshot(
            asset_quantities=dict(self._holdings),
            settled_cash=self._settled_cash,
            dividend_receivables=dict(self._receivables),
            total_nav=nav,
        )


def _as_date(name: str, value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO date") from exc
    raise ValueError(f"{name} must be a date or ISO date string")


def _positive(name: str, value: float) -> float:
    number = _number(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(name: str, value: float) -> float:
    number = _number(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _validated_balances(
    name: str,
    values: Optional[Mapping[str, float]],
    *,
    allow_zero: bool = True,
) -> dict[str, float]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for symbol, raw_value in values.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"{name} keys must be non-empty strings")
        value = _number(f"{name}[{symbol!r}]", raw_value)
        if value < 0.0 or (not allow_zero and value <= 0.0):
            qualifier = "positive" if not allow_zero else "non-negative"
            raise ValueError(f"{name}[{symbol!r}] must be {qualifier}")
        result[symbol] = value
    return result
