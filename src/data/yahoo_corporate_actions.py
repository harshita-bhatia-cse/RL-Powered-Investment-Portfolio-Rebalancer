"""Yahoo Finance corporate-action candidates and issuer verification workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import pandas as pd

from src.data.corporate_action_provider import (
    ASSET_UNIVERSE,
    CorporateActionValidationError,
)
from src.strategies.allocation import CorporateActionEvent, CorporateActionType

YAHOO_CANDIDATE_COLUMNS = (
    "symbol",
    "event_type",
    "source_id",
    "currency",
    "announcement_date",
    "ex_date",
    "payment_date",
    "effective_date",
    "distribution_per_share",
    "split_ratio",
    "source_field",
    "verification_status",
    "missing_fields",
)
ISSUER_VERIFICATION_COLUMNS = (
    "symbol",
    "event_type",
    "action_date",
    "source_field",
    "yahoo_reported_value",
    "source_id",
    "currency",
    "distribution_per_share",
    "payment_date",
    "split_ratio",
    "verification_source",
    "verification_reference",
)
VERIFIED_EVENT_COLUMNS = (
    "symbol",
    "event_type",
    "source_id",
    "currency",
    "announcement_date",
    "ex_date",
    "payment_date",
    "effective_date",
    "distribution_per_share",
    "split_ratio",
)
ACTION_COLUMNS = {
    "Dividends": (CorporateActionType.ETF_DISTRIBUTION, "distribution"),
    "Capital Gains": (CorporateActionType.ETF_DISTRIBUTION, "capital_gain"),
    "Stock Splits": (CorporateActionType.SPLIT, "split"),
}


class YahooCorporateActionError(RuntimeError):
    """Base error for Yahoo Finance action acquisition."""


class YahooResponseError(YahooCorporateActionError):
    """Yahoo returned an empty, malformed, or incomplete history response."""


@dataclass(frozen=True)
class YahooActionCandidate:
    """Yahoo-reported event data, preserving fields Yahoo does not provide as null."""

    symbol: str
    event_type: CorporateActionType
    source_id: Optional[str]
    currency: Optional[str]
    announcement_date: Optional[date]
    ex_date: Optional[date]
    payment_date: Optional[date]
    effective_date: Optional[date]
    distribution_per_share: Optional[float]
    split_ratio: Optional[float]
    source_field: str
    raw_value: float
    verification_status: str
    missing_fields: tuple[str, ...]

    @property
    def action_date(self) -> date:
        action_date = self.effective_date or self.ex_date
        if action_date is None:
            raise CorporateActionValidationError(
                f"Yahoo candidate for {self.symbol} has no action date"
            )
        return action_date


@dataclass(frozen=True)
class YahooSymbolOutcome:
    symbol: str
    status: str
    price_rows: int
    price_start: Optional[str]
    price_end: Optional[str]
    currency: Optional[str]
    dividend_count: int
    capital_gain_count: int
    split_count: int
    duplicate_candidate_count: int
    error: Optional[str] = None


@dataclass(frozen=True)
class YahooAcquisitionBatch:
    provider: str
    requested_symbols: tuple[str, ...]
    start_date: str
    end_date: str
    retrieved_at: str
    candidates: tuple[YahooActionCandidate, ...]
    raw_records: tuple[Mapping[str, Any], ...]
    outcomes: tuple[YahooSymbolOutcome, ...]
    price_frames: Mapping[str, pd.DataFrame]

    @property
    def complete(self) -> bool:
        expected = set(self.requested_symbols)
        actual = {outcome.symbol for outcome in self.outcomes}
        return (
            actual == expected
            and len(self.outcomes) == len(expected)
            and all(outcome.status == "success" for outcome in self.outcomes)
        )


class YahooFinanceCorporateActionProvider:
    """Retrieve Yahoo Finance action rows via yfinance without inventing fields.

    Yahoo action rows generally expose event dates and amounts/ratios but not
    payment dates or stable event IDs. Such fields remain ``None``. Returned
    candidates are not ledger-ready; the issuer verification workflow must
    supply real identifiers and confirm distributions before ledger use.
    """

    def __init__(
        self,
        *,
        ticker_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        if ticker_factory is None:
            try:
                import yfinance
            except ImportError as exc:
                raise RuntimeError(
                    "yfinance is required; install the repository-declared dependency"
                ) from exc
            ticker_factory = yfinance.Ticker
        self._ticker_factory = ticker_factory

    def acquire(
        self,
        symbols: Sequence[str] = ASSET_UNIVERSE,
        start_date: date | str = "2018-01-03",
        end_date: date | str = "2026-11-11",
    ) -> YahooAcquisitionBatch:
        """Download action-enabled, unadjusted daily histories for each symbol."""
        start = _as_date("start_date", start_date)
        end = _as_date("end_date", end_date)
        if start > end:
            raise ValueError("start_date must not be after end_date")
        requested = _validate_symbols(symbols)

        candidates: list[YahooActionCandidate] = []
        raw_records: list[Mapping[str, Any]] = []
        outcomes: list[YahooSymbolOutcome] = []
        price_frames: dict[str, pd.DataFrame] = {}

        for symbol in requested:
            try:
                ticker = self._ticker_factory(symbol)
                frame = ticker.history(
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    actions=True,
                    auto_adjust=False,
                    back_adjust=False,
                    repair=False,
                    keepna=False,
                )
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    raise YahooResponseError(
                        f"Yahoo returned no price history for {symbol}; "
                        "empty actions cannot be treated as no events"
                    )
                missing_columns = {"Dividends", "Stock Splits"} - set(frame.columns)
                if missing_columns:
                    raise YahooResponseError(
                        f"Yahoo history omitted action columns for {symbol}: "
                        + ", ".join(sorted(missing_columns))
                    )
                normalized_frame = _normalize_yahoo_frame(
                    frame,
                    symbol=symbol,
                    start=start,
                    end=end,
                )
                if normalized_frame.empty:
                    raise YahooResponseError(
                        f"Yahoo returned no in-range price rows for {symbol}"
                    )
                metadata = getattr(ticker, "history_metadata", {}) or {}
                currency_raw = metadata.get("currency")
                currency = (
                    currency_raw.strip().upper()
                    if isinstance(currency_raw, str) and currency_raw.strip()
                    else None
                )
                if currency is not None and currency != "USD":
                    raise CorporateActionValidationError(
                        f"Yahoo reports unsupported currency {currency!r} for {symbol}"
                    )
                price_frames[symbol] = normalized_frame
                symbol_candidates, symbol_raw = _extract_candidates(
                    symbol,
                    normalized_frame,
                    currency=currency,
                )
                candidates.extend(symbol_candidates)
                raw_records.extend(symbol_raw)
                counts = {
                    "distribution": sum(
                        item.source_field == "Dividends"
                        for item in symbol_candidates
                    ),
                    "capital_gain": sum(
                        item.source_field == "Capital Gains"
                        for item in symbol_candidates
                    ),
                    "split": sum(
                        item.source_field == "Stock Splits"
                        for item in symbol_candidates
                    ),
                }
                outcomes.append(
                    YahooSymbolOutcome(
                        symbol=symbol,
                        status="success",
                        price_rows=len(normalized_frame),
                        price_start=normalized_frame.index.min().isoformat(),
                        price_end=normalized_frame.index.max().isoformat(),
                        currency=currency,
                        dividend_count=counts["distribution"],
                        capital_gain_count=counts["capital_gain"],
                        split_count=counts["split"],
                        duplicate_candidate_count=_duplicate_count(
                            symbol_candidates
                        ),
                    )
                )
            except Exception as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                outcomes.append(
                    YahooSymbolOutcome(
                        symbol=symbol,
                        status="failed",
                        price_rows=0,
                        price_start=None,
                        price_end=None,
                        currency=None,
                        dividend_count=0,
                        capital_gain_count=0,
                        split_count=0,
                        duplicate_candidate_count=0,
                        error=(
                            str(exc)
                            if isinstance(exc, (YahooResponseError, CorporateActionValidationError))
                            else f"yfinance request failed ({type(exc).__name__})"
                        ),
                    )
                )

        candidates.sort(key=_candidate_sort_key)
        return YahooAcquisitionBatch(
            provider="Yahoo Finance via yfinance",
            requested_symbols=requested,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            candidates=tuple(candidates),
            raw_records=tuple(raw_records),
            outcomes=tuple(outcomes),
            price_frames=price_frames,
        )


def validate_yahoo_candidates(
    candidates: Iterable[YahooActionCandidate],
    *,
    start_date: date | str,
    end_date: date | str,
    market_sessions: Iterable[date | str],
) -> tuple[YahooActionCandidate, ...]:
    """Validate Yahoo candidates without requiring absent IDs or payment dates."""
    start = _as_date("start_date", start_date)
    end = _as_date("end_date", end_date)
    sessions = {_as_date("market session", value) for value in market_sessions}
    if start > end:
        raise ValueError("start_date must not be after end_date")
    if not sessions:
        raise CorporateActionValidationError("market_sessions must not be empty")

    seen: dict[tuple[Any, ...], YahooActionCandidate] = {}
    seen_action_keys: dict[tuple[Any, ...], YahooActionCandidate] = {}
    accepted: list[YahooActionCandidate] = []
    for candidate in candidates:
        if not isinstance(candidate, YahooActionCandidate):
            raise CorporateActionValidationError(
                "candidates must contain YahooActionCandidate records"
            )
        if candidate.symbol not in ASSET_UNIVERSE:
            raise CorporateActionValidationError(
                f"Unsupported Yahoo action symbol {candidate.symbol!r}"
            )
        if candidate.event_type not in {
            CorporateActionType.ETF_DISTRIBUTION,
            CorporateActionType.CASH_DIVIDEND,
            CorporateActionType.SPLIT,
        }:
            raise CorporateActionValidationError(
                f"Unsupported event type for {candidate.symbol}"
            )
        action_date = candidate.action_date
        if not start <= action_date <= end:
            raise CorporateActionValidationError(
                f"Yahoo event date {action_date} is outside the requested range"
            )
        if action_date not in sessions:
            raise CorporateActionValidationError(
                f"Yahoo event date {action_date} does not align with a market session"
            )
        if candidate.currency not in {None, "USD"}:
            raise CorporateActionValidationError(
                f"Unsupported Yahoo currency {candidate.currency!r}"
            )
        if candidate.source_id is not None and not candidate.source_id.strip():
            raise CorporateActionValidationError("source_id cannot be blank")
        if candidate.verification_status != "issuer_verification_required":
            raise CorporateActionValidationError(
                f"Unexpected verification status for {candidate.symbol}"
            )
        if candidate.event_type is CorporateActionType.SPLIT:
            _require_positive(candidate.split_ratio, "split_ratio")
            if candidate.distribution_per_share is not None:
                raise CorporateActionValidationError(
                    "Split candidate cannot contain a distribution amount"
                )
        else:
            _require_positive(
                candidate.distribution_per_share,
                "distribution_per_share",
            )
            if candidate.payment_date is not None and candidate.payment_date < action_date:
                raise CorporateActionValidationError(
                    "payment_date cannot precede the distribution ex-date"
                )
        key = _candidate_key(candidate)
        action_key = (
            candidate.symbol,
            candidate.event_type.value,
            candidate.action_date,
            candidate.source_field,
        )
        previous = seen_action_keys.get(action_key)
        if previous is not None and previous.raw_value != candidate.raw_value:
            raise CorporateActionValidationError(
                "Conflicting Yahoo action candidates: "
                f"{candidate.symbol} {candidate.action_date} "
                f"{candidate.source_field}"
            )
        if key in seen:
            raise CorporateActionValidationError(
                "Duplicate Yahoo action candidates require issuer review: "
                f"{candidate.symbol} {candidate.action_date} "
                f"{candidate.source_field}"
            )
        seen[key] = candidate
        seen_action_keys[action_key] = candidate
        accepted.append(candidate)
    return tuple(sorted(accepted, key=_candidate_sort_key))


def apply_issuer_verification(
    candidates: Iterable[YahooActionCandidate],
    verification_csv: str | Path,
    *,
    start_date: date | str,
    end_date: date | str,
    market_sessions: Iterable[date | str],
    amount_tolerance: float = 1e-8,
) -> tuple[CorporateActionEvent, ...]:
    """Build ledger events only from explicit issuer-sourced verification rows.

    Verification rows must provide a source identifier and source reference.
    Distribution rows must supply a verified amount and payment date matching
    the Yahoo-reported amount within tolerance. No identifiers or payment dates
    are synthesized by this workflow.
    """
    if not math.isfinite(float(amount_tolerance)) or amount_tolerance < 0:
        raise ValueError("amount_tolerance must be finite and non-negative")
    validated_candidates = validate_yahoo_candidates(
        candidates,
        start_date=start_date,
        end_date=end_date,
        market_sessions=market_sessions,
    )
    rows = _read_verification_csv(Path(verification_csv))
    by_key: dict[tuple[str, str, str, str, float], Mapping[str, str]] = {}
    for row in rows:
        key = _verification_key(row)
        if key in by_key:
            raise CorporateActionValidationError(
                f"Duplicate issuer-verification row for {key}"
            )
        by_key[key] = row

    expected_keys = {
        (
            candidate.symbol,
            candidate.event_type.value,
            candidate.action_date.isoformat(),
            candidate.source_field,
            candidate.raw_value,
        )
        for candidate in validated_candidates
    }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        extra = sorted(set(by_key) - expected_keys)
        raise CorporateActionValidationError(
            f"Issuer verification must cover every candidate exactly once; "
            f"missing={missing}, extra={extra}"
        )

    verified: list[CorporateActionEvent] = []
    for candidate in validated_candidates:
        key = (
            candidate.symbol,
            candidate.event_type.value,
            candidate.action_date.isoformat(),
            candidate.source_field,
            candidate.raw_value,
        )
        row = by_key[key]
        source_id = row["source_id"].strip()
        verification_source = row["verification_source"].strip()
        verification_reference = row["verification_reference"].strip()
        if not source_id:
            raise CorporateActionValidationError(
                f"Issuer source_id is required for {candidate.symbol} {candidate.action_date}"
            )
        if not verification_source or not verification_reference:
            raise CorporateActionValidationError(
                "Issuer verification source and reference are required"
            )
        currency = row["currency"].strip().upper()
        if currency != "USD" or candidate.currency != "USD":
            raise CorporateActionValidationError(
                f"USD currency must be confirmed for {candidate.symbol}"
            )

        if candidate.event_type is CorporateActionType.SPLIT:
            verified_ratio = _csv_positive(row["split_ratio"], "split_ratio")
            if not math.isclose(
                verified_ratio,
                float(candidate.split_ratio),
                rel_tol=0.0,
                abs_tol=amount_tolerance,
            ):
                raise CorporateActionValidationError(
                    f"Issuer split ratio does not match Yahoo candidate for "
                    f"{candidate.symbol} {candidate.action_date}"
                )
            verified.append(
                CorporateActionEvent(
                    symbol=candidate.symbol,
                    event_type=CorporateActionType.SPLIT,
                    source_id=source_id,
                    currency=currency,
                    effective_date=candidate.action_date,
                    split_ratio=verified_ratio,
                )
            )
            continue

        verified_amount = _csv_positive(
            row["distribution_per_share"],
            "distribution_per_share",
        )
        if not math.isclose(
            verified_amount,
            float(candidate.distribution_per_share),
            rel_tol=0.0,
            abs_tol=amount_tolerance,
        ):
            raise CorporateActionValidationError(
                f"Issuer distribution amount does not match Yahoo candidate for "
                f"{candidate.symbol} {candidate.action_date}"
            )
        payment_date = _as_date("payment_date", row["payment_date"])
        if payment_date < candidate.action_date:
            raise CorporateActionValidationError(
                f"Payment date precedes ex-date for {candidate.symbol}"
            )
        verified.append(
            CorporateActionEvent(
                symbol=candidate.symbol,
                event_type=candidate.event_type,
                source_id=source_id,
                currency=currency,
                ex_date=candidate.action_date,
                payment_date=payment_date,
                distribution_per_share=verified_amount,
            )
        )

    event_ids = [event.event_id for event in verified]
    if len(event_ids) != len(set(event_ids)):
        raise CorporateActionValidationError(
            "Issuer source identifiers must be unique within each symbol"
        )
    return tuple(sorted(verified, key=_event_sort_key))


def audit_yahoo_against_local_prices(
    batch: YahooAcquisitionBatch,
    local_price_data: Mapping[str, pd.DataFrame],
    *,
    market_sessions: Iterable[date | str],
    adjustment_jump_tolerance: float = 0.0025,
) -> dict[str, Any]:
    """Compare Yahoo action candidates, Yahoo prices, and local raw/adjusted closes.

    Diagnostics are not proof of an event or source completeness. They expose
    missing local sessions, large price-source differences, unexplained
    adjusted/raw factor changes, and event-day adjustment-factor comparisons.
    """
    if not math.isfinite(float(adjustment_jump_tolerance)) or adjustment_jump_tolerance <= 0:
        raise ValueError("adjustment_jump_tolerance must be finite and positive")
    sessions = sorted({_as_date("market session", value) for value in market_sessions})
    session_set = set(sessions)
    candidates_by_symbol: dict[str, list[YahooActionCandidate]] = {
        symbol: [] for symbol in batch.requested_symbols
    }
    for candidate in batch.candidates:
        candidates_by_symbol[candidate.symbol].append(candidate)

    symbol_audits: dict[str, Any] = {}
    event_diagnostics: list[dict[str, Any]] = []
    unexplained: dict[str, list[str]] = {}

    for symbol in batch.requested_symbols:
        if symbol not in local_price_data or symbol not in batch.price_frames:
            symbol_audits[symbol] = {
                "status": "missing_price_source",
                "local_sessions_missing_from_yahoo": [],
                "max_close_relative_difference": None,
                "max_adj_close_relative_difference": None,
            }
            unexplained[symbol] = []
            continue

        local = _normalize_price_frame(local_price_data[symbol], symbol)
        yahoo = batch.price_frames[symbol]
        expected = {
            session
            for session in session_set
            if batch.start_date <= session.isoformat() <= batch.end_date
        }
        yahoo_dates = set(yahoo.index)
        local_dates = set(local.index)
        missing_yahoo = sorted(expected - yahoo_dates)
        missing_local = sorted(expected - local_dates)
        common = sorted(yahoo_dates & local_dates & expected)

        close_diffs: list[float] = []
        adj_diffs: list[float] = []
        for day in common:
            for col, yahoo_col, diffs in (
                ("close", "Close", close_diffs),
                ("adj_close", "Adj Close", adj_diffs),
            ):
                local_value = float(local.loc[day, col])
                yahoo_value = float(yahoo.loc[day, yahoo_col])
                if local_value > 0.0 and yahoo_value > 0.0:
                    diffs.append(abs(yahoo_value / local_value - 1.0))

        symbol_events = candidates_by_symbol[symbol]
        event_day_set = {candidate.action_date for candidate in symbol_events}
        factor_jumps: list[tuple[date, float]] = []
        previous_date: Optional[date] = None
        previous_factor: Optional[float] = None
        for day in sorted(expected & local_dates):
            close = float(local.loc[day, "close"])
            adj_close = float(local.loc[day, "adj_close"])
            factor = adj_close / close
            if previous_factor is not None and previous_date is not None:
                change = factor / previous_factor - 1.0
                if abs(change) >= adjustment_jump_tolerance:
                    if day not in event_day_set:
                        factor_jumps.append((day, change))
            previous_date = day
            previous_factor = factor

        previous_dates = sorted(expected & local_dates)
        for candidate in symbol_events:
            action_day = candidate.action_date
            prior_dates = [day for day in previous_dates if day < action_day]
            prior_day = prior_dates[-1] if prior_dates else None
            if action_day not in local.index or prior_day is None:
                event_diagnostics.append(
                    {
                        "symbol": symbol,
                        "event_type": candidate.event_type.value,
                        "action_date": action_day.isoformat(),
                        "source_id": candidate.source_id,
                        "status": "missing_local_price_rows",
                    }
                )
                continue
            before = local.loc[prior_day]
            after = local.loc[action_day]
            raw_ratio = float(after["close"]) / float(before["close"])
            adj_ratio = float(after["adj_close"]) / float(before["adj_close"])
            factor_ratio = (
                float(after["adj_close"]) / float(after["close"])
            ) / (float(before["adj_close"]) / float(before["close"]))
            if candidate.event_type is CorporateActionType.SPLIT:
                expected_factor_ratio = None
                expected_unadjusted_raw_ratio = 1.0 / float(
                    candidate.split_ratio
                )
                if abs(raw_ratio / expected_unadjusted_raw_ratio - 1.0) <= 0.05:
                    split_basis = "raw_price_change_consistent_with_split"
                elif abs(raw_ratio - 1.0) <= 0.05:
                    split_basis = (
                        "no_split_price_change_detected; price series may be "
                        "split-adjusted or event requires issuer review"
                    )
                else:
                    split_basis = "inconclusive; issuer and price-source review required"
            else:
                expected_unadjusted_raw_ratio = None
                split_basis = None
                yield_value = float(candidate.distribution_per_share) / float(
                    before["close"]
                )
                expected_factor_ratio = (
                    1.0 / (1.0 - yield_value) if 0.0 <= yield_value < 1.0 else None
                )
            relative_error = (
                factor_ratio / expected_factor_ratio - 1.0
                if expected_factor_ratio is not None and expected_factor_ratio > 0
                else None
            )
            event_diagnostics.append(
                {
                    "symbol": symbol,
                    "event_type": candidate.event_type.value,
                    "source_field": candidate.source_field,
                    "action_date": action_day.isoformat(),
                    "source_id": candidate.source_id,
                    "prior_date": prior_day.isoformat(),
                    "prior_raw_close": float(before["close"]),
                    "event_raw_close": float(after["close"]),
                    "prior_adj_close": float(before["adj_close"]),
                    "event_adj_close": float(after["adj_close"]),
                    "raw_close_ratio": raw_ratio,
                    "adjusted_close_ratio": adj_ratio,
                    "observed_adjustment_factor_ratio": factor_ratio,
                    "expected_adjustment_factor_ratio": expected_factor_ratio,
                    "relative_factor_difference": relative_error,
                    "split_ratio": candidate.split_ratio,
                    "expected_unadjusted_raw_close_ratio": (
                        expected_unadjusted_raw_ratio
                    ),
                    "split_price_basis_assessment": split_basis,
                    "status": "review_required",
                }
            )

        symbol_audits[symbol] = {
            "status": "compared",
            "local_sessions_missing_from_yahoo": [
                day.isoformat() for day in missing_yahoo
            ],
            "yahoo_sessions_missing_from_local": [
                day.isoformat() for day in missing_local
            ],
            "common_price_sessions": len(common),
            "max_close_relative_difference": max(close_diffs, default=None),
            "max_adj_close_relative_difference": max(adj_diffs, default=None),
            "unexplained_adjustment_factor_jumps": [
                {
                    "date": day.isoformat(),
                    "factor_change": change,
                }
                for day, change in factor_jumps
            ],
        }
        unexplained[symbol] = [day.isoformat() for day, _ in factor_jumps]

    return {
        "interpretation": (
            "Diagnostic comparisons only; price movements and adjustment factors "
            "do not independently prove corporate-action completeness."
        ),
        "symbols": symbol_audits,
        "event_diagnostics": event_diagnostics,
        "possible_unexplained_adjustment_dates": unexplained,
    }


def export_yahoo_acquisition(
    batch: YahooAcquisitionBatch,
    output_dir: str | Path,
    *,
    market_sessions: Iterable[date | str],
    local_price_data: Mapping[str, pd.DataFrame],
    issuer_verification_csv: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Write Yahoo candidates, raw action rows, comparison audit, and manifest."""
    if not batch.complete:
        raise CorporateActionValidationError(
            "Refusing to export incomplete Yahoo symbol queries"
        )
    candidates = validate_yahoo_candidates(
        batch.candidates,
        start_date=batch.start_date,
        end_date=batch.end_date,
        market_sessions=market_sessions,
    )
    audit = audit_yahoo_against_local_prices(
        batch,
        local_price_data,
        market_sessions=market_sessions,
    )

    verified_events: tuple[CorporateActionEvent, ...] = ()
    if issuer_verification_csv is not None:
        verified_events = apply_issuer_verification(
            candidates,
            issuer_verification_csv,
            start_date=batch.start_date,
            end_date=batch.end_date,
            market_sessions=market_sessions,
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidates": output / "yahoo_action_candidates.csv",
        "raw": output / "yahoo_action_raw.jsonl",
        "verification_template": output / "issuer_verification_template.csv",
        "price_audit": output / "price_comparison.json",
        "manifest": output / "manifest.json",
    }
    _write_candidates(paths["candidates"], candidates)
    _write_raw(paths["raw"], batch.raw_records)
    _write_verification_template(paths["verification_template"], candidates)
    _write_json(paths["price_audit"], audit)
    if issuer_verification_csv is not None:
        verified_path = output / "corporate_actions_issuer_verified.csv"
        _write_verified_events(verified_path, verified_events)
        paths["verified_events"] = verified_path
        verification_audit_path = output / "issuer_verification_audit.csv"
        _copy_verification_csv(Path(issuer_verification_csv), verification_audit_path)
        paths["verification_audit"] = verification_audit_path

    outcomes_by_symbol = {item.symbol: item for item in batch.outcomes}
    per_symbol: dict[str, Any] = {}
    for symbol in batch.requested_symbols:
        outcome = outcomes_by_symbol[symbol]
        symbol_candidates = [
            item for item in candidates if item.symbol == symbol
        ]
        per_symbol[symbol] = {
            "dividend_records": sum(
                item.source_field == "Dividends" for item in symbol_candidates
            ),
            "capital_gain_distribution_records": sum(
                item.source_field == "Capital Gains" for item in symbol_candidates
            ),
            "split_records": sum(
                item.source_field == "Stock Splits" for item in symbol_candidates
            ),
            "event_date_start": min(
                (item.action_date.isoformat() for item in symbol_candidates),
                default=None,
            ),
            "event_date_end": max(
                (item.action_date.isoformat() for item in symbol_candidates),
                default=None,
            ),
            "missing_payment_date_count": sum(
                item.source_field in {"Dividends", "Capital Gains"}
                and item.payment_date is None
                for item in symbol_candidates
            ),
            "missing_source_identifier_count": sum(
                item.source_id is None for item in symbol_candidates
            ),
            "duplicate_candidate_count": outcome.duplicate_candidate_count,
            "price_history": {
                "rows": outcome.price_rows,
                "start": outcome.price_start,
                "end": outcome.price_end,
                "currency": outcome.currency,
                "status": outcome.status,
            },
            "price_audit": audit["symbols"].get(symbol),
        }

    artifact_names = (
        "candidates",
        "raw",
        "verification_template",
        "price_audit",
    )
    manifest = {
        "manifest_version": 1,
        "source": batch.provider,
        "library": "yfinance",
        "endpoint": "Yahoo Finance chart/history via yfinance Ticker.history",
        "pagination": "Not exposed by yfinance; one history query per symbol",
        "retrieved_at_utc": batch.retrieved_at,
        "requested_symbols": list(batch.requested_symbols),
        "requested_date_range": {
            "start_inclusive": batch.start_date,
            "end_inclusive": batch.end_date,
        },
        "query_complete": batch.complete,
        "per_symbol": per_symbol,
        "validation": {
            "candidate_count": len(candidates),
            "duplicate_candidates": sum(
                outcome.duplicate_candidate_count for outcome in batch.outcomes
            ),
            "invalid_records": 0,
            "failed_symbols": [
                outcome.symbol
                for outcome in batch.outcomes
                if outcome.status != "success"
            ],
            "issuer_verification_required": len(candidates) - len(verified_events),
            "ledger_ready_event_count": len(verified_events),
        },
        "coverage_status": (
            "Yahoo history retrieval completed; event completeness and "
            "issuer-level payment/amount verification remain unverified"
        ),
        "price_comparison": audit,
        "artifacts": {
            path.name: {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name in artifact_names
            for path in (paths[name],)
        },
        "redistribution": (
            "Local acquisition output only; review Yahoo Finance terms before "
            "retention, use, or redistribution."
        ),
    }
    for name in ("verified_events", "verification_audit"):
        if name in paths:
            path = paths[name]
            manifest["artifacts"][path.name] = {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
    _write_json(paths["manifest"], manifest)
    return manifest


def _extract_candidates(
    symbol: str,
    frame: pd.DataFrame,
    *,
    currency: Optional[str],
) -> tuple[list[YahooActionCandidate], list[Mapping[str, Any]]]:
    candidates: list[YahooActionCandidate] = []
    raw_records: list[Mapping[str, Any]] = []
    for timestamp, row in frame.iterrows():
        event_date = _timestamp_date(timestamp)
        for field, (event_type, subtype) in ACTION_COLUMNS.items():
            if field not in frame.columns:
                continue
            raw_value = _action_number(row[field], field, symbol, event_date)
            if raw_value == 0.0:
                continue
            if raw_value < 0.0:
                raise CorporateActionValidationError(
                    f"Yahoo returned a negative {field} value for {symbol} on {event_date}"
                )
            if event_type is CorporateActionType.SPLIT:
                missing_fields = ("source_id",)
                if currency is None:
                    missing_fields += ("currency",)
                candidate = YahooActionCandidate(
                    symbol=symbol,
                    event_type=event_type,
                    source_id=None,
                    currency=currency,
                    announcement_date=None,
                    ex_date=None,
                    payment_date=None,
                    effective_date=event_date,
                    distribution_per_share=None,
                    split_ratio=raw_value,
                    source_field=field,
                    raw_value=raw_value,
                    verification_status="issuer_verification_required",
                    missing_fields=missing_fields,
                )
            else:
                missing_fields = ("source_id", "payment_date")
                if currency is None:
                    missing_fields += ("currency",)
                candidate = YahooActionCandidate(
                    symbol=symbol,
                    event_type=event_type,
                    source_id=None,
                    currency=currency,
                    announcement_date=None,
                    ex_date=event_date,
                    payment_date=None,
                    effective_date=None,
                    distribution_per_share=raw_value,
                    split_ratio=None,
                    source_field=field,
                    raw_value=raw_value,
                    verification_status="issuer_verification_required",
                    missing_fields=missing_fields,
                )
            candidates.append(candidate)
            raw_records.append(
                {
                    "symbol": symbol,
                    "source": "Yahoo Finance via yfinance",
                    "source_field": field,
                    "subtype": subtype,
                    "event_date": event_date.isoformat(),
                    "source_id": None,
                    "currency": currency,
                    "raw_value": raw_value,
                    "verification_status": "issuer_verification_required",
                    "raw_yahoo_row": {
                        _stringify(key): _json_value(value)
                        for key, value in row.items()
                    },
                }
            )
    return candidates, raw_records


def _normalize_yahoo_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    normalized = frame.copy()
    dates = pd.DatetimeIndex(pd.to_datetime(normalized.index, errors="coerce"))
    if dates.isna().any():
        raise YahooResponseError(f"Yahoo returned invalid dates for {symbol}")
    normalized.index = [_timestamp_date(value) for value in dates]
    if normalized.index.has_duplicates:
        raise CorporateActionValidationError(
            f"Yahoo returned duplicate price dates for {symbol}"
        )
    normalized.index.name = "date"
    normalized = normalized.sort_index()
    normalized = normalized.loc[
        (normalized.index >= start) & (normalized.index <= end)
    ]
    for column in ("Close", "Adj Close", "Dividends", "Stock Splits"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    required_price_columns = {"Close", "Adj Close"}
    if not required_price_columns.issubset(normalized.columns):
        raise YahooResponseError(
            f"Yahoo history omitted close fields for {symbol}"
        )
    if normalized[["Close", "Adj Close"]].isna().any().any():
        raise YahooResponseError(
            f"Yahoo history contains invalid close values for {symbol}"
        )
    if (normalized[["Close", "Adj Close"]] <= 0.0).any().any():
        raise YahooResponseError(
            f"Yahoo history contains non-positive close values for {symbol}"
        )
    return normalized


def _normalize_price_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    local = frame.copy()
    if "date" in local.columns:
        local["date"] = pd.to_datetime(local["date"], errors="coerce")
        local = local.set_index("date")
    local.index = [_timestamp_date(value) for value in pd.to_datetime(local.index)]
    local.index.name = "date"
    if local.index.has_duplicates:
        raise CorporateActionValidationError(
            f"Local prices have duplicate dates for {symbol}"
        )
    aliases = {
        "close": ("close", "Close", "prccd"),
        "adj_close": ("adj_close", "Adj Close"),
    }
    result = pd.DataFrame(index=local.index)
    for target, options in aliases.items():
        source = next((name for name in options if name in local.columns), None)
        if source is None:
            raise CorporateActionValidationError(
                f"Local prices for {symbol} lack {target}"
            )
        result[target] = pd.to_numeric(local[source], errors="coerce")
    if not result.notna().all().all() or (result <= 0.0).any().any():
        raise CorporateActionValidationError(
            f"Local prices for {symbol} contain invalid close values"
        )
    return result.sort_index()


def _write_candidates(
    path: Path, candidates: Sequence[YahooActionCandidate]
) -> None:
    with _atomic_text_writer(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=YAHOO_CANDIDATE_COLUMNS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "symbol": candidate.symbol,
                    "event_type": candidate.event_type.value,
                    "source_id": candidate.source_id,
                    "currency": candidate.currency,
                    "announcement_date": _date_value(candidate.announcement_date),
                    "ex_date": _date_value(candidate.ex_date),
                    "payment_date": _date_value(candidate.payment_date),
                    "effective_date": _date_value(candidate.effective_date),
                    "distribution_per_share": candidate.distribution_per_share,
                    "split_ratio": candidate.split_ratio,
                    "source_field": candidate.source_field,
                    "verification_status": candidate.verification_status,
                    "missing_fields": "|".join(candidate.missing_fields),
                }
            )


def _write_verified_events(
    path: Path,
    events: Sequence[CorporateActionEvent],
) -> None:
    with _atomic_text_writer(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=VERIFIED_EVENT_COLUMNS)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "symbol": event.symbol,
                    "event_type": event.event_type.value,
                    "source_id": event.source_id,
                    "currency": event.currency,
                    "announcement_date": _date_value(event.announcement_date),
                    "ex_date": _date_value(event.ex_date),
                    "payment_date": _date_value(event.payment_date),
                    "effective_date": _date_value(event.effective_date),
                    "distribution_per_share": event.distribution_per_share,
                    "split_ratio": event.split_ratio,
                }
            )


def _copy_verification_csv(source: Path, destination: Path) -> None:
    rows = _read_verification_csv(source)
    with _atomic_text_writer(destination) as handle:
        writer = csv.DictWriter(handle, fieldnames=ISSUER_VERIFICATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_verification_template(
    path: Path, candidates: Sequence[YahooActionCandidate]
) -> None:
    with _atomic_text_writer(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=ISSUER_VERIFICATION_COLUMNS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "symbol": candidate.symbol,
                    "event_type": candidate.event_type.value,
                    "action_date": candidate.action_date.isoformat(),
                    "source_field": candidate.source_field,
                    "yahoo_reported_value": candidate.raw_value,
                    "source_id": "",
                    "currency": candidate.currency or "",
                    "distribution_per_share": "",
                    "payment_date": "",
                    "split_ratio": "",
                    "verification_source": "",
                    "verification_reference": "",
                }
            )


def _read_verification_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ISSUER_VERIFICATION_COLUMNS:
            raise CorporateActionValidationError(
                "Issuer verification CSV has an unexpected header"
            )
        return list(reader)


def _verification_key(
    row: Mapping[str, str],
) -> tuple[str, str, str, str, float]:
    symbol = row.get("symbol", "").strip()
    event_type = row.get("event_type", "").strip()
    action_date = row.get("action_date", "").strip()
    if symbol not in ASSET_UNIVERSE:
        raise CorporateActionValidationError(
            f"Unsupported issuer-verification symbol {symbol!r}"
        )
    if event_type not in {item.value for item in CorporateActionType}:
        raise CorporateActionValidationError(
            f"Unsupported issuer-verification event type {event_type!r}"
        )
    parsed_date = _as_date("action_date", action_date)
    source_field = row.get("source_field", "").strip()
    if source_field not in ACTION_COLUMNS:
        raise CorporateActionValidationError(
            f"Unsupported Yahoo source field {source_field!r}"
        )
    yahoo_value = _csv_positive(
        row.get("yahoo_reported_value", ""),
        "yahoo_reported_value",
    )
    expected_type, _ = ACTION_COLUMNS[source_field]
    if event_type != expected_type.value:
        raise CorporateActionValidationError(
            f"Source field {source_field!r} conflicts with event type"
        )
    return (
        symbol,
        event_type,
        parsed_date.isoformat(),
        source_field,
        yahoo_value,
    )


def _candidate_key(candidate: YahooActionCandidate) -> tuple[Any, ...]:
    return (
        candidate.symbol,
        candidate.event_type.value,
        candidate.action_date,
        candidate.source_field,
        candidate.raw_value,
    )


def _duplicate_count(candidates: Sequence[YahooActionCandidate]) -> int:
    keys = [_candidate_key(candidate) for candidate in candidates]
    return len(keys) - len(set(keys))


def _candidate_sort_key(
    candidate: YahooActionCandidate,
) -> tuple[str, str, str, str, float]:
    return (
        candidate.action_date.isoformat(),
        candidate.symbol,
        candidate.event_type.value,
        candidate.source_field,
        candidate.raw_value,
    )


def _event_sort_key(
    event: CorporateActionEvent,
) -> tuple[str, str, str, str]:
    return (
        event.action_date.isoformat(),
        event.symbol,
        event.event_type.value,
        event.source_id,
    )


def _action_number(value: Any, field: str, symbol: str, event_date: date) -> float:
    if isinstance(value, bool):
        raise CorporateActionValidationError(
            f"Invalid {field} value for {symbol} on {event_date}"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CorporateActionValidationError(
            f"Invalid {field} value for {symbol} on {event_date}"
        ) from exc
    if not math.isfinite(number):
        raise CorporateActionValidationError(
            f"Non-finite {field} value for {symbol} on {event_date}"
        )
    return number


def _require_positive(value: Optional[float], field: str) -> None:
    if value is None or isinstance(value, bool):
        raise CorporateActionValidationError(
            f"{field} must be a positive finite number"
        )
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise CorporateActionValidationError(
            f"{field} must be a positive finite number"
        )


def _csv_positive(value: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CorporateActionValidationError(
            f"Issuer {field} must be a positive finite number"
        ) from exc
    _require_positive(result, field)
    return result


def _as_date(field: str, value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise CorporateActionValidationError(
                f"{field} must be an ISO YYYY-MM-DD date"
            ) from exc
        if parsed.isoformat() != value:
            raise CorporateActionValidationError(
                f"{field} must use YYYY-MM-DD"
            )
        return parsed
    raise CorporateActionValidationError(
        f"{field} must be a date or ISO date string"
    )


def _timestamp_date(value: Any) -> date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise CorporateActionValidationError("Invalid Yahoo action date")
    if timestamp.tzinfo is not None:
        return timestamp.date()
    return timestamp.normalize().date()


def _stringify(value: Any) -> str:
    return str(value)


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return _timestamp_date(value).isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def _date_value(value: Optional[date]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _write_raw(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with _atomic_text_writer(path) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with _atomic_text_writer(path) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _atomic_text_writer(path: Path):
    class AtomicWriter:
        def __enter__(self):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            self.temporary_path = Path(temporary_name)
            self.handle = os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="",
            )
            return self.handle

        def __exit__(self, exception_type, *_unused):
            self.handle.close()
            if exception_type is None:
                self.temporary_path.replace(path)
            else:
                self.temporary_path.unlink(missing_ok=True)
            return False

    return AtomicWriter()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    if isinstance(symbols, (str, bytes)) or not symbols:
        raise ValueError("symbols must be a non-empty sequence")
    requested = tuple(symbols)
    if any(not isinstance(symbol, str) or symbol not in ASSET_UNIVERSE for symbol in requested):
        raise ValueError("symbols contain an unsupported asset")
    if len(set(requested)) != len(requested):
        raise ValueError("symbols must not contain duplicates")
    return requested
