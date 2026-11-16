"""Provider-neutral acquisition and validation for historical corporate actions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence
from urllib.parse import parse_qs, urlparse

import requests

from src.strategies.allocation import CorporateActionEvent, CorporateActionType

ASSET_UNIVERSE = ("SPY", "QQQ", "IEF", "GLD", "XLU", "XLV")
MASSIVE_API_BASE = "https://api.massive.com"
DIVIDENDS_ENDPOINT = "/stocks/v1/dividends"
SPLITS_ENDPOINT = "/stocks/v1/splits"
EVENT_COLUMNS = (
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


class CorporateActionProviderError(RuntimeError):
    """Base error that never includes a credential-bearing request URL."""


class ProviderAccessError(CorporateActionProviderError):
    """The API rejected access, subscription, or ticker entitlement."""


class ProviderUnsupportedError(CorporateActionProviderError):
    """The provider does not expose the requested endpoint or resource."""


class ProviderRequestError(CorporateActionProviderError):
    """A network request failed after the configured retry policy."""


class ProviderResponseError(CorporateActionProviderError):
    """The provider returned an invalid response or unsupported endpoint."""


class CorporateActionValidationError(ValueError):
    """A retrieved event set failed normalization or completeness checks."""


class CorporateActionProvider(Protocol):
    """Interface implemented by historical corporate-action providers."""

    def acquire(
        self,
        symbols: Sequence[str],
        start_date: date | str,
        end_date: date | str,
    ) -> "AcquisitionBatch":
        """Retrieve dividend and split records for each requested symbol."""


@dataclass(frozen=True)
class QueryOutcome:
    symbol: str
    category: str
    endpoint: str
    status: str
    pages: int
    records_received: int
    error: Optional[str] = None


@dataclass(frozen=True)
class AcquisitionBatch:
    provider: str
    requested_symbols: tuple[str, ...]
    start_date: str
    end_date: str
    events: tuple[CorporateActionEvent, ...]
    raw_records: tuple[Mapping[str, Any], ...]
    outcomes: tuple[QueryOutcome, ...]
    exact_duplicate_count: int
    retrieved_at: str

    @property
    def complete(self) -> bool:
        expected = {
            (symbol, category)
            for symbol in self.requested_symbols
            for category in ("dividend", "split")
        }
        actual = {(outcome.symbol, outcome.category) for outcome in self.outcomes}
        return (
            len(self.outcomes) == len(expected)
            and actual == expected
            and all(outcome.status in {"success", "empty"} for outcome in self.outcomes)
        )


class MassiveCorporateActionProvider:
    """Client for Massive's documented historical dividend and split endpoints.

    The API key is supplied at construction and is never logged, returned in
    results, or written to generated files. All pages for a query are consumed
    before the query is considered successful.
    """

    def __init__(
        self,
        api_key: str,
        *,
        session: Optional[requests.Session] = None,
        max_attempts: int = 4,
        backoff_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("A non-empty Massive API key is required")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError("max_attempts must be a positive integer")
        if max_attempts < 1 or max_attempts > 10:
            raise ValueError("max_attempts must be between 1 and 10")
        if not math.isfinite(float(backoff_seconds)) or backoff_seconds < 0:
            raise ValueError("backoff_seconds must be finite and non-negative")
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

        self._api_key = api_key.strip()
        self._session = session or requests.Session()
        self._max_attempts = max_attempts
        self._backoff_seconds = min(float(backoff_seconds), 60.0)
        self._timeout_seconds = float(timeout_seconds)
        self._sleep = sleep

    def acquire(
        self,
        symbols: Sequence[str],
        start_date: date | str,
        end_date: date | str,
    ) -> AcquisitionBatch:
        """Query both event endpoints for every symbol, retaining per-query status.

        A successful empty result is represented by status ``empty``. API,
        access, pagination, or response failures are represented explicitly in
        outcomes and make ``batch.complete`` false; partial batches must not be
        exported as complete datasets.
        """
        start = _as_iso_date("start_date", start_date)
        end = _as_iso_date("end_date", end_date)
        if start > end:
            raise ValueError("start_date must not be after end_date")
        requested_symbols = _validate_symbols(symbols)

        events_by_id: dict[str, CorporateActionEvent] = {}
        raw_records: list[Mapping[str, Any]] = []
        outcomes: list[QueryOutcome] = []
        exact_duplicates = 0

        for symbol in requested_symbols:
            for category, endpoint in (
                ("dividend", DIVIDENDS_ENDPOINT),
                ("split", SPLITS_ENDPOINT),
            ):
                pages = 0
                records_received = 0
                date_field = (
                    "ex_dividend_date"
                    if category == "dividend"
                    else "execution_date"
                )
                params = {
                    "ticker": symbol,
                    f"{date_field}.gte": start,
                    f"{date_field}.lte": end,
                    "limit": 5000,
                    "sort": f"{date_field}.asc",
                }
                try:
                    records, pages = self._fetch_all_pages(endpoint, params)
                    records_received = len(records)
                    raw_records.extend(
                        {
                            "symbol": symbol,
                            "category": category,
                            "endpoint": endpoint,
                            "record": dict(record),
                        }
                        for record in records
                    )
                    normalized: list[CorporateActionEvent] = []
                    for record in records:
                        normalized.append(
                            _normalize_record(
                                symbol,
                                category,
                                record,
                                start_date=start,
                                end_date=end,
                            )
                        )

                    for event in normalized:
                        existing = events_by_id.get(event.event_id)
                        if existing is None:
                            events_by_id[event.event_id] = event
                        elif existing == event:
                            exact_duplicates += 1
                        else:
                            raise CorporateActionValidationError(
                                f"Conflicting duplicate event identity {event.event_id!r}"
                            )

                    outcomes.append(
                        QueryOutcome(
                            symbol=symbol,
                            category=category,
                            endpoint=endpoint,
                            status="success" if records else "empty",
                            pages=pages,
                            records_received=len(records),
                        )
                    )
                except CorporateActionValidationError as exc:
                    outcomes.append(
                        QueryOutcome(
                            symbol=symbol,
                            category=category,
                            endpoint=endpoint,
                            status="validation_failed",
                            pages=pages,
                            records_received=records_received,
                            error=str(exc),
                        )
                    )
                except CorporateActionProviderError as exc:
                    outcomes.append(
                        QueryOutcome(
                            symbol=symbol,
                            category=category,
                            endpoint=endpoint,
                            status=_failure_status(exc),
                            pages=pages,
                            records_received=records_received,
                            error=str(exc),
                        )
                    )

        ordered_events = tuple(
            sorted(events_by_id.values(), key=_event_sort_key)
        )
        return AcquisitionBatch(
            provider="Massive",
            requested_symbols=requested_symbols,
            start_date=start,
            end_date=end,
            events=ordered_events,
            raw_records=tuple(raw_records),
            outcomes=tuple(outcomes),
            exact_duplicate_count=exact_duplicates,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )

    def _fetch_all_pages(
        self, endpoint: str, initial_params: Mapping[str, Any]
    ) -> tuple[list[Mapping[str, Any]], int]:
        url = f"{MASSIVE_API_BASE}{endpoint}"
        params: Optional[Mapping[str, Any]] = {
            **initial_params,
            "apiKey": self._api_key,
        }
        records: list[Mapping[str, Any]] = []
        visited: set[str] = set()
        pages = 0

        while url:
            _validate_page_url(url, endpoint)
            if url in visited:
                raise ProviderResponseError("Provider pagination loop detected")
            visited.add(url)
            response = self._request(url, params)
            params = None
            payload = _decode_json(response)
            status = payload.get("status")
            if status != "OK":
                raise ProviderResponseError(
                    f"Massive returned a non-success status for {endpoint}"
                )
            page_records = payload.get("results")
            if not isinstance(page_records, list):
                raise ProviderResponseError(
                    f"Massive response omitted the results array for {endpoint}"
                )
            if any(not isinstance(record, Mapping) for record in page_records):
                raise ProviderResponseError(
                    f"Massive returned a malformed record for {endpoint}"
                )

            records.extend(page_records)
            pages += 1
            next_url = payload.get("next_url")
            if next_url is not None and (
                not isinstance(next_url, str) or not next_url.strip()
            ):
                raise ProviderResponseError("Massive returned an invalid next_url")
            if next_url:
                url = next_url
                params = (
                    None
                    if parse_qs(urlparse(url).query).get("apiKey")
                    else {"apiKey": self._api_key}
                )
            else:
                url = ""

        return records, pages

    def _request(
        self, url: str, params: Optional[Mapping[str, Any]]
    ) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(self._max_attempts):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._timeout_seconds,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 == self._max_attempts:
                    raise ProviderRequestError(
                        "Massive request failed after retry attempts"
                    ) from None
                self._sleep(min(60.0, self._backoff_seconds * (2**attempt)))
                continue

            if 300 <= response.status_code < 400:
                raise ProviderResponseError(
                    "Massive returned an unexpected redirect; refusing to follow it"
                )
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt + 1 == self._max_attempts:
                    if response.status_code == 429:
                        raise ProviderRequestError(
                            "Massive rate limit persisted after retry attempts"
                        )
                    raise ProviderRequestError(
                        f"Massive server error HTTP {response.status_code} "
                        "persisted after retry attempts"
                    )
                self._sleep(
                    _retry_delay(
                        response,
                        min(60.0, self._backoff_seconds * (2**attempt)),
                    )
                )
                continue

            if response.status_code in {401, 403}:
                raise ProviderAccessError(
                    f"Massive denied access (HTTP {response.status_code}); "
                    "check endpoint entitlement and ticker coverage"
                )
            if response.status_code == 404:
                raise ProviderUnsupportedError(
                    "Massive endpoint or requested resource is unsupported or not found"
                )
            if response.status_code >= 400:
                raise ProviderResponseError(
                    f"Massive rejected the request (HTTP {response.status_code})"
                )
            return response

        raise ProviderRequestError("Massive request failed") from last_error


def validate_events(
    events: Iterable[CorporateActionEvent],
    *,
    start_date: date | str,
    end_date: date | str,
    market_sessions: Iterable[date | str],
) -> tuple[CorporateActionEvent, ...]:
    """Validate normalized events and require action dates to be market sessions."""
    start = _as_iso_date("start_date", start_date)
    end = _as_iso_date("end_date", end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")
    sessions = {
        date.fromisoformat(_as_iso_date("market session", session))
        for session in market_sessions
    }
    if not sessions:
        raise CorporateActionValidationError("market_sessions must not be empty")

    validated: dict[str, CorporateActionEvent] = {}
    for event in events:
        if not isinstance(event, CorporateActionEvent):
            raise CorporateActionValidationError(
                "events must contain CorporateActionEvent records"
            )
        if event.symbol not in ASSET_UNIVERSE:
            raise CorporateActionValidationError(
                f"Unsupported symbol {event.symbol!r}"
            )
        if event.event_type not in {
            CorporateActionType.SPLIT,
            CorporateActionType.CASH_DIVIDEND,
            CorporateActionType.ETF_DISTRIBUTION,
        }:
            raise CorporateActionValidationError(
                f"Unsupported event type for {event.event_id!r}"
            )
        if event.currency != "USD":
            raise CorporateActionValidationError(
                f"Non-USD event {event.event_id!r} is unsupported"
            )
        action_date = event.action_date.isoformat()
        if not start <= action_date <= end:
            raise CorporateActionValidationError(
                f"Event {event.event_id!r} is outside the requested date range"
            )
        if event.event_type is CorporateActionType.SPLIT:
            if event.split_ratio is None or not math.isfinite(event.split_ratio) or event.split_ratio <= 0:
                raise CorporateActionValidationError(
                    f"Invalid split ratio for {event.event_id!r}"
                )
        else:
            if (
                event.distribution_per_share is None
                or not math.isfinite(event.distribution_per_share)
                or event.distribution_per_share <= 0
            ):
                raise CorporateActionValidationError(
                    f"Invalid distribution amount for {event.event_id!r}"
                )
            if event.payment_date is None:
                raise CorporateActionValidationError(
                    f"Missing payment date for {event.event_id!r}"
                )
        if event.action_date not in sessions:
            raise CorporateActionValidationError(
                f"Action date {action_date} for {event.event_id!r} "
                "does not align with a supplied market session"
            )

        previous = validated.get(event.event_id)
        if previous is None:
            validated[event.event_id] = event
        elif previous != event:
            raise CorporateActionValidationError(
                f"Conflicting duplicate event identity {event.event_id!r}"
            )

    return tuple(sorted(validated.values(), key=_event_sort_key))


def load_normalized_events(
    path: str | Path,
    *,
    start_date: date | str,
    end_date: date | str,
    market_sessions: Iterable[date | str],
) -> tuple[CorporateActionEvent, ...]:
    """Load the normalized CSV schema and validate it for environment use."""
    events: list[CorporateActionEvent] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EVENT_COLUMNS:
            raise CorporateActionValidationError(
                "Normalized corporate-action CSV has an unexpected header"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                event_type = CorporateActionType(row["event_type"])
                event = CorporateActionEvent(
                    symbol=row["symbol"],
                    event_type=event_type,
                    source_id=row["source_id"],
                    currency=row["currency"],
                    announcement_date=_csv_date(row["announcement_date"]),
                    ex_date=_csv_date(row["ex_date"]),
                    payment_date=_csv_date(row["payment_date"]),
                    effective_date=_csv_date(row["effective_date"]),
                    distribution_per_share=_csv_number(
                        row["distribution_per_share"]
                    ),
                    split_ratio=_csv_number(row["split_ratio"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise CorporateActionValidationError(
                    f"Invalid normalized event on CSV row {row_number}"
                ) from exc
            events.append(event)
    return validate_events(
        events,
        start_date=start_date,
        end_date=end_date,
        market_sessions=market_sessions,
    )


def export_acquisition(
    batch: AcquisitionBatch,
    output_dir: str | Path,
    *,
    market_sessions: Iterable[date | str],
) -> dict[str, Any]:
    """Validate and write normalized events, raw records, and a checksummed manifest.

    Incomplete API batches are never exported. The manifest distinguishes
    successful empty queries from failed/unsupported queries and explicitly
    does not certify provider-level historical completeness.
    """
    if not batch.complete:
        raise CorporateActionValidationError(
            "Refusing to export an incomplete acquisition with failed or unsupported queries"
        )
    events = validate_events(
        batch.events,
        start_date=batch.start_date,
        end_date=batch.end_date,
        market_sessions=market_sessions,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    events_path = output_path / "corporate_actions.csv"
    raw_path = output_path / "corporate_actions_raw.jsonl"
    manifest_path = output_path / "manifest.json"

    _write_events_csv(events_path, events)
    _write_raw_jsonl(raw_path, batch.raw_records)
    symbol_summaries = _symbol_summaries(batch, events)
    validation_summary = {
        "normalized_event_count": len(events),
        "exact_duplicate_count": batch.exact_duplicate_count,
        "conflicting_duplicates": 0,
        "invalid_records": 0,
        "date_range_violations": 0,
        "market_session_misalignment": 0,
    }
    manifest = {
        "manifest_version": 1,
        "provider": batch.provider,
        "base_url": MASSIVE_API_BASE,
        "endpoints": {
            "dividends": DIVIDENDS_ENDPOINT,
            "splits": SPLITS_ENDPOINT,
        },
        "normalization": {
            "dividend_amount_field": "cash_amount",
            "split_ratio_formula": "split_to / split_from",
            "split_currency": "USD account currency (not reported by split endpoint)",
        },
        "requested_symbols": list(batch.requested_symbols),
        "requested_date_range": {
            "start_inclusive": batch.start_date,
            "end_inclusive": batch.end_date,
        },
        "retrieved_at_utc": batch.retrieved_at,
        "query_complete": batch.complete,
        "pagination": {
            "all_pages_consumed": all(
                outcome.status in {"success", "empty"} for outcome in batch.outcomes
            ),
            "queries": [asdict(outcome) for outcome in batch.outcomes],
        },
        "coverage_status": (
            "queries_complete; provider event coverage not independently verified"
        ),
        "per_symbol": symbol_summaries,
        "validation": validation_summary,
        "artifacts": {
            events_path.name: {
                "sha256": _sha256_file(events_path),
                "bytes": events_path.stat().st_size,
            },
            raw_path.name: {
                "sha256": _sha256_file(raw_path),
                "bytes": raw_path.stat().st_size,
            },
        },
        "credential_policy": "API credentials are excluded from artifacts and logs",
    }
    _write_json(manifest_path, manifest)
    return manifest


def _normalize_record(
    symbol: str,
    category: str,
    record: Mapping[str, Any],
    *,
    start_date: str,
    end_date: str,
) -> CorporateActionEvent:
    if symbol not in ASSET_UNIVERSE:
        raise CorporateActionValidationError(f"Unsupported symbol {symbol!r}")
    returned_symbol = record.get("ticker")
    if returned_symbol is not None and returned_symbol != symbol:
        raise CorporateActionValidationError(
            f"Provider returned an unexpected ticker for requested symbol {symbol}"
        )
    source_id = record.get("id")
    if not isinstance(source_id, (str, int)) or not str(source_id).strip():
        raise CorporateActionValidationError(
            f"Missing stable event ID for {symbol} {category} record"
        )
    source_id = str(source_id)

    if category == "dividend":
        ex_date = _provider_date(record, "ex_dividend_date", required=True)
        payment_date = _provider_date(record, "pay_date", required=True)
        amount = _positive_provider_number(record.get("cash_amount"), "cash_amount")
        currency = record.get("currency")
        if not isinstance(currency, str) or currency.strip().upper() != "USD":
            raise CorporateActionValidationError(
                f"Dividend {symbol}:{source_id} has unsupported currency"
            )
        if payment_date < ex_date:
            raise CorporateActionValidationError(
                f"Payment date precedes ex-date for {symbol}:{source_id}"
            )
        event_type = CorporateActionType.ETF_DISTRIBUTION
        if not start_date <= ex_date.isoformat() <= end_date:
            raise CorporateActionValidationError(
                f"Dividend {symbol}:{source_id} is outside the requested date range"
            )
        return CorporateActionEvent(
            symbol=symbol,
            event_type=event_type,
            source_id=source_id,
            currency="USD",
            announcement_date=_provider_date(
                record, "declaration_date", required=False
            ),
            ex_date=ex_date,
            payment_date=payment_date,
            distribution_per_share=amount,
        )

    if category == "split":
        effective_date = _provider_date(record, "execution_date", required=True)
        split_from = _positive_provider_number(record.get("split_from"), "split_from")
        split_to = _positive_provider_number(record.get("split_to"), "split_to")
        ratio = split_to / split_from
        if not math.isfinite(ratio) or ratio <= 0:
            raise CorporateActionValidationError(
                f"Invalid split ratio for {symbol}:{source_id}"
            )
        if not start_date <= effective_date.isoformat() <= end_date:
            raise CorporateActionValidationError(
                f"Split {symbol}:{source_id} is outside the requested date range"
            )
        return CorporateActionEvent(
            symbol=symbol,
            event_type=CorporateActionType.SPLIT,
            source_id=source_id,
            currency="USD",
            announcement_date=_provider_date(
                record, "declaration_date", required=False
            ),
            effective_date=effective_date,
            split_ratio=ratio,
        )

    raise CorporateActionValidationError(
        f"Unsupported provider event category {category!r}"
    )


def _provider_date(
    record: Mapping[str, Any], field: str, *, required: bool
) -> Optional[date]:
    value = record.get(field)
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise CorporateActionValidationError(
            f"Missing or invalid provider date field {field!r}"
        )
    return _parse_date(value, field)


def _csv_date(value: Optional[str]) -> Optional[date]:
    if value in {None, ""}:
        return None
    return _parse_date(value, "CSV date")


def _csv_number(value: Optional[str]) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise CorporateActionValidationError(
            "CSV amount and ratio fields must be numeric"
        ) from exc


def _parse_date(value: str, field: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise CorporateActionValidationError(
            f"Invalid ISO date in provider field {field!r}"
        ) from exc
    if parsed.isoformat() != value:
        raise CorporateActionValidationError(
            f"Provider field {field!r} must use YYYY-MM-DD"
        )
    return parsed


def _as_iso_date(field: str, value: date | str) -> str:
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    elif isinstance(value, str):
        parsed = _parse_date(value, field)
    else:
        raise ValueError(f"{field} must be a date or ISO date string")
    return parsed.isoformat()


def _positive_provider_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise CorporateActionValidationError(
            f"Provider field {field!r} must be a positive finite number"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CorporateActionValidationError(
            f"Provider field {field!r} must be a positive finite number"
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise CorporateActionValidationError(
            f"Provider field {field!r} must be a positive finite number"
        )
    return number


def _validate_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    if isinstance(symbols, (str, bytes)) or not symbols:
        raise ValueError("symbols must be a non-empty sequence")
    requested = tuple(symbols)
    if any(not isinstance(symbol, str) or not symbol.strip() for symbol in requested):
        raise ValueError("symbols must contain non-empty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("symbols must not contain duplicates")
    unsupported = sorted(set(requested) - set(ASSET_UNIVERSE))
    if unsupported:
        raise ValueError("Unsupported symbols: " + ", ".join(unsupported))
    return requested


def _validate_page_url(url: str, endpoint: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.massive.com"
        or parsed.path != endpoint
    ):
        raise ProviderResponseError(
            "Massive returned an unexpected pagination URL; refusing to follow it"
        )


def _decode_json(response: requests.Response) -> Mapping[str, Any]:
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise ProviderResponseError("Massive returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ProviderResponseError("Massive returned a non-object JSON response")
    return payload


def _retry_delay(response: requests.Response, fallback: float) -> float:
    header = response.headers.get("Retry-After")
    if header is None:
        return fallback
    try:
        value = float(header)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(header)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            value = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return fallback
    return max(0.0, min(value, 60.0))


def _failure_status(error: CorporateActionProviderError) -> str:
    if isinstance(error, ProviderAccessError):
        return "access_denied"
    if isinstance(error, ProviderUnsupportedError):
        return "unsupported_or_not_found"
    if isinstance(error, ProviderRequestError):
        return "request_failed"
    return "invalid_response"


def _event_sort_key(event: CorporateActionEvent) -> tuple[str, str, str, str]:
    return (
        event.action_date.isoformat(),
        event.symbol,
        event.event_type.value,
        event.source_id,
    )


def _symbol_summaries(
    batch: AcquisitionBatch,
    events: Sequence[CorporateActionEvent],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for symbol in batch.requested_symbols:
        symbol_events = [event for event in events if event.symbol == symbol]
        symbol_outcomes = [
            outcome for outcome in batch.outcomes if outcome.symbol == symbol
        ]
        summaries[symbol] = {
            "dividend_records": sum(
                event.event_type is CorporateActionType.ETF_DISTRIBUTION
                for event in symbol_events
            ),
            "split_records": sum(
                event.event_type is CorporateActionType.SPLIT
                for event in symbol_events
            ),
            "event_date_start": min(
                (event.action_date.isoformat() for event in symbol_events),
                default=None,
            ),
            "event_date_end": max(
                (event.action_date.isoformat() for event in symbol_events),
                default=None,
            ),
            "query_status": {
                outcome.category: outcome.status for outcome in symbol_outcomes
            },
        }
    return summaries


def _write_events_csv(
    path: Path, events: Sequence[CorporateActionEvent]
) -> None:
    with _atomic_text_writer(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
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


def _write_raw_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with _atomic_text_writer(path) as handle:
        for record in records:
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
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
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            self.temporary_path = Path(temporary_name)
            self.handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
            return self.handle

        def __exit__(self, exception_type, exception, traceback):
            self.handle.close()
            if exception_type is None:
                self.temporary_path.replace(path)
            else:
                self.temporary_path.unlink(missing_ok=True)
            return False

    return AtomicWriter()


def _date_value(value: Optional[date]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
