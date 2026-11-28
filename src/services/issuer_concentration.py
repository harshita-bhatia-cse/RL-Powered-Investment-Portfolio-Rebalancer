"""Verified local reference data and issuer concentration assessment."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any


ISSUER_CONCENTRATION_POLICY_VERSION = "issuer-concentration-v1"
ISSUER_REFERENCE_PATH = Path(__file__).with_name("issuer_reference_v1.json")
_PERCENT = Decimal("100")
_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9&.-]{0,31}\Z")
_VERSION = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")
_RECORD_FIELDS = {
    "exchange",
    "symbol",
    "instrument_identifier",
    "instrument_identifier_type",
    "issuer_id",
    "issuer_id_type",
    "effective_from",
    "effective_to",
    "source",
    "source_reference",
    "verified_on",
    "reference_data_version",
}
_INSTRUMENT_ID_TYPES = {"ISIN", "OTHER_VERIFIED"}
_ISSUER_ID_TYPES = {"CIN", "LEI", "OTHER_VERIFIED"}


class IssuerReferenceDataError(ValueError):
    """Raised when local issuer reference data is malformed or ambiguous."""


@dataclass(frozen=True)
class _ReferenceRecord:
    exchange: str
    symbol: str
    instrument_identifier: str
    instrument_identifier_type: str
    issuer_id: str
    issuer_id_type: str
    effective_from: date
    effective_to: date | None
    source: str
    source_reference: str
    verified_on: date


@dataclass(frozen=True)
class _ReferenceData:
    version: str
    records: tuple[_ReferenceRecord, ...]


def _calendar_date(value: object, field: str, *, optional: bool = False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise IssuerReferenceDataError(f"{field} must be an ISO calendar date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise IssuerReferenceDataError(
            f"{field} must be an ISO calendar date"
        ) from error


def _reference_version(value: object) -> str:
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        raise IssuerReferenceDataError("reference_data_version is invalid")
    return value


def _expanded_identifier_digits(value: str) -> str:
    return "".join(
        str(int(character, 36)) if character.isalpha() else character
        for character in value
    )


def _has_luhn_checksum(value: str) -> bool:
    digits = _expanded_identifier_digits(value)
    total = 0
    for offset, character in enumerate(reversed(digits)):
        digit = int(character)
        if offset % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _has_mod97_checksum(value: str) -> bool:
    remainder = 0
    for character in value:
        expanded = str(int(character, 36)) if character.isalpha() else character
        for digit in expanded:
            remainder = (remainder * 10 + int(digit)) % 97
    return remainder == 1


def is_valid_instrument_identifier(identifier_type: object, value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if identifier_type == "ISIN":
        return bool(
            re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", value)
            and _has_luhn_checksum(value)
        )
    return (
        identifier_type == "OTHER_VERIFIED"
        and len(value) <= 64
        and bool(value.strip())
    )


def is_valid_issuer_identifier(identifier_type: object, value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if identifier_type == "CIN":
        return bool(
            re.fullmatch(r"[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}", value)
        )
    if identifier_type == "LEI":
        return bool(
            re.fullmatch(r"[A-Z0-9]{4}00[A-Z0-9]{12}[0-9]{2}", value)
            and _has_mod97_checksum(value)
        )
    return (
        identifier_type == "OTHER_VERIFIED"
        and len(value) <= 128
        and bool(value.strip())
    )


def _validate_reference_data(value: object) -> _ReferenceData:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "reference_data_version",
        "records",
    }:
        raise IssuerReferenceDataError("issuer reference document is invalid")
    if (
        isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != 1
    ):
        raise IssuerReferenceDataError("issuer reference schema is unsupported")
    version = _reference_version(value.get("reference_data_version"))
    raw_records = value.get("records")
    if not isinstance(raw_records, list):
        raise IssuerReferenceDataError("issuer reference records must be a list")

    records = []
    seen_records = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping) or set(raw) != _RECORD_FIELDS:
            raise IssuerReferenceDataError("issuer reference record is invalid")

        exchange = raw.get("exchange")
        symbol = raw.get("symbol")
        instrument_type = raw.get("instrument_identifier_type")
        instrument_id = raw.get("instrument_identifier")
        issuer_type = raw.get("issuer_id_type")
        issuer_id = raw.get("issuer_id")
        source = raw.get("source")
        source_reference = raw.get("source_reference")
        if (
            not isinstance(exchange, str)
            or exchange not in {"NSE", "BSE"}
            or not isinstance(symbol, str)
            or not _SYMBOL.fullmatch(symbol)
            or not isinstance(instrument_type, str)
            or instrument_type not in _INSTRUMENT_ID_TYPES
            or not is_valid_instrument_identifier(instrument_type, instrument_id)
            or not isinstance(issuer_type, str)
            or issuer_type not in _ISSUER_ID_TYPES
            or not is_valid_issuer_identifier(issuer_type, issuer_id)
            or not isinstance(source, str)
            or not source.strip()
            or len(source) > 512
            or not isinstance(source_reference, str)
            or not source_reference.strip()
            or len(source_reference) > 512
            or raw.get("reference_data_version") != version
        ):
            raise IssuerReferenceDataError("issuer reference record is invalid")

        effective_from = _calendar_date(raw.get("effective_from"), "effective_from")
        effective_to = _calendar_date(
            raw.get("effective_to"), "effective_to", optional=True
        )
        verified_on = _calendar_date(raw.get("verified_on"), "verified_on")
        if effective_to is not None and effective_to < effective_from:
            raise IssuerReferenceDataError("issuer reference validity range is invalid")

        record = _ReferenceRecord(
            exchange=exchange,
            symbol=symbol,
            instrument_identifier=instrument_id,
            instrument_identifier_type=instrument_type,
            issuer_id=issuer_id,
            issuer_id_type=issuer_type,
            effective_from=effective_from,
            effective_to=effective_to,
            source=source.strip(),
            source_reference=source_reference.strip(),
            verified_on=verified_on,
        )
        identity = (
            record.exchange,
            record.symbol,
            record.instrument_identifier,
            record.issuer_id_type,
            record.issuer_id,
            record.effective_from,
            record.effective_to,
            record.source,
            record.source_reference,
            record.verified_on,
        )
        if identity in seen_records:
            raise IssuerReferenceDataError("duplicate issuer reference record")
        seen_records.add(identity)
        records.append(record)

    by_listing: dict[tuple[str, str], list[_ReferenceRecord]] = defaultdict(list)
    by_instrument: dict[tuple[str, str], list[_ReferenceRecord]] = defaultdict(list)
    for record in records:
        by_listing[(record.exchange, record.symbol)].append(record)
        by_instrument[
            (record.instrument_identifier_type, record.instrument_identifier)
        ].append(record)

    def overlaps(left: _ReferenceRecord, right: _ReferenceRecord) -> bool:
        return (
            (left.effective_to is None or right.effective_from <= left.effective_to)
            and (right.effective_to is None or left.effective_from <= right.effective_to)
        )

    for listing_records in by_listing.values():
        for index, current in enumerate(listing_records):
            if any(overlaps(current, previous) for previous in listing_records[:index]):
                raise IssuerReferenceDataError(
                    "issuer reference listing validity ranges overlap"
                )

    for instrument_records in by_instrument.values():
        for index, current in enumerate(instrument_records):
            for previous in instrument_records[:index]:
                previous_issuer = (previous.issuer_id_type, previous.issuer_id)
                current_issuer = (current.issuer_id_type, current.issuer_id)
                if overlaps(previous, current) and previous_issuer != current_issuer:
                    raise IssuerReferenceDataError(
                        "instrument identifier has conflicting issuer mappings"
                    )

    records.sort(
        key=lambda row: (
            row.exchange,
            row.symbol,
            row.effective_from,
            row.instrument_identifier_type,
            row.instrument_identifier,
        )
    )
    return _ReferenceData(version=version, records=tuple(records))


def load_issuer_reference_data(
    path: Path = ISSUER_REFERENCE_PATH,
) -> dict[str, Any]:
    """Load the versioned local file; this performs no network or DB access."""
    try:
        contents = path.read_text(encoding="utf-8")
        value = json.loads(contents)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IssuerReferenceDataError(
            "local issuer reference data is unavailable"
        ) from error
    _validate_reference_data(value)
    return value


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{field} must be an exact decimal value")
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


def _decimal_string(value: Decimal) -> str:
    return format(value, "f")


def _version_from_untrusted(value: object) -> str:
    if isinstance(value, Mapping):
        try:
            return _reference_version(value.get("reference_data_version"))
        except IssuerReferenceDataError:
            pass
    return "unavailable"


def _not_evaluated(
    *,
    session_date: date,
    evaluated_at: datetime,
    reference_data_version: str,
    reason_codes: Sequence[str],
    unresolved_listings: Sequence[dict[str, str]] = (),
) -> dict[str, Any]:
    return {
        "assessment_status": "NOT_EVALUATED",
        "policy_version": ISSUER_CONCENTRATION_POLICY_VERSION,
        "reference_data_version": reference_data_version,
        "evaluated_at": evaluated_at.isoformat(),
        "session_date": session_date.isoformat(),
        "portfolio_nav": None,
        "total_equity_market_value": None,
        "listing_exposures": [],
        "issuer_exposures": [],
        "alerts": [],
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "unresolved_listings": list(unresolved_listings),
    }


def assess_portfolio_issuer_concentration(
    *,
    session_date: date,
    valuation_complete: bool,
    portfolio_nav: Decimal | str | int | None,
    cash_balance: Decimal | str | int,
    holdings: Sequence[Mapping[str, Any]],
    reference_data: Mapping[str, Any] | None,
    ledger_reconciliation_available: bool = True,
    ledger_reconciled: bool = True,
    evaluated_at: datetime | None = None,
) -> dict[str, Any]:
    """Calculate issuer weights from the complete monitoring valuation snapshot."""
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

    normalized_holdings = []
    seen_listings = set()
    for row in holdings:
        if not isinstance(row, Mapping):
            raise ValueError("each holding must be a mapping")
        exchange = row.get("exchange")
        symbol = row.get("symbol")
        if (
            not isinstance(exchange, str)
            or exchange not in {"NSE", "BSE"}
            or not isinstance(symbol, str)
            or not _SYMBOL.fullmatch(symbol)
        ):
            raise ValueError("each holding must have a valid NSE/BSE listing")
        listing = (exchange, symbol)
        if listing in seen_listings:
            raise ValueError(f"duplicate holding listing {exchange}:{symbol}")
        seen_listings.add(listing)
        quantity = _decimal(row.get("quantity"), "holding quantity")
        if quantity <= 0:
            raise ValueError("holding quantity must be positive")

        if row.get("valuation_status") == "UNAVAILABLE":
            if row.get("market_value") is not None:
                raise ValueError("an unavailable holding cannot have market_value")
            if valuation_complete:
                raise ValueError("complete valuation contains an unavailable holding")
            continue
        if row.get("valuation_status") != "VALUED":
            raise ValueError("holding valuation_status must be VALUED or UNAVAILABLE")
        market_date = row.get("market_date")
        if isinstance(market_date, str):
            market_date = _calendar_date(market_date, "holding market_date")
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
                "market_value": market_value,
            }
        )

    if not valuation_complete:
        return _not_evaluated(
            session_date=session_date,
            evaluated_at=assessment_time,
            reference_data_version=_version_from_untrusted(reference_data),
            reason_codes=["valuation_incomplete"],
        )
    if nav is None:
        return _not_evaluated(
            session_date=session_date,
            evaluated_at=assessment_time,
            reference_data_version=_version_from_untrusted(reference_data),
            reason_codes=["portfolio_nav_unavailable"],
        )
    if nav <= 0:
        return _not_evaluated(
            session_date=session_date,
            evaluated_at=assessment_time,
            reference_data_version=_version_from_untrusted(reference_data),
            reason_codes=["portfolio_nav_not_positive"],
        )
    if cash + sum(
        (row["market_value"] for row in normalized_holdings), Decimal("0")
    ) != nav:
        return _not_evaluated(
            session_date=session_date,
            evaluated_at=assessment_time,
            reference_data_version=_version_from_untrusted(reference_data),
            reason_codes=["valuation_nav_mismatch"],
        )

    try:
        validated_reference = _validate_reference_data(reference_data)
    except IssuerReferenceDataError:
        return _not_evaluated(
            session_date=session_date,
            evaluated_at=assessment_time,
            reference_data_version=_version_from_untrusted(reference_data),
            reason_codes=["issuer_reference_invalid"],
        )

    if not ledger_reconciliation_available or not ledger_reconciled:
        reason = (
            "ledger_reconciliation_unavailable"
            if not ledger_reconciliation_available
            else "ledger_discrepancy"
        )
        return _not_evaluated(
            session_date=session_date,
            evaluated_at=assessment_time,
            reference_data_version=validated_reference.version,
            reason_codes=[reason],
        )

    active_by_listing: dict[tuple[str, str], _ReferenceRecord] = {}
    for record in validated_reference.records:
        if (
            record.effective_from <= session_date
            and (record.effective_to is None or session_date <= record.effective_to)
        ):
            active_by_listing[(record.exchange, record.symbol)] = record

    unresolved = []
    listing_records = []
    issuer_values: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    issuer_listings: dict[tuple[str, str], list[str]] = defaultdict(list)
    for holding in normalized_holdings:
        record = active_by_listing.get((holding["exchange"], holding["symbol"]))
        market_value = holding["market_value"]
        if record is None and market_value > 0:
            unresolved.append(
                {
                    "exchange": holding["exchange"],
                    "symbol": holding["symbol"],
                    "listing": holding["listing"],
                    "reason": "No verified issuer mapping applies to this session.",
                }
            )
        issuer_id = record.issuer_id if record is not None else None
        issuer_id_type = record.issuer_id_type if record is not None else None
        instrument_identifier = (
            record.instrument_identifier if record is not None else None
        )
        instrument_identifier_type = (
            record.instrument_identifier_type if record is not None else None
        )
        listing_records.append(
            {
                "exchange": holding["exchange"],
                "symbol": holding["symbol"],
                "listing": holding["listing"],
                "market_value": _decimal_string(market_value),
                "issuer_id": issuer_id,
                "issuer_id_type": issuer_id_type,
                "instrument_identifier": instrument_identifier,
                "instrument_identifier_type": instrument_identifier_type,
                "source": record.source if record is not None else None,
                "source_reference": (
                    record.source_reference if record is not None else None
                ),
                "verified_on": (
                    record.verified_on.isoformat() if record is not None else None
                ),
            }
        )
        if record is not None and market_value > 0:
            issuer_key = (record.issuer_id_type, record.issuer_id)
            issuer_values[issuer_key] += market_value
            issuer_listings[issuer_key].append(holding["listing"])

    if unresolved:
        return _not_evaluated(
            session_date=session_date,
            evaluated_at=assessment_time,
            reference_data_version=validated_reference.version,
            reason_codes=["issuer_mapping_unresolved"],
            unresolved_listings=unresolved,
        )

    listing_records.sort(
        key=lambda row: (
            -Decimal(row["market_value"]),
            row["exchange"],
            row["symbol"],
        )
    )
    with localcontext() as context:
        context.prec = 50
        issuer_exposures = []
        for (issuer_id_type, issuer_id), value in issuer_values.items():
            issuer_exposures.append(
                {
                    "issuer_id": issuer_id,
                    "issuer_id_type": issuer_id_type,
                    "market_value": _decimal_string(value),
                    "portfolio_weight_pct": _decimal_string(value / nav * _PERCENT),
                    "listings": sorted(issuer_listings[(issuer_id_type, issuer_id)]),
                }
            )
    issuer_exposures.sort(
        key=lambda row: (
            -Decimal(row["market_value"]),
            row["issuer_id_type"],
            row["issuer_id"],
        )
    )
    return {
        "assessment_status": "EVALUATED",
        "policy_version": ISSUER_CONCENTRATION_POLICY_VERSION,
        "reference_data_version": validated_reference.version,
        "evaluated_at": assessment_time.isoformat(),
        "session_date": session_date.isoformat(),
        "portfolio_nav": _decimal_string(nav),
        "total_equity_market_value": _decimal_string(
            sum(
                (row["market_value"] for row in normalized_holdings),
                Decimal("0"),
            )
        ),
        "listing_exposures": listing_records,
        "issuer_exposures": issuer_exposures,
        "alerts": [],
        "reason_codes": [],
        "unresolved_listings": [],
    }
