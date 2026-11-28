"""Versioned project taxonomy and issuer-level sector concentration assessment."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from src.services.issuer_concentration import (
    _decimal,
    _money,
    is_valid_issuer_identifier,
)


SECTOR_CONCENTRATION_POLICY_VERSION = "sector-concentration-v1"
SECTOR_REFERENCE_PATH = Path(__file__).with_name("sector_reference_v1.json")
SECTOR_TAXONOMY_NAME = "project-india-equity-sectors"
SECTOR_TAXONOMY_VERSION = "1.0"
SECTOR_TAXONOMY = {
    "BANKING_FINANCE": "Banking and Financial Services",
    "CONSUMER": "Consumer Products and Services",
    "ENERGY_UTILITIES": "Energy and Utilities",
    "HEALTHCARE": "Healthcare and Life Sciences",
    "INDUSTRIALS": "Industrial Products and Services",
    "MATERIALS": "Materials and Natural Resources",
    "TECHNOLOGY": "Technology and Digital Services",
    "TELECOM_MEDIA": "Telecommunications and Media",
    "TRANSPORT_LOGISTICS": "Transportation and Logistics",
    "DIVERSIFIED": "Diversified",
}
_VERSION = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")
_SECTOR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_MAPPING_FIELDS = {
    "issuer_id_type",
    "issuer_id",
    "sector_code",
    "sector_name",
    "effective_from",
    "effective_to",
    "source",
    "source_reference",
    "source_published_on",
    "verified_on",
    "reference_data_version",
}
_URL = re.compile(r"https?://\S+\Z")


def _has_document_locator(value: str) -> bool:
    return any(_URL.fullmatch(part) for part in value.split()) and bool(
        re.search(r"\b(?:page|pages|p\.|section|sec\.)\b", value, re.IGNORECASE)
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _mapping_record_document(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "issuer_id_type": record["issuer_id_type"],
        "issuer_id": record["issuer_id"],
        "sector_code": record["sector_code"],
        "sector_name": record["sector_name"],
        "effective_from": record["effective_from"].isoformat(),
        "effective_to": (
            record["effective_to"].isoformat()
            if record["effective_to"] is not None
            else None
        ),
        "source": record["source"],
        "source_reference": record["source_reference"],
        "source_published_on": record["source_published_on"].isoformat(),
        "verified_on": record["verified_on"].isoformat(),
        "reference_data_version": record["reference_data_version"],
    }


def _validated_reference_snapshot(
    reference_data: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    validated = _validate_document(reference_data)
    records = [
        _mapping_record_document(record) for record in validated["records"]
    ]
    records.sort(key=_canonical_json)
    snapshot = {
        "schema_version": 1,
        "reference_data_version": validated["reference_data_version"],
        "taxonomy": {
            "name": validated["taxonomy_name"],
            "version": validated["taxonomy_version"],
            "sectors": [
                {"code": code, "name": name}
                for code, name in sorted(SECTOR_TAXONOMY.items())
            ],
        },
        "records": records,
    }
    fingerprint = hashlib.sha256(
        _canonical_json(snapshot).encode("utf-8")
    ).hexdigest()
    return snapshot, fingerprint


def _sector_date(value: object, field: str, *, optional: bool = False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise SectorReferenceDataError(f"{field} must be an ISO calendar date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise SectorReferenceDataError(
            f"{field} must be an ISO calendar date"
        ) from error
    if parsed.isoformat() != value:
        raise SectorReferenceDataError(f"{field} must be an ISO calendar date")
    return parsed


class SectorReferenceDataError(ValueError):
    """Raised when local sector reference data is malformed or ambiguous."""


def _validate_document(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "reference_data_version",
        "taxonomy",
        "records",
    }:
        raise SectorReferenceDataError("sector reference document is invalid")
    schema_version = value.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise SectorReferenceDataError("sector reference schema is unsupported")
    reference_version = value.get("reference_data_version")
    if not isinstance(reference_version, str) or not _VERSION.fullmatch(
        reference_version
    ):
        raise SectorReferenceDataError("sector reference version is invalid")

    taxonomy = value.get("taxonomy")
    if not isinstance(taxonomy, Mapping) or set(taxonomy) != {
        "name",
        "version",
        "sectors",
    }:
        raise SectorReferenceDataError("sector taxonomy is invalid")
    if (
        taxonomy.get("name") != SECTOR_TAXONOMY_NAME
        or taxonomy.get("version") != SECTOR_TAXONOMY_VERSION
    ):
        raise SectorReferenceDataError("sector taxonomy version is unsupported")
    raw_sectors = taxonomy.get("sectors")
    if not isinstance(raw_sectors, list):
        raise SectorReferenceDataError("taxonomy sectors must be a list")
    declared_sectors: dict[str, str] = {}
    seen_names: set[str] = set()
    for sector in raw_sectors:
        if not isinstance(sector, Mapping) or set(sector) != {"code", "name"}:
            raise SectorReferenceDataError("taxonomy sector declaration is invalid")
        code = sector.get("code")
        name = sector.get("name")
        if (
            not isinstance(code, str)
            or not _SECTOR_CODE.fullmatch(code)
            or code not in SECTOR_TAXONOMY
            or not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or len(name) > 128
            or SECTOR_TAXONOMY.get(code) != name
            or code in declared_sectors
            or name in seen_names
        ):
            raise SectorReferenceDataError("taxonomy sector declaration is invalid")
        declared_sectors[code] = name
        seen_names.add(name)
    if declared_sectors != SECTOR_TAXONOMY:
        raise SectorReferenceDataError("project taxonomy declaration is incomplete")

    raw_records = value.get("records")
    if not isinstance(raw_records, list):
        raise SectorReferenceDataError("sector reference records must be a list")
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    by_issuer: dict[tuple[str, str], list[tuple[date, date | None]]] = defaultdict(list)
    for raw in raw_records:
        if not isinstance(raw, Mapping) or set(raw) != _MAPPING_FIELDS:
            raise SectorReferenceDataError("sector mapping record is invalid")
        issuer_type = raw.get("issuer_id_type")
        issuer_id = raw.get("issuer_id")
        code = raw.get("sector_code")
        name = raw.get("sector_name")
        source = raw.get("source")
        source_reference = raw.get("source_reference")
        if (
            not is_valid_issuer_identifier(issuer_type, issuer_id)
            or not isinstance(code, str)
            or code not in declared_sectors
            or not isinstance(name, str)
            or name != declared_sectors.get(code)
            or not isinstance(source, str)
            or not source.strip()
            or source != source.strip()
            or len(source) > 512
            or not isinstance(source_reference, str)
            or not source_reference.strip()
            or source_reference != source_reference.strip()
            or len(source_reference) > 1024
            or not _has_document_locator(source_reference)
            or raw.get("reference_data_version") != reference_version
        ):
            raise SectorReferenceDataError("sector mapping record is invalid")
        effective_from = _sector_date(raw.get("effective_from"), "effective_from")
        effective_to = _sector_date(
            raw.get("effective_to"), "effective_to", optional=True
        )
        source_published_on = _sector_date(
            raw.get("source_published_on"), "source_published_on"
        )
        verified_on = _sector_date(raw.get("verified_on"), "verified_on")
        if effective_to is not None and effective_to < effective_from:
            raise SectorReferenceDataError("sector mapping date range is invalid")
        identity = (
            issuer_type,
            issuer_id,
            code,
            effective_from,
            effective_to,
            source,
            source_reference,
            source_published_on,
            verified_on,
        )
        if identity in seen:
            raise SectorReferenceDataError("duplicate sector mapping")
        seen.add(identity)
        key = (issuer_type, issuer_id)
        for previous_from, previous_to in by_issuer[key]:
            if (
                (previous_to is None or effective_from <= previous_to)
                and (effective_to is None or previous_from <= effective_to)
            ):
                raise SectorReferenceDataError(
                    "issuer sector mapping validity ranges overlap"
                )
        by_issuer[key].append((effective_from, effective_to))
        records.append(
            {
                "issuer_id_type": issuer_type,
                "issuer_id": issuer_id,
                "sector_code": code,
                "sector_name": name,
                "effective_from": effective_from,
                "effective_to": effective_to,
                "source": source,
                "source_reference": source_reference,
                "source_published_on": source_published_on,
                "verified_on": verified_on,
                "reference_data_version": reference_version,
            }
        )
    return {
        "reference_data_version": reference_version,
        "taxonomy_name": SECTOR_TAXONOMY_NAME,
        "taxonomy_version": SECTOR_TAXONOMY_VERSION,
        "records": records,
    }


def load_sector_reference_data(
    path: Path = SECTOR_REFERENCE_PATH,
) -> dict[str, Any]:
    """Load local, versioned sector data without network or database access."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SectorReferenceDataError(
            "local sector reference data is unavailable"
        ) from error
    _validate_document(value)
    return value


def _version_from_untrusted(value: object) -> str:
    if isinstance(value, Mapping):
        candidate = value.get("reference_data_version")
        if isinstance(candidate, str) and _VERSION.fullmatch(candidate):
            return candidate
    return "unavailable"


def _not_evaluated(
    *,
    session_date: date,
    evaluated_at: datetime,
    reference_data_version: str,
    reason_codes: list[str],
    unresolved_issuers: list[dict[str, str]] | None = None,
    zero_value_listings: list[dict[str, Any]] | None = None,
    reference_snapshot: dict[str, Any] | None = None,
    reference_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "assessment_status": "NOT_EVALUATED",
        "policy_version": SECTOR_CONCENTRATION_POLICY_VERSION,
        "reference_data_version": reference_data_version,
        "reference_snapshot": reference_snapshot,
        "reference_snapshot_sha256": reference_snapshot_sha256,
        "taxonomy_name": SECTOR_TAXONOMY_NAME,
        "taxonomy_version": SECTOR_TAXONOMY_VERSION,
        "evaluated_at": evaluated_at.isoformat(),
        "session_date": session_date.isoformat(),
        "portfolio_nav": None,
        "total_equity_market_value": None,
        "sector_exposures": [],
        "largest_sector_weight_pct": None,
        "top_three_sector_weight_pct": None,
        "sector_hhi": None,
        "issuer_contributions": [],
        "zero_value_listings": zero_value_listings or [],
        "alerts": [],
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "unresolved_issuers": unresolved_issuers or [],
    }


def _issuer_snapshot(
    value: object,
    session_date: date,
) -> tuple[Decimal, Decimal, list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(value, Mapping) or value.get("assessment_status") != "EVALUATED":
        raise ValueError("issuer assessment is unavailable")
    if value.get("session_date") != session_date.isoformat():
        raise ValueError("issuer assessment session date does not match")
    nav = _money(value.get("portfolio_nav"), "portfolio_nav")
    equity = _money(value.get("total_equity_market_value"), "total_equity_market_value")
    if nav <= 0 or equity < 0 or equity > nav:
        raise ValueError("issuer assessment NAV or equity is invalid")
    raw_issuers = value.get("issuer_exposures")
    raw_listings = value.get("listing_exposures")
    if not isinstance(raw_issuers, list) or not isinstance(raw_listings, list):
        raise ValueError("issuer assessment exposures are invalid")
    issuers: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    total = Decimal("0")
    for item in raw_issuers:
        if not isinstance(item, Mapping):
            raise ValueError("issuer assessment entry is invalid")
        issuer_type, issuer_id = item.get("issuer_id_type"), item.get("issuer_id")
        if not is_valid_issuer_identifier(issuer_type, issuer_id):
            raise ValueError("issuer assessment identifier is invalid")
        key = (issuer_type, issuer_id)
        if key in seen:
            raise ValueError("issuer assessment contains duplicate issuers")
        seen.add(key)
        market_value = _money(item.get("market_value"), "issuer market value")
        weight = _decimal(item.get("portfolio_weight_pct"), "issuer weight")
        with localcontext() as context:
            context.prec = 50
            expected_weight = market_value / nav * Decimal("100")
        if market_value <= 0 or weight != expected_weight:
            raise ValueError("issuer assessment value or weight is inconsistent")
        listings = item.get("listings")
        if (
            not isinstance(listings, list)
            or not listings
            or any(not isinstance(listing, str) for listing in listings)
            or len(set(listings)) != len(listings)
        ):
            raise ValueError("issuer assessment listing trace is invalid")
        issuers.append(
            {
                "issuer_id_type": issuer_type,
                "issuer_id": issuer_id,
                "market_value": market_value,
            }
        )
        total += market_value
    if total != equity:
        raise ValueError("issuer assessment equity total is inconsistent")
    zero_listings = []
    for item in raw_listings:
        if not isinstance(item, Mapping):
            raise ValueError("issuer listing trace is invalid")
        market_value = _money(item.get("market_value"), "listing market value")
        if market_value != 0:
            continue
        exchange, symbol = item.get("exchange"), item.get("symbol")
        if (
            exchange not in {"NSE", "BSE"}
            or not isinstance(symbol, str)
            or not re.fullmatch(r"[A-Z0-9][A-Z0-9&.-]{0,31}", symbol)
        ):
            raise ValueError("zero-value listing identity is invalid")
        issuer_type = item.get("issuer_id_type")
        issuer_id = item.get("issuer_id")
        if (
            (issuer_id is None) != (issuer_type is None)
            or issuer_id is not None
            and not is_valid_issuer_identifier(issuer_type, issuer_id)
        ):
            raise ValueError("zero-value listing issuer is invalid")
        zero_listings.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "listing": f"{exchange}:{symbol}",
                "market_value": "0",
                "issuer_id_type": issuer_type,
                "issuer_id": issuer_id,
            }
        )
    return nav, equity, issuers, zero_listings


def assess_portfolio_sector_concentration(
    *,
    session_date: date,
    issuer_assessment: Mapping[str, Any] | None,
    reference_data: Mapping[str, Any] | None,
    evaluated_at: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate evaluated Phase 6.2B issuer values into sector exposures."""
    if not isinstance(session_date, date) or isinstance(session_date, datetime):
        raise ValueError("session_date must be a calendar date")
    assessment_time = evaluated_at or datetime.now(timezone.utc)
    if assessment_time.tzinfo is None:
        raise ValueError("evaluated_at must be timezone-aware")
    reference_version = _version_from_untrusted(reference_data)
    reference = None
    reference_snapshot = None
    reference_snapshot_sha256 = None
    if isinstance(reference_data, Mapping):
        try:
            reference = _validate_document(reference_data)
            reference_snapshot, reference_snapshot_sha256 = (
                _validated_reference_snapshot(reference_data)
            )
            reference_version = reference["reference_data_version"]
        except SectorReferenceDataError:
            reference = None
    try:
        nav, equity, issuers, zero_listings = _issuer_snapshot(
            issuer_assessment, session_date
        )
    except (ArithmeticError, TypeError, ValueError):
        return _not_evaluated(
            session_date=session_date,
            evaluated_at=assessment_time,
            reference_data_version=reference_version,
            reason_codes=["issuer_assessment_unavailable"],
            reference_snapshot=reference_snapshot,
            reference_snapshot_sha256=reference_snapshot_sha256,
        )
    if reference is None:
        return _not_evaluated(
            session_date=session_date,
            evaluated_at=assessment_time,
            reference_data_version=reference_version,
            reason_codes=["sector_reference_invalid"],
            zero_value_listings=zero_listings,
        )

    mappings: dict[tuple[str, str], dict[str, Any]] = {}
    for record in reference["records"]:
        if (
            record["effective_from"] <= session_date
            and (record["effective_to"] is None or session_date <= record["effective_to"])
            and record["source_published_on"] <= session_date
        ):
            mappings[(record["issuer_id_type"], record["issuer_id"])] = record

    unresolved: list[dict[str, str]] = []
    contributions: list[dict[str, Any]] = []
    values_by_sector: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    issuers_by_sector: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for issuer in issuers:
        key = (issuer["issuer_id_type"], issuer["issuer_id"])
        mapping = mappings.get(key)
        if mapping is None:
            unresolved.append(
                {
                    "issuer_id_type": key[0],
                    "issuer_id": key[1],
                    "reason": "No eligible verified sector mapping applies to this session.",
                }
            )
            continue
        amount = issuer["market_value"]
        sector_key = (mapping["sector_code"], mapping["sector_name"])
        values_by_sector[sector_key] += amount
        issuer_reference = {
            "issuer_id_type": key[0],
            "issuer_id": key[1],
        }
        issuers_by_sector[sector_key].append(issuer_reference)
        contributions.append(
            {
                **issuer_reference,
                "market_value": format(amount, "f"),
                "sector_code": mapping["sector_code"],
                "sector_name": mapping["sector_name"],
                "mapping": {
                    "effective_from": mapping["effective_from"].isoformat(),
                    "effective_to": (
                        mapping["effective_to"].isoformat()
                        if mapping["effective_to"] is not None
                        else None
                    ),
                    "source": mapping["source"],
                    "source_reference": mapping["source_reference"],
                    "source_published_on": mapping[
                        "source_published_on"
                    ].isoformat(),
                    "verified_on": mapping["verified_on"].isoformat(),
                    "reference_data_version": mapping["reference_data_version"],
                    "mapping_sha256": hashlib.sha256(
                        _canonical_json(
                            _mapping_record_document(mapping)
                        ).encode("utf-8")
                    ).hexdigest(),
                },
            }
        )
    if unresolved:
        return _not_evaluated(
            session_date=session_date,
            evaluated_at=assessment_time,
            reference_data_version=reference["reference_data_version"],
            reason_codes=["issuer_sector_mapping_unresolved"],
            unresolved_issuers=unresolved,
            zero_value_listings=zero_listings,
        )

    contributions.sort(key=lambda row: (row["issuer_id_type"], row["issuer_id"]))
    with localcontext() as context:
        context.prec = 50
        sector_rows = []
        for (code, name), amount in values_by_sector.items():
            sector_rows.append(
                {
                    "sector_code": code,
                    "sector_name": name,
                    "market_value": format(amount, "f"),
                    "portfolio_weight_pct": format(amount / nav * Decimal("100"), "f"),
                    "issuers": sorted(
                        issuers_by_sector[(code, name)],
                        key=lambda item: (item["issuer_id_type"], item["issuer_id"]),
                    ),
                }
            )
        sector_rows.sort(
            key=lambda row: (
                -Decimal(row["market_value"]),
                row["sector_code"],
                row["sector_name"],
            )
        )
        largest = (
            Decimal(sector_rows[0]["market_value"]) / nav * Decimal("100")
            if sector_rows
            else Decimal("0")
        )
        top_three = (
            sum(
                (Decimal(row["market_value"]) for row in sector_rows[:3]),
                Decimal("0"),
            )
            / nav
            * Decimal("100")
        )
        sector_hhi = (
            sum(
                (
                    (Decimal(row["market_value"]) / equity) ** 2
                    for row in sector_rows
                ),
                Decimal("0"),
            )
            if equity > 0
            else None
        )
    return {
        "assessment_status": "EVALUATED",
        "policy_version": SECTOR_CONCENTRATION_POLICY_VERSION,
        "reference_data_version": reference["reference_data_version"],
        "reference_snapshot": reference_snapshot,
        "reference_snapshot_sha256": reference_snapshot_sha256,
        "taxonomy_name": reference["taxonomy_name"],
        "taxonomy_version": reference["taxonomy_version"],
        "evaluated_at": assessment_time.isoformat(),
        "session_date": session_date.isoformat(),
        "portfolio_nav": format(nav, "f"),
        "total_equity_market_value": format(equity, "f"),
        "sector_exposures": sector_rows,
        "largest_sector_weight_pct": format(largest, "f"),
        "top_three_sector_weight_pct": format(top_three, "f"),
        "sector_hhi": format(sector_hhi, "f") if sector_hhi is not None else None,
        "issuer_contributions": contributions,
        "zero_value_listings": zero_listings,
        "alerts": [],
        "reason_codes": [],
        "unresolved_issuers": [],
    }


def safe_sector_concentration_assessment(
    stored: object,
    issuer_assessment: object,
) -> dict[str, Any] | None:
    """Validate persisted sector aggregates against the safe issuer snapshot."""
    try:
        return _validate_sector_assessment(stored, issuer_assessment)
    except (ArithmeticError, KeyError, TypeError, ValueError):
        return None


def _stored_decimal(value: object, field: str, *, paise: bool = False) -> Decimal:
    if not isinstance(value, str) or not value or len(value) > 80:
        raise ValueError(f"{field} is invalid")
    parsed = _decimal(value, field)
    if parsed < 0:
        raise ValueError(f"{field} cannot be negative")
    if paise:
        _money(parsed, field)
    return parsed


def _stored_date(value: object, field: str, *, optional: bool = False):
    if optional and value is None:
        return None
    parsed = _sector_date(value, field)
    return parsed


def _active_mapping_records(
    reference: Mapping[str, Any],
    session_date: date,
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (record["issuer_id_type"], record["issuer_id"]):
        _mapping_record_document(record)
        for record in reference["records"]
        if (
            record["effective_from"] <= session_date
            and (
                record["effective_to"] is None
                or session_date <= record["effective_to"]
            )
            and record["source_published_on"] <= session_date
        )
    }


def _unresolved_issuer_keys(
    issuer_assessment: Mapping[str, Any],
    reference: Mapping[str, Any],
    session_date: date,
) -> set[tuple[str, str]] | None:
    exposures = issuer_assessment.get("issuer_exposures")
    if not isinstance(exposures, list):
        return None
    expected_keys = set()
    for item in exposures:
        if not isinstance(item, Mapping):
            return None
        key = (item.get("issuer_id_type"), item.get("issuer_id"))
        if not is_valid_issuer_identifier(*key) or key in expected_keys:
            return None
        expected_keys.add(key)
    return expected_keys - set(_active_mapping_records(reference, session_date))


def _validate_sector_assessment(
    stored: object,
    issuer_assessment: object,
) -> dict[str, Any] | None:
    if (
        not isinstance(stored, Mapping)
        or "reference_snapshot" not in stored
        or "reference_snapshot_sha256" not in stored
    ):
        return None
    status_value = stored.get("assessment_status")
    reference_version = stored.get("reference_data_version")
    if (
        status_value not in {"EVALUATED", "NOT_EVALUATED"}
        or stored.get("policy_version") != SECTOR_CONCENTRATION_POLICY_VERSION
        or not isinstance(reference_version, str)
        or not _VERSION.fullmatch(reference_version)
        or stored.get("taxonomy_name") != SECTOR_TAXONOMY_NAME
        or stored.get("taxonomy_version") != SECTOR_TAXONOMY_VERSION
        or not isinstance(stored.get("session_date"), str)
        or not isinstance(stored.get("evaluated_at"), str)
        or len(stored["evaluated_at"]) > 40
    ):
        return None
    raw_snapshot = stored.get("reference_snapshot")
    snapshot_sha256 = stored.get("reference_snapshot_sha256")
    reference_snapshot = None
    validated_reference = None
    if raw_snapshot is None:
        if snapshot_sha256 is not None:
            return None
    else:
        if (
            not isinstance(snapshot_sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", snapshot_sha256)
            or not isinstance(raw_snapshot, Mapping)
        ):
            return None
        reference_snapshot, expected_fingerprint = _validated_reference_snapshot(
            raw_snapshot
        )
        if (
            snapshot_sha256 != expected_fingerprint
            or reference_snapshot["reference_data_version"] != reference_version
        ):
            return None
        validated_reference = _validate_document(reference_snapshot)
    if status_value == "EVALUATED" and reference_snapshot is None:
        return None
    session_date = _stored_date(stored["session_date"], "session_date")
    evaluated_at = datetime.fromisoformat(stored["evaluated_at"])
    if evaluated_at.tzinfo is None:
        return None

    issuer_is_evaluated = (
        isinstance(issuer_assessment, Mapping)
        and issuer_assessment.get("assessment_status") == "EVALUATED"
    )
    issuer_session = (
        issuer_assessment.get("session_date")
        if isinstance(issuer_assessment, Mapping)
        else None
    )
    if issuer_is_evaluated and issuer_session != session_date:
        return None

    raw_reasons = stored.get("reason_codes")
    raw_sectors = stored.get("sector_exposures")
    raw_contributions = stored.get("issuer_contributions")
    raw_alerts = stored.get("alerts")
    raw_unresolved = stored.get("unresolved_issuers")
    raw_zero_listings = stored.get("zero_value_listings")
    allowed_reasons = {
        "issuer_assessment_unavailable",
        "sector_reference_invalid",
        "issuer_sector_mapping_unresolved",
    }
    if (
        not isinstance(raw_reasons, list)
        or any(not isinstance(item, str) or item not in allowed_reasons for item in raw_reasons)
        or not isinstance(raw_sectors, list)
        or not isinstance(raw_contributions, list)
        or raw_alerts != []
        or not isinstance(raw_unresolved, list)
        or not isinstance(raw_zero_listings, list)
    ):
        return None

    safe_zero_listings: list[dict[str, Any]] = []
    expected_zero_listings: list[dict[str, Any]] = []
    if issuer_is_evaluated:
        raw_issuer_listings = issuer_assessment.get("listing_exposures")
        if not isinstance(raw_issuer_listings, list):
            return None
        for item in raw_issuer_listings:
            if not isinstance(item, Mapping):
                return None
            value = _stored_decimal(item.get("market_value"), "listing market value", paise=True)
            if value != 0:
                continue
            exchange, symbol = item.get("exchange"), item.get("symbol")
            issuer_type, issuer_id = item.get("issuer_id_type"), item.get("issuer_id")
            if (
                exchange not in {"NSE", "BSE"}
                or not isinstance(symbol, str)
                or not re.fullmatch(r"[A-Z0-9][A-Z0-9&.-]{0,31}", symbol)
                or (issuer_id is None) != (issuer_type is None)
                or issuer_id is not None
                and not is_valid_issuer_identifier(issuer_type, issuer_id)
            ):
                return None
            expected_zero_listings.append(
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "listing": f"{exchange}:{symbol}",
                    "market_value": Decimal("0"),
                    "issuer_id_type": issuer_type,
                    "issuer_id": issuer_id,
                }
            )
    expected_zero_listings.sort(key=lambda item: item["listing"])
    for item in raw_zero_listings:
        if not isinstance(item, Mapping) or set(item) != {
            "exchange",
            "symbol",
            "listing",
            "market_value",
            "issuer_id_type",
            "issuer_id",
        }:
            return None
        value = _stored_decimal(item.get("market_value"), "zero listing value", paise=True)
        exchange, symbol = item.get("exchange"), item.get("symbol")
        if (
            value != 0
            or exchange not in {"NSE", "BSE"}
            or not isinstance(symbol, str)
            or not re.fullmatch(r"[A-Z0-9][A-Z0-9&.-]{0,31}", symbol)
            or item.get("listing") != f"{exchange}:{symbol}"
        ):
            return None
        issuer_type, issuer_id = item.get("issuer_id_type"), item.get("issuer_id")
        if (
            (issuer_id is None) != (issuer_type is None)
            or issuer_id is not None
            and not is_valid_issuer_identifier(issuer_type, issuer_id)
        ):
            return None
        safe_zero_listings.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "listing": f"{exchange}:{symbol}",
                "market_value": value,
                "issuer_id_type": issuer_type,
                "issuer_id": issuer_id,
            }
        )
    if safe_zero_listings != expected_zero_listings:
        return None

    metric_names = (
        "portfolio_nav",
        "total_equity_market_value",
        "largest_sector_weight_pct",
        "top_three_sector_weight_pct",
        "sector_hhi",
    )
    safe_metrics: dict[str, Decimal | None] = {}
    for name in metric_names:
        value = stored.get(name)
        safe_metrics[name] = (
            None
            if value is None
            else _stored_decimal(
                value,
                name,
                paise=name in {"portfolio_nav", "total_equity_market_value"},
            )
        )

    safe_unresolved: list[dict[str, str]] = []
    seen_unresolved: set[tuple[str, str]] = set()
    for item in raw_unresolved:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"issuer_id_type", "issuer_id", "reason"}
            or not is_valid_issuer_identifier(
                item.get("issuer_id_type"), item.get("issuer_id")
            )
            or item.get("reason")
            != "No eligible verified sector mapping applies to this session."
        ):
            return None
        key = (item["issuer_id_type"], item["issuer_id"])
        if key in seen_unresolved:
            return None
        seen_unresolved.add(key)
        safe_unresolved.append(
            {
                "issuer_id_type": key[0],
                "issuer_id": key[1],
                "reason": item["reason"],
            }
        )
    safe_unresolved.sort(key=lambda item: (item["issuer_id_type"], item["issuer_id"]))

    if status_value == "NOT_EVALUATED":
        binding_unavailable_is_valid = (
            reference_snapshot is None
            and (
                "sector_reference_invalid" in raw_reasons
                or (
                    "issuer_assessment_unavailable" in raw_reasons
                    and reference_version == "unavailable"
                )
            )
        )
        if (
            not raw_reasons
            or raw_sectors
            or raw_contributions
            or any(value is not None for value in safe_metrics.values())
            or (
                reference_snapshot is None
                and not binding_unavailable_is_valid
            )
            or (
                reference_snapshot is not None
                and "sector_reference_invalid" in raw_reasons
            )
            or (
                "issuer_assessment_unavailable" in raw_reasons
                and issuer_is_evaluated
            )
            or (
                "issuer_sector_mapping_unresolved" in raw_reasons
                and (
                    not issuer_is_evaluated
                    or validated_reference is None
                    or _unresolved_issuer_keys(
                        issuer_assessment, validated_reference, session_date
                    )
                    != seen_unresolved
                )
            )
            or (
                "sector_reference_invalid" in raw_reasons
                and (not issuer_is_evaluated or safe_unresolved)
            )
        ):
            return None
        return {
            "assessment_status": status_value,
            "policy_version": SECTOR_CONCENTRATION_POLICY_VERSION,
            "reference_data_version": reference_version,
            "reference_snapshot": reference_snapshot,
            "reference_snapshot_sha256": snapshot_sha256,
            "taxonomy_name": SECTOR_TAXONOMY_NAME,
            "taxonomy_version": SECTOR_TAXONOMY_VERSION,
            "evaluated_at": evaluated_at,
            "session_date": session_date,
            **safe_metrics,
            "sector_exposures": [],
            "issuer_contributions": [],
            "zero_value_listings": safe_zero_listings,
            "alerts": [],
            "reason_codes": raw_reasons,
            "unresolved_issuers": safe_unresolved,
        }

    if (
        not issuer_is_evaluated
        or raw_reasons
        or safe_unresolved
        or safe_metrics["portfolio_nav"] is None
        or safe_metrics["total_equity_market_value"] is None
        or safe_metrics["largest_sector_weight_pct"] is None
        or safe_metrics["top_three_sector_weight_pct"] is None
    ):
        return None
    nav = safe_metrics["portfolio_nav"]
    equity = safe_metrics["total_equity_market_value"]
    if (
        nav <= 0
        or equity < 0
        or equity > nav
        or nav != Decimal(issuer_assessment["portfolio_nav"])
        or equity != Decimal(issuer_assessment["total_equity_market_value"])
    ):
        return None
    raw_issuer_exposures = issuer_assessment.get("issuer_exposures")
    if not isinstance(raw_issuer_exposures, list):
        return None
    expected_issuers: dict[tuple[str, str], Decimal] = {}
    for item in raw_issuer_exposures:
        if not isinstance(item, Mapping):
            return None
        key = (item.get("issuer_id_type"), item.get("issuer_id"))
        if not is_valid_issuer_identifier(*key):
            return None
        value = _stored_decimal(item.get("market_value"), "issuer market value", paise=True)
        if value <= 0 or key in expected_issuers:
            return None
        expected_issuers[key] = value
    if sum(expected_issuers.values(), Decimal("0")) != equity:
        return None
    if validated_reference is None:
        return None
    active_mappings = _active_mapping_records(validated_reference, session_date)

    safe_contributions: list[dict[str, Any]] = []
    contribution_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in raw_contributions:
        if not isinstance(item, Mapping) or set(item) != {
            "issuer_id_type",
            "issuer_id",
            "market_value",
            "sector_code",
            "sector_name",
            "mapping",
        }:
            return None
        key = (item.get("issuer_id_type"), item.get("issuer_id"))
        code, name = item.get("sector_code"), item.get("sector_name")
        amount = _stored_decimal(item.get("market_value"), "issuer contribution", paise=True)
        mapping = item.get("mapping")
        if (
            not is_valid_issuer_identifier(*key)
            or key in contribution_by_key
            or key not in expected_issuers
            or amount != expected_issuers[key]
            or not isinstance(code, str)
            or SECTOR_TAXONOMY.get(code) != name
            or not isinstance(mapping, Mapping)
            or set(mapping) != {
                "effective_from",
                "effective_to",
                "source",
                "source_reference",
                "source_published_on",
                "verified_on",
                "reference_data_version",
                "mapping_sha256",
            }
            or mapping.get("reference_data_version") != reference_version
            or not isinstance(mapping.get("source"), str)
            or not mapping["source"].strip()
            or mapping["source"] != mapping["source"].strip()
            or len(mapping["source"]) > 512
            or not isinstance(mapping.get("source_reference"), str)
            or not _has_document_locator(mapping["source_reference"])
            or len(mapping["source_reference"]) > 1024
        ):
            return None
        effective_from = _stored_date(mapping.get("effective_from"), "effective_from")
        effective_to = _stored_date(
            mapping.get("effective_to"), "effective_to", optional=True
        )
        published = _stored_date(
            mapping.get("source_published_on"), "source_published_on"
        )
        verified = _stored_date(mapping.get("verified_on"), "verified_on")
        if (
            effective_to is not None and effective_to < effective_from
            or not effective_from <= session_date
            or effective_to is not None and session_date > effective_to
            or published > session_date
        ):
            return None
        safe_mapping = {
            "effective_from": effective_from,
            "effective_to": effective_to,
            "source": mapping["source"],
            "source_reference": mapping["source_reference"],
            "source_published_on": published,
            "verified_on": verified,
            "reference_data_version": reference_version,
        }
        mapping_record = {
            "issuer_id_type": key[0],
            "issuer_id": key[1],
            "sector_code": code,
            "sector_name": name,
            "effective_from": effective_from.isoformat(),
            "effective_to": (
                effective_to.isoformat() if effective_to is not None else None
            ),
            "source": mapping["source"],
            "source_reference": mapping["source_reference"],
            "source_published_on": published.isoformat(),
            "verified_on": verified.isoformat(),
            "reference_data_version": reference_version,
        }
        expected_mapping = active_mappings.get(key)
        mapping_digest = hashlib.sha256(
            _canonical_json(mapping_record).encode("utf-8")
        ).hexdigest()
        if (
            expected_mapping is None
            or mapping_record != expected_mapping
            or mapping.get("mapping_sha256") != mapping_digest
        ):
            return None
        safe_mapping["mapping_sha256"] = mapping_digest
        contribution = {
            "issuer_id_type": key[0],
            "issuer_id": key[1],
            "market_value": amount,
            "sector_code": code,
            "sector_name": name,
            "mapping": safe_mapping,
        }
        contribution_by_key[key] = contribution
        safe_contributions.append(contribution)
    if set(contribution_by_key) != set(expected_issuers):
        return None

    grouped_values: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    grouped_issuers: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for key, contribution in contribution_by_key.items():
        sector_key = (contribution["sector_code"], contribution["sector_name"])
        grouped_values[sector_key] += contribution["market_value"]
        grouped_issuers[sector_key].append(
            {"issuer_id_type": key[0], "issuer_id": key[1]}
        )
    if sum(grouped_values.values(), Decimal("0")) != equity:
        return None

    safe_sectors: list[dict[str, Any]] = []
    for item in raw_sectors:
        if not isinstance(item, Mapping) or set(item) != {
            "sector_code",
            "sector_name",
            "market_value",
            "portfolio_weight_pct",
            "issuers",
        }:
            return None
        code, name = item.get("sector_code"), item.get("sector_name")
        key = (code, name)
        amount = _stored_decimal(item.get("market_value"), "sector market value", paise=True)
        weight = _stored_decimal(item.get("portfolio_weight_pct"), "sector weight")
        issuers = item.get("issuers")
        with localcontext() as context:
            context.prec = 50
            expected_weight = grouped_values.get(key, Decimal("-1")) / nav * Decimal("100")
        expected_members = sorted(
            grouped_issuers.get(key, []),
            key=lambda row: (row["issuer_id_type"], row["issuer_id"]),
        )
        if (
            not isinstance(code, str)
            or SECTOR_TAXONOMY.get(code) != name
            or key not in grouped_values
            or amount != grouped_values[key]
            or weight != expected_weight
            or not isinstance(issuers, list)
            or issuers != expected_members
            or key in {(row["sector_code"], row["sector_name"]) for row in safe_sectors}
        ):
            return None
        safe_sectors.append(
            {
                "sector_code": code,
                "sector_name": name,
                "market_value": amount,
                "portfolio_weight_pct": weight,
                "issuers": expected_members,
            }
        )
    if set((row["sector_code"], row["sector_name"]) for row in safe_sectors) != set(
        grouped_values
    ):
        return None
    expected_order = sorted(
        safe_sectors,
        key=lambda row: (
            -row["market_value"],
            row["sector_code"],
            row["sector_name"],
        ),
    )
    if safe_sectors != expected_order:
        return None

    with localcontext() as context:
        context.prec = 50
        expected_largest = (
            safe_sectors[0]["market_value"] / nav * Decimal("100")
            if safe_sectors
            else Decimal("0")
        )
        expected_top_three = (
            sum(
                (row["market_value"] for row in safe_sectors[:3]),
                Decimal("0"),
            )
            / nav
            * Decimal("100")
        )
        expected_hhi = (
            sum(
                ((row["market_value"] / equity) ** 2 for row in safe_sectors),
                Decimal("0"),
            )
            if equity > 0
            else None
        )
    if (
        safe_metrics["largest_sector_weight_pct"] != expected_largest
        or safe_metrics["top_three_sector_weight_pct"] != expected_top_three
        or safe_metrics["sector_hhi"] != expected_hhi
    ):
        return None
    safe_contributions.sort(key=lambda row: (row["issuer_id_type"], row["issuer_id"]))
    return {
        "assessment_status": status_value,
        "policy_version": SECTOR_CONCENTRATION_POLICY_VERSION,
        "reference_data_version": reference_version,
        "reference_snapshot": reference_snapshot,
        "reference_snapshot_sha256": snapshot_sha256,
        "taxonomy_name": SECTOR_TAXONOMY_NAME,
        "taxonomy_version": SECTOR_TAXONOMY_VERSION,
        "evaluated_at": evaluated_at,
        "session_date": session_date,
        **safe_metrics,
        "sector_exposures": safe_sectors,
        "issuer_contributions": safe_contributions,
        "zero_value_listings": safe_zero_listings,
        "alerts": [],
        "reason_codes": raw_reasons,
        "unresolved_issuers": safe_unresolved,
    }
