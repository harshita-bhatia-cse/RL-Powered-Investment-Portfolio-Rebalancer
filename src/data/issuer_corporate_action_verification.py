"""Build an auditable issuer-verified corporate-action dataset from SSGA XLSX."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.data.corporate_action_provider import (
    ASSET_UNIVERSE,
    CorporateActionValidationError,
    EVENT_COLUMNS,
)

SSGA_LANDING_URL = (
    "https://www.ssga.com/us/en/intermediary/resources/documents/"
    "etf-dividend-distributions"
)
SSGA_WORKBOOK_URL = (
    "https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/"
    "etfs/us/spdr-etf-historical-distributions.xlsx"
)
GLD_PROSPECTUS_URL = (
    "https://files.spdrgoldshares.com/spdr-cms/2025-10/"
    "GLD_Prospectus_10042022_as_filed.pdf"
)
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
UNVERIFIED_COLUMNS = (*YAHOO_CANDIDATE_COLUMNS, "verification_reason")
EVIDENCE_COLUMNS = (
    *EVENT_COLUMNS,
    "record_date",
    "official_provider_event_id",
    "identifier_origin",
    "official_source_url",
    "evidence_reference",
    "source_worksheet",
    "source_row",
    "source_amount_column",
    "yahoo_candidate_status",
)
RESEARCH_FILE = "issuer_research.json"
DEFAULT_START = date(2018, 1, 3)
DEFAULT_END = date(2026, 11, 11)

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

_AMOUNT_COLUMNS = {
    "DIVIDEND ($)": ("Dividends", "dividend"),
    "SHORT TERM CAPITAL GAIN ($)": ("Capital Gains", "short_term_capital_gain"),
    "LONG TERM CAPITAL GAIN ($)": ("Capital Gains", "long_term_capital_gain"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_date(value: str, field: str) -> date:
    text = value.strip()
    for pattern in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            pass
    try:
        serial = Decimal(text)
    except InvalidOperation as exc:
        raise CorporateActionValidationError(
            f"Invalid {field} date in issuer workbook: {value!r}"
        ) from exc
    if not serial.is_finite() or serial < 1:
        raise CorporateActionValidationError(
            f"Invalid {field} date in issuer workbook: {value!r}"
        )
    return date(1899, 12, 30) + timedelta(days=int(serial))


def _column_index(cell_reference: str) -> int:
    letters = "".join(char for char in cell_reference if char.isalpha())
    if not letters:
        raise CorporateActionValidationError(
            f"Invalid spreadsheet cell reference: {cell_reference!r}"
        )
    value = 0
    for letter in letters.upper():
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def _read_xlsx_rows(path: Path) -> tuple[str, list[tuple[int, dict[str, str]]]]:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise CorporateActionValidationError(
            f"Cannot read issuer workbook {path}"
        ) from exc

    with archive:
        try:
            strings_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
            rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        except (KeyError, ET.ParseError) as exc:
            raise CorporateActionValidationError(
                "Issuer workbook is missing required XLSX metadata"
            ) from exc

        strings = [
            "".join(node.text or "" for node in item.findall(f".//{{{_MAIN_NS}}}t"))
            for item in strings_root.findall(f"{{{_MAIN_NS}}}si")
        ]
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in rels_root.findall(f"{{{_PKG_REL_NS}}}Relationship")
        }
        sheet = workbook_root.find(f".//{{{_MAIN_NS}}}sheet")
        if sheet is None:
            raise CorporateActionValidationError(
                "Issuer workbook does not contain a worksheet"
            )
        sheet_name = sheet.attrib.get("name", "")
        relationship_id = sheet.attrib.get(f"{{{_REL_NS}}}id")
        target = targets.get(relationship_id or "")
        if not target:
            raise CorporateActionValidationError(
                "Issuer workbook worksheet target is missing"
            )
        worksheet_path = str(Path("xl") / target).replace("\\", "/")
        try:
            worksheet_root = ET.fromstring(archive.read(worksheet_path))
        except (KeyError, ET.ParseError) as exc:
            raise CorporateActionValidationError(
                "Issuer workbook worksheet cannot be parsed"
            ) from exc

        row_elements = worksheet_root.findall(
            f".//{{{_MAIN_NS}}}sheetData/{{{_MAIN_NS}}}row"
        )
        if not row_elements:
            raise CorporateActionValidationError(
                "Issuer workbook worksheet is empty"
            )

        def row_values(element: ET.Element) -> list[str]:
            cells: dict[int, str] = {}
            for cell in element.findall(f"{{{_MAIN_NS}}}c"):
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    text = "".join(
                        node.text or ""
                        for node in cell.findall(f".//{{{_MAIN_NS}}}t")
                    )
                else:
                    value_element = cell.find(f"{{{_MAIN_NS}}}v")
                    text = value_element.text if value_element is not None else ""
                    if cell_type == "s" and text:
                        try:
                            text = strings[int(text)]
                        except (IndexError, ValueError) as exc:
                            raise CorporateActionValidationError(
                                "Issuer workbook contains an invalid shared string"
                            ) from exc
                cells[_column_index(cell.attrib.get("r", ""))] = (text or "").strip()
            return [
                cells.get(index, "")
                for index in range(max(cells, default=-1) + 1)
            ]

        headers = row_values(row_elements[0])
        if len(headers) != len(set(headers)) or not {
            "TICKER",
            "EX-DATE",
            "RECORD DATE",
            "PAYABLE DATE",
            *_AMOUNT_COLUMNS,
        }.issubset(headers):
            raise CorporateActionValidationError(
                "Issuer workbook has an unexpected distributions header"
            )
        rows: list[tuple[int, dict[str, str]]] = []
        for row_element in row_elements[1:]:
            values = row_values(row_element)
            rows.append(
                (
                    int(row_element.attrib.get("r", "0")),
                    {
                        header: values[index] if index < len(values) else ""
                        for index, header in enumerate(headers)
                    },
                )
            )
        return sheet_name, rows


def _canonical_event_id(
    *,
    symbol: str,
    event_type: str,
    ex_date: str,
    payment_date: str,
    amount: str,
    currency: str,
    source_reference: str,
) -> str:
    canonical = json.dumps(
        {
            "amount": amount,
            "currency": currency,
            "event_type": event_type,
            "ex_date": ex_date,
            "payment_date": payment_date,
            "source_reference": source_reference,
            "symbol": symbol,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "local-ssga-sha256-" + hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _parse_issuer_rows(
    workbook_path: Path,
    *,
    start_date: date,
    end_date: date,
    market_sessions: Optional[Mapping[str, Iterable[date | str]]],
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    sheet_name, rows = _read_xlsx_rows(workbook_path)
    sessions = {
        symbol: {
            value if isinstance(value, date) else date.fromisoformat(str(value))
            for value in values
        }
        for symbol, values in (market_sessions or {}).items()
    }
    parsed: list[dict[str, Any]] = []
    duplicate_count = 0
    source_keys: dict[tuple[str, str, str, str], str] = {}
    date_type_amounts: dict[tuple[str, str, str], str] = {}

    for row_number, raw in rows:
        symbol = raw.get("TICKER", "").strip().upper()
        if symbol not in ASSET_UNIVERSE:
            continue
        if not raw.get("EX-DATE", "").strip():
            raise CorporateActionValidationError(
                f"Missing ex-date for {symbol} in worksheet row {row_number}"
            )
        ex_date = _as_date(raw["EX-DATE"], "ex-date")
        if not start_date <= ex_date <= end_date:
            continue
        if market_sessions is not None and ex_date not in sessions.get(symbol, set()):
            raise CorporateActionValidationError(
                f"Issuer ex-date {ex_date} for {symbol} is not a local market session"
            )
        payment_text = raw.get("PAYABLE DATE", "").strip()
        payment_date = _as_date(payment_text, "payment") if payment_text else None
        if payment_date is None or payment_date < ex_date:
            raise CorporateActionValidationError(
                f"Missing or invalid payable date for {symbol} on {ex_date}"
            )
        record_text = raw.get("RECORD DATE", "").strip()
        record_date = _as_date(record_text, "record") if record_text else None

        for amount_column, (candidate_field, kind) in _AMOUNT_COLUMNS.items():
            amount_text = raw.get(amount_column, "").strip().replace(",", "")
            if not amount_text:
                continue
            try:
                amount = Decimal(amount_text)
            except InvalidOperation as exc:
                raise CorporateActionValidationError(
                    f"Invalid {amount_column} for {symbol} on {ex_date}"
                ) from exc
            if not amount.is_finite() or amount < 0:
                raise CorporateActionValidationError(
                    f"Invalid non-positive {amount_column} for {symbol} on {ex_date}"
                )
            if amount == 0:
                continue

            amount_fixed = format(amount.normalize(), "f")
            source_reference = (
                f"SSGA historical-distributions workbook:"
                f"{sheet_name}!row={row_number};column={amount_column}"
            )
            event_type = "etf_distribution"
            duplicate_key = (
                symbol,
                ex_date.isoformat(),
                amount_column,
                amount_fixed,
            )
            conflict_key = (symbol, ex_date.isoformat(), amount_column)
            if duplicate_key in source_keys:
                duplicate_count += 1
                raise CorporateActionValidationError(
                    f"Duplicate issuer record for {symbol} on {ex_date}: "
                    f"rows {source_keys[duplicate_key]} and {row_number}"
                )
            if conflict_key in date_type_amounts:
                previous_amount = date_type_amounts[conflict_key]
                raise CorporateActionValidationError(
                    f"Conflicting {amount_column} records for {symbol} on "
                    f"{ex_date}: {previous_amount} versus {amount_fixed}"
                )
            source_keys[duplicate_key] = str(row_number)
            date_type_amounts[conflict_key] = amount_fixed
            source_id = _canonical_event_id(
                symbol=symbol,
                event_type=event_type,
                ex_date=ex_date.isoformat(),
                payment_date=payment_date.isoformat(),
                amount=amount_fixed,
                currency="USD",
                source_reference=source_reference,
            )
            parsed.append(
                {
                    "symbol": symbol,
                    "event_type": event_type,
                    "source_id": source_id,
                    "currency": "USD",
                    "announcement_date": "",
                    "ex_date": ex_date.isoformat(),
                    "payment_date": payment_date.isoformat(),
                    "effective_date": "",
                    "distribution_per_share": amount_fixed,
                    "split_ratio": "",
                    "record_date": (
                        record_date.isoformat() if record_date is not None else ""
                    ),
                    "official_provider_event_id": "",
                    "identifier_origin": "locally generated deterministic SHA-256",
                    "official_source_url": SSGA_WORKBOOK_URL,
                    "evidence_reference": source_reference,
                    "source_worksheet": sheet_name,
                    "source_row": row_number,
                    "source_amount_column": amount_column,
                    "candidate_field": candidate_field,
                    "amount_kind": kind,
                }
            )

    counts = {
        "worksheet_rows": len(rows),
        "issuer_records_parsed": len(parsed),
        "duplicate_records": duplicate_count,
        "invalid_records": 0,
    }
    return sheet_name, parsed, counts


def _read_candidates(
    path: Path, *, start_date: date, end_date: date
) -> tuple[list[dict[str, str]], dict[str, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != YAHOO_CANDIDATE_COLUMNS:
            raise CorporateActionValidationError(
                "Yahoo candidate file has an unexpected header"
            )
        candidates = [dict(row) for row in reader]
    unique_candidates: dict[tuple[str, str, str, str], str] = {}
    for row in candidates:
        if row.get("symbol") not in ASSET_UNIVERSE:
            raise CorporateActionValidationError(
                f"Unsupported symbol in Yahoo candidate file: {row.get('symbol')!r}"
            )
        if row.get("currency") != "USD":
            raise CorporateActionValidationError(
                f"Unsupported Yahoo candidate currency: {row.get('currency')!r}"
            )
        if row.get("event_type") not in ("etf_distribution", "cash_dividend", "split"):
            raise CorporateActionValidationError(
                f"Unsupported Yahoo candidate event type: {row.get('event_type')!r}"
            )
        date_text = row.get("effective_date") or row.get("ex_date") or ""
        try:
            action_date = date.fromisoformat(date_text)
        except ValueError as exc:
            raise CorporateActionValidationError(
                f"Invalid Yahoo candidate action date: {date_text!r}"
            ) from exc
        if not start_date <= action_date <= end_date:
            raise CorporateActionValidationError(
                f"Yahoo candidate date {action_date} is outside the requested range"
            )
        numeric_field = (
            "split_ratio" if row["event_type"] == "split" else "distribution_per_share"
        )
        try:
            amount = Decimal(row.get(numeric_field, ""))
        except InvalidOperation as exc:
            raise CorporateActionValidationError(
                f"Invalid Yahoo candidate {numeric_field} for {row['symbol']}"
            ) from exc
        if not amount.is_finite() or amount <= 0:
            raise CorporateActionValidationError(
                f"Yahoo candidate {numeric_field} must be positive and finite"
            )
        signature = json.dumps(
            {
                "amount": str(amount.normalize()),
                "currency": row["currency"],
                "event_type": row["event_type"],
                "source_field": row.get("source_field", ""),
            },
            sort_keys=True,
        )
        key = (
            row["symbol"],
            row["event_type"],
            action_date.isoformat(),
            row.get("source_field", ""),
        )
        previous = unique_candidates.get(key)
        if previous is not None:
            if previous == signature:
                raise CorporateActionValidationError(
                    f"Duplicate Yahoo candidate for {row['symbol']} on {action_date}"
                )
            raise CorporateActionValidationError(
                f"Conflicting Yahoo candidates for {row['symbol']} on {action_date}"
            )
        unique_candidates[key] = signature
        row["_action_date"] = action_date.isoformat()
    return candidates, {
        "duplicate_candidates": 0,
        "conflicting_candidates": 0,
        "invalid_candidates": 0,
    }


def _candidate_matches_issuer(
    candidate: Mapping[str, str], issuer_records: Sequence[Mapping[str, Any]]
) -> tuple[bool, str]:
    symbol = candidate["symbol"]
    action_date = candidate["_action_date"]
    category = candidate.get("source_field", "")
    matching = [
        record
        for record in issuer_records
        if record["symbol"] == symbol
        and record["ex_date"] == action_date
        and record["candidate_field"] == category
    ]
    if not matching:
        if candidate.get("event_type") == "split":
            return False, "SSGA distribution workbook does not verify split events"
        return False, "No matching official issuer distribution row was available"
    try:
        reported = Decimal(candidate.get("distribution_per_share", ""))
    except InvalidOperation:
        return False, "Yahoo amount is missing or invalid"
    official = sum(
        (Decimal(str(record["distribution_per_share"])) for record in matching),
        Decimal("0"),
    )
    # Yahoo publishes these candidate amounts at three decimal places.
    if reported.quantize(Decimal("0.001")) == official.quantize(Decimal("0.001")):
        return True, "Issuer date and amount match Yahoo at Yahoo's 0.001 precision"
    if symbol == "XLU" and category == "Dividends":
        return (
            False,
            "Issuer amount differs from Yahoo; 2:1 share-basis change is "
            "consistent with the reported split but remains unverified",
        )
    return False, "Issuer amount does not match Yahoo candidate at 0.001 precision"


def _price_split_comparison(price_dir: Path) -> dict[str, Any]:
    path = price_dir / "XLU_daily.csv"
    if not path.is_file():
        return {
            "status": "local_price_file_missing",
            "symbol": "XLU",
            "event_date": "2025-12-05",
        }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_date = {row.get("date", ""): row for row in rows}
    previous_date = "2025-12-04"
    event_date = "2025-12-05"
    if previous_date not in by_date or event_date not in by_date:
        return {
            "status": "comparison_dates_missing",
            "symbol": "XLU",
            "previous_date": previous_date,
            "event_date": event_date,
        }
    before = by_date[previous_date]
    after = by_date[event_date]
    comparison: dict[str, Any] = {
        "status": "compared",
        "symbol": "XLU",
        "previous_date": previous_date,
        "event_date": event_date,
        "split_candidate_ratio": 2.0,
        "source_file": str(path),
    }
    for column in ("open", "high", "low", "close", "adj_close"):
        before_value = float(before[column])
        after_value = float(after[column])
        comparison[f"{column}_before"] = before_value
        comparison[f"{column}_after"] = after_value
        comparison[f"{column}_ratio_after_over_before"] = after_value / before_value
    comparison["interpretation"] = (
        "Neither local close series shows the approximately 0.5 one-session "
        "price ratio expected from an unadjusted 2:1 split. This is consistent "
        "with an already split-adjusted price history or an invalid Yahoo event; "
        "it does not distinguish those cases or verify the split."
    )
    return comparison


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_issuer_verification_dataset(
    *,
    candidates_path: str | Path,
    issuer_workbook_path: str | Path,
    output_dir: str | Path,
    local_price_dir: str | Path,
    market_sessions: Optional[Mapping[str, Iterable[date | str]]] = None,
    start_date: date | str = DEFAULT_START,
    end_date: date | str = DEFAULT_END,
    retrieved_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """Create separate verified events, candidate disposition, and audit outputs.

    Only the official State Street distributions workbook is parsed into
    ledger-ready events. The source workbook and original Yahoo candidate file
    are not modified; the workbook is copied into the output evidence folder.
    """
    start = start_date if isinstance(start_date, date) else date.fromisoformat(start_date)
    end = end_date if isinstance(end_date, date) else date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")
    candidate_file = Path(candidates_path)
    workbook_file = Path(issuer_workbook_path)
    output = Path(output_dir)
    prices = Path(local_price_dir)
    if not candidate_file.is_file():
        raise FileNotFoundError(candidate_file)
    if not workbook_file.is_file():
        raise FileNotFoundError(workbook_file)
    output.mkdir(parents=True, exist_ok=True)
    evidence_dir = output / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    preserved_workbook = evidence_dir / "ssga_spdr_historical_distributions.xlsx"
    if workbook_file.resolve() != preserved_workbook.resolve():
        shutil.copy2(workbook_file, preserved_workbook)

    sheet_name, issuer_records, issuer_counts = _parse_issuer_rows(
        workbook_file,
        start_date=start,
        end_date=end,
        market_sessions=market_sessions,
    )
    candidates, candidate_validation = _read_candidates(
        candidate_file, start_date=start, end_date=end
    )
    candidate_match: dict[tuple[str, str, str, str], tuple[bool, str]] = {}
    unverified_rows: list[dict[str, str]] = []
    verified_candidate_keys: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        matched, reason = _candidate_matches_issuer(candidate, issuer_records)
        key = (
            candidate["symbol"],
            candidate.get("event_type", ""),
            candidate["_action_date"],
            candidate.get("source_field", ""),
        )
        candidate_match[key] = (matched, reason)
        if matched:
            verified_candidate_keys.add(key)
        else:
            unverified_rows.append(
                {
                    **{
                        column: candidate.get(column, "")
                        for column in YAHOO_CANDIDATE_COLUMNS
                    },
                    "verification_reason": reason,
                }
            )

    normalized_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for record in issuer_records:
        row = {column: record.get(column, "") for column in EVENT_COLUMNS}
        normalized_rows.append(row)
        candidate_key = (
            record["symbol"],
            "etf_distribution",
            record["ex_date"],
            record["candidate_field"],
        )
        match = candidate_match.get(
            candidate_key,
            (False, "Issuer record found; no Yahoo candidate existed for this event"),
        )
        evidence_rows.append(
            {
                **row,
                "record_date": record["record_date"],
                "official_provider_event_id": "",
                "identifier_origin": record["identifier_origin"],
                "official_source_url": record["official_source_url"],
                "evidence_reference": record["evidence_reference"],
                "source_worksheet": record["source_worksheet"],
                "source_row": record["source_row"],
                "source_amount_column": record["source_amount_column"],
                "yahoo_candidate_status": (
                    "verified_at_yahoo_precision" if match[0] else match[1]
                ),
            }
        )

    normalized_rows.sort(
        key=lambda row: (
            row["ex_date"] or row["effective_date"],
            row["symbol"],
            row["source_id"],
        )
    )
    evidence_rows.sort(
        key=lambda row: (
            row["ex_date"] or row["effective_date"],
            row["symbol"],
            row["source_id"],
        )
    )
    _write_csv(output / "verified_normalized_events.csv", EVENT_COLUMNS, normalized_rows)
    _write_csv(output / "verified_events_evidence.csv", EVIDENCE_COLUMNS, evidence_rows)
    _write_csv(output / "unverified_candidates.csv", UNVERIFIED_COLUMNS, unverified_rows)

    raw_evidence_path = output / "issuer_source_records.jsonl"
    with raw_evidence_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in issuer_records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")

    per_symbol: dict[str, Any] = {}
    for symbol in ASSET_UNIVERSE:
        symbol_events = [
            record for record in issuer_records if record["symbol"] == symbol
        ]
        symbol_candidates = [row for row in candidates if row["symbol"] == symbol]
        verified_rows = [
            row for row in symbol_candidates
            if (
                row["symbol"],
                row.get("event_type", ""),
                row["_action_date"],
                row.get("source_field", ""),
            ) in verified_candidate_keys
        ]
        event_dates = [row["ex_date"] for row in symbol_events]
        candidate_dates = [row["_action_date"] for row in symbol_candidates]
        per_symbol[symbol] = {
            "yahoo_candidate_count": len(symbol_candidates),
            "yahoo_candidate_verified_count": len(verified_rows),
            "yahoo_candidate_unverified_count": len(symbol_candidates) - len(verified_rows),
            "issuer_verified_dividend_records": sum(
                row["source_amount_column"] == "DIVIDEND ($)"
                for row in symbol_events
            ),
            "issuer_verified_capital_gain_records": sum(
                row["source_amount_column"]
                in ("SHORT TERM CAPITAL GAIN ($)", "LONG TERM CAPITAL GAIN ($)")
                for row in symbol_events
            ),
            "issuer_verified_split_records": 0,
            "ledger_ready_event_count": len(symbol_events),
            "issuer_event_date_range": (
                [min(event_dates), max(event_dates)] if event_dates else None
            ),
            "yahoo_candidate_date_range": (
                [min(candidate_dates), max(candidate_dates)]
                if candidate_dates else None
            ),
            "missing_yahoo_payment_dates": sum(
                row.get("event_type") != "split"
                and not row.get("payment_date", "")
                for row in symbol_candidates
            ),
            "missing_yahoo_source_ids": sum(
                not row.get("source_id", "") for row in symbol_candidates
            ),
        }

    xlu_split = _price_split_comparison(prices)
    research = {
        "as_of": (
            retrieved_at or datetime.now(timezone.utc)
        ).astimezone(timezone.utc).isoformat(),
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "verified_sources": [
            {
                "provider": "State Street Global Advisors",
                "landing_page": SSGA_LANDING_URL,
                "workbook_url": SSGA_WORKBOOK_URL,
                "worksheet": sheet_name,
                "coverage_symbols": ["SPY", "XLU", "XLV"],
                "coverage_limitation": (
                    "The workbook contains distribution records; it does not "
                    "provide a complete split history or announcement dates."
                ),
                "identifier_limitation": (
                    "No official event IDs are published in the workbook. "
                    "source_id values are local deterministic hashes, not issuer IDs."
                ),
                "currency_note": (
                    "Workbook amounts are labeled with '$' on the U.S. issuer "
                    "distribution sheet; normalized as USD."
                ),
            }
        ],
        "unresolved_issuer_sources": {
            "QQQ": {
                "official_source": "https://www.invesco.com/qqq-etf/en/home.html",
                "status": "unverified",
                "limitation": (
                    "The accessible official product information reviewed did "
                    "not expose event-level historical distribution records."
                ),
            },
            "IEF": {
                "official_source": (
                    "https://www.ishares.com/us/products/239456/"
                    "ishares-710-year-treasury-bond-etf"
                ),
                "status": "unverified",
                "limitation": (
                    "The official historical-download request returned an HTML "
                    "product page instead of parseable distribution CSV data; "
                    "no event was verified from that response."
                ),
            },
            "GLD": {
                "official_source": (
                    "https://www.ssga.com/us/en/intermediary/etfs/"
                    "spdr-gold-shares-gld"
                ),
                "prospectus": GLD_PROSPECTUS_URL,
                "status": "no Yahoo candidates; full period not verified",
                "evidence": (
                    "The inspected official prospectus states that the Trust "
                    "does not generate income and that holders will not receive "
                    "dividends. This supports the no-cash-dividend expectation "
                    "in that prospectus, but does not establish a complete "
                    "2018-2026 distribution and split history."
                ),
                "prospectus_page_references": {
                    "trust_does_not_generate_income": 16,
                    "shareholders_will_not_receive_dividends": 18,
                },
                "preserved_as": (
                    "evidence/GLD_Prospectus_10042022_as_filed.pdf"
                    if (evidence_dir / "GLD_Prospectus_10042022_as_filed.pdf").is_file()
                    else None
                ),
            },
            "XLU_split": {
                "status": "unverified",
                "reported_event": "2:1, effective 2025-12-05",
                "official_split_notice": None,
                "price_comparison": xlu_split,
                "issuer_distribution_comparison": (
                    "For XLU, older Yahoo dividend amounts are approximately "
                    "half of the corresponding official State Street amounts; "
                    "after the reported date, values align at Yahoo's precision. "
                    "This is consistent with a share-basis adjustment but is "
                    "not a split notice and does not prove event validity."
                ),
            },
        },
        "unverified_candidate_count": len(unverified_rows),
        "per_symbol": per_symbol,
    }
    research_path = output / RESEARCH_FILE
    research_path.write_text(
        json.dumps(research, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact_paths = [
        preserved_workbook,
        output / "verified_normalized_events.csv",
        output / "verified_events_evidence.csv",
        output / "unverified_candidates.csv",
        raw_evidence_path,
        research_path,
    ]
    prospectus_path = evidence_dir / "GLD_Prospectus_10042022_as_filed.pdf"
    if prospectus_path.is_file():
        artifact_paths.append(prospectus_path)
    manifest = {
        "provider": "State Street Global Advisors; Yahoo candidates used only for comparison",
        "source_workbook_url": SSGA_WORKBOOK_URL,
        "source_landing_page": SSGA_LANDING_URL,
        "retrieved_at": research["as_of"],
        "requested_symbols": list(ASSET_UNIVERSE),
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "yahoo_candidate_input": {
            "path": str(candidate_file),
            "sha256": _sha256(candidate_file),
            "row_count": len(candidates),
            "modified": False,
        },
        "issuer_workbook": {
            "preserved_as": str(preserved_workbook.relative_to(output)),
            "sha256": _sha256(preserved_workbook),
            "worksheet": sheet_name,
            **issuer_counts,
        },
        "counts": {
            "yahoo_candidates": len(candidates),
            "issuer_verified_normalized_events": len(normalized_rows),
            "issuer_verified_dividend_records": sum(
                row["event_type"] == "etf_distribution"
                and row["source_amount_column"] == "DIVIDEND ($)"
                for row in evidence_rows
            ),
            "issuer_verified_split_records": 0,
            "yahoo_candidates_unverified": len(unverified_rows),
            "missing_yahoo_payment_dates": sum(
                row.get("event_type") != "split"
                and not row.get("payment_date", "")
                for row in candidates
            ),
            "missing_yahoo_source_ids": sum(
                not row.get("source_id", "") for row in candidates
            ),
            "verified_events_missing_announcement_dates": sum(
                not row["announcement_date"] for row in evidence_rows
            ),
            "verified_events_missing_official_provider_ids": sum(
                not row["official_provider_event_id"] for row in evidence_rows
            ),
        },
        "validation": {
            "source_duplicates": issuer_counts["duplicate_records"],
            "source_invalid_records": issuer_counts["invalid_records"],
            **candidate_validation,
            "candidate_rows_retained": len(candidates),
            "unverified_rows_retained": len(unverified_rows),
            "ledger_schema_columns": list(EVENT_COLUMNS),
            "identifier_policy": (
                "Deterministic local SHA-256 IDs include canonical event fields "
                "and the official workbook worksheet/row reference; they are "
                "not issuer/provider event IDs."
            ),
        },
        "per_symbol": per_symbol,
        "pagination": {
            "mechanism": "Single official XLSX workbook; no paginated API query",
            "worksheet": sheet_name,
            "worksheet_data_rows": issuer_counts["worksheet_rows"],
        },
        "price_comparison": xlu_split,
        "artifacts": {
            str(path.relative_to(output)): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
        "limitations": [
            "Issuer event coverage is incomplete for QQQ and IEF.",
            "GLD's complete 2018-2026 corporate-action history was not verified.",
            "The XLU split candidate is not verified by an official split record.",
            "Yahoo candidate absence or failed source access is not proof of no events.",
            "Do not claim total-return accuracy or use for historical PPO training.",
        ],
        "redistribution": (
            "Generated under the repository's ignored data/ tree. Do not commit "
            "or redistribute issuer or Yahoo data without confirming applicable rights."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_local_market_sessions(
    price_dir: str | Path, symbols: Sequence[str] = ASSET_UNIVERSE
) -> dict[str, tuple[date, ...]]:
    """Load per-symbol session dates from existing local FMP daily CSV files."""
    root = Path(price_dir)
    sessions: dict[str, tuple[date, ...]] = {}
    for symbol in symbols:
        path = root / f"{symbol}_daily.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "date" not in (reader.fieldnames or ()):
                raise CorporateActionValidationError(
                    f"Price CSV has no date column: {path}"
                )
            values = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    values.append(date.fromisoformat(row["date"]))
                except (TypeError, ValueError) as exc:
                    raise CorporateActionValidationError(
                        f"Invalid market date in {path} row {row_number}"
                    ) from exc
        sessions[symbol] = tuple(values)
    return sessions
