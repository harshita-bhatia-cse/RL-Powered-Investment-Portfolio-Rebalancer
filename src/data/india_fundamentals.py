"""CSV-first ingestion pipeline for normalized Indian company fundamentals."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "india_fundamentals"

NORMALIZED_COLUMNS = (
    "Company Name",
    "Symbol",
    "Exchange",
    "NSE Symbol",
    "BSE Symbol",
    "ISIN",
    "Reporting Period",
    "Period Type",
    "Reporting Date",
    "Publication Date",
    "Fiscal Year",
    "Currency",
    "Units",
    "Revenue",
    "Operating Profit",
    "Net Profit",
    "EPS",
    "Total Assets",
    "Total Debt",
    "Shareholder Equity",
    "Operating Cash Flow",
    "ROE",
    "ROCE",
    "P/E",
    "P/B",
    "Shares Outstanding",
    "Market Price",
    "Source",
    "Source URL",
    "Source Document",
    "Retrieval Timestamp",
    "Provenance Status",
)

FIELD_ALIASES = {
    "Company Name": ("company_name", "company", "name"),
    "Symbol": ("symbol", "company_symbol", "ticker"),
    "Exchange": ("exchange", "listing_exchange"),
    "NSE Symbol": ("nse_symbol", "nse"),
    "BSE Symbol": ("bse_symbol", "bse"),
    "ISIN": ("isin",),
    "Reporting Period": ("reporting_period", "period", "quarter"),
    "Period Type": ("period_type", "frequency", "reporting_type"),
    "Reporting Date": ("reporting_date", "period_end_date", "datadate"),
    "Publication Date": ("publication_date", "filing_date", "published_date"),
    "Fiscal Year": ("fiscal_year", "year"),
    "Currency": ("currency",),
    "Units": ("units", "unit"),
    "Revenue": ("revenue", "total_revenue", "sales"),
    "Operating Profit": ("operating_profit", "ebit", "operating_income"),
    "Net Profit": ("net_profit", "net_income", "profit_after_tax", "pat"),
    "EPS": ("eps", "earnings_per_share"),
    "Total Assets": ("total_assets", "assets"),
    "Total Debt": ("total_debt", "debt"),
    "Shareholder Equity": ("shareholder_equity", "stockholder_equity", "net_worth", "book_equity"),
    "Operating Cash Flow": ("operating_cash_flow", "cash_flow_from_operations", "cfo"),
    "ROE": ("roe", "return_on_equity"),
    "ROCE": ("roce", "return_on_capital_employed"),
    "P/E": ("pe", "p/e", "price_earnings", "price_to_earnings"),
    "P/B": ("pb", "p/b", "price_book", "price_to_book"),
    "Shares Outstanding": ("shares_outstanding", "outstanding_shares"),
    "Market Price": ("market_price", "share_price", "close", "current_price"),
    "Source": ("source", "provider"),
    "Source URL": ("source_url", "url"),
    "Source Document": ("source_document", "document", "filing"),
    "Retrieval Timestamp": ("retrieval_timestamp", "retrieved_at"),
    "Provenance Status": ("provenance_status", "license_status"),
}


class IndiaFundamentalProvider(Protocol):
    """Future external providers implement fetch() and return raw records."""

    def fetch(self) -> pd.DataFrame:
        ...


@dataclass(frozen=True)
class IndiaFundamentalCSVProvider:
    """Read-only CSV provider for the initial implementation."""

    input_path: str | Path

    def fetch(self) -> pd.DataFrame:
        path = Path(self.input_path)
        if not path.is_file():
            raise FileNotFoundError(f"India fundamentals CSV not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError("India fundamental input must be a CSV file")
        return pd.read_csv(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _first_alias(frame: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    lower_columns = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        column = lower_columns.get(alias.strip().lower())
        if column is not None:
            return frame[column]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _numeric(value: pd.Series) -> pd.Series:
    converted = pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return converted


def _ratio_numeric(value: pd.Series) -> pd.Series:
    converted = pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return converted


def normalize_india_fundamentals(
    raw: pd.DataFrame,
    *,
    default_source: str = "CSV",
    retrieval_timestamp: str | None = None,
) -> pd.DataFrame:
    """Normalize financial records without filling missing financial values."""
    if raw.empty:
        raise ValueError("India fundamentals input contains no records")
    if not isinstance(raw, pd.DataFrame):
        raise ValueError("India fundamentals input must be tabular")

    retrieved = retrieval_timestamp or _utc_timestamp()
    normalized = pd.DataFrame(index=raw.index)

    for target, aliases in FIELD_ALIASES.items():
        normalized[target] = _first_alias(raw, aliases)

    normalized["Company Name"] = normalized["Company Name"].apply(
        lambda value: str(value).strip() if pd.notna(value) and str(value).strip() else pd.NA
    )
    normalized["Symbol"] = normalized["Symbol"].apply(
        lambda value: str(value).strip().upper() if pd.notna(value) and str(value).strip() else pd.NA
    )

    for column in ("NSE Symbol", "BSE Symbol"):
        normalized[column] = normalized[column].apply(
            lambda value: str(value).strip().upper()
            if pd.notna(value) and str(value).strip()
            else pd.NA
        )

    exchange = normalized["Exchange"].apply(
        lambda value: str(value).strip().upper()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )
    normalized["Exchange"] = exchange

    for index in normalized.index:
        symbol = normalized.at[index, "Symbol"]
        nse = normalized.at[index, "NSE Symbol"]
        bse = normalized.at[index, "BSE Symbol"]
        listing_exchange = normalized.at[index, "Exchange"]
        if pd.isna(nse) and pd.notna(symbol) and str(listing_exchange).upper() == "NSE":
            normalized.at[index, "NSE Symbol"] = symbol
        if pd.isna(bse) and pd.notna(symbol) and str(listing_exchange).upper() == "BSE":
            normalized.at[index, "BSE Symbol"] = symbol

    if normalized["Symbol"].isna().all():
        raise ValueError("At least one NSE/BSE company symbol or ticker is required")
    if normalized["Company Name"].isna().all():
        raise ValueError("At least one company name is required")

    period_type = normalized["Period Type"].apply(
        lambda value: str(value).strip().lower()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )
    period_type = period_type.replace({
        "year": "annual",
        "yearly": "annual",
        "q": "quarterly",
        "quarter": "quarterly",
    })
    invalid_period_type = period_type.notna() & ~period_type.isin(("annual", "quarterly"))
    if invalid_period_type.any():
        raise ValueError("Period Type must be annual or quarterly")
    normalized["Period Type"] = period_type

    for column in ("Reporting Date", "Publication Date"):
        parsed = pd.to_datetime(normalized[column], errors="coerce")
        normalized[column] = parsed.dt.strftime("%Y-%m-%d")

    normalized["Fiscal Year"] = normalized["Fiscal Year"].apply(
        lambda value: str(value).strip()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )
    normalized["Reporting Period"] = normalized["Reporting Period"].apply(
        lambda value: str(value).strip()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )

    if normalized["Reporting Date"].isna().all() and normalized["Reporting Period"].isna().all():
        raise ValueError("Reporting Date or Reporting Period is required")

    normalized["Currency"] = normalized["Currency"].apply(
        lambda value: str(value).strip().upper()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )
    normalized["Units"] = normalized["Units"].apply(
        lambda value: str(value).strip()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )

    financial_columns = (
        "Revenue",
        "Operating Profit",
        "Net Profit",
        "EPS",
        "Total Assets",
        "Total Debt",
        "Shareholder Equity",
        "Operating Cash Flow",
        "Shares Outstanding",
        "Market Price",
    )
    for column in financial_columns:
        values = normalized[column]
        normalized[column] = _numeric(values)

    source_supplied_roe = _ratio_numeric(normalized["ROE"])
    source_supplied_roce = _ratio_numeric(normalized["ROCE"])
    source_supplied_pe = _ratio_numeric(normalized["P/E"])
    source_supplied_pb = _ratio_numeric(normalized["P/B"])

    net_profit = normalized["Net Profit"]
    equity = normalized["Shareholder Equity"]
    operating_profit = normalized["Operating Profit"]
    debt = normalized["Total Debt"]
    eps = normalized["EPS"]
    shares = normalized["Shares Outstanding"]
    market_price = normalized["Market Price"]

    calculated_roe = net_profit / equity * 100.0
    calculated_roe = calculated_roe.where(equity > 0)
    calculated_roce = operating_profit / (equity + debt) * 100.0
    calculated_roce = calculated_roce.where((equity + debt) > 0)
    calculated_pe = market_price / eps
    calculated_pe = calculated_pe.where(eps > 0)
    units = normalized["Units"].astype("string").str.strip().str.lower()
    unit_multipliers = pd.Series(1.0, index=normalized.index)
    unit_multipliers = unit_multipliers.mask(
        units.eq("crore"),
        10_000_000.0,
    ).mask(units.eq("lakhs"), 100_000.0).mask(
        units.eq("million"),
        1_000_000.0,
    ).mask(units.eq("billion"), 1_000_000_000.0)
    equity_in_base_currency = equity * unit_multipliers
    book_value_per_share = equity_in_base_currency / shares
    calculated_pb = market_price / book_value_per_share
    calculated_pb = calculated_pb.where(
        (shares > 0) & (equity > 0) & (market_price > 0)
    )

    normalized["ROE"] = source_supplied_roe.where(source_supplied_roe.notna(), calculated_roe)
    normalized["ROCE"] = source_supplied_roce.where(source_supplied_roce.notna(), calculated_roce)
    normalized["P/E"] = source_supplied_pe.where(source_supplied_pe.notna(), calculated_pe)
    normalized["P/B"] = source_supplied_pb.where(source_supplied_pb.notna(), calculated_pb)

    for column in ("ROE", "ROCE", "P/E", "P/B"):
        normalized[column] = _ratio_numeric(normalized[column])

    normalized["Source"] = normalized["Source"].apply(
        lambda value: str(value).strip()
        if pd.notna(value) and str(value).strip()
        else default_source
    )
    normalized["Source URL"] = normalized["Source URL"].apply(
        lambda value: str(value).strip()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )
    normalized["Source Document"] = normalized["Source Document"].apply(
        lambda value: str(value).strip()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )
    normalized["Retrieval Timestamp"] = retrieved

    provenance = normalized["Provenance Status"].apply(
        lambda value: str(value).strip().lower()
        if pd.notna(value) and str(value).strip()
        else "unverified"
    )
    normalized["Provenance Status"] = provenance

    normalized["ISIN"] = normalized["ISIN"].apply(
        lambda value: str(value).strip().upper()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )

    normalized = normalized.loc[:, list(NORMALIZED_COLUMNS)]
    normalized = normalized.sort_values(
        ["Symbol", "Reporting Date", "Period Type"],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    return normalized


def _missing_field_counts(normalized: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in NORMALIZED_COLUMNS:
        counts[column] = int(normalized[column].isna().sum())
    return counts


def ingest_india_fundamentals(
    provider: IndiaFundamentalProvider,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    source_name: str = "CSV",
) -> dict[str, object]:
    """Preserve raw CSV/provider records and write normalized fundamentals."""
    raw = provider.fetch()
    if not isinstance(raw, pd.DataFrame):
        raise ValueError("Fundamental provider fetch() must return a DataFrame")

    output_path = Path(output_dir)
    raw_dir = output_path / "raw"
    normalized_dir = output_path / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    retrieval_timestamp = _utc_timestamp()
    raw_path: Path | None = None
    if isinstance(provider, IndiaFundamentalCSVProvider):
        source_path = Path(provider.input_path)
        raw_path = raw_dir / source_path.name
        if raw_path.exists():
            if _sha256(raw_path) != _sha256(source_path):
                raw_path = raw_dir / f"{source_path.stem}-{_sha256(source_path)[:12]}{source_path.suffix}"
        if source_path.resolve() != raw_path.resolve():
                shutil.copy2(source_path, raw_path)

    normalized = normalize_india_fundamentals(
        raw,
        default_source=source_name,
        retrieval_timestamp=retrieval_timestamp,
    )

    normalized_path = normalized_dir / "india_fundamentals_normalized.csv"
    normalized.to_csv(normalized_path, index=False)

    source_coverage = {
        str(symbol): int(len(group))
        for symbol, group in normalized.groupby("Symbol", dropna=True)
    }
    period_coverage = {
        period_type: int(
            len(normalized.loc[normalized["Period Type"] == period_type])
        )
        for period_type in ("annual", "quarterly")
    }
    missing_fields = _missing_field_counts(normalized)
    present_fields = {
        column: int(len(normalized) - count)
        for column, count in missing_fields.items()
    }

    manifest = {
        "generated_at_utc": retrieval_timestamp,
        "timezone": "Asia/Kolkata",
        "source_name": source_name,
        "raw_input": str(raw_path) if raw_path is not None else None,
        "raw_sha256": _sha256(raw_path) if raw_path is not None else None,
        "normalized_output": str(normalized_path),
        "normalized_schema": list(NORMALIZED_COLUMNS),
        "record_count": int(len(normalized)),
        "company_coverage": source_coverage,
        "period_type_coverage": period_coverage,
        "missing_field_counts": missing_fields,
        "populated_field_counts": present_fields,
        "missing_value_policy": (
            "Missing values are preserved as blank CSV fields. "
            "No financial figure is forward-filled or invented."
        ),
        "calculated_ratios": {
            "ROE": "Net Profit / Shareholder Equity * 100",
            "ROCE": "Operating Profit / (Shareholder Equity + Total Debt) * 100",
            "P/E": "Market Price / EPS",
            "P/B": "Market Price / (Shareholder Equity / Shares Outstanding)",
        },
        "ratio_policy": (
            "Source-supplied ROE, ROCE, P/E and P/B are retained. "
            "Calculated values are only created when all required underlying "
            "values are positive and valid."
        ),
        "limitations": [
            "CSV ingestion does not independently verify source filings.",
            "Units and currency must be reviewed for cross-company consistency.",
            "External provider integration requires implementing IndiaFundamentalProvider.",
            "No paid API or live external provider is used by this module.",
        ],
    }
    manifest_path = output_path / "india_fundamentals_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "record_count": int(len(normalized)),
        "raw_output": str(raw_path) if raw_path is not None else None,
        "normalized_output": str(normalized_path),
        "manifest": str(manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Indian company fundamental data from CSV"
    )
    parser.add_argument("input_csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--source", default="CSV")
    args = parser.parse_args(argv)
    result = ingest_india_fundamentals(
        IndiaFundamentalCSVProvider(args.input_csv),
        output_dir=args.output_dir,
        source_name=args.source,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
