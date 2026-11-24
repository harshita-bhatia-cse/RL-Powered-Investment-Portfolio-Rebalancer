"""Explicit representative NSE stock universe configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import argparse
import json

import pandas as pd


SUPPORTED_SECTORS = (
    "Banking",
    "IT",
    "Energy",
    "Pharma",
    "Automobile",
    "FMCG",
    "Metals",
    "Infrastructure",
)


@dataclass(frozen=True)
class IndiaStock:
    company_name: str
    nse_symbol: str
    sector: str
    instrument_type: str = "stock"
    listing_date: str | None = None
    listing_status: str = "listing-date-not-verified"
    historical_scope: str = "separate; no pre-demerger alias mapping"

    def validate(self) -> None:
        if not self.company_name.strip():
            raise ValueError("company name is required")
        if not self.nse_symbol.strip().upper().endswith(".NS"):
            raise ValueError(
                f"invalid NSE symbol {self.nse_symbol!r}; expected an .NS suffix"
            )
        if self.sector not in SUPPORTED_SECTORS:
            raise ValueError(
                f"unsupported sector {self.sector!r}; expected one of {SUPPORTED_SECTORS}"
            )
        if self.instrument_type not in ("stock", "etf"):
            raise ValueError(
                f"unsupported instrument type {self.instrument_type!r}"
            )
        if self.listing_date is not None:
            parsed = pd.to_datetime(self.listing_date, errors="coerce")
            if pd.isna(parsed):
                raise ValueError(
                    f"invalid listing date {self.listing_date!r}"
                )

    @property
    def instrument_metadata(self) -> dict[str, str | None]:
        return {
            "company_name": self.company_name,
            "nse_symbol": self.nse_symbol,
            "exchange": "NSE",
            "sector": self.sector,
            "instrument_type": self.instrument_type,
            "listing_date": self.listing_date,
            "listing_status": self.listing_status,
            "historical_scope": self.historical_scope,
        }


UNIVERSE_SOURCE_METADATA = {
    "mapping_source": "owner-maintained representative universe",
    "mapping_status": "explicit local configuration",
    "source": "NSE symbol names and sectors require independent verification",
    "live_verification_status": "not performed",
    "mapping_version": "1.1",
}


INDIA_STOCK_UNIVERSE: tuple[IndiaStock, ...] = (
    IndiaStock("HDFC Bank", "HDFCBANK.NS", "Banking"),
    IndiaStock("ICICI Bank", "ICICIBANK.NS", "Banking"),
    IndiaStock("State Bank of India", "SBIN.NS", "Banking"),
    IndiaStock("Tata Consultancy Services", "TCS.NS", "IT"),
    IndiaStock("Infosys", "INFY.NS", "IT"),
    IndiaStock("HCL Technologies", "HCLTECH.NS", "IT"),
    IndiaStock("Reliance Industries", "RELIANCE.NS", "Energy"),
    IndiaStock("Oil and Natural Gas Corporation", "ONGC.NS", "Energy"),
    IndiaStock("NTPC", "NTPC.NS", "Energy"),
    IndiaStock("Sun Pharmaceutical Industries", "SUNPHARMA.NS", "Pharma"),
    IndiaStock("Dr. Reddy's Laboratories", "DRREDDY.NS", "Pharma"),
    IndiaStock("Cipla", "CIPLA.NS", "Pharma"),
    IndiaStock("Maruti Suzuki India", "MARUTI.NS", "Automobile"),
    IndiaStock("Mahindra and Mahindra", "M&M.NS", "Automobile"),
    IndiaStock(
        "Tata Motors Passenger Vehicles Limited",
        "TMPV.NS",
        "Automobile",
        listing_status="listing-date-not-verified",
        historical_scope="separate post-demerger security; no inherited Tata Motors history",
    ),
    IndiaStock(
        "Tata Motors Limited (Commercial Vehicles)",
        "TMCV.NS",
        "Automobile",
        listing_status="listing-date-not-verified",
        historical_scope="separate post-demerger security; no inherited Tata Motors history",
    ),
    IndiaStock("Hindustan Unilever", "HINDUNILVR.NS", "FMCG"),
    IndiaStock("ITC", "ITC.NS", "FMCG"),
    IndiaStock("Nestle India", "NESTLEIND.NS", "FMCG"),
    IndiaStock("Tata Steel", "TATASTEEL.NS", "Metals"),
    IndiaStock("Hindalco Industries", "HINDALCO.NS", "Metals"),
    IndiaStock("JSW Steel", "JSWSTEEL.NS", "Metals"),
    IndiaStock("Larsen and Toubro", "LT.NS", "Infrastructure"),
    IndiaStock("UltraTech Cement", "ULTRACEMCO.NS", "Infrastructure"),
    IndiaStock("DLF", "DLF.NS", "Infrastructure"),
)


def validate_india_stock_universe(
    universe: Iterable[IndiaStock] = INDIA_STOCK_UNIVERSE,
) -> tuple[IndiaStock, ...]:
    entries = tuple(universe)
    if not entries:
        raise ValueError("India stock universe is empty")
    symbols: set[str] = set()
    for entry in entries:
        entry.validate()
        symbol = entry.nse_symbol.strip().upper()
        if symbol in symbols:
            raise ValueError(f"duplicate NSE symbol in India universe: {symbol}")
        symbols.add(symbol)
    return entries


def get_india_stock_universe(
    sector: str | None = None,
) -> tuple[IndiaStock, ...]:
    universe = validate_india_stock_universe()
    if sector is None:
        return universe
    if sector not in SUPPORTED_SECTORS:
        raise ValueError(f"unsupported sector: {sector}")
    return tuple(entry for entry in universe if entry.sector == sector)


def india_stock_universe_symbols(sector: str | None = None) -> tuple[str, ...]:
    return tuple(
        entry.nse_symbol
        for entry in get_india_stock_universe(sector=sector)
    )


def universe_to_sector_metadata() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for entry in validate_india_stock_universe():
        result[entry.nse_symbol] = {
            "Sector": entry.sector,
            "Industry": "not specified",
        }
    return result


def universe_to_dataframe(
    universe: Iterable[IndiaStock] = INDIA_STOCK_UNIVERSE,
) -> pd.DataFrame:
    entries = validate_india_stock_universe(universe)
    return pd.DataFrame([
        {
            "Company Name": entry.company_name,
            "NSE Symbol": entry.nse_symbol,
            "Sector": entry.sector,
            "Instrument Type": entry.instrument_type,
            "Exchange": "NSE",
            "Listing Date": entry.listing_date,
            "Listing Status": entry.listing_status,
            "Historical Scope": entry.historical_scope,
            "Source/Provenance": UNIVERSE_SOURCE_METADATA["mapping_source"],
            "Live Verification Status": UNIVERSE_SOURCE_METADATA["live_verification_status"],
        }
        for entry in entries
    ])


def build_india_universe_report(
    *,
    market_csv: str | Path | None = None,
    universe: Iterable[IndiaStock] = INDIA_STOCK_UNIVERSE,
) -> dict[str, object]:
    entries = validate_india_stock_universe(universe)
    symbols = tuple(entry.nse_symbol for entry in entries)
    available: set[str] = set()
    if market_csv is not None:
        path = Path(market_csv)
        if path.is_file():
            frame = pd.read_csv(path)
            if "Symbol" in frame.columns:
                available = {
                    str(symbol).strip().upper()
                    for symbol in frame["Symbol"].dropna().tolist()
                }

    available_symbols = set(symbols).intersection(available)
    return {
        "source_metadata": UNIVERSE_SOURCE_METADATA,
        "symbol_count": len(symbols),
        "symbols": list(symbols),
        "sector_counts": universe_to_dataframe(entries)["Sector"].value_counts().to_dict(),
        "rows": [
            {
                "Company Name": entry.company_name,
                "NSE Symbol": entry.nse_symbol,
                "Sector": entry.sector,
                "Instrument Type": entry.instrument_type,
                "Market Data Status": (
                    "available"
                    if entry.nse_symbol in available
                    else "not available"
                ),
            }
            for entry in entries
        ],
        "market_data_available_count": len(available_symbols),
        "market_data_unavailable_symbols": [
            symbol for symbol in symbols if symbol not in available
        ] if market_csv is not None else [],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate the configured representative NSE stock universe"
    )
    parser.add_argument("--sector", default=None)
    parser.add_argument(
        "--market-csv",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "india_daily"
            / "validated"
            / "indian_etf_validated.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "india_daily"
            / "stock_universe"
        ),
    )
    args = parser.parse_args(argv)
    universe = (
        get_india_stock_universe(args.sector)
        if args.sector
        else INDIA_STOCK_UNIVERSE
    )
    report = build_india_universe_report(
        market_csv=args.market_csv,
        universe=universe,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    universe_csv = output_dir / "india_stock_universe.csv"
    universe_to_dataframe(universe).to_csv(universe_csv, index=False)

    status_csv = output_dir / "india_stock_universe_market_status.csv"
    pd.DataFrame(report["rows"]).to_csv(status_csv, index=False)
    report["universe_csv"] = str(universe_csv)
    report["market_status_csv"] = str(status_csv)
    manifest_path = output_dir / "india_stock_universe_manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "symbol_count": report["symbol_count"],
        "sector_counts": report["sector_counts"],
        "market_data_available_count": report["market_data_available_count"],
        "market_data_unavailable_symbols": report["market_data_unavailable_symbols"],
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
