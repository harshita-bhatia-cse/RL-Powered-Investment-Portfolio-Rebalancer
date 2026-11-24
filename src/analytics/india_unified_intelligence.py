"""Chronological integration of India market, sector, fundamental and news data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKET_INDICATORS = (
    PROJECT_ROOT
    / "data"
    / "india_daily"
    / "analysis"
    / "india_market_indicators.csv"
)
DEFAULT_STOCK_INTELLIGENCE = (
    PROJECT_ROOT
    / "data"
    / "india_daily"
    / "sector_analysis"
    / "india_stock_opportunities.csv"
)
DEFAULT_FUNDAMENTALS = (
    PROJECT_ROOT
    / "data"
    / "india_fundamentals"
    / "normalized"
    / "india_fundamentals_normalized.csv"
)
DEFAULT_NEWS = (
    PROJECT_ROOT
    / "data"
    / "india_news"
    / "normalized"
    / "india_news_normalized.csv"
)
DEFAULT_MARKET_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "india_daily"
    / "indian_etf_ingestion_manifest.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "india_daily"
    / "unified_intelligence"
)

TECHNICAL_COLUMNS = (
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "EMA_50",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Daily_Return",
    "Rolling_Volatility_20",
    "Momentum_20",
)

FUNDAMENTAL_COLUMNS = (
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
)

NEWS_COLUMNS = (
    "Positive Count",
    "Negative Count",
    "Neutral Count",
    "Risk-Related Count",
    "Unclassified Count",
)


def _read_optional_csv(path: str | Path) -> tuple[pd.DataFrame | None, str]:
    candidate = Path(path)
    if not candidate.is_file():
        return None, "missing"
    try:
        frame = pd.read_csv(candidate)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None, "unavailable"
    if frame.empty:
        return frame, "empty"
    return frame, "available"


def _read_manifest(path: str | Path) -> dict[str, object] | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _source_type_for_fundamentals(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "missing"
    sources = (
        frame["Source"].dropna().astype(str).str.strip().str.lower()
        if "Source" in frame.columns
        else pd.Series(dtype=str)
    )
    if sources.empty:
        return "unknown"
    if sources.str.contains("sample", case=False, regex=False).any():
        return "sample"
    return "provider"


def _source_type_for_news(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "missing"
    dataset_types = (
        frame["Dataset Type"].dropna().astype(str).str.strip().str.lower()
        if "Dataset Type" in frame.columns
        else pd.Series(dtype=str)
    )
    if dataset_types.empty:
        return "unknown"
    if dataset_types.eq("sample").any() or dataset_types.eq("synthetic").any():
        return "sample"
    return "provider"


def _latest_fundamentals_as_of(
    market_rows: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach only fundamentals published on or before each market date."""
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame(index=market_rows.index)

    working = fundamentals.copy()
    for column in ("Publication Date", "Reporting Date", "Retrieval Timestamp"):
        if column not in working.columns:
            working[column] = pd.NA
    working["Publication Date"] = pd.to_datetime(
        working["Publication Date"],
        errors="coerce",
    )
    working["Reporting Date"] = pd.to_datetime(
        working["Reporting Date"],
        errors="coerce",
    )
    working["Symbol"] = working["Symbol"].astype("string").str.strip().str.upper()

    result = pd.DataFrame(index=market_rows.index)
    for column in FUNDAMENTAL_COLUMNS:
        result["Fundamental_" + column] = np.nan
    result["Fundamental_Publication_Date"] = pd.NA
    result["Fundamental_Reporting_Period"] = pd.NA
    result["Fundamental_Source"] = pd.NA
    result["Fundamental_Source_URL"] = pd.NA
    result["Fundamental_Retrieval_Timestamp"] = pd.NA
    result["Fundamental_Provenance_Status"] = pd.NA
    result["Fundamental_Data_Available"] = False

    for index, market_row in market_rows.iterrows():
        symbol = str(market_row["Symbol"]).strip().upper()
        date = market_row["Date"]
        candidates = working.loc[
            (working["Symbol"] == symbol)
            & working["Publication Date"].notna()
            & (working["Publication Date"] <= date)
        ].sort_values("Publication Date", kind="stable")
        if candidates.empty:
            continue

        latest = candidates.iloc[-1]
        for column in FUNDAMENTAL_COLUMNS:
            if column in latest.index:
                value = pd.to_numeric(pd.Series([latest[column]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    result.at[index, "Fundamental_" + column] = float(value)
        result.at[index, "Fundamental_Publication_Date"] = (
            latest["Publication Date"].strftime("%Y-%m-%d")
        )
        result.at[index, "Fundamental_Reporting_Period"] = (
            str(latest["Reporting Period"])
            if pd.notna(latest["Reporting Period"])
            else pd.NA
        )
        result.at[index, "Fundamental_Source"] = (
            str(latest["Source"]) if pd.notna(latest["Source"]) else pd.NA
        )
        result.at[index, "Fundamental_Source_URL"] = (
            str(latest["Source URL"]) if pd.notna(latest["Source URL"]) else pd.NA
        )
        result.at[index, "Fundamental_Retrieval_Timestamp"] = (
            str(latest["Retrieval Timestamp"])
            if pd.notna(latest["Retrieval Timestamp"])
            else pd.NA
        )
        result.at[index, "Fundamental_Provenance_Status"] = (
            str(latest["Provenance Status"])
            if pd.notna(latest["Provenance Status"])
            else "unverified"
        )
        result.at[index, "Fundamental_Data_Available"] = True

    return result


def _news_summary_as_of(
    market_rows: pd.DataFrame,
    news: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach only original news published on or before each market date."""
    result = pd.DataFrame(index=market_rows.index)
    for column in NEWS_COLUMNS:
        result["News_" + column.replace(" Count", "_Count")] = 0
    result["News_Original_Article_Count"] = 0
    result["News_Duplicate_Article_Count"] = 0
    result["News_Missing_Timestamp_Count"] = 0
    result["News_First_Publication_Date"] = pd.NA
    result["News_Last_Publication_Date"] = pd.NA
    result["News_Data_Available"] = False

    if news is None or news.empty:
        return result

    working = news.copy()
    working["Publication Timestamp"] = pd.to_datetime(
        working["Publication Timestamp"],
        errors="coerce",
        utc=True,
    )
    working["Publication Date"] = working["Publication Timestamp"].dt.tz_convert(
        "Asia/Kolkata"
    ).dt.normalize().dt.tz_localize(None)
    working["Symbol"] = working["Symbol"].astype("string").str.strip().str.upper()
    if "Duplicate Status" not in working.columns:
        working["Duplicate Status"] = "original"
    if "Classification" not in working.columns:
        working["Classification"] = "unclassified"

    count_columns = {
        "positive": "News_Positive_Count",
        "negative": "News_Negative_Count",
        "neutral": "News_Neutral_Count",
        "risk-related": "News_Risk-Related_Count",
        "unclassified": "News_Unclassified_Count",
    }

    for index, market_row in market_rows.iterrows():
        symbol = str(market_row["Symbol"]).strip().upper()
        date = market_row["Date"]
        candidates = working.loc[
            (working["Symbol"] == symbol)
            & working["Publication Date"].notna()
            & (working["Publication Date"] <= date)
        ]
        if candidates.empty:
            continue

        result.at[index, "News_Data_Available"] = True
        original = candidates.loc[candidates["Duplicate Status"].eq("original")]
        result.at[index, "News_Original_Article_Count"] = int(len(original))
        result.at[index, "News_Duplicate_Article_Count"] = int(
            candidates["Duplicate Status"].eq("duplicate").sum()
        )
        result.at[index, "News_Missing_Timestamp_Count"] = int(
            candidates["Publication Timestamp"].isna().sum()
        )
        classifications = candidates["Classification"].fillna("unclassified")
        for classification, output_column in count_columns.items():
            result.at[index, output_column] = int(
                classifications.eq(classification).sum()
            )
        publication_dates = candidates["Publication Date"].dropna()
        if not publication_dates.empty:
            result.at[index, "News_First_Publication_Date"] = (
                publication_dates.min().strftime("%Y-%m-%d")
            )
            result.at[index, "News_Last_Publication_Date"] = (
                publication_dates.max().strftime("%Y-%m-%d")
            )

    return result


def build_unified_india_intelligence(
    market_indicators: pd.DataFrame,
    sector_opportunities: pd.DataFrame | None,
    fundamentals: pd.DataFrame | None,
    news: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join components chronologically using NSE/BSE symbols."""
    if market_indicators.empty:
        raise ValueError("India market indicators are empty")
    required = {"Date", "Symbol", "Adjusted Close"}
    missing = sorted(required - set(market_indicators.columns))
    if missing:
        raise ValueError(
            "India market indicators are missing required columns: "
            + ", ".join(missing)
        )

    market = market_indicators.copy()
    market["Date"] = pd.to_datetime(market["Date"], errors="coerce")
    market["Symbol"] = market["Symbol"].astype("string").str.strip().str.upper()
    if market["Date"].isna().any():
        raise ValueError("India market indicators contain invalid dates")
    market = market.sort_values(["Symbol", "Date"], kind="stable").reset_index(drop=True)

    technical = market.loc[:, ["Date", "Symbol", "Adjusted Close"]].copy()
    for column in TECHNICAL_COLUMNS:
        technical[column] = (
            pd.to_numeric(market[column], errors="coerce")
            if column in market.columns
            else np.nan
        )

    if sector_opportunities is not None and not sector_opportunities.empty:
        sector = sector_opportunities.copy()
        sector["Symbol"] = sector["Symbol"].astype("string").str.strip().str.upper()
        sector_columns = [
            column
            for column in (
                "Sector",
                "Industry",
                "Trend",
                "Sector_Momentum_20",
                "Sector_Relative_Performance",
                "Explainable_Signal",
            )
            if column in sector.columns
        ]
        technical = technical.merge(
            sector.loc[:, ["Symbol", *sector_columns]],
            on="Symbol",
            how="left",
        )
    else:
        technical["Sector"] = pd.NA
        technical["Industry"] = pd.NA
        technical["Sector_Momentum_20"] = np.nan
        technical["Sector_Relative_Performance"] = np.nan
        technical["Explainable_Signal"] = "insufficient evidence"

    fundamental_join = _latest_fundamentals_as_of(technical, fundamentals)
    news_join = _news_summary_as_of(technical, news)

    unified = pd.concat([technical, fundamental_join, news_join], axis=1)
    unified["Sector_Data_Status"] = (
        unified["Sector"].notna()
        .map({True: "available", False: "missing"})
    )
    unified["Fundamental_Data_Source_Type"] = (
        _source_type_for_fundamentals(fundamentals)
    )
    unified["News_Data_Source_Type"] = _source_type_for_news(news)
    unified["Missing_Data_Indicator"] = (
        unified.isna().any(axis=1).map(
            {True: "contains missing values", False: "complete"}
        )
    )
    unified["Data_Freshness"] = (
        "historical; components require publication-date checks"
    )
    unified["Look_Ahead_Control"] = (
        "fundamental and news publications included only through market date"
    )

    first_columns = [
        "Date",
        "Symbol",
        "Adjusted Close",
        "Sector",
        "Industry",
        "Sector_Data_Status",
    ]
    remaining = [
        column
        for column in unified.columns
        if column not in first_columns
    ]
    return unified.loc[:, first_columns + remaining].sort_values(
        ["Symbol", "Date"],
        kind="stable",
    ).reset_index(drop=True)


def analyze_unified_india_intelligence(
    *,
    market_indicators_csv: str | Path = DEFAULT_MARKET_INDICATORS,
    sector_opportunities_csv: str | Path = DEFAULT_STOCK_INTELLIGENCE,
    fundamentals_csv: str | Path = DEFAULT_FUNDAMENTALS,
    news_csv: str | Path = DEFAULT_NEWS,
    market_manifest: str | Path = DEFAULT_MARKET_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Generate the consolidated India stock-level intelligence outputs."""
    market, market_status = _read_optional_csv(market_indicators_csv)
    sector, sector_status = _read_optional_csv(sector_opportunities_csv)
    fundamentals, fundamentals_status = _read_optional_csv(fundamentals_csv)
    news, news_status = _read_optional_csv(news_csv)
    if market is None:
        raise FileNotFoundError(
            f"India market indicator input not found: {market_indicators_csv}"
        )

    market_manifest_payload = _read_manifest(market_manifest)
    unified = build_unified_india_intelligence(
        market,
        sector,
        fundamentals,
        news,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    consolidated_path = output_path / "india_unified_stock_intelligence.csv"
    unified.to_csv(consolidated_path, index=False, date_format="%Y-%m-%d")

    market_source = (
        market_manifest_payload.get("provider")
        if market_manifest_payload is not None
        else "unknown"
    )
    market_retrieved = (
        market_manifest_payload.get("retrieval_timestamp")
        if market_manifest_payload is not None
        else None
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
        "timezone": "Asia/Kolkata",
        "input_status": {
            "market_indicators": market_status,
            "sector_opportunities": sector_status,
            "fundamentals": fundamentals_status,
            "news": news_status,
        },
        "inputs": {
            "market_indicators": str(market_indicators_csv),
            "sector_opportunities": str(sector_opportunities_csv),
            "fundamentals": str(fundamentals_csv),
            "news": str(news_csv),
            "market_manifest": str(market_manifest),
        },
        "provenance": {
            "market_provider": market_source,
            "market_retrieval_timestamp": market_retrieved,
            "market_data_type": "validated provider market data",
            "sector_data_type": "owner-maintained explicit mappings",
            "fundamental_data_type": _source_type_for_fundamentals(fundamentals),
            "news_data_type": _source_type_for_news(news),
        },
        "availability": {
            "technical_observations": int(len(unified)),
            "symbols": sorted(unified["Symbol"].dropna().unique().tolist()),
            "symbols_with_sector": sorted(
                unified.loc[
                    unified["Sector_Data_Status"].eq("available"),
                    "Symbol",
                ].dropna().unique().tolist()
            ),
            "symbols_with_fundamentals": sorted(
                unified.loc[
                    unified["Fundamental_Data_Available"].eq(True),
                    "Symbol",
                ].dropna().unique().tolist()
            ),
            "symbols_with_news": sorted(
                unified.loc[
                    unified["News_Data_Available"].eq(True),
                    "Symbol",
                ].dropna().unique().tolist()
            ),
        },
        "look_ahead_controls": [
            "Market data is sorted chronologically by Symbol and Date.",
            "Fundamentals are attached only when Publication Date <= market Date.",
            "News is attached only when Publication Timestamp/Date <= market Date.",
            "News duplicates are counted separately and do not replace originals.",
            "Missing publication dates make the record unavailable, not backfilled.",
        ],
        "component_columns": {
            "technical": list(TECHNICAL_COLUMNS),
            "fundamental": list(FUNDAMENTAL_COLUMNS),
            "news": list(NEWS_COLUMNS),
        },
        "limitations": [
            "The fundamentals input is explicitly a small labelled sample.",
            "The news input is explicitly synthetic sample data.",
            "The sector mappings are not from a live or authoritative provider.",
            "This output contains component metrics only.",
            "No prediction score or buy/sell recommendation is generated.",
        ],
        "analysis_only": True,
        "ppo_training_executed": False,
        "trades_executed": False,
        "consolidated_output": str(consolidated_path),
    }

    manifest_path = output_path / "india_unified_intelligence_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "observation_count": int(len(unified)),
        "symbols": manifest["availability"]["symbols"],
        "symbols_with_fundamentals": manifest["availability"]["symbols_with_fundamentals"],
        "symbols_with_news": manifest["availability"]["symbols_with_news"],
        "consolidated_output": str(consolidated_path),
        "manifest": str(manifest_path),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build unified chronological India investment intelligence"
    )
    parser.add_argument("--market-indicators", default=str(DEFAULT_MARKET_INDICATORS))
    parser.add_argument("--sector-opportunities", default=str(DEFAULT_STOCK_INTELLIGENCE))
    parser.add_argument("--fundamentals", default=str(DEFAULT_FUNDAMENTALS))
    parser.add_argument("--news", default=str(DEFAULT_NEWS))
    parser.add_argument("--market-manifest", default=str(DEFAULT_MARKET_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(
        analyze_unified_india_intelligence(
            market_indicators_csv=args.market_indicators,
            sector_opportunities_csv=args.sector_opportunities,
            fundamentals_csv=args.fundamentals,
            news_csv=args.news,
            market_manifest=args.market_manifest,
            output_dir=args.output_dir,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
