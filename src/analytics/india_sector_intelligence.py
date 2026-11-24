"""Offline India sector performance and explainable stock-opportunity screening."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDICATORS = (
    PROJECT_ROOT
    / "data"
    / "india_daily"
    / "analysis"
    / "india_market_indicators.csv"
)
DEFAULT_STOCK_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "india_daily"
    / "analysis"
    / "india_stock_summary.csv"
)
DEFAULT_METADATA = Path(__file__).resolve().with_name("india_sector_metadata.json")
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "india_daily" / "sector_analysis"


def load_sector_metadata(
    metadata_path: str | Path = DEFAULT_METADATA,
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    """Load explicit symbol-to-sector metadata; never infer a sector from a name."""
    path = Path(metadata_path)
    if not path.is_file():
        return {}, {
            "mapping_version": "unavailable",
            "mapping_source": "metadata file not found",
            "missing_metadata": True,
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    sectors = payload.get("sectors", {})
    from src.data.india_stock_universe import universe_to_sector_metadata

    universe_sectors = universe_to_sector_metadata()
    for symbol, details in universe_sectors.items():
        sectors.setdefault(symbol, details)
    if not isinstance(sectors, dict):
        return {}, {
            "mapping_version": str(payload.get("mapping_version", "unavailable")),
            "mapping_source": str(payload.get("mapping_source", "unknown")),
            "missing_metadata": True,
        }

    normalized: dict[str, dict[str, str]] = {}
    for symbol, details in sectors.items():
        if not isinstance(details, dict):
            continue
        sector = str(details.get("sector", "")).strip()
        industry = str(details.get("industry", "")).strip()
        if symbol and sector:
            normalized[str(symbol).strip().upper()] = {
                "Sector": sector,
                "Industry": industry or "not specified",
            }
    return normalized, {
        "mapping_version": str(payload.get("mapping_version", "unavailable")),
        "mapping_source": str(payload.get("mapping_source", "unknown")),
        "missing_metadata": False,
    }


def calculate_sector_intelligence(
    indicators: pd.DataFrame,
    stock_summary: pd.DataFrame,
    sector_metadata: Mapping[str, Mapping[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate sector metrics and an explainable stock opportunity screen."""
    indicators_required = {
        "Date",
        "Symbol",
        "Adjusted Close",
        "Daily_Return",
        "Momentum_20",
        "Rolling_Volatility_20",
    }
    missing = sorted(indicators_required - set(indicators.columns))
    if missing:
        raise ValueError(
            "India indicator input is missing required columns: " + ", ".join(missing)
        )

    indicators = indicators.copy()
    indicators["Date"] = pd.to_datetime(indicators["Date"], errors="coerce")
    indicators["Symbol"] = indicators["Symbol"].astype("string").str.strip().str.upper()
    if indicators["Date"].isna().any():
        raise ValueError("India indicator input contains invalid dates")

    available_symbols = set(indicators["Symbol"].dropna().tolist())
    if "Symbol" in stock_summary.columns:
        available_symbols.update(
            str(symbol).strip().upper()
            for symbol in stock_summary["Symbol"].dropna().tolist()
        )
    mapped_symbols = set(sector_metadata)
    missing_mappings = sorted(available_symbols - mapped_symbols)

    missing_rows = [
        {
            "Symbol": symbol,
            "Sector": "unavailable",
            "Industry": "unavailable",
            "Status": "explicit sector mapping unavailable",
        }
        for symbol in missing_mappings
    ]
    missing_frame = pd.DataFrame(
        missing_rows,
        columns=["Symbol", "Sector", "Industry", "Status"],
    )

    mapped = indicators[
        indicators["Symbol"].isin(mapped_symbols)
    ].copy()
    mapped["Sector"] = mapped["Symbol"].map(
        lambda symbol: sector_metadata[symbol]["Sector"]
    )
    mapped["Industry"] = mapped["Symbol"].map(
        lambda symbol: sector_metadata[symbol]["Industry"]
    )

    sector_rows: list[dict[str, object]] = []
    for sector, sector_frame in mapped.groupby("Sector", sort=True):
        sector_symbols = sorted(sector_frame["Symbol"].unique().tolist())
        daily_pivot = sector_frame.pivot_table(
            index="Date",
            columns="Symbol",
            values="Daily_Return",
            aggfunc="mean",
        )
        sector_daily = daily_pivot.mean(axis=1, skipna=True).dropna()

        if len(sector_daily) >= 20:
            wealth = (1.0 + sector_daily).cumprod()
            sector_return_20 = float(wealth.iloc[-1] / wealth.iloc[-21] - 1.0) if len(wealth) >= 21 else np.nan
            sector_volatility_20 = float(
                sector_daily.tail(20).std(ddof=1) * np.sqrt(252.0)
            ) if sector_daily.tail(20).count() > 1 else np.nan
            period_return = float(wealth.iloc[-1] - 1.0)
        else:
            sector_return_20 = np.nan
            sector_volatility_20 = np.nan
            period_return = np.nan

        latest = sector_frame.sort_values("Date").groupby("Symbol").tail(1)
        momentum_values = pd.to_numeric(
            latest["Momentum_20"],
            errors="coerce",
        ).dropna()
        sector_momentum_20 = (
            float(momentum_values.mean())
            if len(momentum_values) == len(sector_symbols)
            else np.nan
        )
        sector_volatility = sector_volatility_20

        sector_rows.append({
            "Sector": str(sector),
            "Industry": str(sector_metadata[sector_symbols[0]]["Industry"]),
            "Mapped_Stock_Count": int(len(sector_symbols)),
            "Mapped_Symbols": ",".join(sector_symbols),
            "Observation_Count": int(len(sector_frame)),
            "Period_Return": period_return,
            "Sector_Return_20": sector_return_20,
            "Sector_Momentum_20": sector_momentum_20,
            "Sector_Volatility_20": sector_volatility,
            "Evidence_Status": (
                "sufficient evidence"
                if len(sector_daily) >= 20 and pd.notna(sector_momentum_20)
                else "insufficient evidence"
            ),
        })

    sector_frame = pd.DataFrame(
        sector_rows,
        columns=[
            "Sector",
            "Industry",
            "Mapped_Stock_Count",
            "Mapped_Symbols",
            "Observation_Count",
            "Period_Return",
            "Sector_Return_20",
            "Sector_Momentum_20",
            "Sector_Volatility_20",
            "Evidence_Status",
        ],
    )

    if sector_frame.empty:
        sector_frame["Relative_Performance"] = pd.Series(dtype=float)
        sector_frame["Rank_Score"] = pd.Series(dtype=float)
        sector_frame["Sector_Rank"] = pd.Series(dtype=int)
    else:
        evidence_momentum = sector_frame["Sector_Momentum_20"].dropna()
        cross_sector_median = (
            float(evidence_momentum.median())
            if not evidence_momentum.empty
            else np.nan
        )
        sector_frame["Relative_Performance"] = (
            sector_frame["Sector_Momentum_20"] - cross_sector_median
        )
        sector_frame["Rank_Score"] = (
            0.70 * sector_frame["Sector_Momentum_20"].fillna(0.0)
            + 0.30 * sector_frame["Relative_Performance"].fillna(0.0)
        )
        sector_frame["Sector_Rank"] = (
            sector_frame["Rank_Score"].rank(method="min", ascending=False)
            .astype("Int64")
        )
        sector_frame = sector_frame.sort_values(
            ["Sector_Rank", "Sector"],
            na_position="last",
        ).reset_index(drop=True)

    summary_required = {
        "Symbol",
        "Observation_Count",
        "Trend",
        "Momentum_20",
        "Rolling_Volatility_20",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
    }
    missing_summary = sorted(summary_required - set(stock_summary.columns))
    if missing_summary:
        raise ValueError(
            "India stock summary is missing required columns: "
            + ", ".join(missing_summary)
        )

    opportunity_rows: list[dict[str, object]] = []
    for row in stock_summary.to_dict("records"):
        symbol = str(row["Symbol"]).strip().upper()
        mapping = sector_metadata.get(symbol)
        sector = mapping["Sector"] if mapping else "unavailable"
        industry = mapping["Industry"] if mapping else "unavailable"

        momentum = pd.to_numeric(pd.Series([row["Momentum_20"]]), errors="coerce").iloc[0]
        volatility = pd.to_numeric(
            pd.Series([row["Rolling_Volatility_20"]]),
            errors="coerce",
        ).iloc[0]
        rsi14 = pd.to_numeric(pd.Series([row["RSI_14"]]), errors="coerce").iloc[0]
        macd = pd.to_numeric(pd.Series([row["MACD"]]), errors="coerce").iloc[0]
        macd_signal = pd.to_numeric(
            pd.Series([row["MACD_Signal"]]),
            errors="coerce",
        ).iloc[0]
        macd_histogram = pd.to_numeric(
            pd.Series([row["MACD_Histogram"]]),
            errors="coerce",
        ).iloc[0]
        observation_count = int(row["Observation_Count"])
        trend = str(row["Trend"]).strip().lower()

        sector_row = (
            sector_frame.loc[sector_frame["Sector"] == sector].iloc[0]
            if sector != "unavailable"
            and not sector_frame.loc[sector_frame["Sector"] == sector].empty
            else None
        )
        sector_momentum = (
            float(sector_row["Sector_Momentum_20"])
            if sector_row is not None
            and pd.notna(sector_row["Sector_Momentum_20"])
            else np.nan
        )
        sector_relative = (
            float(sector_row["Relative_Performance"])
            if sector_row is not None
            and pd.notna(sector_row["Relative_Performance"])
            else np.nan
        )

        required_values = (
            momentum,
            volatility,
            rsi14,
            macd,
            macd_signal,
            macd_histogram,
            sector_momentum,
        )
        if (
            observation_count < 50
            or mapping is None
            or any(pd.isna(value) for value in required_values)
            or trend == "insufficient history"
        ):
            signal = "insufficient evidence"
        elif (
            momentum > 0
            and sector_momentum > 0
            and sector_relative >= 0
            and trend in ("upward", "strong upward")
            and macd_histogram > 0
            and rsi14 < 70
        ):
            signal = "positive momentum"
        elif (
            momentum < 0
            or sector_momentum < 0
            or trend in ("downward", "strong downward")
            or macd_histogram < 0
        ):
            signal = "weakening trend"
        else:
            signal = "mixed or neutral evidence"

        opportunity_rows.append({
            "Symbol": symbol,
            "Sector": sector,
            "Industry": industry,
            "Observation_Count": observation_count,
            "Trend": trend,
            "Momentum_20": momentum,
            "Rolling_Volatility_20": volatility,
            "RSI_14": rsi14,
            "MACD_Histogram": macd_histogram,
            "Sector_Momentum_20": sector_momentum,
            "Sector_Relative_Performance": sector_relative,
            "Explainable_Signal": signal,
            "Recommendation_Status": "analysis only; not a buy/sell recommendation",
        })

    opportunity_frame = pd.DataFrame(
        opportunity_rows,
        columns=[
            "Symbol",
            "Sector",
            "Industry",
            "Observation_Count",
            "Trend",
            "Momentum_20",
            "Rolling_Volatility_20",
            "RSI_14",
            "MACD_Histogram",
            "Sector_Momentum_20",
            "Sector_Relative_Performance",
            "Explainable_Signal",
            "Recommendation_Status",
        ],
    )
    return sector_frame, opportunity_frame, missing_frame


def analyze_india_sector_intelligence(
    *,
    indicators_csv: str | Path = DEFAULT_INDICATORS,
    stock_summary_csv: str | Path = DEFAULT_STOCK_SUMMARY,
    metadata_path: str | Path = DEFAULT_METADATA,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Reuse India analysis outputs and write sector intelligence CSVs."""
    indicators_path = Path(indicators_csv)
    summary_path = Path(stock_summary_csv)
    if not indicators_path.is_file():
        raise FileNotFoundError(
            f"India indicator output not found: {indicators_path}"
        )
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"India stock summary output not found: {summary_path}"
        )

    indicators = pd.read_csv(indicators_path)
    stock_summary = pd.read_csv(summary_path)
    sector_metadata, metadata_details = load_sector_metadata(metadata_path)
    sector, opportunity, missing = calculate_sector_intelligence(
        indicators,
        stock_summary,
        sector_metadata,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sector_path = output_path / "india_sector_performance.csv"
    opportunity_path = output_path / "india_stock_opportunities.csv"
    missing_path = output_path / "india_missing_sector_mappings.csv"

    sector.to_csv(sector_path, index=False)
    opportunity.to_csv(opportunity_path, index=False)
    missing.to_csv(missing_path, index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
        "timezone": "Asia/Kolkata",
        "inputs": {
            "indicators_csv": str(indicators_path),
            "stock_summary_csv": str(summary_path),
            "sector_metadata": str(Path(metadata_path)),
        },
        "metadata": metadata_details,
        "calculations": {
            "sector_daily_return": "Equal-weight mean of mapped constituent daily adjusted-close returns",
            "sector_period_return": "Compounded sector constituent equal-weight daily returns",
            "sector_return_20": "Compounded sector daily return over the latest 20 observations",
            "sector_momentum_20": "Mean of latest mapped constituent Momentum_20 values",
            "sector_volatility_20": "Annualized standard deviation of latest 20 equal-weight sector daily returns",
            "relative_performance": "Sector Momentum_20 minus median Momentum_20 across sectors",
            "rank_score": "0.70 * Sector_Momentum_20 + 0.30 * Relative_Performance",
            "ranking": "Descending Rank_Score",
            "opportunity_evidence": "Existing trend, momentum, RSI, MACD histogram and sector momentum",
        },
        "outputs": {
            "sector_performance": str(sector_path),
            "stock_opportunities": str(opportunity_path),
            "missing_sector_mappings": str(missing_path),
        },
        "limitations": [
            "Sector mappings are explicit and must be independently verified.",
            "No external sector database or paid API is used.",
            "Sector and stock signals are descriptive, not guaranteed return forecasts.",
            "Insufficient observations produce insufficient-evidence labels.",
            "These outputs are not buy/sell recommendations.",
        ],
        "analysis_only": True,
        "ppo_training_executed": False,
        "trades_executed": False,
    }
    manifest_path = output_path / "india_sector_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "sector_count": int(len(sector)),
        "stock_count": int(len(opportunity)),
        "missing_mapping_count": int(len(missing)),
        "sector_performance": str(sector_path),
        "stock_opportunities": str(opportunity_path),
        "missing_sector_mappings": str(missing_path),
        "manifest": str(manifest_path),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze India sectors and describe stock opportunities"
    )
    parser.add_argument("--indicators", default=str(DEFAULT_INDICATORS))
    parser.add_argument("--stock-summary", default=str(DEFAULT_STOCK_SUMMARY))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(
        analyze_india_sector_intelligence(
            indicators_csv=args.indicators,
            stock_summary_csv=args.stock_summary,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
