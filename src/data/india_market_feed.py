"""Reusable backend service for updating Indian market data.

This service only fetches and stores data. It does not create recommendations,
place orders or start PPO training.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd
from dotenv import load_dotenv

from src.data.india_market_provider import (
    IndiaMarketProviderResponse,
    IndiaMarketSymbol,
    download_india_etf_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "india_daily"

# Load the repository .env before any provider or symbol configuration is read.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


def _symbols_from_environment() -> tuple[IndiaMarketSymbol, ...]:
    use_universe = os.environ.get(
        "INDIA_USE_STOCK_UNIVERSE",
        "false",
    ).strip().lower() in ("1", "true", "yes")
    if use_universe:
        sector = os.environ.get("INDIA_STOCK_UNIVERSE_SECTOR", "").strip()
        from src.data.india_stock_universe import india_stock_universe_symbols

        values = (
            india_stock_universe_symbols(sector or None)
            if sector
            else india_stock_universe_symbols()
        )
        if not values:
            raise ValueError("configured India stock universe contains no symbols")
        return tuple(IndiaMarketSymbol(value) for value in values)

    raw = os.environ.get("INDIA_DATA_SYMBOLS", "").strip()
    for separator in (";", "|"):
        raw = raw.replace(separator, ",")
    values = tuple(
        value.strip()
        for value in raw.split(",")
        if value.strip()
    )
    if not values:
        raise ValueError(
            "INDIA_DATA_SYMBOLS must contain at least one NSE/BSE symbol"
        )
    provider_names = {
        "yahoo",
        "yahoo-finance",
        "yahoo-finance-yfinance",
        "yfinance",
        "alpha-vantage",
        "alphavantage",
        "alpha_vantage",
    }
    invalid = [
        value
        for value in values
        if value.lower().removesuffix(".ns").removesuffix(".bo") in provider_names
    ]
    if invalid:
        raise ValueError(
            "INDIA_DATA_PROVIDER must not be used as a stock symbol; "
            "INDIA_DATA_SYMBOLS contains: " + ", ".join(invalid)
        )
    return tuple(IndiaMarketSymbol(value) for value in values)


def _latest_daily_date(output_dir: Path) -> date | None:
    candidates = (
        output_dir / "validated" / "indian_etf_validated.csv",
        output_dir / "raw" / "india_etf_provider_combined.csv",
    )
    latest: date | None = None
    for path in candidates:
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "Date" not in frame.columns:
            continue
        parsed = pd.to_datetime(frame["Date"], errors="coerce").dropna()
        if parsed.empty:
            continue
        candidate = parsed.max().date()
        latest = candidate if latest is None else max(latest, candidate)
    return latest


def _latest_intraday_timestamp(output_dir: Path, interval: str) -> date | None:
    path = output_dir / "live" / f"india_etf_{interval}.csv"
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
        parsed = pd.to_datetime(frame["Date"], errors="coerce").dropna()
    except Exception:
        return None
    return parsed.max().date() if not parsed.empty else None


def update_indian_market_feed(
    *,
    symbols: Sequence[IndiaMarketSymbol | str] | None = None,
    interval: str | None = None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    output_dir: str | Path | None = None,
    backfill: bool | None = None,
) -> dict[str, object]:
    """Fetch once and merge normalized Indian market data.

    Incremental mode remains the default. Backfill mode is opt-in and always
    requests the full configured start/end range for every selected symbol.
    """
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    selected_symbols = (
        tuple(symbols)
        if symbols is not None
        else _symbols_from_environment()
    )
    selected_interval = str(
        interval or os.environ.get("INDIA_DATA_INTERVAL", "1d")
    ).strip().lower()
    selected_output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )
    if backfill is None:
        backfill = os.environ.get(
            "INDIA_DATA_BACKFILL",
            "false",
        ).strip().lower() in ("1", "true", "yes")
    if not isinstance(backfill, bool):
        raise ValueError("backfill must be a boolean")

    today_india = pd.Timestamp.now(tz="Asia/Kolkata").date()
    configured_end = os.environ.get("INDIA_DATA_END_DATE", "").strip()
    selected_end = (
        date.fromisoformat(end_date)
        if isinstance(end_date, str)
        else end_date or (
            date.fromisoformat(configured_end)
            if configured_end
            else today_india
        )
    )

    if backfill:
        configured_start = os.environ.get("INDIA_DATA_START_DATE", "").strip()
        backfill_start = (
            date.fromisoformat(start_date)
            if isinstance(start_date, str)
            else start_date or (
                date.fromisoformat(configured_start)
                if configured_start
                else None
            )
        )
        if backfill_start is None:
            raise ValueError(
                "Backfill mode requires INDIA_DATA_START_DATE or start_date"
            )
        selected_start = backfill_start
    elif start_date is not None:
        selected_start = (
            date.fromisoformat(start_date)
            if isinstance(start_date, str)
            else start_date
        )
    else:
        latest = (
            _latest_intraday_timestamp(selected_output, selected_interval)
            if selected_interval != "1d"
            else _latest_daily_date(selected_output)
        )
        if latest is not None:
            selected_start = latest + timedelta(days=1)
        else:
            configured_start = os.environ.get(
                "INDIA_DATA_START_DATE",
                (today_india - timedelta(days=365)).isoformat(),
            )
            selected_start = date.fromisoformat(configured_start)

    if selected_start > selected_end:
        return {
            "updated": False,
            "reason": "The normalized dataset is already current.",
            "interval": selected_interval,
            "data_freshness": (
                "historical" if selected_interval == "1d"
                else "delayed_or_unverified"
            ),
            "symbols": [
                symbol.symbol if isinstance(symbol, IndiaMarketSymbol) else str(symbol)
                for symbol in selected_symbols
            ],
            "retrieval_timestamp": None,
        }

    report = download_india_etf_data(
        symbols=selected_symbols,
        start_date=selected_start,
        end_date=selected_end,
        output_dir=selected_output,
        interval=selected_interval,
        backfill=backfill,
        license_status=os.environ.get(
            "INDIA_DATA_LICENSE_STATUS",
            "unverified",
        ),
    )
    report["updated"] = True
    report["interval"] = selected_interval
    report["requested_start_date"] = selected_start.isoformat()
    report["requested_end_date"] = selected_end.isoformat()
    report["timezone"] = "Asia/Kolkata"
    report["feed_mode"] = "backfill" if backfill else "incremental"

    if selected_interval == "1d":
        _save_daily_rows_to_existing_store(report, selected_output)
    return report


def _save_daily_rows_to_existing_store(
    report: dict[str, object],
    output_dir: Path,
) -> None:
    """Append normalized India daily rows to the existing SQLite price store."""
    csv_path = Path(str(report.get("validated_output", "")))
    if not csv_path.is_file():
        return
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return
    store_frame = pd.DataFrame({
        "ticker": frame["Symbol"].astype(str),
        "datadate": pd.to_datetime(frame["Date"], errors="coerce"),
        "prcod": pd.to_numeric(frame["Open"], errors="coerce"),
        "prchd": pd.to_numeric(frame["High"], errors="coerce"),
        "prcld": pd.to_numeric(frame["Low"], errors="coerce"),
        "prccd": pd.to_numeric(frame["Close"], errors="coerce"),
        "adj_close": pd.to_numeric(frame["Adjusted Close"], errors="coerce"),
        "cshtrd": pd.to_numeric(frame["Volume"], errors="coerce"),
    }).dropna()
    if store_frame.empty:
        return
    try:
        from src.data.data_store import DataStore

        store = DataStore(str(output_dir.parent))
        store.save_price_data(store_frame)
    except Exception:
        # CSV remains the primary normalized feed artifact if SQLite is unavailable.
        return


def run_indian_market_feed(
    *,
    once: bool = True,
    refresh_seconds: int | None = None,
) -> dict[str, object]:
    """Run one update or repeatedly update according to configuration."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    interval_seconds = (
        refresh_seconds
        if refresh_seconds is not None
        else int(os.environ.get("INDIA_DATA_REFRESH_SECONDS", "0"))
    )
    if interval_seconds < 0:
        raise ValueError("INDIA_DATA_REFRESH_SECONDS must be non-negative")

    result = update_indian_market_feed()
    if once or interval_seconds == 0:
        return result
    while True:
        time.sleep(interval_seconds)
        result = update_indian_market_feed()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and update normalized Indian market data"
    )
    parser.add_argument("--once", action="store_true", default=True)
    parser.add_argument(
        "--interval",
        choices=("1d", "1min", "5min", "15min", "30min", "60min"),
        default=None,
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    result = update_indian_market_feed(
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    )
    safe_result = dict(result)
    safe_result.pop("raw_payloads", None)
    print(safe_result)


if __name__ == "__main__":
    main()
