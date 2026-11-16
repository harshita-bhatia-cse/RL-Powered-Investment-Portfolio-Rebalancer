"""Acquire Yahoo Finance corporate-action candidates for local review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.corporate_action_provider import ASSET_UNIVERSE  # noqa: E402
from src.data.yahoo_corporate_actions import (  # noqa: E402
    YahooCorporateActionError,
    YahooFinanceCorporateActionProvider,
    export_yahoo_acquisition,
)

DEFAULT_START = "2018-01-03"
DEFAULT_END = "2026-11-11"
SESSION_SYMBOLS = ASSET_UNIVERSE + ("^GSPC", "^VIX")


def _read_market_data(
    data_dir: Path,
) -> tuple[tuple[str, ...], dict[str, pd.DataFrame]]:
    common_dates: set[pd.Timestamp] | None = None
    local_prices: dict[str, pd.DataFrame] = {}
    for symbol in SESSION_SYMBOLS:
        path = data_dir / f"{symbol}_daily.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required market-data file not found: {path}")
        frame = pd.read_csv(path, parse_dates=["date"])
        if frame["date"].isna().any():
            raise ValueError(f"Invalid date values in {path}")
        dates = set(pd.DatetimeIndex(frame["date"]).normalize())
        common_dates = dates if common_dates is None else common_dates & dates
        if symbol in ASSET_UNIVERSE:
            missing = {"close", "adj_close"} - set(frame.columns)
            if missing:
                raise ValueError(
                    f"{path} lacks required local price columns: "
                    + ", ".join(sorted(missing))
                )
            local_prices[symbol] = frame[["date", "close", "adj_close"]].copy()
    if not common_dates:
        raise ValueError("The fixed universe has no common recorded market sessions")
    return tuple(day.date().isoformat() for day in sorted(common_dates)), local_prices


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Yahoo Finance action candidates for the fixed ETF universe. "
            "Results remain unverified and are not directly ledger-ready."
        )
    )
    parser.add_argument(
        "--confirm-express-prior-permission",
        action="store_true",
        help=(
            "Confirm that you have express prior permission for automated Yahoo "
            "Finance access and local retention."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "fmp_daily",
        help="Existing local daily CSV directory for sessions and price comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "yahoo_corporate_actions",
        help="Local output directory (covered by the repository data ignore rule).",
    )
    parser.add_argument(
        "--issuer-verification-csv",
        type=Path,
        help="Optional completed issuer verification CSV; never generated automatically.",
    )
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    args = parser.parse_args()

    if not args.confirm_express_prior_permission:
        parser.error(
            "Yahoo's terms restrict automated data collection without express "
            "prior permission. Do not acquire unless you have that permission; "
            "then pass --confirm-express-prior-permission."
        )

    try:
        sessions, local_prices = _read_market_data(args.data_dir)
        batch = YahooFinanceCorporateActionProvider().acquire(
            ASSET_UNIVERSE,
            args.start_date,
            args.end_date,
        )
        if not batch.complete:
            for outcome in batch.outcomes:
                if outcome.status != "success":
                    print(
                        f"{outcome.symbol}: {outcome.status}: "
                        f"{outcome.error or 'unspecified provider failure'}",
                        file=sys.stderr,
                    )
            print(
                "Acquisition incomplete; no candidate dataset or manifest was written.",
                file=sys.stderr,
            )
            return 2
        manifest = export_yahoo_acquisition(
            batch,
            args.output_dir,
            market_sessions=sessions,
            local_price_data=local_prices,
            issuer_verification_csv=args.issuer_verification_csv,
        )
    except (OSError, ValueError, YahooCorporateActionError) as exc:
        print(f"Acquisition/validation failed: {exc}", file=sys.stderr)
        return 3

    for symbol in ASSET_UNIVERSE:
        summary = manifest["per_symbol"][symbol]
        history = summary["price_history"]
        print(
            f"{symbol}: dividends={summary['dividend_records']}, "
            f"capital-gain distributions="
            f"{summary['capital_gain_distribution_records']}, "
            f"splits={summary['split_records']}, "
            f"price dates={history['start']}..{history['end']}, "
            f"missing payment dates={summary['missing_payment_date_count']}, "
            f"missing source IDs={summary['missing_source_identifier_count']}"
        )
    print(f"Local artifacts written to {args.output_dir}")
    print(
        "Coverage status: Yahoo response retrieval completed; historical event "
        "completeness and issuer verification are not established."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
