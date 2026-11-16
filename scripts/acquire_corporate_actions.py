"""Acquire Massive corporate-action records after licensing and access review."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.corporate_action_provider import (  # noqa: E402
    ASSET_UNIVERSE,
    CorporateActionProviderError,
    MassiveCorporateActionProvider,
    export_acquisition,
)

DEFAULT_START = "2018-01-03"
DEFAULT_END = "2026-11-11"
SESSION_SYMBOLS = ASSET_UNIVERSE + ("^GSPC", "^VIX")


def _market_sessions(data_dir: Path) -> tuple[pd.Timestamp, ...]:
    common_dates: set[pd.Timestamp] | None = None
    for symbol in SESSION_SYMBOLS:
        path = data_dir / f"{symbol}_daily.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required market-data file not found: {path}")
        frame = pd.read_csv(path, usecols=["date"], parse_dates=["date"])
        dates = set(pd.DatetimeIndex(frame["date"]).normalize())
        common_dates = dates if common_dates is None else common_dates & dates
    if not common_dates:
        raise ValueError("The fixed universe has no common recorded market sessions")
    return tuple(sorted(common_dates))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and validate historical Massive corporate actions. "
            "No provider request is made unless licensing is explicitly confirmed."
        )
    )
    parser.add_argument(
        "--confirm-licensed-use",
        action="store_true",
        help="Confirm that your account permits acquisition and local retention.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "fmp_daily",
        help="Existing local daily CSV directory used for trading-session alignment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "corporate_actions",
        help="Output directory (ignored by the repository's existing data rule).",
    )
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    args = parser.parse_args()

    if not args.confirm_licensed_use:
        parser.error(
            "Before acquisition, review provider access and licensing, then pass "
            "--confirm-licensed-use."
        )
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        parser.error("Set MASSIVE_API_KEY in the process environment; it is not logged.")

    sessions = _market_sessions(args.data_dir)
    provider = MassiveCorporateActionProvider(api_key)
    batch = provider.acquire(ASSET_UNIVERSE, args.start_date, args.end_date)
    if not batch.complete:
        for outcome in batch.outcomes:
            if outcome.status not in {"success", "empty"}:
                print(
                    f"{outcome.symbol} {outcome.category}: "
                    f"{outcome.status}: {outcome.error}",
                    file=sys.stderr,
                )
        print(
            "Acquisition incomplete; no dataset or manifest was written.",
            file=sys.stderr,
        )
        return 2

    try:
        manifest = export_acquisition(
            batch,
            args.output_dir,
            market_sessions=sessions,
        )
    except (CorporateActionProviderError, ValueError) as exc:
        print(f"Validation/export failed: {exc}", file=sys.stderr)
        return 3

    for symbol, summary in manifest["per_symbol"].items():
        print(
            f"{symbol}: dividends={summary['dividend_records']}, "
            f"splits={summary['split_records']}, "
            f"status={summary['query_status']}"
        )
    print(f"Dataset written to {args.output_dir}")
    print(
        "Coverage status: "
        "queries complete; provider event coverage not independently verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
