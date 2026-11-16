"""Build local issuer-verified corporate-action audit files; no network access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.issuer_corporate_action_verification import (
    build_issuer_verification_dataset,
    load_local_market_sessions,
)


def main() -> int:
    repository_root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=repository_root / "data/yahoo_corporate_actions/yahoo_action_candidates.csv",
    )
    parser.add_argument("--issuer-workbook", type=Path, required=True)
    parser.add_argument(
        "--prices",
        type=Path,
        default=repository_root / "data/fmp_daily",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "data/issuer_verified_corporate_actions",
    )
    args = parser.parse_args()
    sessions = load_local_market_sessions(args.prices)
    manifest = build_issuer_verification_dataset(
        candidates_path=args.candidates,
        issuer_workbook_path=args.issuer_workbook,
        output_dir=args.output,
        local_price_dir=args.prices,
        market_sessions=sessions,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
