import csv
import hashlib
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.corporate_action_provider import CorporateActionValidationError
from src.data.yahoo_corporate_actions import (
    ISSUER_VERIFICATION_COLUMNS,
    YahooActionCandidate,
    YahooFinanceCorporateActionProvider,
    apply_issuer_verification,
    audit_yahoo_against_local_prices,
    export_yahoo_acquisition,
    validate_yahoo_candidates,
)
from src.strategies.allocation import (
    CorporateActionEvent,
    CorporateActionLedger,
    CorporateActionType,
)

SESSIONS = (
    "2018-01-03",
    "2018-01-04",
    "2018-01-05",
    "2018-01-08",
    "2018-01-09",
    "2018-01-10",
)


def yahoo_frame(
    *,
    dividends=None,
    splits=None,
    capital_gains=None,
    dates=None,
):
    dates = dates or ["2018-01-03", "2018-01-04", "2018-01-05"]
    index = pd.DatetimeIndex(dates, tz="America/New_York", name="Date")
    count = len(index)
    return pd.DataFrame(
        {
            "Open": [100.0] * count,
            "High": [101.0] * count,
            "Low": [99.0] * count,
            "Close": [100.0] * count,
            "Adj Close": [100.0] * count,
            "Volume": [1000] * count,
            "Dividends": dividends or [0.0] * count,
            "Stock Splits": splits or [0.0] * count,
            "Capital Gains": capital_gains or [0.0] * count,
        },
        index=index,
    )


class FakeTicker:
    def __init__(self, symbol, frame, currency="USD"):
        self.symbol = symbol
        self.frame = frame
        self.history_metadata = {"currency": currency}
        self.calls = []

    def history(self, **kwargs):
        self.calls.append(kwargs)
        return self.frame.copy()


def candidate(
    *,
    symbol="SPY",
    event_type=CorporateActionType.ETF_DISTRIBUTION,
    action_date="2018-01-04",
    amount=1.0,
    ratio=None,
    field="Dividends",
    currency="USD",
):
    is_split = event_type is CorporateActionType.SPLIT
    return YahooActionCandidate(
        symbol=symbol,
        event_type=event_type,
        source_id=None,
        currency=currency,
        announcement_date=None,
        ex_date=None if is_split else date.fromisoformat(action_date),
        payment_date=None,
        effective_date=date.fromisoformat(action_date) if is_split else None,
        distribution_per_share=None if is_split else amount,
        split_ratio=ratio if is_split else None,
        source_field="Stock Splits" if is_split else field,
        raw_value=ratio if is_split else amount,
        verification_status="issuer_verification_required",
        missing_fields=("source_id",) if is_split else ("source_id", "payment_date"),
    )


def write_verification(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ISSUER_VERIFICATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class YahooFinanceCorporateActionTests(unittest.TestCase):
    def test_provider_normalizes_actions_without_fabricating_missing_fields(self):
        frame = yahoo_frame(
            dividends=[0.0, 0.25, 0.0],
            splits=[0.0, 0.0, 2.0],
            capital_gains=[0.0, 0.1, 0.0],
        )
        ticker = FakeTicker("SPY", frame)
        provider = YahooFinanceCorporateActionProvider(
            ticker_factory=lambda symbol: ticker
        )

        batch = provider.acquire(
            ("SPY",),
            start_date="2018-01-03",
            end_date="2018-01-05",
        )

        self.assertTrue(batch.complete)
        self.assertEqual(len(batch.candidates), 3)
        by_field = {item.source_field: item for item in batch.candidates}
        distribution = by_field["Dividends"]
        capital_gain = by_field["Capital Gains"]
        split = by_field["Stock Splits"]
        self.assertEqual(distribution.source_id, None)
        self.assertEqual(distribution.payment_date, None)
        self.assertEqual(distribution.announcement_date, None)
        self.assertEqual(distribution.distribution_per_share, 0.25)
        self.assertEqual(distribution.missing_fields, ("source_id", "payment_date"))
        self.assertEqual(capital_gain.source_field, "Capital Gains")
        self.assertEqual(capital_gain.event_type, CorporateActionType.ETF_DISTRIBUTION)
        self.assertEqual(split.source_id, None)
        self.assertEqual(split.split_ratio, 2.0)
        self.assertEqual(distribution.verification_status, "issuer_verification_required")
        self.assertEqual(batch.raw_records[0]["source_id"], None)
        self.assertEqual(batch.outcomes[0].currency, "USD")

        call = ticker.calls[0]
        self.assertEqual(call["start"], "2018-01-03")
        self.assertEqual(call["end"], "2018-01-06")
        self.assertTrue(call["actions"])
        self.assertFalse(call["auto_adjust"])

    def test_empty_or_malformed_yahoo_response_is_not_reported_as_no_events(self):
        provider = YahooFinanceCorporateActionProvider(
            ticker_factory=lambda symbol: FakeTicker("SPY", pd.DataFrame())
        )
        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-05")

        self.assertFalse(batch.complete)
        self.assertEqual(batch.outcomes[0].status, "failed")
        self.assertIn("no price history", batch.outcomes[0].error)

    def test_non_usd_and_invalid_amounts_fail_symbol_query(self):
        non_usd = YahooFinanceCorporateActionProvider(
            ticker_factory=lambda symbol: FakeTicker(
                symbol,
                yahoo_frame(dividends=[0, 1, 0]),
                currency="CAD",
            )
        ).acquire(("SPY",), "2018-01-03", "2018-01-05")
        self.assertFalse(non_usd.complete)
        self.assertIn("unsupported currency", non_usd.outcomes[0].error)

        invalid_amount = YahooFinanceCorporateActionProvider(
            ticker_factory=lambda symbol: FakeTicker(
                symbol,
                yahoo_frame(dividends=[0, -1, 0]),
            )
        ).acquire(("SPY",), "2018-01-03", "2018-01-05")
        self.assertFalse(invalid_amount.complete)
        self.assertIn("negative Dividends", invalid_amount.outcomes[0].error)

    def test_provider_rejects_duplicate_symbols_and_unsupported_universe(self):
        provider = YahooFinanceCorporateActionProvider(
            ticker_factory=lambda symbol: FakeTicker(symbol, yahoo_frame())
        )
        with self.assertRaisesRegex(ValueError, "unsupported asset"):
            provider.acquire(("UNKNOWN",), "2018-01-03", "2018-01-05")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            provider.acquire(("SPY", "SPY"), "2018-01-03", "2018-01-05")

    def test_candidate_validation_checks_dates_amounts_currency_and_duplicates(self):
        valid = candidate()
        checked = validate_yahoo_candidates(
            [valid],
            start_date="2018-01-03",
            end_date="2018-01-10",
            market_sessions=SESSIONS,
        )
        self.assertEqual(checked, (valid,))

        for invalid, message in (
            (candidate(action_date="2018-01-06"), "does not align"),
            (candidate(amount=0), "positive finite"),
            (
                candidate(
                    event_type=CorporateActionType.SPLIT,
                    ratio=0,
                ),
                "positive finite",
            ),
            (candidate(symbol="BAD"), "Unsupported Yahoo action symbol"),
            (candidate(currency="CAD"), "Unsupported Yahoo currency"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                CorporateActionValidationError, message
            ):
                validate_yahoo_candidates(
                    [invalid],
                    start_date="2018-01-03",
                    end_date="2018-01-10",
                    market_sessions=SESSIONS,
                )

        with self.assertRaisesRegex(CorporateActionValidationError, "Duplicate"):
            validate_yahoo_candidates(
                [valid, valid],
                start_date="2018-01-03",
                end_date="2018-01-10",
                market_sessions=SESSIONS,
            )

        with self.assertRaisesRegex(CorporateActionValidationError, "Conflicting"):
            validate_yahoo_candidates(
            [valid, candidate(amount=2.0)],
            start_date="2018-01-03",
            end_date="2018-01-10",
            market_sessions=SESSIONS,
            )

    def test_issuer_verification_supplies_real_fields_for_ledger_events(self):
        dividend = candidate()
        split = candidate(
            event_type=CorporateActionType.SPLIT,
            action_date="2018-01-05",
            ratio=2.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "verification.csv"
            write_verification(
                path,
                [
                    {
                        "symbol": "SPY",
                        "event_type": "etf_distribution",
                        "action_date": "2018-01-04",
                        "source_field": "Dividends",
                        "yahoo_reported_value": "1.0",
                        "source_id": "issuer-dividend-id",
                        "currency": "USD",
                        "distribution_per_share": "1.0",
                        "payment_date": "2018-01-08",
                        "split_ratio": "",
                        "verification_source": "SPY issuer distribution notice",
                        "verification_reference": "https://issuer.example/dividend",
                    },
                    {
                        "symbol": "SPY",
                        "event_type": "split",
                        "action_date": "2018-01-05",
                        "source_field": "Stock Splits",
                        "yahoo_reported_value": "2.0",
                        "source_id": "issuer-split-id",
                        "currency": "USD",
                        "distribution_per_share": "",
                        "payment_date": "",
                        "split_ratio": "2.0",
                        "verification_source": "Issuer split notice",
                        "verification_reference": "https://issuer.example/split",
                    },
                ],
            )
            events = apply_issuer_verification(
                [dividend, split],
                path,
                start_date="2018-01-03",
                end_date="2018-01-10",
                market_sessions=SESSIONS,
            )

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].source_id, "issuer-dividend-id")
        self.assertEqual(events[0].payment_date, date(2018, 1, 8))
        self.assertEqual(events[1].split_ratio, 2.0)

    def test_issuer_verification_rejects_missing_payment_id_or_mismatched_amount(self):
        action = candidate()
        base_row = {
            "symbol": "SPY",
            "event_type": "etf_distribution",
            "action_date": "2018-01-04",
            "source_field": "Dividends",
            "yahoo_reported_value": "1.0",
            "source_id": "issuer-id",
            "currency": "USD",
            "distribution_per_share": "1.0",
            "payment_date": "2018-01-08",
            "split_ratio": "",
            "verification_source": "Issuer document",
            "verification_reference": "https://issuer.example/notice",
        }
        cases = (
            (dict(base_row, source_id=""), "source_id is required"),
            (dict(base_row, payment_date=""), "payment_date"),
            (dict(base_row, distribution_per_share="2.0"), "does not match"),
            (dict(base_row, verification_reference=""), "source and reference"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "verification.csv"
            for row, message in cases:
                write_verification(path, [row])
                with self.subTest(message=message), self.assertRaisesRegex(
                    CorporateActionValidationError, message
                ):
                    apply_issuer_verification(
                        [action],
                        path,
                        start_date="2018-01-03",
                        end_date="2018-01-10",
                        market_sessions=SESSIONS,
                    )

    def test_issuer_verified_events_replay_through_ledger_once(self):
        action = candidate()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "verification.csv"
            write_verification(
                path,
                [
                    {
                        "symbol": "SPY",
                        "event_type": "etf_distribution",
                        "action_date": "2018-01-04",
                        "source_field": "Dividends",
                        "yahoo_reported_value": "1.0",
                        "source_id": "issuer-id",
                        "currency": "USD",
                        "distribution_per_share": "1.0",
                        "payment_date": "2018-01-08",
                        "split_ratio": "",
                        "verification_source": "Issuer",
                        "verification_reference": "https://issuer.example/event",
                    }
                ],
            )
            events = apply_issuer_verification(
                [action],
                path,
                start_date="2018-01-03",
                end_date="2018-01-10",
                market_sessions=SESSIONS,
            )

        ledger = CorporateActionLedger({"SPY": 10.0}, 0.0)
        before = ledger.snapshot({"SPY": 100.0})
        ledger.process_date("2018-01-04", events)
        on_ex_date = ledger.snapshot({"SPY": 99.0})
        ledger.process_date("2018-01-08", events)
        paid = ledger.snapshot({"SPY": 99.0})
        ledger.process_date("2018-01-08", events)
        repeated = ledger.snapshot({"SPY": 99.0})

        self.assertEqual(on_ex_date.dividend_receivables, {"SPY:issuer-id": 10.0})
        self.assertEqual(on_ex_date.total_nav, before.total_nav)
        self.assertEqual(ledger.settled_cash, 10.0)
        self.assertEqual(paid.total_nav, on_ex_date.total_nav)
        self.assertEqual(repeated.total_nav, paid.total_nav)

    def test_same_day_dividend_and_split_preserve_nav_and_entitlement(self):
        dividend = CorporateActionEvent(
            symbol="SPY",
            event_type=CorporateActionType.ETF_DISTRIBUTION,
            source_id="dividend-2018-01-04",
            ex_date="2018-01-04",
            payment_date="2018-01-08",
            distribution_per_share=1.0,
        )
        split = CorporateActionEvent(
            symbol="SPY",
            event_type=CorporateActionType.SPLIT,
            source_id="split-2018-01-04",
            effective_date="2018-01-04",
            split_ratio=2.0,
        )
        ledger = CorporateActionLedger({"SPY": 2.0}, settled_cash=0.0)
        before = ledger.snapshot({"SPY": 100.0})

        ledger.process_date("2018-01-04", [split, dividend])
        ex_date = ledger.snapshot({"SPY": 49.5})
        ledger.process_date("2018-01-04", [split, dividend])
        replayed = ledger.snapshot({"SPY": 49.5})
        ledger.process_date("2018-01-08", [split, dividend])
        paid = ledger.snapshot({"SPY": 49.5})

        self.assertEqual(before.total_nav, 200.0)
        self.assertEqual(ex_date.asset_quantities, {"SPY": 4.0})
        self.assertEqual(ex_date.dividend_receivables, {"SPY:dividend-2018-01-04": 2.0})
        self.assertEqual(ex_date.total_nav, before.total_nav)
        self.assertEqual(replayed.total_nav, ex_date.total_nav)
        self.assertEqual(ledger.settled_cash, 2.0)
        self.assertEqual(paid.dividend_receivables, {})
        self.assertEqual(paid.total_nav, before.total_nav)

    def test_price_audit_compares_adjusted_prices_and_flags_unexplained_jumps(self):
        dates = pd.to_datetime(SESSIONS[:3])
        yahoo = yahoo_frame(dividends=[0.0, 1.0, 0.0], dates=SESSIONS[:3])
        yahoo.index = dates
        yahoo["Close"] = [100.0, 99.0, 100.0]
        yahoo["Adj Close"] = [95.0, 95.0, 97.0]
        provider = YahooFinanceCorporateActionProvider(
            ticker_factory=lambda symbol: FakeTicker("SPY", yahoo)
        )
        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-05")
        local = pd.DataFrame(
            {
                "close": [100.0, 99.0, 100.0],
                "adj_close": [95.0, 95.0, 97.0],
            },
            index=dates,
        )

        audit = audit_yahoo_against_local_prices(
            batch,
            {"SPY": local},
            market_sessions=SESSIONS[:3],
        )

        event = audit["event_diagnostics"][0]
        self.assertEqual(event["status"], "review_required")
        self.assertAlmostEqual(event["relative_factor_difference"], 0.0)
        self.assertEqual(
            audit["possible_unexplained_adjustment_dates"]["SPY"],
            ["2018-01-05"],
        )
        self.assertIn("do not independently prove", audit["interpretation"])

    def test_split_audit_does_not_confuse_split_ratio_with_adjustment_factor(self):
        dates = pd.to_datetime(SESSIONS[:3])
        yahoo = yahoo_frame(splits=[0.0, 2.0, 0.0], dates=SESSIONS[:3])
        yahoo.index = dates
        prices = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0],
                "adj_close": [95.0, 95.0, 95.0],
            },
            index=dates,
        )
        batch = YahooFinanceCorporateActionProvider(
            ticker_factory=lambda symbol: FakeTicker("XLU", yahoo)
        ).acquire(("XLU",), "2018-01-03", "2018-01-05")

        audit = audit_yahoo_against_local_prices(
            batch,
            {"XLU": prices},
            market_sessions=SESSIONS[:3],
        )
        event = audit["event_diagnostics"][0]

        self.assertEqual(event["expected_adjustment_factor_ratio"], None)
        self.assertEqual(event["relative_factor_difference"], None)
        self.assertEqual(event["expected_unadjusted_raw_close_ratio"], 0.5)
        self.assertIn("price series may be split-adjusted", event["split_price_basis_assessment"])

    def test_export_creates_candidates_template_raw_audit_and_manifest(self):
        frame = yahoo_frame(dividends=[0, 0.5, 0])
        ticker = FakeTicker("SPY", frame)
        batch = YahooFinanceCorporateActionProvider(
            ticker_factory=lambda symbol: ticker
        ).acquire(("SPY",), "2018-01-03", "2018-01-05")
        local = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0],
                "adj_close": [95.0, 95.5, 95.5],
            },
            index=pd.to_datetime(SESSIONS[:3]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = export_yahoo_acquisition(
                batch,
                temp_dir,
                market_sessions=SESSIONS,
                local_price_data={"SPY": local},
            )
            root = Path(temp_dir)
            candidates_path = root / "yahoo_action_candidates.csv"
            raw_path = root / "yahoo_action_raw.jsonl"
            template_path = root / "issuer_verification_template.csv"
            manifest_path = root / "manifest.json"
            self.assertEqual(manifest["per_symbol"]["SPY"]["dividend_records"], 1)
            self.assertEqual(
                manifest["per_symbol"]["SPY"]["missing_payment_date_count"], 1
            )
            self.assertEqual(
                manifest["per_symbol"]["SPY"]["missing_source_identifier_count"], 1
            )
            self.assertEqual(manifest["validation"]["ledger_ready_event_count"], 0)
            self.assertTrue(manifest["coverage_status"].startswith("Yahoo history"))
            self.assertEqual(
                manifest["artifacts"]["yahoo_action_candidates.csv"]["sha256"],
                hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
            )
            with candidates_path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["source_id"], "")
            self.assertEqual(row["payment_date"], "")
            self.assertIn("payment_date", row["missing_fields"])
            with template_path.open(encoding="utf-8", newline="") as handle:
                verification = next(csv.DictReader(handle))
            self.assertEqual(verification["source_id"], "")
            self.assertIn("SPY", raw_path.read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                manifest,
            )


if __name__ == "__main__":
    unittest.main()
