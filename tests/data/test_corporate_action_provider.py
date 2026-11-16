import csv
import hashlib
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import requests

from src.data.corporate_action_provider import (
    AcquisitionBatch,
    CorporateActionValidationError,
    MassiveCorporateActionProvider,
    QueryOutcome,
    export_acquisition,
    load_normalized_events,
    validate_events,
)
from src.strategies.allocation import (
    CorporateActionEvent,
    CorporateActionLedger,
    CorporateActionType,
)

DATE_SESSIONS = (
    "2018-01-03",
    "2018-01-04",
    "2018-01-05",
    "2018-01-08",
    "2018-01-09",
    "2018-01-10",
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params=None, timeout=None, allow_redirects=None):
        self.calls.append(
            {
                "url": url,
                "params": params,
                "timeout": timeout,
                "allow_redirects": allow_redirects,
            }
        )
        if not self.responses:
            raise AssertionError("Unexpected provider HTTP request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def provider_with_responses(responses, *, sleeps=None, **kwargs):
    session = FakeSession(responses)
    provider = MassiveCorporateActionProvider(
        "test-only-key",
        session=session,
        backoff_seconds=0,
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
        **kwargs,
    )
    return provider, session


def empty_page():
    return FakeResponse(payload={"status": "OK", "results": []})


def successful_batch(events, *, outcomes=None, raw_records=(), duplicates=0):
    symbol = events[0].symbol if events else "SPY"
    if outcomes is None:
        outcomes = (
            QueryOutcome(symbol, "dividend", "/stocks/v1/dividends", "success", 1, 1),
            QueryOutcome(symbol, "split", "/stocks/v1/splits", "empty", 1, 0),
        )
    return AcquisitionBatch(
        provider="Massive",
        requested_symbols=(symbol,),
        start_date="2018-01-03",
        end_date="2018-01-10",
        events=tuple(events),
        raw_records=tuple(raw_records),
        outcomes=tuple(outcomes),
        exact_duplicate_count=duplicates,
        retrieved_at="2026-11-12T00:00:00+00:00",
    )


class MassiveCorporateActionProviderTests(unittest.TestCase):
    def test_dividend_normalization_uses_cash_amount_and_preserves_dates(self):
        dividend = {
            "id": "div-1",
            "ticker": "SPY",
            "currency": "USD",
            "declaration_date": "2018-01-03",
            "ex_dividend_date": "2018-01-04",
            "pay_date": "2018-01-08",
            "cash_amount": 1.25,
            "split_adjusted_cash_amount": 2.50,
        }
        provider, _ = provider_with_responses(
            [
                FakeResponse(payload={"status": "OK", "results": [dividend]}),
                empty_page(),
            ]
        )

        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")

        self.assertTrue(batch.complete)
        self.assertEqual(len(batch.events), 1)
        event = batch.events[0]
        self.assertEqual(event.event_type, CorporateActionType.ETF_DISTRIBUTION)
        self.assertEqual(event.source_id, "div-1")
        self.assertEqual(event.announcement_date, date(2018, 1, 3))
        self.assertEqual(event.ex_date, date(2018, 1, 4))
        self.assertEqual(event.payment_date, date(2018, 1, 8))
        self.assertEqual(event.distribution_per_share, 1.25)
        self.assertEqual(event.currency, "USD")

    def test_split_normalization_calculates_forward_and_reverse_ratios(self):
        split_records = [
            {
                "id": "forward",
                "ticker": "SPY",
                "execution_date": "2018-01-04",
                "split_from": 1,
                "split_to": 4,
                "adjustment_type": "forward_split",
            },
            {
                "id": "reverse",
                "ticker": "SPY",
                "execution_date": "2018-01-05",
                "split_from": 4,
                "split_to": 1,
                "adjustment_type": "reverse_split",
            },
        ]
        provider, _ = provider_with_responses(
            [
                empty_page(),
                FakeResponse(payload={"status": "OK", "results": split_records}),
            ]
        )

        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")

        self.assertTrue(batch.complete)
        self.assertEqual(
            [event.split_ratio for event in batch.events],
            [4.0, 0.25],
        )
        self.assertEqual(
            [event.event_type for event in batch.events],
            [CorporateActionType.SPLIT, CorporateActionType.SPLIT],
        )

    def test_successful_empty_is_distinct_from_failed_request(self):
        provider, _ = provider_with_responses([empty_page(), empty_page()])

        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")

        self.assertTrue(batch.complete)
        self.assertEqual([outcome.status for outcome in batch.outcomes], ["empty", "empty"])
        self.assertEqual(batch.events, ())

        denied_provider, _ = provider_with_responses(
            [FakeResponse(status_code=403), empty_page()]
        )
        denied = denied_provider.acquire(("SPY",), "2018-01-03", "2018-01-10")
        self.assertFalse(denied.complete)
        self.assertEqual(denied.outcomes[0].status, "access_denied")

        unsupported_provider, _ = provider_with_responses(
            [FakeResponse(status_code=404), empty_page()]
        )
        unsupported = unsupported_provider.acquire(
            ("SPY",), "2018-01-03", "2018-01-10"
        )
        self.assertFalse(unsupported.complete)
        self.assertEqual(
            unsupported.outcomes[0].status,
            "unsupported_or_not_found",
        )

    def test_pagination_is_fully_consumed_and_key_is_added_safely(self):
        dividend_page = FakeResponse(
            payload={
                "status": "OK",
                "results": [
                    {
                        "id": "div-1",
                        "currency": "USD",
                        "declaration_date": "2018-01-03",
                        "ex_dividend_date": "2018-01-04",
                        "pay_date": "2018-01-08",
                        "cash_amount": 1.0,
                    }
                ],
                "next_url": (
                    "https://api.massive.com/stocks/v1/dividends?cursor=page-2"
                ),
            }
        )
        second_page = FakeResponse(
            payload={
                "status": "OK",
                "results": [
                    {
                        "id": "div-2",
                        "currency": "USD",
                        "declaration_date": "2018-01-04",
                        "ex_dividend_date": "2018-01-05",
                        "pay_date": "2018-01-09",
                        "cash_amount": 0.5,
                    }
                ],
            }
        )
        provider, session = provider_with_responses(
            [dividend_page, second_page, empty_page()]
        )

        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")

        self.assertTrue(batch.complete)
        self.assertEqual(len(batch.events), 2)
        self.assertEqual(batch.outcomes[0].pages, 2)
        self.assertEqual(session.calls[0]["params"]["ex_dividend_date.gte"], "2018-01-03")
        self.assertEqual(session.calls[0]["params"]["ex_dividend_date.lte"], "2018-01-10")
        self.assertEqual(session.calls[1]["params"], {"apiKey": "test-only-key"})
        self.assertEqual(session.calls[2]["params"]["execution_date.gte"], "2018-01-03")
        self.assertFalse(session.calls[0]["allow_redirects"])

    def test_rate_limit_is_retried_without_logging_credentials(self):
        sleeps = []
        provider, _ = provider_with_responses(
            [
                FakeResponse(status_code=429, headers={"Retry-After": "0"}),
                empty_page(),
                empty_page(),
            ],
            sleeps=sleeps,
            max_attempts=2,
        )

        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")

        self.assertTrue(batch.complete)
        self.assertEqual(sleeps, [0.0])
        self.assertNotIn("test-only-key", repr(batch))

    def test_transport_failure_is_not_reported_as_empty(self):
        provider, _ = provider_with_responses(
            [
                requests.ConnectionError("private URL should not be exposed"),
                empty_page(),
            ],
            max_attempts=1,
        )

        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")

        self.assertFalse(batch.complete)
        self.assertEqual(batch.outcomes[0].status, "request_failed")
        self.assertNotIn("private URL", batch.outcomes[0].error)

    def test_missing_payment_date_and_invalid_currency_fail_query_validation(self):
        missing_payment = {
            "id": "div-no-pay",
            "currency": "USD",
            "ex_dividend_date": "2018-01-04",
            "cash_amount": 1.0,
        }
        provider, _ = provider_with_responses(
            [
                FakeResponse(payload={"status": "OK", "results": [missing_payment]}),
                empty_page(),
            ]
        )
        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")
        self.assertFalse(batch.complete)
        self.assertEqual(batch.outcomes[0].status, "validation_failed")
        self.assertIn("pay_date", batch.outcomes[0].error)

        bad_currency = dict(missing_payment, pay_date="2018-01-08", currency="CAD")
        provider, _ = provider_with_responses(
            [
                FakeResponse(payload={"status": "OK", "results": [bad_currency]}),
                empty_page(),
            ]
        )
        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")
        self.assertFalse(batch.complete)
        self.assertIn("unsupported currency", batch.outcomes[0].error)

    def test_missing_id_nonpositive_values_and_out_of_range_events_fail(self):
        invalid_records = [
            {
                "currency": "USD",
                "ex_dividend_date": "2018-01-04",
                "pay_date": "2018-01-08",
                "cash_amount": 1.0,
            },
            {
                "id": "zero",
                "currency": "USD",
                "ex_dividend_date": "2018-01-04",
                "pay_date": "2018-01-08",
                "cash_amount": 0,
            },
            {
                "id": "outside",
                "currency": "USD",
                "ex_dividend_date": "2018-01-11",
                "pay_date": "2018-01-12",
                "cash_amount": 1.0,
            },
            {
                "id": "bad-date",
                "currency": "USD",
                "ex_dividend_date": "2018-02-30",
                "pay_date": "2018-03-02",
                "cash_amount": 1.0,
            },
        ]
        for invalid in invalid_records:
            with self.subTest(id=invalid.get("id")):
                provider, _ = provider_with_responses(
                    [
                        FakeResponse(payload={"status": "OK", "results": [invalid]}),
                        empty_page(),
                    ]
                )
                batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")
                self.assertFalse(batch.complete)
                self.assertEqual(batch.outcomes[0].status, "validation_failed")

    def test_zero_split_denominator_fails_validation(self):
        provider, _ = provider_with_responses(
            [
                empty_page(),
                FakeResponse(
                    payload={
                        "status": "OK",
                        "results": [
                            {
                                "id": "zero-denominator",
                                "execution_date": "2018-01-04",
                                "split_from": 0,
                                "split_to": 2,
                            }
                        ],
                    }
                ),
            ]
        )

        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")

        self.assertFalse(batch.complete)
        self.assertEqual(batch.outcomes[1].status, "validation_failed")
        self.assertIn("split_from", batch.outcomes[1].error)

    def test_duplicate_records_are_deduplicated_and_conflicts_fail(self):
        record = {
            "id": "dup-1",
            "currency": "USD",
            "ex_dividend_date": "2018-01-04",
            "pay_date": "2018-01-08",
            "cash_amount": 1.0,
        }
        provider, _ = provider_with_responses(
            [
                FakeResponse(payload={"status": "OK", "results": [record, record]}),
                empty_page(),
            ]
        )
        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")
        self.assertTrue(batch.complete)
        self.assertEqual(len(batch.events), 1)
        self.assertEqual(batch.exact_duplicate_count, 1)

        conflict = dict(record, cash_amount=2.0)
        provider, _ = provider_with_responses(
            [
                FakeResponse(payload={"status": "OK", "results": [record, conflict]}),
                empty_page(),
            ]
        )
        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")
        self.assertFalse(batch.complete)
        self.assertEqual(batch.outcomes[0].status, "validation_failed")
        self.assertIn("Conflicting duplicate", batch.outcomes[0].error)

    def test_unsupported_pagination_url_is_refused(self):
        provider, _ = provider_with_responses(
            [
                FakeResponse(
                    payload={
                        "status": "OK",
                        "results": [],
                        "next_url": "https://attacker.example/steal",
                    }
                ),
                empty_page(),
            ]
        )

        batch = provider.acquire(("SPY",), "2018-01-03", "2018-01-10")

        self.assertFalse(batch.complete)
        self.assertEqual(batch.outcomes[0].status, "invalid_response")

    def test_validation_requires_supported_symbol_usd_and_session_alignment(self):
        event = CorporateActionEvent(
            symbol="SPY",
            event_type=CorporateActionType.ETF_DISTRIBUTION,
            source_id="event-1",
            currency="USD",
            ex_date="2018-01-04",
            payment_date="2018-01-08",
            distribution_per_share=1.0,
        )
        checked = validate_events(
            [event],
            start_date="2018-01-03",
            end_date="2018-01-10",
            market_sessions=DATE_SESSIONS,
        )
        self.assertEqual(checked, (event,))

        for invalid, sessions, message in (
            (
                CorporateActionEvent(
                    symbol="OTHER",
                    event_type=CorporateActionType.ETF_DISTRIBUTION,
                    source_id="bad-symbol",
                    ex_date="2018-01-04",
                    payment_date="2018-01-08",
                    distribution_per_share=1.0,
                ),
                DATE_SESSIONS,
                "Unsupported symbol",
            ),
            (
                CorporateActionEvent(
                    symbol="SPY",
                    event_type=CorporateActionType.ETF_DISTRIBUTION,
                    source_id="bad-currency",
                    currency="CAD",
                    ex_date="2018-01-04",
                    payment_date="2018-01-08",
                    distribution_per_share=1.0,
                ),
                DATE_SESSIONS,
                "Non-USD",
            ),
            (
                event,
                ("2018-01-03", "2018-01-05"),
                "does not align",
            ),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                CorporateActionValidationError, message
            ):
                validate_events(
                    [invalid],
                    start_date="2018-01-03",
                    end_date="2018-01-10",
                    market_sessions=sessions,
                )

    def test_export_writes_manifest_raw_records_and_checksums(self):
        event = CorporateActionEvent(
            symbol="SPY",
            event_type=CorporateActionType.ETF_DISTRIBUTION,
            source_id="fixture-dividend",
            currency="USD",
            announcement_date="2018-01-03",
            ex_date="2018-01-04",
            payment_date="2018-01-08",
            distribution_per_share=0.5,
        )
        batch = successful_batch(
            [event],
            raw_records=(
                {
                    "symbol": "SPY",
                    "category": "dividend",
                    "endpoint": "/stocks/v1/dividends",
                    "record": {
                        "id": "fixture-dividend",
                        "cash_amount": 0.5,
                    },
                },
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = export_acquisition(
                batch,
                temp_dir,
                market_sessions=DATE_SESSIONS,
            )
            root = Path(temp_dir)
            csv_path = root / "corporate_actions.csv"
            raw_path = root / "corporate_actions_raw.jsonl"
            manifest_path = root / "manifest.json"

            self.assertEqual(manifest["per_symbol"]["SPY"]["dividend_records"], 1)
            self.assertEqual(manifest["per_symbol"]["SPY"]["split_records"], 0)
            self.assertIn("not independently verified", manifest["coverage_status"])
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                manifest,
            )
            loaded_events = load_normalized_events(
                csv_path,
                start_date="2018-01-03",
                end_date="2018-01-10",
                market_sessions=DATE_SESSIONS,
            )
            self.assertEqual(loaded_events, (event,))
            self.assertEqual(
                manifest["artifacts"]["corporate_actions.csv"]["sha256"],
                hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["artifacts"]["corporate_actions_raw.jsonl"]["sha256"],
                hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            )
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["distribution_per_share"], "0.5")
            self.assertIn("fixture-dividend", raw_path.read_text(encoding="utf-8"))
            self.assertNotIn("test-only-key", manifest_path.read_text(encoding="utf-8"))

    def test_export_refuses_incomplete_queries_and_non_session_events(self):
        denied = successful_batch(
            [],
            outcomes=(
                QueryOutcome(
                    "SPY",
                    "dividend",
                    "/stocks/v1/dividends",
                    "access_denied",
                    0,
                    0,
                    "account is not entitled",
                ),
                QueryOutcome(
                    "SPY", "split", "/stocks/v1/splits", "empty", 1, 0
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(CorporateActionValidationError, "incomplete"):
                export_acquisition(denied, temp_dir, market_sessions=DATE_SESSIONS)

        misaligned = CorporateActionEvent(
            symbol="SPY",
            event_type=CorporateActionType.SPLIT,
            source_id="off-session",
            effective_date="2018-01-06",
            split_ratio=2.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(CorporateActionValidationError, "does not align"):
                export_acquisition(
                    successful_batch([misaligned]),
                    temp_dir,
                    market_sessions=DATE_SESSIONS,
                )
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_normalized_events_replay_through_ledger_once_with_consistent_nav(self):
        split = CorporateActionEvent(
            symbol="SPY",
            event_type=CorporateActionType.SPLIT,
            source_id="fixture-split",
            effective_date="2018-01-04",
            split_ratio=2.0,
        )
        dividend = CorporateActionEvent(
            symbol="SPY",
            event_type=CorporateActionType.ETF_DISTRIBUTION,
            source_id="fixture-distribution",
            ex_date="2018-01-04",
            payment_date="2018-01-08",
            distribution_per_share=1.0,
        )
        ledger = CorporateActionLedger({"SPY": 10.0}, 0.0)
        before = ledger.snapshot({"SPY": 100.0})

        ledger.process_date("2018-01-04", [split, dividend, split, dividend])
        on_ex_date = ledger.snapshot({"SPY": 49.5})
        ledger.process_date("2018-01-08", [split, dividend])
        after_payment = ledger.snapshot({"SPY": 49.5})
        ledger.process_date("2018-01-08", [split, dividend])
        repeated = ledger.snapshot({"SPY": 49.5})

        self.assertEqual(ledger.holdings["SPY"], 20.0)
        self.assertEqual(on_ex_date.dividend_receivables, {"SPY:fixture-distribution": 10.0})
        self.assertEqual(on_ex_date.total_nav, before.total_nav)
        self.assertEqual(ledger.settled_cash, 10.0)
        self.assertEqual(after_payment.total_nav, on_ex_date.total_nav)
        self.assertEqual(repeated.total_nav, after_payment.total_nav)
        self.assertEqual(repeated.dividend_receivables, {})


if __name__ == "__main__":
    unittest.main()
