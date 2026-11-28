import unittest
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from src.api.account_api import _safe_issuer_concentration_assessment
from src.services.issuer_concentration import (
    IssuerReferenceDataError,
    ISSUER_CONCENTRATION_POLICY_VERSION,
    assess_portfolio_issuer_concentration,
    is_valid_instrument_identifier,
    is_valid_issuer_identifier,
    load_issuer_reference_data,
)


_SESSION_DATE = date(2026, 2, 2)
_EVALUATED_AT = datetime(2026, 2, 2, tzinfo=timezone.utc)
_VERSION = "test-reference-v1"


def _record(
    exchange="NSE",
    symbol="ALFA",
    issuer_id="TEST-ISSUER-1",
    instrument_id="TEST-INSTRUMENT-1",
    effective_from="2020-01-01",
    effective_to=None,
):
    return {
        "exchange": exchange,
        "symbol": symbol,
        "instrument_identifier": instrument_id,
        "instrument_identifier_type": "OTHER_VERIFIED",
        "issuer_id": issuer_id,
        "issuer_id_type": "OTHER_VERIFIED",
        "effective_from": effective_from,
        "effective_to": effective_to,
        "source": "test fixture only",
        "source_reference": "test fixture document 1",
        "verified_on": "2026-01-01",
        "reference_data_version": _VERSION,
    }


def _reference(*records, version=_VERSION):
    return {
        "schema_version": 1,
        "reference_data_version": version,
        "records": list(records),
    }


def _holding(
    exchange="NSE",
    symbol="ALFA",
    value="60.00",
    *,
    company_name=None,
    status="VALUED",
    market_date=_SESSION_DATE,
):
    item = {
        "exchange": exchange,
        "symbol": symbol,
        "quantity": "1",
        "valuation_status": status,
        "market_value": value if status == "VALUED" else None,
        "market_date": market_date.isoformat() if status == "VALUED" else None,
    }
    if company_name is not None:
        item["company_name"] = company_name
    return item


def _assess(holdings=(), *, cash="40.00", reference=None, nav=None, complete=True):
    if nav is None:
        nav = Decimal(cash) + sum(
            (
                Decimal(item["market_value"])
                for item in holdings
                if item["valuation_status"] == "VALUED"
            ),
            Decimal("0"),
        )
    return assess_portfolio_issuer_concentration(
        session_date=_SESSION_DATE,
        valuation_complete=complete,
        portfolio_nav=nav,
        cash_balance=cash,
        holdings=holdings,
        reference_data=reference or _reference(),
        evaluated_at=_EVALUATED_AT,
    )


class IssuerConcentrationTests(unittest.TestCase):
    def test_two_exchange_listings_of_same_verified_issuer_are_aggregated(self):
        holdings = [
            _holding("NSE", "ALFA", "30.00"),
            _holding("BSE", "ALFA", "30.00"),
        ]
        reference = _reference(
            _record("NSE", "ALFA", instrument_id="TEST-NSE-ALFA"),
            _record("BSE", "ALFA", instrument_id="TEST-BSE-ALFA"),
        )

        result = _assess(holdings, reference=reference)

        self.assertEqual(result["assessment_status"], "EVALUATED")
        self.assertEqual(
            [row["listing"] for row in result["listing_exposures"]],
            ["BSE:ALFA", "NSE:ALFA"],
        )
        self.assertEqual(len(result["issuer_exposures"]), 1)
        self.assertEqual(result["issuer_exposures"][0]["market_value"], "60.00")
        self.assertEqual(
            Decimal(result["issuer_exposures"][0]["portfolio_weight_pct"]),
            Decimal("60"),
        )
        self.assertEqual(
            result["issuer_exposures"][0]["listings"],
            ["BSE:ALFA", "NSE:ALFA"],
        )
        self.assertEqual(result["alerts"], [])

    def test_different_issuer_ids_are_not_merged_by_similar_names(self):
        holdings = [
            _holding("NSE", "ALFA", "30.00", company_name="Example Industries"),
            _holding("BSE", "BETA", "30.00", company_name="Example Industry"),
        ]
        reference = _reference(
            _record("NSE", "ALFA", "TEST-ISSUER-1", "TEST-INSTRUMENT-1"),
            _record("BSE", "BETA", "TEST-ISSUER-2", "TEST-INSTRUMENT-2"),
        )

        result = _assess(holdings, reference=reference)

        self.assertEqual(len(result["issuer_exposures"]), 2)
        self.assertEqual(
            {row["issuer_id"] for row in result["issuer_exposures"]},
            {"TEST-ISSUER-1", "TEST-ISSUER-2"},
        )

    def test_missing_mapping_suppresses_all_partial_issuer_metrics(self):
        holdings = [
            _holding("NSE", "ALFA", "30.00"),
            _holding("BSE", "BETA", "30.00"),
        ]
        result = _assess(
            holdings,
            reference=_reference(_record("NSE", "ALFA")),
        )

        self.assertEqual(result["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(result["reason_codes"], ["issuer_mapping_unresolved"])
        self.assertEqual(result["unresolved_listings"][0]["listing"], "BSE:BETA")
        self.assertIsNone(result["portfolio_nav"])
        self.assertEqual(result["listing_exposures"], [])
        self.assertEqual(result["issuer_exposures"], [])
        self.assertEqual(result["alerts"], [])

    def test_conflicting_overlapping_mappings_are_unavailable(self):
        first = _record(
            exchange="NSE",
            symbol="ALFA",
            issuer_id="TEST-ISSUER-1",
            instrument_id="TEST-SHARED-INSTRUMENT",
        )
        conflicting = _record(
            exchange="BSE",
            symbol="BETA",
            issuer_id="TEST-ISSUER-2",
            instrument_id="TEST-SHARED-INSTRUMENT",
            effective_from="2025-01-01",
        )

        result = _assess(
            [_holding()],
            reference=_reference(first, conflicting),
        )

        self.assertEqual(result["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(result["reason_codes"], ["issuer_reference_invalid"])
        self.assertEqual(result["reference_data_version"], _VERSION)
        self.assertEqual(result["issuer_exposures"], [])

    def test_effective_date_boundaries_select_the_correct_verified_mapping(self):
        reference = _reference(
            _record(
                issuer_id="TEST-OLD",
                instrument_id="TEST-OLD-INSTRUMENT",
                effective_from="2025-01-01",
                effective_to="2025-12-31",
            ),
            _record(
                issuer_id="TEST-NEW",
                instrument_id="TEST-NEW-INSTRUMENT",
                effective_from="2026-01-01",
            ),
        )

        on_last_old_day = assess_portfolio_issuer_concentration(
            session_date=date(2025, 12, 31),
            valuation_complete=True,
            portfolio_nav=Decimal("100.00"),
            cash_balance=Decimal("40.00"),
            holdings=[
                _holding(
                    value="60.00",
                    market_date=date(2025, 12, 31),
                )
            ],
            reference_data=reference,
            evaluated_at=_EVALUATED_AT,
        )
        on_first_new_day = _assess([_holding(value="60.00")], reference=reference)

        self.assertEqual(
            on_last_old_day["issuer_exposures"][0]["issuer_id"], "TEST-OLD"
        )
        self.assertEqual(
            on_first_new_day["issuer_exposures"][0]["issuer_id"], "TEST-NEW"
        )

    def test_empty_reference_allows_cash_only_and_unmapped_zero_value(self):
        empty_reference = _reference()
        cash_only = _assess([], cash="100.00", reference=empty_reference)
        zero_value = _assess(
            [_holding(value="0.00")],
            cash="100.00",
            reference=empty_reference,
        )

        self.assertEqual(cash_only["assessment_status"], "EVALUATED")
        self.assertEqual(cash_only["issuer_exposures"], [])
        self.assertEqual(zero_value["assessment_status"], "EVALUATED")
        self.assertIsNone(zero_value["listing_exposures"][0]["issuer_id"])
        self.assertEqual(zero_value["issuer_exposures"], [])

    def test_incomplete_valuation_has_no_partial_metrics_or_mapping_result(self):
        result = _assess(
            [_holding(value="60.00")],
            reference=_reference(_record()),
            complete=False,
        )

        self.assertEqual(result["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(result["reason_codes"], ["valuation_incomplete"])
        self.assertIsNone(result["portfolio_nav"])
        self.assertEqual(result["listing_exposures"], [])
        self.assertEqual(result["issuer_exposures"], [])
        self.assertEqual(result["alerts"], [])

    def test_duplicate_listing_and_negative_values_are_rejected(self):
        with self.assertRaises(ValueError):
            _assess([_holding(), _holding()])
        with self.assertRaises(ValueError):
            _assess([_holding(value="-1.00")], cash="101.00")
        with self.assertRaises(ValueError):
            _assess([_holding()], cash="-1.00", nav=Decimal("59.00"))

    def test_duplicate_reference_record_is_not_evaluated(self):
        record = _record()
        result = _assess(
            [_holding()],
            reference=_reference(record, deepcopy(record)),
        )
        self.assertEqual(result["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(result["reason_codes"], ["issuer_reference_invalid"])

    def test_strict_identifier_provenance_and_date_validation(self):
        self.assertTrue(is_valid_instrument_identifier("ISIN", "US0378331005"))
        self.assertFalse(is_valid_instrument_identifier("ISIN", "US0378331004"))
        self.assertFalse(is_valid_instrument_identifier("OTHER_VERIFIED", "  "))
        self.assertTrue(
            is_valid_issuer_identifier("CIN", "L12345MH2020ABC123456")
        )
        self.assertFalse(is_valid_issuer_identifier("CIN", "L12345MH2020ABC12345"))
        self.assertTrue(
            is_valid_issuer_identifier("LEI", "5493001KJTIIGC8Y1R12")
        )
        self.assertFalse(
            is_valid_issuer_identifier("LEI", "5493001KJTIIGC8Y1R13")
        )
        self.assertFalse(is_valid_issuer_identifier("OTHER_VERIFIED", "   "))

        for field, value in (
            ("source", "   "),
            ("source_reference", "  "),
            ("issuer_id", "  "),
            ("instrument_identifier", " "),
            ("effective_from", "20260101"),
        ):
            record = _record()
            record[field] = value
            with self.subTest(field=field):
                result = _assess([_holding()], reference=_reference(record))
                self.assertEqual(result["assessment_status"], "NOT_EVALUATED")
                self.assertEqual(result["reason_codes"], ["issuer_reference_invalid"])

    def test_invalid_utf8_reference_file_uses_safe_data_error(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "issuer-reference.json"
            path.write_bytes(b"\xff\xfe")
            with self.assertRaises(IssuerReferenceDataError):
                load_issuer_reference_data(path)

    def test_api_allowlist_rejects_malformed_legacy_assessment(self):
        valid = _assess(
            [_holding()],
            cash="40.00",
            reference=_reference(_record()),
        )
        self.assertIsNotNone(_safe_issuer_concentration_assessment(valid))

        malformed = deepcopy(valid)
        malformed["issuer_exposures"][0]["issuer_id"] = []
        self.assertIsNone(_safe_issuer_concentration_assessment(malformed))
        malformed = deepcopy(valid)
        malformed["reference_data_version"] = "internal diagnostic"
        self.assertIsNone(_safe_issuer_concentration_assessment(malformed))

    def test_api_allowlist_rejects_zero_nav_and_inconsistent_issuer_groups(self):
        two_issuer_result = _assess(
            [
                _holding("NSE", "ALFA", "30.00"),
                _holding("BSE", "BETA", "30.00"),
            ],
            reference=_reference(
                _record("NSE", "ALFA", "TEST-ISSUER-1", "TEST-INSTRUMENT-1"),
                _record("BSE", "BETA", "TEST-ISSUER-2", "TEST-INSTRUMENT-2"),
            ),
        )
        self.assertIsNotNone(
            _safe_issuer_concentration_assessment(two_issuer_result)
        )

        zero_nav = deepcopy(two_issuer_result)
        zero_nav["portfolio_nav"] = "0.00"
        self.assertIsNone(_safe_issuer_concentration_assessment(zero_nav))

        excess_equity = deepcopy(two_issuer_result)
        excess_equity["portfolio_nav"] = "50.00"
        self.assertIsNone(_safe_issuer_concentration_assessment(excess_equity))

        duplicate = deepcopy(two_issuer_result)
        duplicate["issuer_exposures"][1] = deepcopy(
            duplicate["issuer_exposures"][0]
        )
        self.assertIsNone(_safe_issuer_concentration_assessment(duplicate))

        omitted_and_duplicated = deepcopy(two_issuer_result)
        omitted_and_duplicated["issuer_exposures"][1] = deepcopy(
            omitted_and_duplicated["issuer_exposures"][0]
        )
        self.assertIsNone(
            _safe_issuer_concentration_assessment(omitted_and_duplicated)
        )

        unexpected = deepcopy(two_issuer_result)
        unexpected["issuer_exposures"][1]["issuer_id"] = "TEST-UNEXPECTED"
        self.assertIsNone(_safe_issuer_concentration_assessment(unexpected))

    def test_api_allowlist_accepts_one_issuer_from_multiple_listings(self):
        result = _assess(
            [
                _holding("NSE", "ALFA", "30.00"),
                _holding("BSE", "ALFA", "30.00"),
            ],
            reference=_reference(
                _record("NSE", "ALFA", instrument_id="TEST-NSE-INSTRUMENT"),
                _record("BSE", "ALFA", instrument_id="TEST-BSE-INSTRUMENT"),
            ),
        )
        safe = _safe_issuer_concentration_assessment(result)
        self.assertIsNotNone(safe)
        self.assertEqual(len(safe["issuer_exposures"]), 1)
        self.assertEqual(
            safe["issuer_exposures"][0]["listings"],
            ["BSE:ALFA", "NSE:ALFA"],
        )

    def test_assessment_has_no_issuer_alert_policy(self):
        result = _assess(
            [_holding(value="90.00")],
            cash="10.00",
            reference=_reference(_record()),
        )
        self.assertEqual(
            result["policy_version"], ISSUER_CONCENTRATION_POLICY_VERSION
        )
        self.assertEqual(result["alerts"], [])


if __name__ == "__main__":
    unittest.main()
