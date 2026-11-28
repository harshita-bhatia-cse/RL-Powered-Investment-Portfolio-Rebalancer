import unittest
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from src.api.account_api import _safe_issuer_concentration_assessment
from src.services.sector_concentration import (
    SECTOR_CONCENTRATION_POLICY_VERSION,
    SECTOR_REFERENCE_PATH,
    SECTOR_TAXONOMY,
    SECTOR_TAXONOMY_NAME,
    SECTOR_TAXONOMY_VERSION,
    SectorReferenceDataError,
    assess_portfolio_sector_concentration,
    load_sector_reference_data,
    safe_sector_concentration_assessment,
)


_SESSION = date(2026, 6, 30)
_EVALUATED_AT = datetime(2026, 7, 1, tzinfo=timezone.utc)
_REFERENCE_VERSION = "test-sector-v1"


def _issuer(issuer_id, market_value, issuer_id_type="OTHER_VERIFIED"):
    return {
        "issuer_id_type": issuer_id_type,
        "issuer_id": issuer_id,
        "market_value": str(market_value),
    }


def _issuer_assessment(
    issuers=(),
    *,
    cash="200.00",
    session_date=_SESSION,
    status="EVALUATED",
    listings=(),
):
    total_equity = sum((Decimal(row["market_value"]) for row in issuers), Decimal("0"))
    nav = Decimal(cash) + total_equity
    entries = []
    listing_entries = list(listings)
    for row in issuers:
        amount = Decimal(row["market_value"])
        with localcontext() as context:
            context.prec = 50
            weight = amount / nav * Decimal("100")
        entries.append(
            {
                **row,
                "portfolio_weight_pct": str(weight),
                "listings": [f"NSE:{row['issuer_id']}"],
            }
        )
        listing_entries.append(
            {
                "exchange": "NSE",
                "symbol": row["issuer_id"],
                "listing": f"NSE:{row['issuer_id']}",
                "quantity": "1",
                "market_value": str(amount),
                "portfolio_weight_pct": str(weight),
                "instrument_identifier": f"TEST-INSTRUMENT-{row['issuer_id']}",
                "instrument_identifier_type": "OTHER_VERIFIED",
                "issuer_id": row["issuer_id"],
                "issuer_id_type": row["issuer_id_type"],
                "source": "test fixture only",
                "source_reference": "test fixture reference",
                "verified_on": "2026-01-15",
            }
        )
    listing_entries.sort(
        key=lambda row: (
            -Decimal(row["market_value"]),
            row["exchange"],
            row["symbol"],
        )
    )
    return {
        "assessment_status": status,
        "policy_version": "issuer-concentration-v1",
        "reference_data_version": "issuer-test-v1",
        "evaluated_at": _EVALUATED_AT.isoformat(),
        "session_date": session_date.isoformat(),
        "portfolio_nav": str(nav) if status == "EVALUATED" else None,
        "total_equity_market_value": (
            str(total_equity) if status == "EVALUATED" else None
        ),
        "listing_exposures": listing_entries,
        "issuer_exposures": entries if status == "EVALUATED" else [],
        "alerts": [],
        "reason_codes": [] if status == "EVALUATED" else ["ledger_discrepancy"],
        "unresolved_listings": [],
    }


def _record(
    issuer_id,
    sector_code="TECHNOLOGY",
    *,
    issuer_id_type="OTHER_VERIFIED",
    effective_from="2020-01-01",
    effective_to=None,
    source_published_on="2025-01-01",
    sector_name=None,
):
    return {
        "issuer_id_type": issuer_id_type,
        "issuer_id": issuer_id,
        "sector_code": sector_code,
        "sector_name": sector_name or SECTOR_TAXONOMY[sector_code],
        "effective_from": effective_from,
        "effective_to": effective_to,
        "source": "Issuer annual report",
        "source_reference": (
            "Annual report, page 17 https://example.org/annual-report.pdf"
        ),
        "source_published_on": source_published_on,
        "verified_on": "2026-01-15",
        "reference_data_version": _REFERENCE_VERSION,
    }


def _reference(*records, version=_REFERENCE_VERSION):
    return {
        "schema_version": 1,
        "reference_data_version": version,
        "taxonomy": {
            "name": SECTOR_TAXONOMY_NAME,
            "version": SECTOR_TAXONOMY_VERSION,
            "sectors": [
                {"code": code, "name": name}
                for code, name in sorted(SECTOR_TAXONOMY.items())
            ],
        },
        "records": list(records),
    }


def _assess(issuers=(), reference=None, **kwargs):
    return assess_portfolio_sector_concentration(
        session_date=_SESSION,
        issuer_assessment=_issuer_assessment(issuers, **kwargs),
        reference_data=reference or _reference(),
        evaluated_at=_EVALUATED_AT,
    )


class SectorConcentrationTests(unittest.TestCase):
    def test_initial_reference_data_is_empty_and_uses_project_taxonomy(self):
        data = load_sector_reference_data(SECTOR_REFERENCE_PATH)
        self.assertEqual(data["records"], [])
        self.assertEqual(
            data["taxonomy"]["name"],
            SECTOR_TAXONOMY_NAME,
        )
        self.assertEqual(len(data["taxonomy"]["sectors"]), len(SECTOR_TAXONOMY))

    def test_loader_rejects_malformed_json_and_invalid_utf8(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            for content in (b"{", b"\xff\xfe"):
                path.write_bytes(content)
                with self.subTest(content=content):
                    with self.assertRaises(SectorReferenceDataError):
                        load_sector_reference_data(path)

    def test_loader_rejects_schema_taxonomy_and_required_field_errors(self):
        bad_documents = []
        unsupported_schema = _reference()
        unsupported_schema["schema_version"] = 2
        bad_documents.append(unsupported_schema)
        unsupported_taxonomy = _reference()
        unsupported_taxonomy["taxonomy"]["version"] = "2.0"
        bad_documents.append(unsupported_taxonomy)
        missing_root_field = _reference()
        del missing_root_field["taxonomy"]
        bad_documents.append(missing_root_field)
        missing_mapping_field = _reference(_record("ISSUER-1"))
        del missing_mapping_field["records"][0]["source_reference"]
        bad_documents.append(missing_mapping_field)
        for document in bad_documents:
            with self.subTest(document=document):
                with self.assertRaises(SectorReferenceDataError):
                    from src.services.sector_concentration import _validate_document

                    _validate_document(document)

    def test_loader_rejects_invalid_identifiers_codes_labels_dates_and_provenance(self):
        cases = []
        for field, value in (
            ("issuer_id", "  "),
            ("issuer_id_type", "UNSUPPORTED"),
            ("sector_code", "UNKNOWN"),
            ("sector_name", "Different label"),
            ("effective_from", "20260630"),
            ("verified_on", "not-a-date"),
            ("source", "  "),
            ("source_reference", "Annual report page 17"),
            ("source_reference", "Annual report https://example.org/report.pdf"),
        ):
            record = _record("ISSUER-1")
            record[field] = value
            cases.append(record)
        for record in cases:
            with self.subTest(record=record):
                with self.assertRaises(SectorReferenceDataError):
                    from src.services.sector_concentration import _validate_document

                    _validate_document(_reference(record))
        invalid_range = _record("ISSUER-1")
        invalid_range["effective_from"] = "2026-01-01"
        invalid_range["effective_to"] = "2020-01-01"
        with self.assertRaises(SectorReferenceDataError):
            from src.services.sector_concentration import _validate_document

            _validate_document(_reference(invalid_range))

    def test_duplicate_and_overlapping_issuer_mappings_are_rejected(self):
        record = _record("ISSUER-1")
        for records in (
            [record, deepcopy(record)],
            [
                _record("ISSUER-1", effective_to="2025-12-31"),
                _record(
                    "ISSUER-1",
                    "CONSUMER",
                    effective_from="2025-12-31",
                ),
            ],
        ):
            with self.subTest(records=records):
                with self.assertRaises(SectorReferenceDataError):
                    from src.services.sector_concentration import _validate_document

                    _validate_document(_reference(*records))

    def test_effective_date_boundaries_and_publication_cutoff(self):
        references = _reference(
            _record(
                "ISSUER-1",
                "CONSUMER",
                effective_from="2020-01-01",
                effective_to="2026-06-29",
            ),
            _record(
                "ISSUER-1",
                "TECHNOLOGY",
                effective_from="2026-06-30",
                source_published_on="2026-06-30",
            ),
        )
        result = _assess([_issuer("ISSUER-1", "80.00")], references)
        self.assertEqual(result["assessment_status"], "EVALUATED")
        self.assertEqual(result["policy_version"], SECTOR_CONCENTRATION_POLICY_VERSION)
        self.assertEqual(result["sector_exposures"][0]["sector_code"], "TECHNOLOGY")
        future_source = deepcopy(references)
        future_source["records"][1]["source_published_on"] = "2026-07-01"
        unavailable = _assess([_issuer("ISSUER-1", "80.00")], future_source)
        self.assertEqual(unavailable["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(
            unavailable["reason_codes"],
            ["issuer_sector_mapping_unresolved"],
        )
        self.assertEqual(unavailable["sector_exposures"], [])

    def test_multiple_issuers_aggregate_and_metrics_use_required_denominators(self):
        issuers = [
            _issuer("ISSUER-A", "300.00"),
            _issuer("ISSUER-B", "200.00"),
            _issuer("ISSUER-C", "150.00"),
            _issuer("ISSUER-D", "100.00"),
            _issuer("ISSUER-E", "50.00"),
        ]
        reference = _reference(
            _record("ISSUER-A", "TECHNOLOGY"),
            _record("ISSUER-B", "TECHNOLOGY"),
            _record("ISSUER-C", "CONSUMER"),
            _record("ISSUER-D", "HEALTHCARE"),
            _record("ISSUER-E", "INDUSTRIALS"),
        )
        result = _assess(issuers, reference, cash="200.00")
        self.assertEqual(result["assessment_status"], "EVALUATED")
        self.assertEqual(
            [row["sector_code"] for row in result["sector_exposures"]],
            ["TECHNOLOGY", "CONSUMER", "HEALTHCARE", "INDUSTRIALS"],
        )
        self.assertEqual(result["sector_exposures"][0]["market_value"], "500.00")
        self.assertEqual(
            Decimal(result["sector_exposures"][0]["portfolio_weight_pct"]),
            Decimal("50"),
        )
        self.assertEqual(
            Decimal(result["largest_sector_weight_pct"]),
            Decimal("50"),
        )
        self.assertEqual(
            Decimal(result["top_three_sector_weight_pct"]),
            Decimal("75"),
        )
        self.assertEqual(Decimal(result["sector_hhi"]), Decimal("0.4453125"))
        self.assertEqual(sum(
            Decimal(row["portfolio_weight_pct"])
            for row in result["sector_exposures"]
        ), Decimal("80"))
        self.assertEqual(result["alerts"], [])

    def test_phase_62b_aggregated_cross_exchange_issuer_is_counted_once(self):
        issuer = _issuer("ISSUER-ONE", "100.00")
        assessment = _issuer_assessment([issuer], cash="0.00")
        assessment["issuer_exposures"][0]["listings"] = ["BSE:AAA", "NSE:AAA"]
        result = assess_portfolio_sector_concentration(
            session_date=_SESSION,
            issuer_assessment=assessment,
            reference_data=_reference(_record("ISSUER-ONE")),
            evaluated_at=_EVALUATED_AT,
        )
        self.assertEqual(len(result["issuer_contributions"]), 1)
        self.assertEqual(len(result["sector_exposures"]), 1)
        self.assertEqual(result["sector_exposures"][0]["market_value"], "100.00")

    def test_positive_unmapped_issuer_suppresses_every_sector_metric(self):
        result = _assess([_issuer("ISSUER-ONE", "80.00")], _reference())
        self.assertEqual(result["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(
            result["reason_codes"],
            ["issuer_sector_mapping_unresolved"],
        )
        self.assertEqual(result["sector_exposures"], [])
        self.assertEqual(result["issuer_contributions"], [])
        self.assertIsNone(result["largest_sector_weight_pct"])
        self.assertEqual(result["alerts"], [])

    def test_cash_only_is_evaluated_and_zero_value_unmapped_listing_is_traced(self):
        cash_only = _assess([], _reference(), cash="100.00")
        self.assertEqual(cash_only["assessment_status"], "EVALUATED")
        self.assertEqual(cash_only["sector_exposures"], [])
        self.assertIsNone(cash_only["sector_hhi"])
        zero_listing = {
            "exchange": "NSE",
            "symbol": "ZERO",
            "listing": "NSE:ZERO",
            "market_value": "0.00",
            "issuer_id_type": None,
            "issuer_id": None,
        }
        assessment = _issuer_assessment([], cash="100.00", listings=[zero_listing])
        traced = assess_portfolio_sector_concentration(
            session_date=_SESSION,
            issuer_assessment=assessment,
            reference_data=_reference(),
            evaluated_at=_EVALUATED_AT,
        )
        self.assertEqual(traced["assessment_status"], "EVALUATED")
        self.assertEqual(traced["zero_value_listings"][0]["listing"], "NSE:ZERO")

    def test_incomplete_issuer_assessment_and_invalid_totals_fail_closed(self):
        for issuer_reason in (
            "valuation_incomplete",
            "ledger_reconciliation_unavailable",
            "ledger_discrepancy",
        ):
            issuer_assessment = _issuer_assessment(
                [_issuer("ISSUER-ONE", "80.00")],
                status="NOT_EVALUATED",
            )
            issuer_assessment["reason_codes"] = [issuer_reason]
            result = assess_portfolio_sector_concentration(
                session_date=_SESSION,
                issuer_assessment=issuer_assessment,
                reference_data=_reference(_record("ISSUER-ONE")),
                evaluated_at=_EVALUATED_AT,
            )
            with self.subTest(issuer_reason=issuer_reason):
                self.assertEqual(result["assessment_status"], "NOT_EVALUATED")
                self.assertEqual(
                    result["reason_codes"],
                    ["issuer_assessment_unavailable"],
                )
                self.assertEqual(result["sector_exposures"], [])
        assessment = _issuer_assessment([_issuer("ISSUER-ONE", "80.00")])
        assessment["total_equity_market_value"] = "79.00"
        inconsistent = assess_portfolio_sector_concentration(
            session_date=_SESSION,
            issuer_assessment=assessment,
            reference_data=_reference(_record("ISSUER-ONE")),
            evaluated_at=_EVALUATED_AT,
        )
        self.assertEqual(inconsistent["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(inconsistent["sector_exposures"], [])

    def test_serializer_accepts_valid_result_and_rejects_forged_sector_groups(self):
        issuers = [_issuer("ISSUER-A", "60.00"), _issuer("ISSUER-B", "40.00")]
        issuer_assessment = _issuer_assessment(issuers, cash="0.00")
        result = assess_portfolio_sector_concentration(
            session_date=_SESSION,
            issuer_assessment=issuer_assessment,
            reference_data=_reference(
                _record("ISSUER-A", "TECHNOLOGY"),
                _record("ISSUER-B", "CONSUMER"),
            ),
            evaluated_at=_EVALUATED_AT,
        )
        safe_issuer = _safe_issuer_concentration_assessment(issuer_assessment)
        self.assertIsNotNone(safe_issuer)
        self.assertIsNotNone(
            safe_sector_concentration_assessment(result, safe_issuer)
        )
        for mutate in (
            lambda data: data["sector_exposures"].append(
                deepcopy(data["sector_exposures"][0])
            ),
            lambda data: data["sector_exposures"].pop(),
            lambda data: data["sector_exposures"][0].update(
                sector_code="MATERIALS",
                sector_name=SECTOR_TAXONOMY["MATERIALS"],
            ),
            lambda data: data.update(portfolio_nav="0.00"),
            lambda data: data["issuer_contributions"].pop(),
        ):
            malformed = deepcopy(result)
            mutate(malformed)
            with self.subTest(malformed=malformed):
                self.assertIsNone(
                    safe_sector_concentration_assessment(
                        malformed,
                        safe_issuer,
                    )
                )

    def test_serializer_rejects_coherently_reassigned_issuer_sector(self):
        issuers = [_issuer("ISSUER-A", "60.00"), _issuer("ISSUER-B", "40.00")]
        issuer_assessment = _issuer_assessment(issuers, cash="0.00")
        result = assess_portfolio_sector_concentration(
            session_date=_SESSION,
            issuer_assessment=issuer_assessment,
            reference_data=_reference(
                _record("ISSUER-A", "TECHNOLOGY"),
                _record("ISSUER-B", "CONSUMER"),
            ),
            evaluated_at=_EVALUATED_AT,
        )
        safe_issuer = _safe_issuer_concentration_assessment(issuer_assessment)
        altered = deepcopy(result)
        altered["issuer_contributions"][0]["sector_code"] = "CONSUMER"
        altered["issuer_contributions"][0]["sector_name"] = SECTOR_TAXONOMY[
            "CONSUMER"
        ]
        altered["sector_exposures"] = [
            {
                "sector_code": "CONSUMER",
                "sector_name": SECTOR_TAXONOMY["CONSUMER"],
                "market_value": "100.00",
                "portfolio_weight_pct": "100",
                "issuers": [
                    {"issuer_id_type": "OTHER_VERIFIED", "issuer_id": "ISSUER-A"},
                    {"issuer_id_type": "OTHER_VERIFIED", "issuer_id": "ISSUER-B"},
                ],
            }
        ]
        altered["largest_sector_weight_pct"] = "100"
        altered["top_three_sector_weight_pct"] = "100"
        altered["sector_hhi"] = "1"
        self.assertIsNone(safe_sector_concentration_assessment(altered, safe_issuer))

    def test_serializer_rejects_missing_or_altered_reference_binding(self):
        issuer_assessment = _issuer_assessment(
            [_issuer("ISSUER-A", "100.00")],
            cash="0.00",
        )
        result = _assess(
            [_issuer("ISSUER-A", "100.00")],
            _reference(_record("ISSUER-A")),
            cash="0.00",
        )
        safe_issuer = _safe_issuer_concentration_assessment(issuer_assessment)
        self.assertIsNotNone(safe_issuer)
        for mutate in (
            lambda data: data.update(
                reference_snapshot_sha256="0" * 64
                if data["reference_snapshot_sha256"] != "0" * 64
                else "1" * 64
            ),
            lambda data: data["issuer_contributions"][0]["mapping"].update(
                source="Altered source"
            ),
            lambda data: data.pop("reference_snapshot_sha256"),
        ):
            malformed = deepcopy(result)
            mutate(malformed)
            with self.subTest(malformed=malformed):
                self.assertIsNone(
                    safe_sector_concentration_assessment(malformed, safe_issuer)
                )

    def test_reference_snapshot_fingerprint_is_canonical_and_content_sensitive(self):
        issuers = [_issuer("ISSUER-A", "60.00"), _issuer("ISSUER-B", "40.00")]
        first = _assess(
            issuers,
            _reference(
                _record("ISSUER-A", "TECHNOLOGY"),
                _record("ISSUER-B", "CONSUMER"),
            ),
            cash="0.00",
        )
        reordered = _assess(
            issuers,
            _reference(
                _record("ISSUER-B", "CONSUMER"),
                _record("ISSUER-A", "TECHNOLOGY"),
            ),
            cash="0.00",
        )
        changed = _assess(
            issuers,
            _reference(
                _record("ISSUER-A", "MATERIALS"),
                _record("ISSUER-B", "CONSUMER"),
            ),
            cash="0.00",
        )
        self.assertEqual(
            first["reference_snapshot_sha256"],
            reordered["reference_snapshot_sha256"],
        )
        self.assertNotEqual(
            first["reference_snapshot_sha256"],
            changed["reference_snapshot_sha256"],
        )

    def test_tied_sector_order_is_deterministic(self):
        result = _assess(
            [_issuer("ISSUER-A", "50.00"), _issuer("ISSUER-B", "50.00")],
            _reference(
                _record("ISSUER-A", "TECHNOLOGY"),
                _record("ISSUER-B", "CONSUMER"),
            ),
            cash="0.00",
        )
        self.assertEqual(
            [row["sector_code"] for row in result["sector_exposures"]],
            ["CONSUMER", "TECHNOLOGY"],
        )
        self.assertEqual(
            Decimal(result["largest_sector_weight_pct"]),
            Decimal("50"),
        )


if __name__ == "__main__":
    unittest.main()
