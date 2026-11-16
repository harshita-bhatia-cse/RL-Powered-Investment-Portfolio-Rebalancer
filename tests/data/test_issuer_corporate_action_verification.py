import csv
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from src.data.corporate_action_provider import (
    EVENT_COLUMNS,
    CorporateActionValidationError,
    load_normalized_events,
)
from src.data.issuer_corporate_action_verification import (
    UNVERIFIED_COLUMNS,
    build_issuer_verification_dataset,
)
from src.strategies.allocation import (
    CorporateActionEvent,
    CorporateActionLedger,
    CorporateActionType,
)

HEADERS = (
    "FUND NAME",
    "TICKER",
    "CUSIP",
    "EX-DATE",
    "RECORD DATE",
    "PAYABLE DATE",
    "DIVIDEND ($)",
    "SHORT TERM CAPITAL GAIN ($)",
    "LONG TERM CAPITAL GAIN ($)",
    "FREQUENCY",
)


def _column_name(index):
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def write_test_workbook(path, rows):
    shared = []
    shared_ids = {}

    def shared_id(value):
        if value not in shared_ids:
            shared_ids[value] = len(shared)
            shared.append(value)
        return shared_ids[value]

    xml_rows = []
    all_rows = [HEADERS, *rows]
    for row_number, values in enumerate(all_rows, start=1):
        cells = []
        for column_number, value in enumerate(values, start=1):
            value = str(value)
            cells.append(
                f'<c r="{_column_name(column_number)}{row_number}" t="s">'
                f"<v>{shared_id(value)}</v></c>"
            )
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    shared_xml = (
        f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(shared)}" uniqueCount="{len(shared)}">'
        + "".join(f"<si><t>{escape(value)}</t></si>" for value in shared)
        + "</sst>"
    )
    sheet_xml = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )
    workbook_xml = (
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="dividend" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels_xml = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Target="worksheets/sheet1.xml" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def write_candidates(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "symbol", "event_type", "source_id", "currency", "announcement_date",
            "ex_date", "payment_date", "effective_date", "distribution_per_share",
            "split_ratio", "source_field", "verification_status", "missing_fields",
        ))
        writer.writeheader()
        writer.writerows(rows)


def candidate(
    symbol="SPY",
    event_type="etf_distribution",
    action_date="2018-01-04",
    amount="1.235",
    split_ratio="",
    source_field="Dividends",
):
    return {
        "symbol": symbol,
        "event_type": event_type,
        "source_id": "",
        "currency": "USD",
        "announcement_date": "",
        "ex_date": action_date if event_type != "split" else "",
        "payment_date": "",
        "effective_date": action_date if event_type == "split" else "",
        "distribution_per_share": amount if event_type != "split" else "",
        "split_ratio": split_ratio,
        "source_field": source_field,
        "verification_status": "issuer_verification_required",
        "missing_fields": "source_id|payment_date",
    }


def issuer_row(symbol, ex_date, amount, payment_date="2018-01-08"):
    return (
        f"{symbol} issuer fund",
        symbol,
        "123456789",
        ex_date,
        "01/05/2018",
        payment_date,
        amount,
        "",
        "",
        "Quarterly",
    )


class IssuerCorporateActionVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.candidates = self.root / "candidates.csv"
        self.workbook = self.root / "issuer.xlsx"
        self.output = self.root / "output"
        self.prices = self.root / "prices"
        self.prices.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def make_market_sessions(self):
        return {
            symbol: (date(2018, 1, 4), date(2018, 1, 8))
            for symbol in ("SPY", "QQQ", "IEF", "GLD", "XLU", "XLV")
        }

    def test_build_uses_precise_issuer_amount_and_local_deterministic_id(self):
        write_test_workbook(
            self.workbook,
            [issuer_row("SPY", "01/04/2018", "1.234567")],
        )
        write_candidates(self.candidates, [candidate()])
        manifest = build_issuer_verification_dataset(
            candidates_path=self.candidates,
            issuer_workbook_path=self.workbook,
            output_dir=self.output,
            local_price_dir=self.prices,
            market_sessions=self.make_market_sessions(),
        )
        with (self.output / "verified_normalized_events.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            events = list(csv.DictReader(handle))
        with (self.output / "verified_events_evidence.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            evidence = list(csv.DictReader(handle))

        self.assertEqual(tuple(events[0]), EVENT_COLUMNS)
        self.assertEqual(events[0]["distribution_per_share"], "1.234567")
        self.assertTrue(events[0]["source_id"].startswith("local-ssga-sha256-"))
        self.assertEqual(events[0]["payment_date"], "2018-01-08")
        self.assertEqual(evidence[0]["record_date"], "2018-01-05")
        self.assertEqual(evidence[0]["official_provider_event_id"], "")
        self.assertEqual(
            evidence[0]["identifier_origin"],
            "locally generated deterministic SHA-256",
        )
        loaded_events = load_normalized_events(
            self.output / "verified_normalized_events.csv",
            start_date="2018-01-03",
            end_date="2018-01-10",
            market_sessions=self.make_market_sessions()["SPY"],
        )
        self.assertEqual(len(loaded_events), 1)
        self.assertEqual(manifest["counts"]["issuer_verified_normalized_events"], 1)
        self.assertEqual(manifest["counts"]["yahoo_candidates_unverified"], 0)

    def test_xlu_basis_difference_and_split_remain_unverified(self):
        write_test_workbook(
            self.workbook,
            [issuer_row("XLU", "01/04/2018", "0.386647")],
        )
        write_candidates(
            self.candidates,
            [
                candidate(
                    symbol="XLU",
                    amount="0.1935",
                ),
                candidate(
                    symbol="XLU",
                    event_type="split",
                    action_date="2018-01-04",
                    split_ratio="2.0",
                    source_field="Stock Splits",
                ),
            ],
        )
        manifest = build_issuer_verification_dataset(
            candidates_path=self.candidates,
            issuer_workbook_path=self.workbook,
            output_dir=self.output,
            local_price_dir=self.prices,
            market_sessions=self.make_market_sessions(),
        )
        with (self.output / "unverified_candidates.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            unresolved = list(csv.DictReader(handle))
        self.assertEqual(len(unresolved), 2)
        self.assertTrue(
            any("share-basis change" in row["verification_reason"] for row in unresolved)
        )
        self.assertTrue(
            any("does not verify split" in row["verification_reason"] for row in unresolved)
        )
        self.assertEqual(manifest["counts"]["issuer_verified_normalized_events"], 1)
        self.assertEqual(manifest["counts"]["issuer_verified_split_records"], 0)
        self.assertEqual(tuple(unresolved[0]), UNVERIFIED_COLUMNS)

    def test_duplicate_issuer_rows_are_rejected_not_silently_deduplicated(self):
        duplicate = issuer_row("SPY", "01/04/2018", "1.234567")
        write_test_workbook(self.workbook, [duplicate, duplicate])
        write_candidates(self.candidates, [])
        with self.assertRaisesRegex(CorporateActionValidationError, "Duplicate"):
            build_issuer_verification_dataset(
                candidates_path=self.candidates,
                issuer_workbook_path=self.workbook,
                output_dir=self.output,
                local_price_dir=self.prices,
                market_sessions=self.make_market_sessions(),
            )

    def test_conflicting_issuer_and_duplicate_candidate_records_are_rejected(self):
        write_test_workbook(
            self.workbook,
            [
                issuer_row("SPY", "01/04/2018", "1.0"),
                issuer_row("SPY", "01/04/2018", "1.1"),
            ],
        )
        write_candidates(self.candidates, [])
        with self.assertRaisesRegex(CorporateActionValidationError, "Conflicting"):
            build_issuer_verification_dataset(
                candidates_path=self.candidates,
                issuer_workbook_path=self.workbook,
                output_dir=self.output,
                local_price_dir=self.prices,
                market_sessions=self.make_market_sessions(),
            )

        write_test_workbook(self.workbook, [])
        duplicate = candidate()
        write_candidates(self.candidates, [duplicate, duplicate])
        with self.assertRaisesRegex(CorporateActionValidationError, "Duplicate Yahoo"):
            build_issuer_verification_dataset(
                candidates_path=self.candidates,
                issuer_workbook_path=self.workbook,
                output_dir=self.output,
                local_price_dir=self.prices,
                market_sessions=self.make_market_sessions(),
            )

    def test_missing_market_session_and_nonpositive_amount_are_rejected(self):
        write_test_workbook(
            self.workbook,
            [issuer_row("SPY", "01/06/2018", "1.0")],
        )
        write_candidates(self.candidates, [])
        with self.assertRaisesRegex(CorporateActionValidationError, "not a local market session"):
            build_issuer_verification_dataset(
                candidates_path=self.candidates,
                issuer_workbook_path=self.workbook,
                output_dir=self.output,
                local_price_dir=self.prices,
                market_sessions=self.make_market_sessions(),
            )

        write_test_workbook(
            self.workbook,
            [issuer_row("SPY", "01/04/2018", "-1.0")],
        )
        with self.assertRaisesRegex(CorporateActionValidationError, "non-positive"):
            build_issuer_verification_dataset(
                candidates_path=self.candidates,
                issuer_workbook_path=self.workbook,
                output_dir=self.output,
                local_price_dir=self.prices,
                market_sessions=self.make_market_sessions(),
            )

    def test_same_day_distributions_and_split_are_idempotent_in_ledger(self):
        events = (
            CorporateActionEvent(
                symbol="SPY",
                event_type=CorporateActionType.ETF_DISTRIBUTION,
                source_id="synthetic-dividend-a",
                ex_date="2018-01-04",
                payment_date="2018-01-08",
                distribution_per_share=0.5,
            ),
            CorporateActionEvent(
                symbol="SPY",
                event_type=CorporateActionType.ETF_DISTRIBUTION,
                source_id="synthetic-dividend-b",
                ex_date="2018-01-04",
                payment_date="2018-01-08",
                distribution_per_share=0.25,
            ),
            CorporateActionEvent(
                symbol="SPY",
                event_type=CorporateActionType.SPLIT,
                source_id="synthetic-split",
                effective_date="2018-01-04",
                split_ratio=2.0,
            ),
        )
        ledger = CorporateActionLedger({"SPY": 2.0}, settled_cash=0.0)
        before = ledger.snapshot({"SPY": 100.0})
        ledger.process_date("2018-01-04", events)
        ex_date = ledger.snapshot({"SPY": 49.625})
        ledger.process_date("2018-01-04", events)
        replayed = ledger.snapshot({"SPY": 49.625})
        ledger.process_date("2018-01-08", events)
        paid = ledger.snapshot({"SPY": 49.625})

        self.assertEqual(ex_date.asset_quantities, {"SPY": 4.0})
        self.assertEqual(len(ex_date.dividend_receivables), 2)
        self.assertEqual(ex_date.total_nav, before.total_nav)
        self.assertEqual(replayed.total_nav, before.total_nav)
        self.assertEqual(ledger.settled_cash, 1.5)
        self.assertEqual(paid.dividend_receivables, {})
        self.assertEqual(paid.total_nav, before.total_nav)


if __name__ == "__main__":
    unittest.main()
