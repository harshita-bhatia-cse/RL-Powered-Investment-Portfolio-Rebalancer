# Phase 5.6 — Existing Dataset Recovery & Feasibility Audit

**Audit date:** 2026-11-13  
**Project:** RL-Powered Investment Portfolio Rebalancer  
**Scope:** Read-only inventory and local-data quality review. No external data access.

## 1. Executive Summary

**Decision: NO-GO for historical PPO training or total-return performance claims.**

The repository already contains daily files for all six required ETFs and both
benchmarks. Each ETF file has 2,479 rows from 2017-01-03 through 2026-11-11;
the S&P 500 file has the same coverage, and the VIX file has 2,482 rows through
2026-11-12. The files therefore contain more than the required latest five
years of nominal daily history.

The files are **not a verified adjusted-OHLC dataset**. They contain raw
`open`, `high`, `low`, `close`, and `volume`, plus `adj_close`; no adjusted
open, high, or low fields exist. The FMP fetcher maps `adjClose` to
`adj_close` and falls back to raw `close` if `adjClose` is absent. The
repository does not retain an FMP response snapshot, price-file manifest, or
adjustment-method/version metadata sufficient to establish how these values
were produced or whether they were later revised.

The price files are technically well-formed in the checks performed: no
missing cells, duplicate dates, nonnumeric/non-finite values, or OHLC-order
violations were found in the eight required files. Technical completeness does
not resolve adjustment methodology, corporate-action completeness, source
rights, or model-use permission.

Issuer corporate-action material is partial. A local State Street workbook
contains distribution data for SPY, XLU, and XLV, and the curated evidence
contains 106 verified dividend records for those three ETFs. It contains no
verified split records; QQQ, IEF, and GLD remain without complete verified
action coverage. Yahoo-derived candidate files are explicitly unverified and
have no ledger-ready events.

The local SQLite database has six application tables, all empty. The
`data/raw/` and `datasets/` directories are absent, and `data/processed/` is
empty. Thus the FMP CSV files are the only local daily price history found.

**Recovery conclusion:** Existing files can support further offline
provenance, rights, and methodology investigation. Their presence does not
make them approved training data. Do not start PPO training, describe the
series as adjusted OHLC/verified total return, or treat synthetic adjusted
prices as actual historical executable prices.

## 2. Dataset Inventory

### Daily price files

There are 25 CSVs in `data/fmp_daily/`. All have columns
`date,open,high,low,close,adj_close,volume`; the ticker is inferred from the
filename rather than stored in a ticker column. The eight required symbols are
summarized individually in Section 3.

| Files | Format and content | Rows / observed dates | Missing cells / duplicate dates |
|---|---|---|---|
| `SPY_daily.csv`, `QQQ_daily.csv`, `IEF_daily.csv`, `GLD_daily.csv`, `XLU_daily.csv`, `XLV_daily.csv` | CSV; raw OHLCV plus `adj_close` | 2,479 each; 2017-01-03–2026-11-11 | 0 / 0 for each |
| `^GSPC_daily.csv` | CSV; OHLCV fields plus `adj_close` | 2,479; 2017-01-03–2026-11-11 | 0 / 0 |
| `^VIX_daily.csv` | CSV; OHLCV fields plus `adj_close` | 2,482; 2017-01-03–2026-11-12 | 0 / 0 |
| `AAPL_daily.csv`, `AMZN_daily.csv`, `BHP_daily.csv`, `COP_daily.csv`, `CVX_daily.csv`, `FCX_daily.csv`, `GOOGL_daily.csv`, `IAU_daily.csv`, `META_daily.csv`, `MSFT_daily.csv`, `NVDA_daily.csv`, `SHY_daily.csv`, `SLV_daily.csv`, `TLT_daily.csv`, `TSLA_daily.csv`, `UUP_daily.csv`, `XOM_daily.csv` | CSV; same seven-column daily-price schema | 2,479 each; 2017-01-03–2026-11-11 | 0 / 0 for each |

These are local files in an FMP-named directory and match the field mapping in
[`data_fetcher.py`](../../src/data/data_fetcher.py). The files do not contain
source, retrieval timestamp, adjustment method, API endpoint/version, or
response checksum fields.

### Other tabular and binary data

| File(s) | Format, size/shape, and contents | Relevance / limitations |
|---|---|---|
| `data/fundamental_data_full.csv` | CSV; 22,909 rows, 64 columns, 715 distinct tickers; `datadate` from 2015-06-30 to 2026-03-31; 24,999 blank cells; no duplicate `(ticker, datadate)` keys | Contains financial/fundamental measures and filing/trading-date fields, not daily OHLCV. None of the eight required symbols occurs as a `ticker`. It cannot replace the required price history. |
| `data/sp500_historical_constituents.csv` | CSV; 2,709 rows; columns `date,tickers`; 1996-01-02 to 2026-04-17; no blank cells or duplicate dates | Historical constituent-list snapshots, not price bars. The file alone does not establish source provenance, completeness, or point-in-time licensing. |
| `examples/data/cache/sp500_components_latest.csv` | CSV; one row, columns `date,tickers`; dated 2025-10-03 | A single constituent snapshot, not historical prices. |
| `data/issuer_verified_corporate_actions/evidence/ssga_spdr_historical_distributions.xlsx` | XLSX; one `dividend` sheet, 13,463 data rows and 10 columns: fund name, ticker, CUSIP, ex-date, record date, payable date, dividend, short-term capital gain, long-term capital gain, frequency | Issuer distribution evidence, not daily prices. The workbook contains 136 SPY, 111 XLU, and 106 XLV rows across its full available history; it has no rows for QQQ, IEF, or GLD. Workbook fields include capital-gain distributions, but those are not represented in the 106 curated verified records described below. |
| `data/finrl_trading.db` | SQLite; 65,536 bytes; six application tables | All tables contain zero rows. The price schema has `ticker,date,open,high,low,close,adj_close,volume` but no source or adjustment-method columns. See Section 2.1. |

No Parquet, Feather/Arrow, NumPy, HDF5, joblib, or other identified binary
historical-price dataset was found in the repository data tree.

### Corporate-action and provenance artifacts

| File(s) | Format and observed contents | Status |
|---|---|---|
| `data/issuer_verified_corporate_actions/verified_normalized_events.csv` | CSV; 106 rows, 10 columns; SPY 35, XLU 35, XLV 36; ex-dates span 2018-03-16–2026-09-21 | Curated verified issuer distributions; all 106 evidence rows map to the workbook’s `DIVIDEND ($)` column. No verified splits are recorded. |
| `data/issuer_verified_corporate_actions/verified_events_evidence.csv` | CSV; 106 rows, 19 columns | Preserves issuer workbook row/column references and verification fields for the curated records. The `source_id` is a deterministic local hash, not an issuer event ID. |
| `data/issuer_verified_corporate_actions/unverified_candidates.csv` | CSV; 175 rows, 14 columns; candidates for IEF, QQQ, SPY, XLU, and XLV | Unverified; not a complete or ledger-ready event set. |
| `data/issuer_verified_corporate_actions/manifest.json`, `issuer_research.json`, `issuer_source_records.jsonl` | JSON/JSONL sidecars; the source-record file has 106 records | Provenance, counts, limitations, and source records; not price bars. The manifest reports zero verified split records and incomplete QQQ/IEF/GLD action coverage. |
| `data/yahoo_corporate_actions/yahoo_action_raw.jsonl`, `yahoo_action_candidates.csv`, `issuer_verification_template.csv` | JSONL: 248 records; CSVs: 248 rows each (13 and 12 columns, respectively) | Yahoo-reported candidates/templates, not issuer-verified events. The manifest reports 0 ledger-ready events and requires issuer verification. GLD is absent from the candidate symbols. |
| `data/yahoo_corporate_actions/manifest.json`, `price_comparison.json` | JSON sidecars | Retrieval/validation and price-comparison metadata; not independent price histories or proof of event completeness. |

The local event artifacts are not folded into the daily price CSVs. They come
from different sources/workflows and must remain distinct until each event,
price basis, and source permission is reconciled.

### 2.1 SQLite tables

Read-only inspection found the following tables and row counts:

| Table | Rows |
|---|---:|
| `price_data` | 0 |
| `sp500_components_details` | 0 |
| `raw_payloads` | 0 |
| `news_articles` | 0 |
| `news_fetch_log` | 0 |
| `fundamental_data` | 0 |

The schema is defined in [`data_store.py`](../../src/data/data_store.py). The
database has no stored prices or raw price response payloads to recover.

### 2.2 Requested directory checks

- `data/raw/`: absent.
- `data/processed/`: exists and is empty.
- `datasets/`: absent.
- Parquet/SQLite alternatives: the only database is `data/finrl_trading.db`;
  there are no Parquet or other alternate daily-price files.

## 3. Required Symbol Coverage Matrix

| Symbol | Local daily file | Records | Earliest–latest | Fields present | Adjusted OHLC? |
|---|---|---:|---|---|---|
| SPY | `data/fmp_daily/SPY_daily.csv` | 2,479 | 2017-01-03–2026-11-11 | Date, raw OHLC, `adj_close`, volume | No |
| QQQ | `data/fmp_daily/QQQ_daily.csv` | 2,479 | 2017-01-03–2026-11-11 | Date, raw OHLC, `adj_close`, volume | No |
| IEF | `data/fmp_daily/IEF_daily.csv` | 2,479 | 2017-01-03–2026-11-11 | Date, raw OHLC, `adj_close`, volume | No |
| GLD | `data/fmp_daily/GLD_daily.csv` | 2,479 | 2017-01-03–2026-11-11 | Date, raw OHLC, `adj_close`, volume | No |
| XLU | `data/fmp_daily/XLU_daily.csv` | 2,479 | 2017-01-03–2026-11-11 | Date, raw OHLC, `adj_close`, volume | No |
| XLV | `data/fmp_daily/XLV_daily.csv` | 2,479 | 2017-01-03–2026-11-11 | Date, raw OHLC, `adj_close`, volume | No |
| ^GSPC | `data/fmp_daily/^GSPC_daily.csv` | 2,479 | 2017-01-03–2026-11-11 | Date, OHLCV fields, `adj_close` | No |
| ^VIX | `data/fmp_daily/^VIX_daily.csv` | 2,482 | 2017-01-03–2026-11-12 | Date, OHLCV fields, `adj_close` | No |

The six ETF files have identical date sets. All eight files share 2,479 dates
from 2017-01-03 through 2026-11-11. The VIX file has three additional dates
through 2026-11-12. The available common history covers the latest five-year
analysis window through its last common session, 2026-11-11, but is not current
through the audit date: no ETF or `^GSPC` row exists for 2026-11-12, while
`^VIX` has a row for that date. Coverage remains subject to the source and
quality limitations in this report.

`adj_close` must not be interpreted as adjusted OHLC. The data contain no
`adj_open`, `adj_high`, or `adj_low`; deriving them by scaling raw OHLC would
create synthetic coordinates and would require validated source semantics and
split-basis checks first.

## 4. Data Quality and Financial Methodology

### File-level checks

For the eight required daily files, local checks found:

- all rows chronologically ordered;
- zero blank cells, duplicate dates, nonnumeric/non-finite rows, or
  non-positive OHLC/adjusted-close values;
- zero violations of high/low ordering; and
- one common 2,479-date intersection across all eight symbols.

These are structural/numerical checks only. They do not prove accuracy,
source rights, adjustment completeness, or historical executable prices.

### Raw versus adjusted fields

The local schema has raw OHLC and a separate `adj_close`. In
[`data_fetcher.py`](../../src/data/data_fetcher.py), the FMP price mapping
stores raw open/high/low/close and volume, then sets `adj_close` from
`adjClose`, falling back to raw `close` when `adjClose` is absent. It does not
create adjusted open/high/low. Therefore, equality between close and
`adj_close` is not proof that no distribution occurred.

Observed rows where `adj_close` differs from `close`:

| Symbol | Rows with difference |
|---|---:|
| SPY | 2,440 |
| QQQ | 2,441 |
| IEF | 2,471 |
| GLD | 0 |
| XLU | 2,441 |
| XLV | 2,441 |
| ^GSPC | 0 |
| ^VIX | 0 |

These observed differences are not a provider methodology statement.
Adjustment direction, inclusion of all ordinary/special/capital-gain
distributions, split treatment, split-adjustment basis, and historical
revision policy remain unverified. No retained response snapshot or versioned
manifest was found for the price files.

### Source mixing, bias, and reproducibility

- The price files are grouped under `data/fmp_daily/` and the fetcher contains
  FMP mapping logic. The file rows themselves do not embed a source identifier,
  so the exact provenance of every row cannot be independently established
  from the CSV alone. No evidence was found that providers were mixed inside
  an individual price file, but the absence of a retained source manifest
  prevents a definitive row-level provenance claim.
- Corporate-action artifacts have separate issuer and Yahoo candidate
  provenance. Do not merge Yahoo candidates into verified events, or combine
  raw and adjusted price bases without an explicit, tested contract.
- Historical adjusted values may be revised after later distributions or
  corrections. With no dated price snapshot or revision log, revision risk
  cannot be measured.
- The fixed current six-ETF universe is used across the history; survivorship
  and selection bias cannot be ruled out. The S&P constituent file may be
  useful for point-in-time research, but this audit does not establish its
  lineage or validate its historical membership.
- `fundamental_data_full.csv` contains filing/accepted/trade-date-related
  fields. Any future feature use must respect information availability at each
  decision date to avoid look-ahead bias.
- The presence of these local files does not establish permission for
  university research, model training, faculty review, dashboard display, or
  redistribution. Provider rights remain subject to the earlier provider
  audits and any written authorization.

## 5. Corporate-Action Findings

1. **Issuer workbook:** The State Street workbook locally contains historical
   distribution fields for SPY, XLU, and XLV. For those tickers it includes
   separate dividend and short-/long-term capital-gain columns. It has no
   records for QQQ, IEF, or GLD. Its existence is evidence for its listed
   records only, not for unlisted funds or complete action coverage.
2. **Curated verified output:** The normalized dataset contains 106 verified
   issuer-backed events from 2018-03-16 through 2026-09-21: SPY 35, XLU 35,
   XLV 36. The accompanying evidence rows identify `DIVIDEND ($)` for all
   106. The workbook’s capital-gain columns are not represented in this
   verified output, so do not claim those distributions are accounted for by
   the 106-event ledger file.
3. **Splits:** The issuer manifest records zero verified splits. The XLU
   2025-12-05 split candidate remains unverified. Applying it would be
   speculative.
4. **Unverified events:** The 175-row unverified-candidate file is not
   authoritative. The Yahoo retrieval produced 248 candidate records across
   SPY, QQQ, IEF, XLU, and XLV; its manifest reports zero ledger-ready events
   and requires issuer verification. Candidate absence for GLD is not proof
   that no events occurred.
5. **Accounting implication:** If raw OHLC is used, missing distributions or
   splits can distort returns, holdings, and portfolio NAV. If a truly
   distribution-/split-adjusted series is used, crediting those events again
   would double-count them. The current files do not support choosing either
   method as a validated end-to-end total-return simulation.

## 6. Missing Data and Evidence Requirements

Before training or reporting strategy performance, the project still needs:

- written confirmation of source rights for local retention, university
  research/model training, faculty evaluation, dashboard demonstration, and
  any public display or derived-output publication;
- a frozen, reproducible price snapshot with source, retrieval time,
  schema/version, checksums, and correction/revision tracking;
- documented adjustment conventions and symbol-specific confirmation that
  raw OHLC and adjusted close use compatible split bases;
- complete issuer-backed dividend, capital-gain/special-distribution, and
  split histories for all six ETFs, or a separately approved adjusted-price
  methodology that demonstrably covers those events;
- a decision to use either raw prices plus a complete event ledger, or a
  consistently adjusted price basis without reapplying those same actions;
- source and coverage confirmation for benchmark series, plus clear
  recognition that `^GSPC` is a price-index benchmark and `^VIX` is a context
  series, not an ETF total-return benchmark; and
- point-in-time controls for universe membership, features, event knowledge,
  and data revisions.

## 7. Recommendation

| Use | Status | Reason |
|---|---|---|
| Recover and inspect the existing files locally | **CONDITIONAL GO** | Files are present and structurally usable for further offline validation; preserve them unchanged and keep provenance labels. |
| Use as a verified adjusted-OHLC / total-return dataset | **NO-GO** | Only `adj_close` is adjusted; methodology, revisions, and split basis are unverified, and corporate-action coverage is incomplete. |
| Historical PPO training or performance claims | **NO-GO** | Training would rely on unresolved data rights, adjustment semantics, and corporate-action coverage. |

**Final decision: NO-GO for PPO training.** Existing local data recover the
required nominal daily date coverage, but not the evidence needed to call the
dataset adjusted OHLC, rights-cleared, revision-controlled, or complete for
ETF total-return simulation. No data acquisition is approved by this audit.

## 8. Inspection and Change Log

Files and directories inspected:

- Dataset inventory under `data/`, `examples/data/`, and repository dataset
  file extensions; `.env` contents were not read.
- `src/data/data_fetcher.py`
- `src/data/data_store.py`
- `src/data/data_processor.py`
- `src/data/issuer_corporate_action_verification.py`
- `src/data/CORPORATE_ACTIONS.md`
- `docs/phase_5_3_final_adjusted_price_audit.md`
- `docs/phase_5_5/free_multi_source_architecture_audit.md`
- Local CSV, SQLite, JSON/JSONL, and issuer XLSX metadata described above.

**Created:** `docs/phase_5_6/existing_dataset_audit.md` (this report only).  
**Existing project files/datasets:** not modified or overwritten.  
**External API calls/downloads, purchases, PPO training, and trades:** none.
