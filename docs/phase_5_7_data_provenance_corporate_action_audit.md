# Phase 5.7 — Existing FMP Data Provenance & Corporate Action Reconciliation Audit

**Audit date:** 2026-11-13  
**Project:** RL-Powered Investment Portfolio Rebalancer  
**Budget:** ₹0  
**Scope:** Read-only repository, local data, and existing documentation inspection. No external market-data requests.

## 1. Executive Summary

**Decision: NO-GO for historical PPO training and total-return performance claims.**

The repository contains daily CSV files for SPY, QQQ, IEF, GLD, XLU, XLV,
`^GSPC`, and `^VIX`. The files are in `data/fmp_daily/` and use the schema
`date,open,high,low,close,adj_close,volume`. The name and current
[`FMPFetcher`](../src/data/data_fetcher.py) implementation make FMP the likely
intended source, but the audit cannot prove that the local CSV rows were
actually downloaded from FMP, when they were retrieved, or which exact
request populated them.

The current fetcher uses FMP's `historical-price-eod/full` path and maps
`adjClose` into `adj_close`, falling back to raw `close` if that response field
is missing. This demonstrates current code behavior, not the provenance of
the existing files. The CSVs are not tracked in Git, no retained price
response, price-specific manifest, acquisition log, or populated price
database rows were found, and local filesystem timestamps are not proof of
provider retrieval.

The observed data have raw OHLCV and adjusted close, **not adjusted OHLC**.
The adjustment factor `adj_close / close` can be calculated from local rows,
and prior Phase 5.3 work found agreement with many verified SPY/XLU/XLV
dividend transitions. That is limited corroboration, not an independent
reconstruction of complete FMP adjustments. Adjustment convention, split
basis, inclusion of all ETF distributions, and revision policy remain
unverified. No adjusted OHLC was generated in this audit.

Corporate-action coverage is incomplete. The local State Street workbook has
issuer-published distribution fields for SPY, XLU, and XLV, including separate
short- and long-term capital-gain columns. The curated 106-event ledger
contains only records mapped from the workbook's `DIVIDEND ($)` column; it
does not contain those capital-gain fields. The workbook does not distinguish
ordinary from special dividends or provide a return-of-capital field. QQQ,
IEF, and GLD are not covered by that workbook; no complete split history is
verified for any of the six ETFs. The XLU 2:1 split is only a Yahoo-derived
candidate.

No evidence was found that the current local presence of FMP or action data
grants permission for local retention, academic research, RL training,
faculty demonstrations, a public dashboard, or derived-output publication.
The provider-use questions remain open.

## 2. Data Provenance

### What the current implementation establishes

In [`src/data/data_fetcher.py`](../src/data/data_fetcher.py),
`FMPFetcher.get_price_data`:

- builds a request to the FMP `historical-price-eod/full` path;
- includes a symbol and `from`/`to` date bounds in the current request
  template;
- reads `open`, `high`, `low`, `close`, and `volume` as raw fields and maps
  `adjClose` to `adj_close`; and
- substitutes raw `close` for `adj_close` if `adjClose` is absent.

The fetcher writes normalized price rows to the local `price_data` table when
used through the database workflow. Its current price path does not preserve
the original FMP EOD response or write a price acquisition manifest.
`raw_payloads` is used for selected fundamentals payloads, not for retaining
these historical price responses.

### What Git history establishes—and does not

- The FMP historical-price request code appears in commit `fa95ffb`
  (`2025-12-07`, “add news sentiment and fix fmp api invokation”). The
  committed version shows an earlier URL query-string form; current code has
  since changed. This records the evolution of application code only.
- `git ls-files data/fmp_daily` returns no tracked price files, and Git history
  contains no commits for those CSV paths. The repository `.gitignore`
  excludes `/data/*`.
- No commit or code-history evidence ties a particular run of the fetcher to
  the current CSV contents.

### Local acquisition evidence

- Filenames are stored under `data/fmp_daily/`; that naming is consistent
  with FMP use but is not a source attestation.
- The CSV rows have no source, endpoint, query, plan, retrieval timestamp,
  API/library version, response ID, or checksum metadata.
- A read-only check found no price rows or raw price responses in
  `data/finrl_trading.db`.
- No project `logs/` or `log/` directory, FMP acquisition log, or
  `data/fmp_daily/` sidecar manifest was found.
- The filesystem exposes creation/last-write timestamps around November 11–12,
  2026. Such timestamps can be changed by copy/synchronization and do not
  identify a provider, API call, or retrieval time.

**Provenance conclusion:** FMP is the likely intended source, based on the
directory and code, but actual source, retrieval dates, exact query
parameters, plan entitlement, and original response content for these local
rows are **UNVERIFIED**. Do not record “downloaded from FMP” as a confirmed
fact without an acquisition record or the original response.

## 3. Adjustment Methodology

### Fields and observed basis

All six ETF files have raw `open`, `high`, `low`, and `close`, a separate
`adj_close`, and `volume`. There are no `adj_open`, `adj_high`, or `adj_low`
columns. The files therefore cannot be called adjusted OHLC.

For the local snapshot, `adj_close` differs from `close` on 2,440 SPY rows,
2,441 QQQ rows, 2,471 IEF rows, 0 GLD rows, 2,441 XLU rows, and 2,441 XLV
rows. Equality for GLD is not evidence that no distribution occurred because
the fetcher has a raw-close fallback when `adjClose` is absent.

### Adjustment-factor reconciliation

The ratio `adj_close / close` is calculable from the local data, but calculating
that ratio is not the same as independently rebuilding the adjustment history
from complete, verified actions. The Phase 5.3 local reconciliation reported
that the one-session factor change was within 1% of the simple dividend
adjustment expression on all 35 verified SPY and all 36 verified XLV event
dates, and on 34 of 35 verified XLU dates. This supports consistency for those
observed dividend records only.

It does not establish:

- the provider's formal backward/forward adjustment convention;
- whether the raw OHLC and `adjClose` use the same split-adjusted share basis;
- treatment of capital-gain, special, or return-of-capital distributions;
- completeness of events for QQQ, IEF, GLD, or splits;
- whether adjusted values have been revised since the files were created; or
- a reproducible snapshot/version of the underlying provider responses.

There is no retained provider adjustment specification or revision history in
the repository. The prior provider audit could not verify FMP's terms and
endpoint details from its official pages at that time. The local files do not
resolve those unknowns.

**No adjusted OHLC derivation was performed.** Do not automatically multiply
raw OHLC by `adj_close / close`. That would create synthetic adjusted-price
coordinates and remains conditional on unverified source and split-basis
assumptions. Adjusted opens, if ever produced, must not be described as actual
historical executable prices.

## 4. Corporate-Action Completeness by ETF

### Evidence classes used

- **Issuer workbook record:** an amount/date is present in the local State
  Street historical-distributions workbook. This is evidence for the
  published record, not proof that every event class or the complete event
  history is captured.
- **Curated verified ledger:** an event appears in
  `verified_normalized_events.csv` and its source row/amount column is
  preserved in `verified_events_evidence.csv`.
- **Unverified candidate:** a Yahoo-derived or other candidate has not passed
  the required issuer verification; it is not ledger-ready.
- **Missing/unverified:** no complete, independently confirmed event record
  was found in the inspected local artifacts. Absence of a candidate is not
  evidence that no event occurred.

| ETF | Ordinary / cash-dividend evidence | Special dividends | Capital-gain distributions | Return of capital | Splits |
|---|---|---|---|---|---|
| **SPY** | **Verified, partial:** 35 curated issuer-backed `DIVIDEND ($)` events, 2018-03-16–2026-09-18. The workbook has 39 dividend-field rows from 2017-01-03 onward. The workbook does not label these ordinary vs. special. | **Unverified classification:** no separate ordinary/special flag in the workbook or normalized ledger. | **Issuer workbook fields present but not in ledger:** 6 nonblank short-term and 4 nonblank long-term capital-gain cells in the requested period. They are not among the 106 normalized events and need reconciliation/normalization. | **Missing:** no dedicated return-of-capital field or verified classification found. | **Unverified:** no issuer-verified split record found. |
| **QQQ** | **Unverified:** no QQQ rows in the State Street workbook; 35 Yahoo dividend candidates are unverified, not issuer-verified ledger events. | **Missing/unverified:** not classified by the available records. | **Missing/unverified:** no verified event-level capital-gain records found in inspected local artifacts. | **Missing/unverified:** no dedicated verified record found. | **Unverified:** no verified split record; lack of Yahoo split candidate is not proof of no split. |
| **IEF** | **Unverified:** no IEF rows in the State Street workbook; 106 Yahoo dividend candidates are unverified, not issuer-verified ledger events. | **Missing/unverified:** not classified by the available records. | **Missing/unverified:** no verified event-level capital-gain records found in inspected local artifacts. | **Missing/unverified:** no dedicated verified record found. | **Unverified:** no verified split record; lack of Yahoo split candidate is not proof of no split. |
| **GLD** | **Partial negative evidence only:** no GLD row in the State Street workbook and no Yahoo candidate. A preserved GLD prospectus reviewed in prior work states that the trust does not generate income and holders will not receive dividends; it does not prove complete 2017–2026 action coverage. | **Missing/unverified:** complete action history not established. | **Missing/unverified:** no verified event-level capital-gain records found. | **Missing/unverified:** no dedicated verified record found. | **Unverified:** no complete issuer-verified split history found. |
| **XLU** | **Verified, partial:** 35 curated issuer-backed `DIVIDEND ($)` events, 2018-03-16–2026-09-21. The workbook has 39 dividend-field rows from 2017-01-03 onward. The workbook does not label these ordinary vs. special. | **Unverified classification:** no separate ordinary/special flag in the workbook or normalized ledger. | **Issuer workbook fields present but not in ledger:** 6 nonblank short-term and 5 nonblank long-term capital-gain cells in the requested period; not represented in the 106 normalized events. | **Missing:** no dedicated return-of-capital field or verified classification found. | **Unverified:** one Yahoo 2:1 split candidate dated 2025-12-05; no official split notice or verified ledger event. |
| **XLV** | **Verified, partial:** 36 curated issuer-backed `DIVIDEND ($)` events, 2018-03-16–2026-09-21. The workbook has 40 dividend-field rows from 2017-01-03 onward. The workbook does not label these ordinary vs. special. | **Unverified classification:** no separate ordinary/special flag in the workbook or normalized ledger. | **Issuer workbook fields present but not in ledger:** 6 nonblank short-term and 5 nonblank long-term capital-gain cells in the requested period; not represented in the 106 normalized events. | **Missing:** no dedicated return-of-capital field or verified classification found. | **Unverified:** no issuer-verified split record found. |

The workbook's short-/long-term capital-gain counts above refer to nonblank
source cells, not necessarily distinct event dates; both fields can occur on
the same date. The normalized event file contains only the 106 records mapped
from `DIVIDEND ($)`. Its manifest reports zero verified split records and
explicitly flags incomplete QQQ, IEF, and GLD coverage.

The Yahoo candidate artifacts contain 247 `Dividends` rows and one `Stock
Splits` row (XLU). Their manifest reports 248 candidates requiring issuer
verification and **zero ledger-ready events**. They are not evidence of
completeness. The workbook has no dedicated return-of-capital classification;
the presence of an unlabeled cash-distribution amount cannot be treated as
proof that return of capital is absent.

## 5. Licensing and Retention Status

| Intended use | Evidence found | Status |
|---|---|---|
| Existing local storage of FMP-named CSVs | Files exist locally; no FMP license, acquisition receipt, or written permission record was found. Prior Phase 5.4 report records FMP official pages were inaccessible during its research and exact project rights remained unresolved. | **Unverified; local presence is not permission.** |
| Academic research and RL/PPO model training | No FMP or other provider document or written approval covering GLA University research/model training was found. | **Unverified / not approved.** |
| Faculty review or private demonstration | No written permission for faculty-facing charts, price-derived outputs, or dashboard review was found. | **Unverified / not approved.** |
| Public dashboard or public demonstration | No public-display or redistribution permission was found. | **Unverified / not approved.** |
| Publication of derived outputs, model weights, signals, or portfolio histories | No terms or written approval addressing these outputs were found. | **Unverified; do not assume derived-output rights.** |
| Raw market-data redistribution or GitHub publication | No redistribution right was found. The repository's Apache 2.0 `LICENSE` is a software license and does not establish rights to third-party market data. `/data/*` is ignored by Git. | **Not approved on current evidence.** |
| State Street workbook retention/reuse | The local manifest identifies State Street and its workbook URL, but explicitly warns not to commit or redistribute issuer data without confirming applicable rights. No project-specific written license was found. | **Use rights unverified.** |
| Yahoo-derived action candidates | The Yahoo manifest says to review Yahoo terms before retention, use, or redistribution; prior audit records that automated collection requires express prior permission and none was verified. | **No verified permission; candidates remain unverified and should not be redistributed.** |

The strict ₹0 budget does not itself establish a free license. Written
permission should explicitly cover local retention (including backups and
post-project retention), non-commercial university research, training and
evaluation of RL/PPO models, faculty access, private and public dashboard
demonstrations, publication of derived outputs/model artifacts, and any raw
data redistribution. Ask whether separate exchange/index terms apply to
`^GSPC` and `^VIX`.

### Provider and issuer references already recorded in the repository

- FMP [Dividend Adjusted Price Chart API](https://site.financialmodelingprep.com/developer/docs/stable/historical-price-eod-dividend-adjusted),
  [FAQ](https://site.financialmodelingprep.com/faqs),
  [pricing plans](https://site.financialmodelingprep.com/pricing-plans), and
  [terms of service](https://site.financialmodelingprep.com/terms-of-service):
  the Phase 5.4 audit records these pages as inaccessible (HTTP 403) on
  2026-11-13. No methodology or project-use permission was confirmed there.
- State Street's [ETF historical-distributions workbook](https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/spdr-etf-historical-distributions.xlsx)
  is named in the local issuer manifest and represented by the local XLSX.
  The manifest cautions against committing or redistributing issuer data
  without confirming applicable rights.
- Yahoo Finance
  [Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)
  are referenced by the local corporate-action documentation. No express
  prior permission for automated collection is recorded in the project.

## 6. Unresolved Financial Risks

1. **Unknown row-level source:** the FMP directory and fetcher suggest FMP,
   but the CSVs cannot be authenticated to a particular FMP response.
2. **Raw/adjusted basis ambiguity:** raw OHLC and `adj_close` coexist, but
   adjustment semantics and compatible split basis are undocumented locally.
3. **Incomplete distributions:** only 106 issuer-backed dividend records are
   in the normalized ledger. Capital-gain cells visible in the State Street
   workbook are not included there; QQQ, IEF, and GLD event coverage remains
   incomplete.
4. **Unclassified distributions:** the source workbook does not distinguish
   ordinary/special dividends or return of capital. Those categories cannot
   be silently inferred from a generic dividend field.
5. **Unverified splits:** no split is verified for any ETF; XLU has an
   unresolved 2:1 candidate. Applying an unverified split, or combining a
   split-adjusted price series with a share-count split, can distort portfolio
   holdings and NAV.
6. **Double counting:** if adjusted prices already include distributions or
   splits, applying the event ledger as well could double-count them. If raw
   prices are used with an incomplete ledger, total returns can be understated
   or holdings can be wrong.
7. **Revisions and reproducibility:** there is no frozen FMP response, dated
   price snapshot manifest, or revision comparison. Historical adjusted
   values may be restated without this audit being able to detect it.
8. **Synthetic execution:** any future adjusted OHLC created by a documented
   method would be synthetic adjusted-price coordinates, not historical
   executable quotes.
9. **Look-ahead and survivorship:** later corrections/distribution knowledge
   and a fixed six-ETF universe can bias a backtest unless event availability,
   point-in-time membership, and feature timestamps are controlled.
10. **Data rights:** model training, demonstrations, public display, and
    derived-output publication remain unapproved on the evidence inspected.

## 7. Recommended Next Steps

1. **Do not train or claim total-return results from the current files.** Keep
   them unchanged and label their provenance and adjustment basis as
   unverified.
2. Search project-owner records for an original FMP acquisition receipt,
   export command/session, saved response, API plan, and written data-use
   terms. Do not infer a provider from directory naming or timestamps.
3. Ask FMP for written confirmation of the exact historical endpoint/plan,
   these symbol/date histories, `adjClose` and raw-OHLC split bases,
   ordinary/special/capital-gain/return-of-capital handling, historical
   revision policy, and permission for local retention, academic RL training,
   faculty evaluation, dashboard display, and derived outputs.
4. Ask GLA University's library/data administrator whether an existing
   institutional entitlement covers these exact data and project uses.
5. Reconcile all workbook capital-gain records with the issuer source and
   define an explicit event taxonomy for ordinary/special distributions,
   capital gains, return of capital, and splits. Obtain issuer-backed
   event-level evidence for QQQ, IEF, and GLD, and verify any split candidates.
6. Only after provenance, licensing, and corporate-action completeness are
   established, freeze an immutable local snapshot with endpoint, plan,
   request parameters, retrieval timestamp, source/version, checksum, and
   correction history. Then choose one consistent basis: raw OHLC plus a
   complete event ledger, or documented adjusted prices without reapplying
   those same actions.

## 8. GO / NO-GO Decision

| Area | Decision | Basis |
|---|---|---|
| Existing files available for offline inspection | **CONDITIONAL GO** | Files exist and have useful coverage, but exact acquisition provenance is not established. |
| Claim that the local CSVs were downloaded from FMP | **NO-GO / unverified** | Code and naming are suggestive; no source-attested export, response, log, or Git-tracked data proves it. |
| Claim of verified adjusted OHLC or complete total return | **NO-GO** | Only adjusted close is present; methodology, split basis, revision history, and full event coverage are unresolved. |
| Local retention, academic training, demonstrations, or public/derived outputs | **NO-GO pending written permission** | No applicable project-specific rights evidence was found. |
| Historical PPO training | **NO-GO** | Financial correctness, provenance, action completeness, and rights gates are unmet. |

**Final recommendation: NO-GO for historical PPO training.** Continue only with
read-only provenance and rights recovery until every evidence gap is resolved.

## 9. Files and Scope Inspected

- `src/data/data_fetcher.py`
- `src/data/data_store.py`
- `src/data/CORPORATE_ACTIONS.md`
- `.gitignore` (data ignore rule only; `.env` contents were not read)
- `LICENSE`
- `data/fmp_daily/*.csv` file metadata and required-symbol rows
- `data/finrl_trading.db` schema and row counts
- `data/issuer_verified_corporate_actions/manifest.json`
- `data/issuer_verified_corporate_actions/issuer_research.json`
- `data/issuer_verified_corporate_actions/verified_normalized_events.csv`
- `data/issuer_verified_corporate_actions/verified_events_evidence.csv`
- `data/issuer_verified_corporate_actions/unverified_candidates.csv`
- `data/issuer_verified_corporate_actions/issuer_source_records.jsonl`
- `data/issuer_verified_corporate_actions/evidence/ssga_spdr_historical_distributions.xlsx`
- `data/yahoo_corporate_actions/manifest.json`
- `data/yahoo_corporate_actions/yahoo_action_candidates.csv`
- `data/yahoo_corporate_actions/yahoo_action_raw.jsonl`
- `docs/phase_5_3_final_adjusted_price_audit.md`
- `docs/phase_5_4_data_provider_feasibility.md`
- `docs/phase_5_5/free_multi_source_architecture_audit.md`
- `docs/phase_5_6/existing_dataset_audit.md`
- Git history for `src/data/data_fetcher.py` and `data/fmp_daily/`
- Repository log-directory and price-file sidecar presence

**Created:** `docs/phase_5_7_data_provenance_corporate_action_audit.md` (this
report only).  
**Existing datasets and production files:** not modified.  
**`.env`:** not opened or read.  
**External API calls, downloads, purchases, PPO training, and trades:** none.
