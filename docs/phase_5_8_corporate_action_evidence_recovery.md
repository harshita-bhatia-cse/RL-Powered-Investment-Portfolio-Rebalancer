# Phase 5.8 — Corporate Action & Adjustment Evidence Recovery

**Audit date:** 2026-11-13  
**Project:** RL-Powered Investment Portfolio Rebalancer  
**Scope:** Read-only examination of existing project files and previously recorded local comparisons. No external market-data requests were made for this audit.

## Executive Summary

**Decision: NO-GO for historical PPO training or total-return claims.**

Local price files exist for all six ETFs and both benchmarks. For the six ETFs,
they contain 2,479 daily rows from 2017-01-03 through 2026-11-11, with raw
OHLCV and `adj_close`. They do not contain adjusted OHLC.

Important provenance evidence was recovered: the existing
`data/yahoo_corporate_actions/price_comparison.json` records that each local
ETF file was compared with Yahoo Finance via yfinance over 2,227 common
sessions, 2018-01-03 through 2026-11-11. It reports **zero relative
difference in raw close** for all six ETFs and maximum adjusted-close relative
differences between zero and approximately 1.03 × 10⁻⁶. This is strong evidence
that the local price series are numerically consistent with that Yahoo
history over the comparison window. It does **not** prove which provider
originally supplied the CSVs: shared upstream values or a later copy remain
possible. It makes the FMP directory name and FMP fetcher insufficient to
assert FMP provenance, and it makes Yahoo acquisition/use rights a material
unresolved risk.

For SPY, XLU, and XLV, a local State Street workbook has dividend and
capital-gain columns. The workbook rows in the audited price range show
positive values in the dividend column, but the populated short- and
long-term capital-gain cells inspected are zero; no positive capital-gain
amount was found in those fields. The normalized issuer event file contains
106 dividend records only: SPY 35, XLU 35, XLV 36. It does not contain
capital-gain events, classified special distributions, return of capital, or
verified splits. QQQ, IEF, and GLD have no issuer-verified distributions in
that file. The State Street workbook does not cover QQQ, IEF, or GLD.

Adjustment-factor changes line up with the observed workbook dividend dates
for SPY/XLU/XLV, and with unverified Yahoo candidate dates for QQQ and IEF,
within the inspected history. This corroborates the local `adj_close` values
against selected dates; it does not reconstruct a complete adjustment ledger
or establish provider methodology. The XLU 2025-12-05 split candidate has no
corresponding material `adj_close / close` factor transition and no issuer
split notice in the local evidence. No missing event has been inferred from
price movement.

No evidence was found for project-specific rights to retain or use the price
data for academic research, RL training, faculty review, dashboard display,
or derived-output publication. Do not treat free or previously downloaded
data as rights-cleared.

## 1. Evidence Discovered

### Price and corporate-action artifacts

| Artifact | What it contains | Evidential limit |
|---|---|---|
| `data/fmp_daily/{SPY,QQQ,IEF,GLD,XLU,XLV}_daily.csv` | Daily date, raw OHLCV, and `adj_close`; 2017-01-03–2026-11-11 | No source, retrieval timestamp, query, plan, adjustment version, or original-response metadata in the files. |
| `data/issuer_verified_corporate_actions/evidence/ssga_spdr_historical_distributions.xlsx` | Local State Street workbook; fields include ticker, ex/record/payable dates, dividend, short-term capital gain, long-term capital gain, and frequency | Covers SPY, XLU, and XLV among the required ETFs; does not establish a complete event history or cover QQQ, IEF, or GLD. No ordinary/special or return-of-capital classification field. |
| `data/issuer_verified_corporate_actions/verified_normalized_events.csv` and `verified_events_evidence.csv` | 106 normalized issuer-backed `DIVIDEND ($)` events with workbook row/column references | SPY 35, XLU 35, XLV 36; all are dividend-column records. No verified capital gains or splits. Local source IDs are generated hashes, not issuer event IDs. |
| `data/issuer_verified_corporate_actions/unverified_candidates.csv` | 175 retained unverified candidates | Not an authoritative event source and not ledger-ready. |
| `data/issuer_verified_corporate_actions/manifest.json` and `issuer_research.json` | Acquisition/validation metadata, source URLs, per-symbol status, limitations, and GLD prospectus findings | The manifest explicitly records zero verified splits and incomplete QQQ/IEF/GLD coverage. |
| `data/issuer_verified_corporate_actions/evidence/GLD_Prospectus_10042022_as_filed.pdf` | Locally preserved prospectus referenced by issuer research metadata | Prior repository research records statements that GLD's trust does not generate income and holders will not receive dividends. This is limited evidence about dividends, not proof of complete five-year corporate-action or split coverage. |
| `data/yahoo_corporate_actions/yahoo_action_candidates.csv`, `yahoo_action_raw.jsonl`, and `issuer_verification_template.csv` | Yahoo/yfinance-derived action candidates and review template | 248 candidates total: 247 dividend rows and one XLU split row; the manifest says 0 ledger-ready events and requires issuer verification. |
| `data/yahoo_corporate_actions/manifest.json` | Previously recorded Yahoo/yfinance history retrieval and comparison provenance | Records one history query per symbol, 2018-01-03–2026-11-11, 2,227 price rows per symbol, and no failed symbols; this audit did not repeat those requests. |
| `data/yahoo_corporate_actions/price_comparison.json` | Prior local-vs-Yahoo raw/adjusted-close comparison and factor-jump diagnostics | Strong numeric cross-source match for the overlap, but not proof of original acquisition source or event completeness. |
| `src/data/issuer_corporate_action_verification.py` and `src/data/CORPORATE_ACTIONS.md` | Workbook parsing, event normalization, validation, provenance, and usage cautions | Code describes the local workflow; it does not prove that unrecorded events do not exist. |
| `docs/phase_5_3_final_adjusted_price_audit.md`, `docs/phase_5_4_data_provider_feasibility.md`, `docs/phase_5_6/existing_dataset_audit.md`, and `docs/phase_5_7_data_provenance_corporate_action_audit.md` | Prior local price, provider, provenance, and event findings | Existing audit conclusions remain conditional on their recorded evidence and are not provider confirmation. |

No new workbook, spreadsheet, acquisition log, or issuer notice was found in
the inspected repository artifacts beyond the items listed above. `.env`
contents were not read.

### Issuer workbook field interpretation

For ex-dates on or after 2017-01-03, the local workbook contains:

| ETF | Dividend-column rows (positive amounts) | Nonblank short-term capital-gain cells | Nonblank long-term capital-gain cells | Positive capital-gain amounts found |
|---|---:|---:|---:|---:|
| SPY | 39 | 6 | 4 | 0 |
| XLU | 39 | 6 | 5 | 0 |
| XLV | 40 | 6 | 5 | 0 |

The capital-gain counts are nonblank cells, not event counts; the populated
cells inspected contain numeric zero. Some cells are blank. This local
workbook evidence does not establish that the issuer has no capital-gain
distributions outside this workbook/version or that an unlabeled dividend
amount has a particular tax classification.

## 2. Six-ETF Corporate-Action Matrix

Status legend:

- **Verified (partial):** event-level evidence is preserved from the local
  issuer workbook, but this does not certify complete period coverage.
- **Unverified candidate:** present in Yahoo-derived candidate data only;
  not issuer-verified or ledger-ready.
- **Not established:** no adequate local evidence to classify or exclude the
  event type.

| ETF | Issuer-verified dividend records | Other dividend evidence / Yahoo candidates | Capital-gain distributions | Special distributions / return of capital | Stock splits |
|---|---|---|---|---|---|
| **SPY** | **Verified (partial):** 35 normalized records, 2018-03-16–2026-09-18; 20 are in the latest-five-year window starting 2021-11-13. The workbook has 39 positive dividend rows from 2017-01-03. | Yahoo has 35 dividend candidates for 2018–2026; these are not independent issuer confirmation. Workbook has no ordinary-vs-special flag. | No positive short-/long-term capital-gain amount found in workbook fields. No capital-gain row is present in normalized events. | **Not established:** no special-distribution classification or return-of-capital field. | **Not established:** no verified split record. |
| **QQQ** | **None found** in the State Street workbook or normalized issuer ledger. | Yahoo has 35 unverified candidate dividends in 2018–2026; 21 candidates fall in the latest-five-year window. | **Not established:** no event-level issuer capital-gain evidence found in inspected local artifacts. | **Not established.** | **Not established:** no verified split record. |
| **IEF** | **None found** in the State Street workbook or normalized issuer ledger. | Yahoo has 106 unverified candidate dividends in 2018–2026; 60 fall in the latest-five-year window. | **Not established:** no event-level issuer capital-gain evidence found. | **Not established.** | **Not established:** no verified split record. |
| **GLD** | **None found** in the workbook, issuer ledger, or Yahoo action candidates. The preserved prospectus provides limited evidence that the trust does not generate income and shareholders will not receive dividends. | No Yahoo candidates; lack of candidates is not complete event-history proof. | **Not established** for the complete period. | **Not established** for the complete period. | **Not established:** no complete issuer-verified split history found. |
| **XLU** | **Verified (partial):** 35 normalized records, 2018-03-16–2026-09-21; 20 are in the latest-five-year window. The workbook has 39 positive dividend rows from 2017-01-03. | Yahoo has 35 dividend candidates plus one split candidate; 21 total candidates are in the latest-five-year window. Yahoo also flags amount/share-basis differences on older distributions; the split explanation is not verified. | No positive short-/long-term capital-gain amount found in workbook fields; not represented in normalized events. | **Not established:** workbook has no special/return-of-capital classification. | **Unverified candidate:** Yahoo reports 2:1 on 2025-12-05. No official split notice is preserved; local price/factor continuity does not resolve whether this was an actual split or an erroneous candidate. |
| **XLV** | **Verified (partial):** 36 normalized records, 2018-03-16–2026-09-21; 20 are in the latest-five-year window. The workbook has 40 positive dividend rows from 2017-01-03. | Yahoo has 36 dividend candidates for 2018–2026; 20 fall in the latest-five-year window. Workbook does not classify ordinary versus special. | No positive short-/long-term capital-gain amount found in workbook fields; not represented in normalized events. | **Not established:** no special/return-of-capital classification. | **Not established:** no verified split record. |

The workbook's 2017 dividend rows for SPY, XLU, and XLV precede the
normalized ledger's configured 2018-01-03 start date. They are present in the
workbook but absent from the curated 106-event ledger. Thus the ledger does
not cover the full 2017 start of the price files even for the three symbols
with issuer evidence.

**No absence is treated as proof of no event.** In particular, GLD's limited
prospectus evidence does not establish its complete split history, and the
absence of QQQ/IEF rows in a State Street-specific workbook is not evidence
about those issuers.

## 3. Price-Date and Adjustment-Factor Comparison

### Alignment and local-versus-Yahoo comparison

- Each required ETF price file has 2,479 dates, 2017-01-03–2026-11-11.
- All 106 normalized issuer events have ex-dates present in their respective
  ETF price histories. All inspected SPY/XLU/XLV workbook amount dates from
  2017 onward also fall on local price dates.
- The prior Yahoo comparison covers 2,227 sessions per ETF from 2018-01-03
  through 2026-11-11. It records no missing common sessions in either
  direction.
- Local versus Yahoo raw `close` has maximum relative difference **0** for
  all six ETFs. Maximum `adj_close` relative differences are:

| ETF | Common sessions | Max raw-close relative difference | Max adjusted-close relative difference |
|---|---:|---:|---:|
| SPY | 2,227 | 0 | 7.92 × 10⁻⁷ |
| QQQ | 2,227 | 0 | 7.66 × 10⁻⁷ |
| IEF | 2,227 | 0 | 1.03 × 10⁻⁶ |
| GLD | 2,227 | 0 | 0 |
| XLU | 2,227 | 0 | 5.53 × 10⁻⁷ |
| XLV | 2,227 | 0 | 6.68 × 10⁻⁷ |

The comparison artifact reports no unexplained adjustment-factor jumps under
its own diagnostic for the 2018–2026 Yahoo overlap. This is evidence of
numerical agreement with the Yahoo history used in that prior comparison,
not independent proof of FMP source, Yahoo source, or complete corporate
actions.

### Factor-transition diagnostics

For this audit, a “material factor transition” means an adjacent-session
change in `adj_close / close` greater than 0.001% in magnitude. This is a
descriptive threshold for separating larger changes from floating-point
rounding noise; it is not an event-detection rule.

| ETF | Material factor-transition dates (2017–2026) | Direction/shape | Event-date comparison |
|---|---:|---|---|
| SPY | 39 | All 39 are upward relative-factor steps; factor ranges approximately 0.85903 to 1.0 | All 39 positive workbook dividend dates (including four in 2017) coincide with a material transition. All 35 normalized issuer dividends reconcile to the simple dividend-factor expression within 1% relative error. |
| QQQ | 39 | All upward; factor approximately 0.93726 to 1.0 | All 35 Yahoo candidate dates coincide with a transition; four other transitions are in 2017, before the candidate artifact's 2018 start. Candidates remain unverified. |
| IEF | 118 | All upward; factor approximately 0.79006 to 1.0 | All 106 Yahoo candidate dates coincide with a transition; 12 other transitions are in 2017. Candidates remain unverified. |
| GLD | 0 | Factor equals 1.0 on every row | No factor changes to reconcile. The fetcher's fallback from missing `adjClose` to raw close means this does not establish absence of adjustments or distributions. |
| XLU | 39 | All upward; factor approximately 0.73666 to 1.0 | All 39 positive workbook dividend dates coincide with transitions. 38 of 39 workbook-date simple dividend-factor comparisons are within 1%; the remaining date is 2020-03-23 at approximately 1.001% relative error. The 2025-12-05 Yahoo split candidate is not a material factor-transition date. |
| XLV | 40 | All upward; factor approximately 0.84962 to 1.0 | All 40 positive workbook dividend dates coincide with transitions. All 36 normalized issuer dividends reconcile to the simple dividend-factor expression within 1%. |

The calculation compared the observed one-session factor ratio with
`1 / (1 - distribution / prior raw close)` on the issuer dividend dates.
This simplified check is a consistency test, not a reconstruction of the
provider's full methodology. It does not test whether later actions rescaled
earlier rows, prove a specific adjustment convention, or independently
validate special distributions, return of capital, or splits.

The following unmatched 2017 factor-transition dates are investigation leads,
not proven missing events:

- SPY: 2017-03-17, 2017-06-16, 2017-09-15, 2017-12-15.
- QQQ: 2017-03-17, 2017-06-16, 2017-09-18, 2017-12-18.
- IEF: first-of-month/near-monthly dates from 2017-02-01 through 2017-12-01,
  plus 2017-12-21.
- XLU: 2017-03-17, 2017-06-16, 2017-09-15, 2017-12-15.
- XLV: 2017-03-17, 2017-06-16, 2017-09-15, 2017-12-15.

The local State Street workbook has matching 2017 dividend dates for SPY,
XLU, and XLV; local Yahoo candidate history starts in 2018. No issuer source
for those 2017 QQQ/IEF dates was found in the inspected artifacts. A factor
change alone does not establish an event or its amount/type.

### XLU split candidate

The local corporate-action research record shows that XLU raw close changed
from about 43.71 on 2025-12-04 to 43.30 on 2025-12-05, not by an approximate
one-half ratio. The `adj_close / close` factor is approximately
0.97226948 on Dec 4 and 0.97226962 on Dec 5, effectively unchanged. The
record says this is consistent with an already split-adjusted price history
or an invalid Yahoo candidate and cannot distinguish those possibilities.
There is no official split notice in the local evidence. **Do not apply the
candidate split to holdings or assert that it occurred.**

## 4. Adjustment, Provenance, and Licensing Findings

### Adjustment methodology

- The CSVs contain raw OHLCV and one adjusted field (`adj_close`), not
  adjusted OHLC.
- The current FMP fetcher calls `historical-price-eod/full`, maps its
  `adjClose` to `adj_close`, and falls back to raw `close` if that field is
  absent. This describes code, not the provenance of these CSVs.
- Local `adj_close / close` factors are numerically consistent with observed
  SPY/XLU/XLV dividend dates. QQQ/IEF factor transitions align with Yahoo
  candidate dates, but those candidates lack issuer verification. GLD's flat
  factor is ambiguous.
- No local source establishes the adjustment treatment of splits,
  ordinary/special distributions, capital gains, or return of capital, or
  whether old adjusted values are revised.
- No adjusted OHLC was derived in this audit. Do not silently use the factor
  to rescale open/high/low; any such values would be synthetic adjusted-price
  coordinates, not historical executable prices.

### Provenance

The `data/fmp_daily/` name and FMP fetcher indicate FMP was an intended
provider, but the CSVs are ignored by Git and have no source manifest,
retrieval timestamp, query parameters, provider version, or raw response.
The previously recorded exact Yahoo raw-close match across 2,227 sessions
provides an alternate plausible provenance signal. It cannot determine
whether the files were originally obtained from Yahoo, FMP, or a common
upstream source. Actual acquisition provenance is **UNVERIFIED**.

### Rights and retention

- No FMP license, acquisition receipt, or written permission for retention,
  academic research, RL training, faculty review, public dashboard display,
  derived output, or redistribution was found.
- The Phase 5.4 audit records that the supplied FMP official pages were
  inaccessible during that research; therefore the local records do not fill
  the rights gap.
- The Yahoo manifest records a prior yfinance history retrieval and the
  comparison artifact records Yahoo price values. The local corporate-action
  documentation says express prior permission for automated Yahoo collection
  was not verified and warns against retention/use/redistribution without
  terms review. No such permission was found in the inspected records.
- The State Street workbook is preserved locally, but its manifest says not
  to commit or redistribute issuer data without confirming applicable rights.
- The repository's Apache software license does not grant market-data rights.

**Status:** local possession is not evidence of permission. Do not publish raw
data or assume academic training, private faculty display, public dashboards,
or model/derived-output publication are permitted.

## 5. Missing Events and Unresolved Classifications

1. The 106-event ledger starts in 2018 even though the price files begin in
   2017; the 2017 workbook dividend rows for SPY, XLU, and XLV are not in that
   ledger.
2. No issuer-verified QQQ or IEF distribution history is in the local
   normalized event dataset. Their Yahoo candidates cover only returned rows
   and do not establish completeness or issuer amounts/payment dates.
3. GLD's prospectus is limited negative evidence about income/dividends, not
   a complete action history. No split history is established.
4. The three State Street dividend series have no explicit ordinary vs.
   special classification. Do not assume every `DIVIDEND ($)` entry is an
   ordinary dividend.
5. No positive short-/long-term capital-gain amount was observed in the
   inspected State Street workbook fields, but this does not prove no such
   event occurred outside the workbook/version or through another category.
   No capital-gain event is in the normalized ledger.
6. No dedicated return-of-capital classification or complete return-of-capital
   history was found for any ETF.
7. No split is issuer-verified for any of the six ETFs. The XLU split is a
   candidate only.
8. A factor step that lacks a verified event is not itself proof of a missing
   distribution. Conversely, factor agreement on observed dates does not
   prove the event history is complete.
9. If adjusted prices already include actions and an event ledger is also
   applied, returns may be double-counted. If raw prices are used with an
   incomplete ledger, total returns or share counts may be wrong.

## 6. Recommended Next Steps

1. Preserve current files; do not generate adjusted OHLC, revise the ledger,
   or train PPO from these data.
2. Ask the project owner to recover any original FMP/Yahoo download
   transaction record, export script, provider plan, snapshot, or written
   rights approval. Treat the price-vs-Yahoo match as a provenance clue, not
   a source certificate.
3. Obtain written provider confirmation for the actual source and for local
   retention, university model training, faculty access, private/public
   demonstrations, derived outputs, and redistribution. Resolve whether
   Yahoo's automated-collection terms apply to the existing price files.
4. Reconcile all six ETFs with issuer event records: obtain QQQ and IEF
   distribution histories; obtain GLD split/action evidence; confirm the
   XLU split candidate; and determine distribution tax/event classifications,
   including capital gains and return of capital.
5. Compare complete issuer events to a versioned price snapshot by ex-date
   and split effective date. Keep raw prices/actions separate from adjusted
   prices. Do not infer missing events from factors.
6. Only after rights, source provenance, adjustment semantics, and action
   completeness pass review, preserve an immutable, checksummed snapshot with
   retrieval timestamp and version metadata. Use one consistent methodology:
   raw OHLC plus a complete event ledger, or a documented adjusted series
   without reapplying those same actions.

## 7. GO / NO-GO Decision

| Activity | Decision | Basis |
|---|---|---|
| Further read-only local evidence recovery | **CONDITIONAL GO** | Existing artifacts can be reconciled, but unknowns must remain explicit. |
| Claim that files are definitely FMP-originated | **NO-GO / unverified** | FMP code/name suggest intent; the Yahoo comparison is an exact raw-close match for 2018–2026 and source snapshots are absent. |
| Claim of complete adjusted OHLC or total-return data | **NO-GO** | Only adjusted close is present; event completeness, classifications, split basis, and revision policy are unresolved. |
| Use in university research/training/demo/publication | **NO-GO pending written permission** | No applicable project-use grant was found. |
| Historical PPO training or performance claims | **NO-GO** | Provenance, rights, and financial event-accounting gates are unmet. |

**Final recommendation: NO-GO for historical PPO training.** No new data was
acquired, no event was inferred, and no adjusted OHLC was produced.

## 8. Inspection and Change Log

Inspected local price files, action CSVs, manifests, JSON/JSONL evidence,
issuer workbook and preserved GLD prospectus metadata, corporate-action
documentation/code, provider and prior audit reports, and prior local Yahoo
comparison metadata. `.env` contents were not read.

**Created:** `docs/phase_5_8_corporate_action_evidence_recovery.md` (this
report only).  
**Existing datasets and production code:** not modified.  
**External API calls, downloads, purchases, PPO training, and trades:** none.
