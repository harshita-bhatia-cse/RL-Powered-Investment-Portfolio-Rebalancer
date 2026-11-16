# Phase 5.3 — Final Adjusted Total-Return Data Audit

## 1. Executive Summary

**Historical PPO training remains NO-GO.** The local FMP files are complete and
well-formed over their observed range, and their `adj_close / close` factors
are positive, finite, and preserve OHLC ordering when applied as a common
per-date multiplier. Some factor transitions align closely with verified
State Street distributions. These observations make adjusted OHLC a
testable research hypothesis, not a validated total-return dataset.

The project has no retained FMP price-response sample or local FMP adjustment
specification establishing:

- whether `adjClose` includes all ETF distributions and splits;
- whether the adjustment is backward- or forward-adjusted under a formally
  documented convention;
- whether raw OHLC and `adjClose` use the same split basis;
- whether the provider revises old adjusted prices, and under what policy; or
- whether an adjusted-open fill is intended to represent a historical
  executable quote.

The FMP fetcher also falls back to raw `close` if the response omits `adjClose`.
That fallback means equality of `close` and `adj_close` is not proof that no
distributions occurred.

**Recommended target:** Approach C, a separately versioned adjusted OHLC
dataset from a source that documents its adjustment semantics and permits the
intended research use. It is **CONDITIONAL GO as a data-engineering direction**,
not a go-live or PPO-training approval. It must pass the validation plan in
Section 8. Approach B is a useful diagnostic/control, but is currently NO-GO
as a claimed total-return methodology. Approach A is financially interpretable
in principle but currently NO-GO because event coverage is incomplete.

## 2. Existing Data and Code Evidence

### Local price rows

The six ETF files in `data/fmp_daily/` each contain 2,479 chronologically
ordered observations from 2017-01-03 through 2026-11-11. The ETF date sets
are identical. Their columns are `date, open, high, low, close, adj_close,
volume`.

| Symbol | Rows | Missing values | Duplicate dates | OHLC violations | `adj_close != close` rows |
|---|---:|---:|---:|---:|---:|
| SPY | 2,479 | 0 | 0 | 0 | 2,440 |
| QQQ | 2,479 | 0 | 0 | 0 | 2,441 |
| IEF | 2,479 | 0 | 0 | 0 | 2,471 |
| GLD | 2,479 | 0 | 0 | 0 | 0 |
| XLU | 2,479 | 0 | 0 | 0 | 2,441 |
| XLV | 2,479 | 0 | 0 | 0 | 2,441 |

All audited prices were positive and finite; volumes were finite and
non-negative. Each raw bar satisfied `high >= max(open, low, close)` and
`low <= min(open, high, close)`. These checks describe the files as they exist
locally; they do not establish provenance or the source's adjustment policy.

The context files are not identical in date coverage: `^GSPC` has 2,479 rows
through 2026-11-11, and `^VIX` has 2,482 rows through 2026-11-12. The
environment aligns frames on common dates, so dates outside the shared range
are not eligible for a complete portfolio transition.

### FMP field mapping and storage

In `src/data/data_fetcher.py`, `FMPFetcher.get_price_data` reads FMP's
`open`, `high`, `low`, `close`, and `volume`, and maps `adjClose` to
`adj_close`. It uses `item.get('adjClose', item['close'])`, so a missing
`adjClose` field silently defaults to raw close. `_standardize_price_data`
stores raw OHLC and `adj_close`; it does not produce adjusted open, high, or
low.

`src/data/data_store.py` defines `price_data` with raw `open, high, low,
close`, `adj_close`, `volume`, and `created_at`; it has no source-response
version, adjustment-method field, or adjusted OHLC fields. Its
`raw_payloads` helpers are documented and used for tabular fundamentals, not
for preserving the FMP EOD price response. A read-only inspection of
`data/finrl_trading.db` found no FMP raw-payload records and no price rows for
the six ETFs or the two context symbols. The CSVs therefore have no retained
FMP response sample or database provenance record available for this audit.

`src/data/data_processor.py::_clean_price_data` derives adjusted OHLC by
dividing Compustat-style `prccd/prcod/prchd/prcld` fields by `ajexdi`. That is
a separate input convention; it does not document how FMP's `adjClose` relates
to the FMP raw OHLC fields.

### Existing RL behavior and event status

`src/rl/environment/portfolio_env.py::PortfolioTradingEnv` uses
`adj_close` for asset and benchmark trailing features, but executes at raw
next-session `open` and marks portfolio holdings using raw `close`. It also
supports supplied corporate actions through `CorporateActionLedger`.
Consequently, the current environment mixes adjusted-close features with
raw-price account transitions; it is not an adjusted-price-only simulation.

The existing verified-event artifacts contain 106 State Street distributions
for SPY (35), XLU (35), and XLV (36). QQQ, IEF, and complete GLD coverage remain
unverified; the XLU 2025-12-05 2:1 candidate remains unverified. This audit did
not add events or change those conclusions.

## 3. FMP Adjustment Methodology Findings

### Verified from repository code and local data

- The FMP fetcher requests `historical-price-eod/full` and maps the response's
  `adjClose` field into local `adj_close`.
- If `adjClose` is absent, the fetcher substitutes `close`.
- Local files contain raw OHLC, volume, and `adj_close`; they contain no
  adjusted OHLC or source/method/version metadata.
- In the current local snapshot, `adj_close / close` ranges are:

  | Symbol | Minimum factor | Maximum factor | Factor on last ETF row |
  |---|---:|---:|---:|
  | SPY | 0.859029 | 1.000000 | 1.000000 |
  | QQQ | 0.937264 | 1.000000 | 1.000000 |
  | IEF | 0.790064 | 1.000000 | 1.000000 |
  | GLD | 1.000000 | 1.000000 | 1.000000 |
  | XLU | 0.736664 | 1.000000 | 1.000000 |
  | XLV | 0.849619 | 1.000000 | 1.000000 |

- These factors are positive and finite in every audited ETF row. A positive
  common multiplier preserves the order of OHLC values mathematically; an
  explicit local calculation of scaled bars found no high/low ordering
  violations.
- State Street's verified distribution dates can be compared with the
  local FMP factor changes. On all 35 SPY and all 36 XLV issuer distribution
  dates, the observed one-session factor ratio was within 1% of the simple
  dividend adjustment expression
  `1 / (1 - distribution_per_share / prior_raw_close)`. On 34 of 35 XLU
  issuer distribution dates it was within 1%; the median absolute relative
  difference for XLU was about 0.78%, with a maximum about 1.00%. This is
  meaningful corroborating evidence for those observed distribution
  transitions, not evidence that all events are present or that every ETF uses
  the same complete adjustment convention.
- The factor is 1.0 at the latest row for each ETF and below or equal to 1.0
  throughout this snapshot. Earlier factors generally rise at distribution
  dates. This is empirically consistent with historical prices being
  backward-adjusted toward a latest-date basis, but the code/data do not
  establish FMP's formal convention or revision policy.
- For XLU, the factor is approximately 0.97226948 on 2025-12-04 and
  0.97226962 on 2025-12-05, a relative factor change of about 0.000014%.
  Raw close moved from 43.71 to 43.30 and adjusted close from 42.497898 to
  42.099274; neither series shows an approximately half-price day-to-day
  change. This does not resolve whether the local raw prices were already
  split-adjusted or whether Yahoo's candidate split is erroneous.

### Provider-documented facts

No FMP methodology document or provider statement about `adjClose` was found
in the inspected repository files, local price files, or the inspected local
database. The report therefore makes no provider-documentation claim about
FMP's dividend/split inclusion, adjustment direction, split basis, or
historical revisions. Those properties are **UNVERIFIED** here.

### Unknowns and assumptions

It is unknown whether FMP adjusted close includes every dividend, ETF
distribution, capital-gain distribution, split, or reverse split for these
symbols and dates. It is also unknown whether the FMP raw OHLC fields have
already been split-adjusted or whether their units match `adjClose` exactly
through all corporate actions. Observed factor behavior and close continuity
are hypotheses/evidence for investigation, not a substitute for provider
documentation and an independent event audit.

The exact historical snapshot retrieval time and API response were not
retained with the CSVs. Whether FMP revises older adjusted observations is
therefore also **UNVERIFIED**.

## 4. Adjusted OHLC Derivation Assessment

The candidate transform is:

```text
factor[t] = adj_close[t] / close[t]
adjusted_open[t] = open[t] * factor[t]
adjusted_high[t] = high[t] * factor[t]
adjusted_low[t] = low[t] * factor[t]
adjusted_close[t] = adj_close[t]
```

### Numerical and geometric properties

For these local files, `factor[t]` is positive and finite for all six ETFs.
Multiplication by one positive scalar per bar preserves that bar's inequalities:
`adjusted_low <= adjusted_open, adjusted_close <= adjusted_high`, and the
high/low bounds remain valid. This is a mathematical/geometric property, not
proof that FMP intends the resulting values as adjusted OHLC.

### Economic and source limitations

- The transform assumes raw OHLC and `adj_close` use the same quote currency
  and split-adjusted share basis. That is not established by local FMP
  documentation.
- If `adjClose` reflects cash distributions, the factor changes encode a
  total-return-like adjustment. The close/adjustment-factor reconciliation
  against issuer distributions is encouraging for SPY, XLU, and XLV, but there
  are no equivalent issuer-verified distributions for QQQ and IEF, and no
  complete GLD event history.
- A factor does not identify a split. The XLU factor is essentially unchanged
  across the reported 2025-12-05 event. If the underlying raw history has
  already been split-adjusted, derived adjusted OHLC could inherit that basis;
  if it has not, the transform does not independently supply the missing split
  adjustment. The event cannot be validated from this calculation.
- Adjusted open/high/low are synthetic back-adjusted price coordinates, not
  the actual historical quotes at which an order could have filled. A future
  distribution can rescale earlier bars. Applying percentage fees/slippage
  to such coordinates can yield a consistent synthetic portfolio calculation,
  but not a claim about actual fill prices or dollar market impact.
- In the current snapshot, GLD and `^GSPC` have `adj_close == close` on every
  row, while the other ETFs do not. Given the fetcher's fallback when
  `adjClose` is missing, equality for GLD is not proof of no actions.

**Decision on derivation:** not defensible as a certified production
total-return data set for all six ETFs yet. It is suitable as a research
diagnostic after the split-basis and distribution checks in Section 8 are
completed. Do not write adjusted OHLC files from it before those conditions
are met.

## 5. Comparison of Three Data Methodologies

| Dimension | A. Raw OHLC + fully verified ledger | B. FMP raw OHLC + derived adjusted OHLC | C. Separate, documented adjusted OHLC source |
|---|---|---|---|
| Current coverage | Local raw bars cover all six ETFs over 2017-01-03–2026-11-11. The verified event set is partial. | Same local bars and adjusted closes are present; derived adjusted OHLC can be calculated for the same dates, but completeness is unverified. | Not acquired or locally demonstrated in this audit. Coverage must be independently verified for every ETF and context series. |
| Dividends | Explicit ex-date entitlement, receivable, and payment-date settlement; currently incomplete source coverage. | Implicitly represented only if FMP `adjClose` includes all distributions. That convention is undocumented here. | Potentially implicit if the provider documents cash distributions and reinvestment convention; must be checked against issuer events. |
| Splits | Explicit and auditable when verified; XLU split and broader event histories remain unresolved. | May inherit an adjustment from FMP, but split basis and inclusion are unknown. Derived factor alone cannot identify or repair a missing split. | Potentially consistent if split adjustment rules are documented and tested against official split records. |
| Execution realism | Raw OHLC approximates historical quote prices; costs can be applied to raw notional. | Adjusted open is synthetic, not actual historical executable open. | Adjusted open is also synthetic unless provider supplies separately defined tradable raw prices; use is total-return simulation, not fill reconstruction. |
| Total-return accuracy | Can be high with complete official events, correct price basis, settlement, and reinvestment convention. Current history is not complete enough. | Unknown until FMP methodology and every asset's event/revision behavior are verified. | Conditional on documented and validated methodology, coverage, and snapshot integrity. |
| Reproducibility | Strong when raw prices and event evidence/IDs are retained and checksummed. | Current CSV lacks acquisition timestamp, response payload, provider version, and adjustment metadata. | Strong only if exact adjusted dataset, provider/version, terms, timestamp, and checksums are retained. |
| Licensing/retention | Depends on the raw-price and event-source terms. | FMP terms and retention rights are not documented in this local evidence; verify before use or redistribution. | Must confirm research use, local retention, and redistribution rights before acquisition. |
| Revision risk | Event corrections are explicit and auditable; raw price revisions still need snapshots. | Retrospective adjustment revisions are undocumented here. | Provider-specific; snapshot/version and revision comparison are required. |
| Complexity | Highest: event coverage, ordering, receivables, settlement, splits, and fractional treatment. | Medium: transform, source-basis guards, revision tracking, and ledger exclusion. | Medium: ingest and validate one coherent source, then enforce basis/version controls. |
| Academic suitability now | **NO-GO** for PPO training because action coverage is incomplete. | **NO-GO** as a claimed total-return method until FMP semantics and split treatment are established. | **CONDITIONAL GO** as the preferred direction if all evidence and validation gates pass. |

Approach A and an adjusted total-return approach answer related but not identical
questions. An adjusted series typically models distributions as reinvested in a
price index; the ledger models holdings, receivables, and payment timing.
Comparisons must align reinvestment timing, fractional shares, fees, and
valuation convention.

## 6. yfinance Assessment

The project declares `yfinance>=0.2.0` in `requirements.txt`; the local
environment inspected for this audit has yfinance 1.8.0. No yfinance market
history was requested or downloaded.

The locally installed `yfinance.utils.auto_adjust` implementation computes
`Adj Close / Close`, multiplies `Open`, `High`, and `Low` by that ratio, drops
the unadjusted OHLC fields, and renames the adjusted values to OHLC. Therefore,
**the local implementation does adjust all OHLC fields** and does not mix its
returned OHLC fields with separate FMP raw prices if used as one complete
dataset. This describes the library transformation; it does not independently
certify Yahoo's underlying event coverage, adjustment methodology, or the
result for each of the six ETFs.

The declared lower bound is broad, not a frozen version. Reproduction would
require recording the exact yfinance version, Python/dependency lock, raw
response or immutable output dataset, query dates/parameters, timezone/date
convention, and SHA-256 checksums. The resulting historical dataset would need
validation against issuer records before use.

The repository's `src/data/CORPORATE_ACTIONS.md` records that Yahoo's Terms of
Service restrict automated collection without express prior permission and
that such permission has not been verified for this project. The right to
automatically collect, retain, or redistribute yfinance data for the proposed
research is therefore **UNRESOLVED**. No live Yahoo calls or data downloads
were made.

## 7. Implications for Our RL Environment

The existing transition in
`src/rl/environment/portfolio_env.py::PortfolioTradingEnv.step` observes
through session `t` close, executes at `t+1` raw open, and marks at `t+1` raw
close. `reset` initializes ledger balances at the reset-date close. The
allocator in `src/strategies/allocation/rebalance_engine.py::rebalance`
accepts separate execution and mark price maps; `RebalanceConfig` controls
fractional precision, transaction cost rate, and slippage.

A future adjusted-price mode would have to be a separate, explicit valuation
mode—not a silent change to existing price mappings:

1. Use one adjusted OHLC snapshot for all six tradable ETFs, with one provider,
   one adjustment convention, and identical trading-session alignment.
2. Observe data through adjusted close at `t`; execute simulated trades at
   adjusted open on `t+1`; mark holdings at adjusted close on `t+1`.
3. Use fractional shares consistently. With adjusted prices, shares are
   synthetic units on the adjusted-price basis; they are not necessarily
   historical legal share counts.
4. Treat distributions as implicitly reinvested in the adjusted price path.
   Do not submit dividend or split events to `CorporateActionLedger`; reject
   a nonempty event list in this mode rather than risk double application.
5. Apply allocator fees and slippage once to adjusted notional. Do not
   separately reduce the reward for transaction costs. State that the costs
   are a simulation on synthetic price units, not evidence of actual fills.
6. Preserve an all-cash reset state. A reset-date close establishes NAV and
   does not inherit pre-existing shares or dividend receivables. The
   adjusted-price model then assumes reinvested-return exposure from that
   reset onward.
7. Keep features trailing-only and fit scalers on training rows only. Freeze
   the price snapshot and scaler versions for train, validation, and test.
   Adjusted series are commonly restated when future actions occur; although
   a future action may multiply all earlier dates by a common factor (which
   cancels in many return features), this invariance must be tested rather
   than assumed. Do not use future-period data to choose an adjustment,
   event, or feature rule.
8. Add an explicit `price_basis`/mode to data loading, environment metadata,
   and experiment manifests. Validate that all execution and mark values for
   an asset come from that same basis. Never combine raw open/close with
   adjusted OHLC or adjusted close. Retain the source dataset unchanged.

An adjusted open is synthetic because the adjusted price is a rescaled
historical coordinate that can incorporate distributions (and perhaps splits)
after the quote date. It is useful for return accounting only under the
documented adjustment convention; it is not an observed transaction price.

## 8. Required Validation Tests

Before any adjusted-price backtest, preregister thresholds and run a test suite
at the data-ingestion/accounting level. Do not train PPO as part of these
checks.

1. **Adjustment-factor consistency.** Require every factor to be finite and
   strictly positive; require `abs(adj_close - close * factor) / adj_close <=
   1e-10` for every row (or explain provider rounding and set a source-precision
   tolerance). A missing or fallback `adjClose` must be distinguishable from a
   genuine factor of 1.
2. **Adjusted OHLC ordering.** For every row, require
   `low <= min(open, close) <= max(open, close) <= high` on both raw and
   adjusted bars, with a floating tolerance of `1e-10 * max(1, abs(high))`.
   Check positive finite values and non-negative finite volume.
3. **Split transitions.** For each issuer-verified split, compare raw price
   basis, adjusted basis, share volume, and factor immediately before/after
   the effective date. Require no unexplained ratio near the split factor
   (within 1%) after the documented convention is applied. A reported but
   unverified split, including XLU 2025-12-05, blocks acceptance rather than
   being “fixed” by a guessed factor.
4. **Dividend reconciliation.** For every official distribution with
   verified amount and ex-date, compare the observed adjustment-factor change
   with the expected dividend adjustment using the prior raw close and a
   fixed, documented formula. Require at least 99% of events within 1% relative
   error; manually resolve every remaining event. Include capital-gain
   distributions and verify share basis. The current observed SPY/XLV/XLU
   results are encouraging but do not validate QQQ, IEF, or GLD.
5. **Independent total-return comparison.** Build a separate benchmark from
   raw prices plus independently verified distributions/splits under an
   explicitly matched reinvestment convention. Compare daily returns outside
   event dates within 5 basis points absolute; compare event-day returns
   within 1 basis point after matching reinvestment timing; require annualized
   return difference <= 10 basis points and cumulative-return difference <=
   25 basis points over each one-year validation window. Investigate rather
   than waive any threshold failure. Use a genuine independent source, not a
   second field from the same price endpoint.
6. **Missing-event detection.** Obtain authoritative, period-complete
   distribution and split coverage for all six ETFs. Match all normalized
   events by symbol, event date, type, and amount/ratio; require zero
   unexplained missing, extra, or duplicate events. A successful empty query
   is insufficient evidence of no events.
7. **Historical revision checks.** Store immutable snapshots with provider,
   retrieval timestamp, version, parameters, terms/rights, row counts, and
   checksums. On refresh, compare all historical rows. Require exact checksum
   equality for a claimed frozen reproduction; any changed row must be
   quantified, explained by provider revision, and recorded as a new version.
8. **Feature leakage checks.** Recompute every feature using data available
   through its decision date only. Fit scalers exclusively on training dates.
   Perturb or append post-decision data and require previous observations and
   features to remain unchanged within `1e-8`, except for a documented
   scale-invariant global restatement that leaves ratios unchanged. Verify
   that scaler means/scales are identical in validation and test.
9. **Transaction-cost handling.** With identical actions and prices, verify
   fees/slippage enter NAV exactly once and that the reward equals
   `log(next_nav / current_nav)` with no additional cost penalty. The measured
   NAV difference must equal the allocator's recorded fee plus slippage to
   a numerical tolerance of one cent or `1e-8` of NAV, whichever is larger.
10. **Raw-versus-adjusted accounting consistency.** On intervals with complete
    verified events, compare (i) raw-price accounting with distributions
    reinvested using a stated convention and (ii) adjusted-price accounting.
    Align ex-date/payment timing, split basis, fractional-share precision,
    fees, and marks. Require cumulative NAV difference <= 25 basis points per
    year and explain residual timing/rounding differences. A comparison that
    omits events or mixes providers is not a pass.

These are proposed acceptance thresholds. They must be fixed before
validation results are inspected; do not tune them after observing a preferred
result.

## 9. Risks and Unresolved Questions

- FMP's documented adjustment convention is unavailable in the local
  repository evidence. Dividend and split coverage, factor direction, OHLC
  basis compatibility, and revision policy remain unverified.
- The `adjClose` fallback to `close` can conceal missing adjustment data.
  GLD's equality of close and adjusted close must not be interpreted as proof
  of no events.
- FMP price CSVs have no stored retrieval timestamp, response payload,
  provider/schema version, or checksum manifest in the inspected artifacts.
- FMP adjusted history may be revised retrospectively; no pair of dated
  snapshots is available to measure revisions.
- Corporate-action evidence is incomplete for QQQ and IEF; GLD has no
  complete issuer action history; XLU's reported split is unresolved.
- Adjusted OHLC-derived fills are not actual historical executable prices.
  Fees/slippage on these prices are synthetic assumptions.
- yfinance's local transform documents how the library scales OHLC, not
  whether Yahoo's underlying adjustments are complete, stable, or licensed for
  this use. Automated-access permission and research retention terms remain
  unresolved.
- Any independent total-return comparison must avoid circular validation
  (e.g. comparing one adjusted field to another field produced by the same
  underlying adjustment feed).
- This audit did not fetch external documentation or make a provider request;
  consequently, provider-contract and licensing questions are explicitly
  left unresolved rather than inferred.

## 10. Final GO / NO-GO Decision

| Method / readiness dimension | Decision | Basis |
|---|---|---|
| A. Raw OHLC + fully verified corporate-action ledger | **NO-GO now** | Only 106 distributions for three ETFs are verified; required event coverage remains incomplete for QQQ, IEF, GLD, and the XLU split. |
| B. FMP raw OHLC + `adj_close / close`-derived adjusted OHLC | **NO-GO now** | Factors are numerically usable and some issuer events reconcile, but FMP's methodology, split basis, completeness, and revisions are undocumented in local evidence. |
| C. Separate, explicitly documented adjusted OHLC source | **CONDITIONAL GO** | Preferred research direction only if full-period coverage, adjustment rules, independent reconciliation, immutable snapshots, and usage rights are established. No such qualifying dataset has been evidenced yet. |
| Data-engineering readiness | **CONDITIONAL GO** | Existing rows are clean; a versioned adjusted dataset and strict basis-mode guards can be designed, but adjusted OHLC is not currently present or certified. |
| Financial-accounting readiness | **NO-GO** | No complete verified event history or validated adjusted-price accounting convention exists for the full universe. |
| Research-backtesting readiness | **NO-GO** | Independent total-return, revision, execution-cost, and raw-versus-adjusted validations remain outstanding. |
| PPO-training readiness | **NO-GO** | Training would rely on unresolved coverage and adjustment assumptions, so reported performance would not yet be defensible. |

Before PPO training begins, require:

1. A selected provider/source with written adjustment methodology covering
   dividends, ETF/capital-gain distributions, splits, split basis, and
   retrospective revisions.
2. Complete, period-aligned adjusted OHLC and volume for all six ETFs, plus
   documented context data, from one coherent source or a formally reconciled
   multi-source pipeline.
3. Verified all-symbol corporate-action coverage (or source-backed evidence
   that the adjusted series incorporates those events), including an official
   resolution of XLU's 2025-12-05 candidate and explicit GLD coverage evidence.
4. Permission for automated acquisition, local retention, and any intended
   redistribution or publication.
5. Passing results for the Section 8 validation suite, retained data/scaler
   snapshots and checksums, and an independent benchmark comparison.
6. An explicit environment price-basis mode that cannot mix raw and adjusted
   prices or apply the corporate-action ledger in adjusted-price mode.

## 11. Recommended Next Steps

1. Do not train PPO or relabel current FMP adjusted close as a verified
   total-return series.
2. Obtain and archive FMP's applicable written `adjClose` methodology and
   licensing/retention terms through authorized channels; if unavailable,
   treat Approach B as rejected for total-return claims.
3. Prefer Approach C only after identifying a provider with documented
   adjusted OHLC semantics and acceptable research rights. Acquire no data
   until authorized and approved.
4. Resolve issuer-event completeness, especially QQQ, IEF, GLD, and the XLU
   split, or demonstrate from provider documentation and independent records
   that the chosen adjusted series incorporates them.
5. Create a separate immutable dataset version with source, version,
   retrieval time, date range, conventions, licensing status, row/session
   audit, and SHA-256 checksums. Never overwrite the existing raw/adjusted
   close files.
6. Implement and pass Section 8 validation before changing the environment.
   Only then decide whether adjusted-price simulation is acceptable for
   academic PPO research; continue to describe it as a synthetic total-return
   simulation rather than actual historical fills.

### Audit trail: files inspected

Repository files:

- `requirements.txt`
- `setup.py`
- `src/data/data_fetcher.py`
- `src/data/data_store.py`
- `src/data/data_processor.py`
- `src/data/CORPORATE_ACTIONS.md`
- `src/rl/environment/portfolio_env.py`
- `src/rl/environment/README.md`
- `src/strategies/allocation/models.py`
- `src/strategies/allocation/rebalance_engine.py`
- `src/data/issuer_corporate_action_verification.py`
- `README.md`

Local data and retained artifacts:

- `data/fmp_daily/SPY_daily.csv`
- `data/fmp_daily/QQQ_daily.csv`
- `data/fmp_daily/IEF_daily.csv`
- `data/fmp_daily/GLD_daily.csv`
- `data/fmp_daily/XLU_daily.csv`
- `data/fmp_daily/XLV_daily.csv`
- `data/fmp_daily/^GSPC_daily.csv`
- `data/fmp_daily/^VIX_daily.csv`
- `data/finrl_trading.db` (opened read-only)
- `data/issuer_verified_corporate_actions/verified_normalized_events.csv`
- `data/yahoo_corporate_actions/price_comparison.json`

Installed yfinance implementation inspected locally:

- `.venv/Lib/site-packages/yfinance/utils.py` (`auto_adjust`)

No local FMP adjustment-methodology document, FMP price-response sample, or
versioned FMP price manifest was found in the inspected files/artifacts.
