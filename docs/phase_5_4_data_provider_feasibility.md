# Phase 5.4 — Final Data Provider Feasibility Audit

**Audit date:** 2026-11-13
**Scope:** Provider research only. No market-data requests, downloads, purchases, code changes, PPO training, or trades were performed.

## 1. Executive Summary

**Do not acquire data or begin historical PPO training yet.** No provider is a GO today. The unresolved points are not cosmetic: exact coverage for the six ETFs, access to the full required period, source-specific rights for academic and public demonstration, and reproducible snapshots have not been established for the leading candidates.

Provisional ranking, based on the documented product shape rather than confirmed project entitlements:

1. **FMP — CONDITIONAL GO; preferred candidate for written confirmation.** The supplied official reference identifies a Dividend Adjusted Price Chart API with `adjOpen`, `adjHigh`, `adjLow`, and `adjClose`, and the supplied FAQ describes split-adjusted `close` and split-and-dividend-adjusted `adjClose`. Those claims could avoid deriving adjusted O/H/L from `adj_close / close`. The referenced FMP pages returned HTTP 403 during this audit, so endpoint semantics, ETF and index coverage, plan access, history, volume, revisions, and rights were not independently verified. Do not treat this ranking as an acquisition approval.
2. **Tiingo — CONDITIONAL GO; fallback candidate.** The project brief describes Tiingo's EOD documentation as offering adjusted O/H/L/C and a CRSP-based split/dividend method. The official documentation and pricing pages returned a JavaScript application shell rather than the referenced documentation details. No ticker-specific access, history, ETF distribution treatment, free-tier entitlement, retention right, or academic/public-use permission was verified.
3. **Alpha Vantage — CONDITIONAL GO as a candidate for further written confirmation, not selected.** Its official support FAQ states that open, high, low, close, and volume are adjusted for splits and cash-dividend events. Its documented Daily Adjusted endpoint, however, describes raw OHLCV plus adjusted close and split/dividend events—not native adjusted O/H/L fields. The exact ETF/index entitlements, adjusted-OHLC output route, non-ordinary ETF distribution behavior, and use rights remain unverified.
4. **Massive — NO-GO for the requested single total-return OHLC dataset.** Its official stock aggregate documentation says the `adjusted` parameter adjusts for splits. It does not document cash-distribution adjustment in these bars. Its indices aggregate endpoint describes index-value bars, but the published plan history starts at 2023-02-14, short of the required 2017 context history. Separate stock and indices entitlements, and data rights, also need confirmation.
5. **Yahoo Finance / yfinance — NO-GO for acquisition under current evidence.** The local yfinance transformation can adjust all OHLC fields using adjusted-close/close, but that demonstrates a library operation, not Yahoo event completeness or license rights. Yahoo's official terms prohibit automated collection without express prior permission; such permission is not documented for this project.

The only defensible next step is written provider confirmation for FMP and Tiingo, with Alpha Vantage as another candidate if its endpoint shape and usage rights can be clarified. A separately documented adjusted-OHLC source would be preferable to silently transforming local FMP raw bars. Adjusted opens are synthetic adjusted-price coordinates, not actual historical executable prices.

## 2. Project Data Requirements

The requested dataset is daily adjusted OHLC for **SPY, QQQ, IEF, GLD, XLU, and XLV**, preferably from 2017-01-03 through 2026-11-11, with the market context symbols **^GSPC** and **^VIX**. A usable source must establish:

- ETF-specific symbol support and actual observations over the complete common period;
- documented split and distribution adjustments, including treatment of ETF cash distributions and non-ordinary distributions;
- consistent adjusted O/H/L/C fields from one source and one price basis;
- separate, compatible access for the benchmark and VIX series;
- a plan that actually entitles the project to the endpoint and complete history;
- a stable snapshot/versioning mechanism or permission to retain a reproducible snapshot; and
- written permission compatible with university research, model training, faculty review, demonstration, and any intended public display.

**Local repository evidence:** `requirements.txt` declares `yfinance>=0.2.0`; it does not pin an exact yfinance version. `setup.py` reads that dependency file. In `src/data/data_fetcher.py`, `FMPFetcher.get_price_data` calls the FMP `historical-price-eod/full` endpoint and maps raw `open/high/low/close/volume` plus `adjClose` into `adj_close`. If `adjClose` is absent, the mapping falls back to raw `close`. No native adjusted O/H/L fields are stored by this mapping. `src/data/data_store.py`'s `price_data` table stores O/H/L/C, `adj_close`, and volume, with no adjustment-method or response-version field. The prior local audit, `docs/phase_5_3_final_adjusted_price_audit.md`, records that the existing CSVs contain raw OHLC and adjusted close and that the local FMP adjustment methodology was not established.

Those local facts do not demonstrate that any provider's current endpoint contains all requested symbols or grants the project data rights.

## 3. FMP Native Adjusted OHLC Assessment

**Official references supplied and attempted (access date 2026-11-13):**

- [Dividend Adjusted Price Chart API](https://site.financialmodelingprep.com/developer/docs/stable/historical-price-eod-dividend-adjusted)
- [FMP FAQs](https://site.financialmodelingprep.com/faqs)
- [FMP pricing plans](https://site.financialmodelingprep.com/pricing-plans)
- [FMP Terms of Service](https://site.financialmodelingprep.com/terms-of-service)

Each page request returned HTTP 403 in this audit. Therefore, the following distinctions are important:

| Question | Audit result |
|---|---|
| Does the referenced endpoint document `adjOpen`, `adjHigh`, `adjLow`, and `adjClose`? | Those field names are identified in the project brief and URL description, but the official page could not be retrieved for independent verification. **Not independently verified here.** |
| Is the endpoint intended for ETFs, and does it cover each of the six ETFs? | **Unverified.** Endpoint naming or general stock coverage is not evidence of ETF support. |
| Does the endpoint apply both dividend and split adjustments to all four adjusted OHLC fields? | **Unverified from accessible documentation.** The brief reports that FMP FAQ says `close` is split-adjusted and `adjClose` is split- and dividend-adjusted; that does not prove the native adjusted-OHLC endpoint applies both kinds of adjustment to every field. |
| Are ordinary and non-ordinary ETF distributions included? | **Unverified.** No accessible source establishes treatment of capital-gain, return-of-capital, special, or other ETF distributions. |
| Are the six ETF histories complete from 2017-01-03 through 2026-11-11 under an available plan? | **Unverified.** Pricing-page access and ticker-specific coverage were not established. Do not infer this from a general plan description. |
| Are `^GSPC` and `^VIX` available through this endpoint or compatible plan? | **Unverified.** |
| Does the response include adjusted volume, and how should it be used? | **Unverified.** |
| Can historical adjusted values be revised, and are point-in-time versions available? | **Unverified.** No revision policy or immutable snapshot facility was verified. |
| Does this remove the need for `adj_close / close` derivation? | **Conditionally.** If the official endpoint returns correctly adjusted O/H/L/C for every ETF and period under one documented convention, its native O/H/L fields would remove the need for that proposed local derivation. Those prerequisites have not been established. |
| What does the subscription permit for this project? | **Unverified.** FMP terms were inaccessible. Local storage, research use, model training, faculty demonstration, public dashboard display, and derived-output rights require written clarification. |

**Assessment:** Highest-priority candidate because it may provide native adjusted OHLC, but there is currently insufficient evidence to request or accept a dataset as fit for the project. Its licensing and actual plan entitlements are decision gates, not implementation details.

## 4. Tiingo Adjusted OHLC Assessment

**Official references supplied and attempted (access date 2026-11-13):**

- [Tiingo End-of-Day documentation](https://www.tiingo.com/documentation/end-of-day)
- [Tiingo pricing](https://www.tiingo.com/pricing)

The pages returned an HTML/JavaScript application shell titled “Stock Market Tools | Tiingo” instead of the referenced documentation or pricing content. Alternative official documentation paths redirected to the same shell or were unavailable. No live EOD data endpoint was queried.

The project brief describes the EOD documentation as stating that Tiingo provides raw and adjusted prices, adjusted O/H/L/C, a CRSP-based adjustment methodology incorporating splits and dividends, and dividend/split fields. Because those statements could not be checked against the accessible page content in this audit, they are recorded as **brief-supplied descriptions, not independently verified provider facts**.

| Question | Audit result |
|---|---|
| Do all six ETF tickers have EOD data? | **Unverified ticker by ticker.** Aggregate coverage statistics cannot prove support. |
| Is each ETF available over 2017-01-03 through 2026-11-11? | **Unverified.** No symbol-specific history or plan entitlement was available. |
| Does the adjusted methodology apply to ETFs, and how are ETF cash, capital-gain, return-of-capital, and special distributions handled? | **Unverified.** The brief's description of split/dividend adjustment and a CRSP method does not establish ETF-specific classification or coverage of every distribution type. |
| Does the free individual plan provide the required EOD API, depth, and request entitlement? | **Unverified.** Pricing content could not be read; do not assume free-plan access from documentation describing a product. |
| May the project retain data locally, train on it, show derived outputs to faculty, publish a dashboard, or redistribute it? | **Unverified.** No accessible terms or written permission established these rights. |
| Are academic/educational licenses available? | **Unverified.** |
| Are `^GSPC` and `^VIX` available under the same plan and compatible terms? | **Unverified.** |
| Can historical adjusted data be revised or snapshotted? | **Unverified.** |

**Assessment:** Fallback candidate only. Before ranking it as operationally feasible, obtain accessible official documentation, exact ETF and date coverage, and written data-use permission.

## 5. Alternative Provider Comparison

| Provider | Adjustment and endpoint evidence | Main fit / blockers | Status |
|---|---|---|---|
| **FMP** | The project brief identifies a dividend-adjusted OHLC endpoint; its pages were blocked in this audit. Existing local integration stores raw OHLC plus `adj_close`, not adjusted OHLC. | Best apparent product fit if its native fields, ETF coverage, plan, index coverage, history, revisions, and license are confirmed. | **CONDITIONAL GO** |
| **Tiingo** | The project brief describes adjusted OHLC and CRSP-based split/dividend adjustment; pages served an app shell, so the claims were not independently verified. | Potentially suitable if ETF-specific method, full history, free/affordable entitlement, retention, and academic/public-use rights are confirmed. | **CONDITIONAL GO** |
| **Alpha Vantage** | Official FAQ says it adjusts O/H/L/C/volume for splits and cash-dividend events. Official `TIME_SERIES_DAILY_ADJUSTED` description says raw OHLCV plus adjusted close and historical split/dividend events, with 25+ years. | The daily endpoint's described response is not native adjusted O/H/L. ETF special distributions, endpoint entitlement, exact index access, and usage permissions remain unresolved. | **CONDITIONAL GO** |
| **Massive** | Official stock aggregate bars return OHLCV and state that `adjusted` means split adjustment. Separate index aggregates return index OHLC; index plan history begins 2023-02-14. | Split-adjusted stock bars do not by themselves provide a documented dividend-adjusted total-return path. Index history misses the required 2017 start. Separate stocks/indices plans and rights complicate a consistent dataset. | **NO-GO** |
| **Yahoo Finance / yfinance** | The local yfinance implementation applies `Adj Close / Close` to O/H/L and substitutes adjusted close as close when auto-adjusting; the broad package version is unpinned. Yahoo's official terms prohibit automated collection without express prior permission. | Underlying action completeness, a frozen source snapshot, and project permissions are unverified. The current record does not show express permission. | **NO-GO** |

### Three data methodologies

| Method | Coverage | Accounting / execution | Reproducibility and rights | Current assessment |
|---|---|---|---|---|
| **A. Raw OHLC + fully verified corporate-action ledger** | Raw histories exist locally, but Phase 5.2 verification is partial: QQQ/IEF, complete GLD history, and XLU split remain unresolved. | Explicit cash distributions and splits can model entitlement and settlement, but only with complete verified events. Raw next-session opens are closer to historical price quotes than adjusted opens; fills remain simulated. | Requires immutable raw data, source references, complete event files, and licenses for both prices and events. | **NO-GO now**: event completeness has not been achieved. |
| **B. FMP raw OHLC + `adj_close / close`-derived OHLC** | Existing local FMP bars cover a substantial period, but only adjusted close is stored. | Applying the same ratio to O/H/L is a hypothesis; it does not prove compatible split bases, event coverage, or the provider's adjustment convention. Resulting open prices are synthetic. | Existing code lacks method/version metadata and response snapshots. Current FMP rights and revision policy are unknown. | **NO-GO as a claimed adjusted total-return dataset.** Keep as diagnostic only. |
| **C. Provider-supplied, documented adjusted OHLC** | Potentially the best fit, but no candidate's exact six-ETF and index coverage was verified. | Avoids locally inventing the O/H/L transformation only if provider documentation establishes the adjustments, including ETF distributions. Adjusted opens remain synthetic. | Requires written research/retention/display rights and fixed snapshots/checksums; revision policy must be known. | **CONDITIONAL GO as the preferred direction, not acquisition approval.** |

No alternative is declared valid on the present evidence.

## 6. Pricing and Historical Coverage

| Provider | Public pricing/history evidence available in this audit | Consequence for requested period |
|---|---|---|
| FMP | The official pricing and documentation pages returned HTTP 403. No reliable current price or plan-specific history entitlement was verified. | Do not infer that the native adjusted endpoint or 2017–2026 ETF history is included in a plan. |
| Tiingo | The official pricing URL returned an application shell. Specific free-plan rates, history, and API access could not be established. | Cannot establish that the free plan includes all six EOD histories or the full target period. |
| Alpha Vantage | Official support describes 25 API requests/day for most free datasets and unlimited requests for verified open-source or educational projects. Its Daily Adjusted endpoint is labeled “Trending Premium” in the documentation. The premium page displays paid plans starting at $49.99/month, but does not prove that a particular tier includes the required function or symbols. | The documented endpoint claims 25+ years of history, but exact ETF coverage, premium-function access for this project, and the historical index entitlements remain unverified. |
| Massive | Official stock aggregate docs say the endpoint is included in all Stocks plans and list plan history as 2 years (Basic), 5 years (Starter), 10 years (Developer), and all history (Advanced). Official indices aggregate docs state history from 2023-02-14 and list “all history” for the indices plans. The pricing page did not expose a usable plan price in this retrieval. | Target ETF span is approximately 9 years 10 months and therefore would require a plan with at least 10 years of stocks history, subject to symbol access. The indices endpoint's documented start misses 2017-01-03, so it cannot alone meet the context requirement. |
| Yahoo / yfinance | No current price or entitlement audit was performed. | Coverage does not establish rights, reproducibility, or adjusted-price completeness. |

The user-reported local FMP file period does not imply that the same provider plan or endpoint can supply the native adjusted OHLC fields over that period. No market data were requested to test these coverage claims.

## 7. Licensing and Academic Use

This section identifies provider terms evidence and open questions; it is not legal advice.

| Provider | What the reviewed evidence says | Rights that still need explicit confirmation |
|---|---|---|
| FMP | Terms URL returned HTTP 403. | Local research retention; storing source rows in Git-ignored project data; training an RL model; faculty review; public dashboard/screenshots; publishing source code and derived results; raw-data redistribution; derived outputs/model parameters; post-subscription retention; separate index data rights. |
| Tiingo | Pricing and EOD documentation pages returned an app shell; no usable terms were verified. | All uses above, including local caching, model training, classroom use, screenshots/public display, derived outputs, retention after termination, and whether an academic license exists. |
| Alpha Vantage | Its support page offers an “Educator / Student” option and says free access covers most datasets at 25 requests/day, with unlimited requests for verified open-source or educational projects. This is an API-frequency statement, not a grant of data redistribution or public-display rights. The official Terms of Service URL returned a PDF that was not parsed. Its real-time data policy describes personal non-commercial use and directs business/commercial use to sales; that page does not settle historical EOD rights for this university project. | Whether the student project qualifies for free access to the adjusted daily function and index history; local retention; training; faculty presentation; public dashboard/screenshots; derived data/model outputs; raw-data redistribution; and post-termination retention. |
| Massive | Individual Terms, last updated 2026-11-03, limit individual plans to personal, non-commercial, non-business use. They state that access to information is governed by separate Market Data Terms and that entitlements depend on the selected plan and source-specific qualifications. Business Terms exist separately. | Whether a university project or faculty/public demonstration must use a business/institutional agreement; the market-data terms and third-party data conditions for the relevant exchanges; storage; model training; dashboards; derived results; redistribution; and retention. Individual-plan wording must not be assumed to authorize organizational/university use. |
| Yahoo / yfinance | Yahoo's official terms, Section 2.4.9, prohibit automated collection from its services without express prior permission. The local project documentation records that this permission has not been verified. | Express prior permission for automated collection; local retention; research/model training; publication/dashboard display; data or derived-output distribution. Without permission, no collection should occur. |

For any provider that requires written approval, ask whether the proposed university project is permitted to: (1) retain historical data locally for the project and after the subscription ends; (2) use it to train and evaluate a noncommercial RL model; (3) show raw or adjusted price charts, screenshots, and metrics to faculty; (4) publish a public demonstration/dashboard and source code; (5) publish model weights, trades, signals, benchmark comparisons, or other derived outputs; and (6) redistribute raw or transformed price records. Ask about attribution, display limits, exchange/source-specific terms, fees, approved plan, and whether a written academic/educational license is available. Request written confirmation for both ETF and index datasets.

## 8. Provider Ranking

| Rank | Provider | Status | Basis and gate before any acquisition |
|---:|---|---|---|
| 1 | FMP | **CONDITIONAL GO** | Preferred to investigate because the referenced API may supply native adjusted OHLC. Require accessible authoritative field/method documentation, confirmation for all ETF and context symbols and target dates, plan entitlement/pricing, volume/revision details, and written academic/public-use rights. |
| 2 | Tiingo | **CONDITIONAL GO** | Fallback because the brief describes an adjusted-OHLC product and split/dividend methodology. Require accessible docs, ticker/date confirmation, ETF distribution treatment, plan pricing, snapshot/revision details, and written retention and demonstration permissions. |
| 3 | Alpha Vantage | **CONDITIONAL GO** | Adjustment FAQ and long history are promising, but the documented daily-adjusted response is raw OHLCV plus adjusted close, and the exact adjusted-OHLC route/plan and ETF/index data rights are unsettled. |
| 4 | Massive | **NO-GO** | Its stock-bars documentation only asserts split adjustment; the requested total-return cash-distribution treatment is absent from that endpoint description, and its indices history begins after the required context start. |
| 5 | Yahoo Finance / yfinance | **NO-GO** | Current record lacks express prior permission for automated collection, and no provider completeness or snapshot guarantee was established. |

**Preferred provider:** FMP, provisionally and only for written verification.
**Fallback provider:** Tiingo, provisionally and only for written verification.
**Acquisition approval:** None. No provider is sufficiently verified for acquisition today.

## 9. Final Recommendation

The provisional FMP-then-Tiingo ranking is about the shape of the promised data products, not a finding that either is available, affordable, licensed, complete, or financially accurate for these symbols. Do not transform the existing FMP files, request data, or start training on the basis of this audit.

Before any acquisition, obtain written evidence of:

1. adjusted O/H/L/C semantics and exact treatment of splits, ordinary ETF distributions, capital gains, return of capital, and special distributions;
2. support and actual 2017-01-03–2026-11-11 coverage for each of the six named ETFs, plus compatible ^GSPC and ^VIX history;
3. endpoint and plan entitlements, explicit price, rate limits, output fields including volume, and any history limits;
4. whether old values are restated, availability of as-of history, snapshot/retention rights, and reproducible query/version identifiers;
5. written permissions covering local retention, academic/model use, faculty evaluation, public displays, source-code publication, derived outputs, and redistribution; and
6. a fixed data snapshot that passes symbol-by-symbol completeness, date alignment, price-quality, distribution-reconciliation, and independent total-return validation.

Adjusted prices are restated price units designed to express a price path under a provider's adjustment convention. A next-session adjusted open is a synthetic rescaled price, not proof of an actual historical market fill. A simulation using it must not also apply distributions or splits separately, and must not mix adjusted prices with raw prices. Those modeling controls cannot compensate for undocumented adjustment semantics or insufficient data rights.

## 10. Questions Requiring Written Provider Confirmation

Send the following questions to FMP and Tiingo first, then to Alpha Vantage if it remains a candidate:

1. Is daily adjusted OHLC explicitly supported for each of SPY, QQQ, IEF, GLD, XLU, and XLV, with complete observations from 2017-01-03 to 2026-11-11? Please provide symbol-specific coverage dates and identify any gaps.
2. Are `open`, `high`, `low`, and `close` all adjusted for both splits and cash distributions? How are ordinary dividends, capital-gain distributions, return of capital, special distributions, and other ETF corporate actions treated?
3. Are ^GSPC and ^VIX available for the same period under a compatible endpoint, plan, and license? Are they separate products or entitlements?
4. Does the response include volume, and if so, is it raw, split-adjusted, or otherwise adjusted? What does that volume mean when prices are distribution-adjusted?
5. What plans include the required endpoints, historical depth, symbols, and request volume? What is the current total price, including index add-ons and academic or public-display charges?
6. Are historical values revised after corporate actions or corrections? Is there point-in-time/as-of access, a dated immutable snapshot, a revision log, or a method to reproduce the exact historical response?
7. May a student retain data locally, train/evaluate an RL model, show charts and metrics privately to faculty, and keep the data after the project or subscription ends?
8. May we publish a GitHub repository containing code but no raw data, and publish derived results, performance statistics, model weights, signals, or portfolio histories? May we show price charts, dashboard screenshots, or the demonstration publicly?
9. May raw or transformed price records be redistributed? If not, what source-data deletion, cache, backup, or access-control conditions apply?
10. Is there a written student, educational, or academic license covering all those uses, and do separate exchange or third-party source terms impose additional restrictions?

Do not treat an email describing general API availability or educational request limits as permission for retention, model training, or public display unless it explicitly covers those uses.

## 11. Next Steps

1. Request written method, symbol/date, plan, revision, index, and license answers from FMP and Tiingo; keep responses with the project audit record.
2. In parallel, ask Alpha Vantage whether its educational entitlement includes the exact daily-adjusted endpoint and index data, and whether it returns usable adjusted O/H/L or only raw OHLC plus adjusted close.
3. Compare provider responses against the required distribution and split evidence. Keep unverified events visible; do not use price-factor changes as proof of corporate actions.
4. Select a provider only after rights and exact coverage are confirmed. Then obtain an immutable, checksummed snapshot with provider, endpoint, plan, query parameters, retrieval date, symbol/date counts, and adjustment metadata recorded.
5. Run the pre-training data validation and economic accounting tests from the Phase 5.3 audit. Only after data engineering, financial accounting, research-use permissions, and reproducibility pass should PPO training readiness be reconsidered.

### Files inspected

- `requirements.txt`
- `setup.py`
- `src/data/data_fetcher.py`
- `src/data/data_store.py`
- `src/data/CORPORATE_ACTIONS.md`
- `src/rl/environment/portfolio_env.py`
- `docs/phase_5_3_final_adjusted_price_audit.md`

### File created

- `docs/phase_5_4_data_provider_feasibility.md` (this report)

No existing project files or datasets were modified. No API calls for market data, market-data downloads, purchases, training, or trades occurred. Provider documentation and terms pages only were accessed for research; several were inaccessible or returned incomplete application shells, as described above.
