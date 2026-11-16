# Phase 5.5 — Next Free Provider Investigation

**Investigation date:** 2026-11-13  
**Budget constraint:** ₹0; no paid subscriptions considered or purchased.

## Executive summary

- The one authorized Alpha Vantage request for `TIME_SERIES_DAILY`, symbol SPY, with `outputsize=compact` and `datatype=json` returned **HTTP 200 and 100 actual daily price records**. The API identified the ticker as SPY. Records span **2026-06-24 through 2026-11-12**.
- Returned fields were open, high, low, close, and volume. **No adjusted close or adjusted OHLC fields** were in this response. The endpoint returned raw/as-traded data, consistent with Alpha Vantage's official documentation.
- Alpha Vantage documentation states that compact daily output (latest 100 observations) is available to free and premium keys; daily `outputsize=full` is available only to premium keys. Thus the free daily endpoint does not provide the requested five-year daily history in a single response.
- Alpha Vantage weekly/monthly endpoints document long historical series, but the reviewed material does **not** establish free-key entitlement for these specific functions. The adjusted weekly/monthly descriptions include OHLC plus an adjusted-close field; they do not establish that the returned OHLC fields themselves are adjusted. Lower-frequency bars would not meet a daily OHLC requirement.
- Tiingo's official EOD and pricing pages returned a JavaScript application shell in this audit, so free access, exact ticker coverage, date depth, adjustment semantics, and academic/dashboard rights remain unverified.
- Stooq's official site and download URLs could not be retrieved through the available research tool; no provider terms or ticker/date/adjustment guarantees could be verified.
- Yahoo's official terms prohibit automated data collection without express prior permission. yfinance is a third-party library for Yahoo Finance data and defers data-use rights to Yahoo's terms. No such permission is established for this project.

**Recommendation:** **CONDITIONAL GO** for Alpha Vantage `TIME_SERIES_DAILY` only as a raw-price, 100-record connectivity check or exploratory source. **NO-GO** as the sole source for the required five-year daily history or adjusted total-return OHLC under the current free plan. Do not begin historical PPO training from this response. No additional symbol or endpoint was tested.

## 1. Alpha Vantage free daily SPY test

### Request and credential handling

- **Endpoint:** `https://www.alphavantage.co/query`
- **Function:** `TIME_SERIES_DAILY`
- **Symbol:** SPY
- **Parameters:** `outputsize=compact`, `datatype=json`
- **Authentication:** Used the existing `ALPHA_VANTAGE_API_KEY` from `.env`. The credential and credential-bearing URL were not printed or saved.
- **Requests made for this investigation:** One market-data API request only.
- **Response body handling:** Parsed in memory and reported as a summary; no full response was displayed or written to disk.

### Observed response

| Check | Result |
|---|---|
| HTTP status | 200 |
| Returned ticker | SPY |
| Actual daily records | Yes |
| Number of records | 100 |
| Latest record date | 2026-11-12 |
| Earliest record date | 2026-06-24 |
| Fields observed | `1. open`, `2. high`, `3. low`, `4. close`, `5. volume` |
| Adjusted close | Not present |
| Adjusted open/high/low | Not present |
| Premium/API/rate-limit restriction in response | No restriction message; time-series data was returned |

This confirms the endpoint worked for this single SPY compact request at the time of test. It does **not** prove five-year coverage, complete ETF or index coverage, future availability, total-return adjustment, stable revisions, or licensing rights.

## 2. Alpha Vantage historical and adjusted endpoint investigation

Official references, accessed 2026-11-13:

- [Alpha Vantage API documentation](https://www.alphavantage.co/documentation/)
- [Alpha Vantage support and FAQ](https://www.alphavantage.co/support/)
- [Alpha Vantage premium plans](https://www.alphavantage.co/premium/)

| Question | Finding |
|---|---|
| Is daily `outputsize=full` free for educational accounts? | The official Daily documentation explicitly says `compact` (latest 100 points) is available to free and premium API keys and `full` (25+ years) is available to premium keys. The support FAQ mentions free service for most datasets and unlimited request frequency for verified open-source/educational projects, but this does not override the Daily endpoint's premium-only `full` entitlement. A free educational request-frequency exception is not evidence that premium features are included. |
| Do weekly/monthly endpoints document long history? | Yes. The official weekly and monthly descriptions state 25+ years of historical data. Unlike Daily, the fetched descriptions do not state a `compact`/`full` parameter. |
| Are weekly/monthly functions guaranteed free? | **Not established.** The docs provide free-key links and example calls, and the support page says most datasets are free, but neither source explicitly confirms that each weekly/monthly function is included with this account's free entitlement. No calls were made to these endpoints. |
| Do weekly/monthly adjusted endpoints provide adjusted OHLC? | **Not established.** The official descriptions list weekly/monthly open, high, low, close, adjusted close, volume, and dividend. Listing a distinct adjusted-close field does not prove that the OHLC fields are adjusted. The support FAQ makes a general statement that Alpha Vantage adjusts OHLCV for splits and cash dividends in supported adjusted-data APIs, but it does not resolve the field semantics or access entitlement for each weekly/monthly endpoint. |
| Does any verified free endpoint provide adjusted daily OHLC? | No. The tested free daily endpoint returned raw OHLCV only. `TIME_SERIES_DAILY_ADJUSTED` was premium-restricted in the prior SPY test and its documented response is raw OHLCV plus adjusted close and split/dividend events. No reviewed evidence identifies a free daily endpoint returning adjusted O/H/L/C. |
| Could weekly/monthly substitute for the goal? | No for the stated daily OHLC requirement. Weekly/monthly bars cannot reconstruct daily execution prices, and any free eligibility/adjusted field semantics need written confirmation. |

The Alpha Vantage support page describes free service as covering “the majority” of datasets, typically with 25 requests/day, and says verified educational/open-source projects may receive unlimited requests. The same page's broad usage statement should not be read as an entitlement to a premium endpoint or full daily output.

## 3. Free-provider comparison

Coverage below means **officially established evidence for this audit**, not a guess based on a provider's general market coverage. No provider data were downloaded or queried for symbols beyond SPY at Alpha Vantage.

| Provider | Six ETFs and five-year daily history | OHLC and adjustments | ^GSPC / ^VIX | Academic and dashboard permission | Current finding |
|---|---|---|---|---|---|
| **Alpha Vantage** | The one SPY compact request yielded only the latest 100 records. No requests were made for QQQ, IEF, GLD, XLU, XLV, ^GSPC, or ^VIX. Daily full history is documented as premium-only. | Free daily SPY response has raw OHLCV and no adjusted close. The tested Daily Adjusted endpoint is premium-restricted and its documentation describes raw OHLCV plus adjusted close/events, not adjusted O/H/L. | The provider offers separate index data APIs, but exact free access and required symbols/history were not checked in this investigation. | Free key acquisition does not by itself establish local retention, model use, public display, screenshots, redistribution, or derived-output rights. Obtain and review current terms for the intended uses. | **CONDITIONAL GO** for the raw compact connectivity test; **NO-GO** as the complete five-year adjusted daily source at ₹0. |
| **Tiingo** | Official EOD documentation/pricing pages returned the SPA shell; ticker-specific six-ETF support and five-year history could not be verified. | Project's earlier provider audit described a documented adjusted OHLC product, but direct documentation could not be read in this session. ETF distribution treatment and free-plan availability are unresolved. | ^GSPC/^VIX availability and compatible free entitlement are unverified. | Pricing/documentation shell gave no dependable terms. Local retention, academic training, public dashboards, screenshots, and redistribution require explicit confirmation. | **NO-GO pending verification**; potentially worth revisiting only if the official docs and free entitlement can be confirmed. |
| **Stooq** | Official site/download URLs were blocked by the research tool. No evidence established exact availability or five-year coverage for the six ETF symbols. | No official documentation was accessible to verify daily OHLC field definitions, adjustment basis, dividend handling, or split conventions. | Exact index/VIX symbols, history, and terms were not verified. | No accessible official terms or license material established research, retention, dashboard, or redistribution permissions. | **NO-GO pending authoritative evidence**; do not infer “free” from a reachable or commonly used CSV URL. |
| **Yahoo Finance / yfinance** | No data queries were made. Exact six-ETF and index/VIX coverage and five-year history were not established by this investigation. | yfinance's package documentation describes downloading Yahoo Finance market data. The local yfinance implementation applies an adjusted-close/close ratio to OHLC when auto-adjusting, but that is a library transformation and does not prove Yahoo completeness or provider-certified adjusted OHLC. | Exact history/rights for ^GSPC and ^VIX were not tested. | Yahoo's [Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html), Section 2.4.9, prohibit automated collection without express prior permission. The [yfinance project page](https://pypi.org/project/yfinance/) directs users to Yahoo terms for rights to data. No express prior permission for this project is documented. | **NO-GO** for automated acquisition absent express permission; no dashboard/publication right established. |

### Provider source access notes

- Tiingo pages attempted: [EOD documentation](https://www.tiingo.com/documentation/end-of-day), [pricing](https://www.tiingo.com/account/billing/pricing). Both returned the site's JavaScript application shell rather than the referenced product details. This is a research access limitation, not evidence that Tiingo lacks the product.
- Stooq pages attempted: [official site](https://stooq.com/), [daily CSV download endpoint](https://stooq.com/q/d/l/?s=spy.us&i=d), and [historical data area](https://stooq.com/db/h/). Requests were blocked by the page retriever's network-resolution restriction. No endpoint was used to download data. Stooq licensing and product specifics remain unverified.
- Yahoo/yfinance references: [Yahoo Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html), [Yahoo Developer API Terms](https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html), [yfinance PyPI page](https://pypi.org/project/yfinance/), and [yfinance history reference](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html). yfinance is a third-party library and its own documentation says users should refer to Yahoo terms for data rights.

## 4. Recommendation and unresolved questions

### Recommendation

1. **Alpha Vantage:** The free compact daily endpoint is technically accessible for SPY and returned five raw daily fields, but only 100 observations. It is not sufficient for five-year daily training. Daily `full` is premium-only; the adjusted daily endpoint is premium-restricted. **Do not use as the full-history source under the strict ₹0 budget.**
2. **Tiingo:** Do not select until accessible official docs and written confirmation establish a free plan with all six ETFs, complete five-year daily history, any required index/VIX series, adjustment semantics, and the project's academic/dashboard rights.
3. **Stooq:** Do not select until official documentation, licensing, symbol/date coverage, and adjustment policy can be checked. Current inability to access official materials leaves these claims unresolved.
4. **Yahoo/yfinance:** Do not automate data collection without express prior Yahoo permission. The free/accessible nature of a tool does not settle permission to retain or display its data.
5. **Historical PPO training:** **NO-GO**. No complete, rights-cleared five-year daily dataset with defensible adjusted OHLC has been established.

### Still unresolved

- Whether Alpha Vantage grants these specific free-key endpoints to this educational project beyond compact daily; whether the weekly/monthly adjusted functions are free; and exact ETF/index symbol support.
- Whether any source may be retained locally, used for model training, shown to faculty, displayed in a public dashboard, or used to publish derived results.
- ETF-specific treatment of distributions, including return of capital, capital gains, and special distributions.
- Reproducible snapshots, data revision policy, and whether future historical revisions can be detected.
- Complete ^GSPC and ^VIX coverage under the same acceptable free arrangement.

## Scope and safeguards

- Only SPY was queried, once, at Alpha Vantage using `TIME_SERIES_DAILY`, `compact`, JSON.
- No other symbols or endpoints were queried.
- No full history or dataset was downloaded, stored, or exported.
- No API key, complete response, or credential-bearing URL appears in this report.
- No production code, existing dataset, PPO code, or trading integration was modified.
- No subscription or trade occurred.

