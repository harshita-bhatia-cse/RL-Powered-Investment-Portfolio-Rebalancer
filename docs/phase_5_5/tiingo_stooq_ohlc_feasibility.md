# Phase 5.5 — Tiingo, OHLC.io, and Stooq Feasibility

**Research date:** 2026-11-13  
**Project:** RL-Powered Investment Portfolio Rebalancer, GLA University  
**Constraint:** ₹0 budget; no purchases or market-data downloads

## 1. Executive Summary

**No provider reviewed here is approved as a complete, rights-cleared source for this project.**

- **Tiingo:** The official EOD documentation, pricing page, and attempted terms locations were not readable by the available documentation retriever; the pages returned a JavaScript application shell or 404. Therefore the existence/terms of a free Starter entitlement, six-ETF support, five-year history, adjusted OHLC fields, request limits, index/VIX availability, and academic/public-display rights cannot be verified from accessible official evidence. Treat as **NO-GO pending written confirmation**.
- **OHLC.io:** Its official pricing page explicitly limits the $0 Free plan to seven predefined tickers: AAPL, NVDA, JNJ, PEP, MSFT, SPY, and QQQ. It offers five years of history, EOD, 12,500 credits/day, 30 requests/minute, and two concurrent requests, but says personal use only. Of the eight required symbols, only SPY and QQQ are on the published free list. The four remaining ETFs and both context benchmarks are not. It is therefore **NO-GO** for the mandatory universe, independent of the five-year depth.
- **Stooq:** Official documentation, terms, and historical-download pages could not be retrieved because the research tool blocked resolution of `stooq.com`. Exact ticker support, five-year depth, adjustment conventions, automated-download permission, and academic/public-display rights remain **UNVERIFIED**. Treat as **NO-GO pending official evidence and written rights clarification**.

**Recommendation:** Do not build the required portfolio dataset from any of these free offers at this time. OHLC.io's free plan is demonstrably incomplete for the required symbols. Tiingo or Stooq can be reconsidered only after official written confirmation addresses coverage, adjustments, and use rights. Historical PPO training remains **NO-GO**.

No email was sent, no credential was used, no market-data API was called, and no market data were downloaded.

## 2. Required Universe and Test for Feasibility

Tradable ETFs: **SPY, QQQ, IEF, GLD, XLU, XLV**.  
Context benchmarks: **^GSPC and ^VIX**.  
Required data: five years of **daily OHLC**, with a sufficiently explicit adjustment methodology for portfolio backtesting.

The current five-year window as of the research date is approximately **2021-11-13 through 2026-11-13**. A provider cutoff described only as “five years” still requires confirmation of exact first available date and daily trading-session coverage for every symbol.

A provider is not feasible merely because it has some free data or a free plan. It must support all eight required symbols, the full period, daily OHLC, acceptable split/distribution treatment, and the project's academic research and demonstration uses.

## 3. Tiingo Feasibility

### Official sources attempted

Access date for all references: 2026-11-13.

- [Tiingo EOD documentation](https://www.tiingo.com/documentation/end-of-day)
- [Tiingo Daily API documentation](https://api.tiingo.com/documentation/tiingo/daily)
- [Tiingo pricing page](https://www.tiingo.com/account/billing/pricing)
- [Tiingo documentation home](https://www.tiingo.com/documentation/)
- [Tiingo terms location attempted](https://www.tiingo.com/terms)

The EOD and pricing URLs returned the Tiingo “Stock Market Tools” HTML/JavaScript application shell rather than the referenced product documentation or a readable pricing table. An attempted terms URL returned 404. Alternative documentation paths redirected to the same shell. No Tiingo API/data endpoint was called.

### Findings and confidence

| Requirement | Finding from accessible official evidence |
|---|---|
| Free Starter plan exists and its exact eligibility | **UNVERIFIED.** The current official pricing content could not be read. Do not rely on the plan name or an old plan description without confirmation. |
| SPY, QQQ, IEF, GLD, XLU, XLV supported | **UNVERIFIED ticker by ticker.** No symbol-level coverage evidence was accessible. |
| Five years of daily history for every required ETF | **UNVERIFIED.** No plan-specific history window or ticker-specific start date was accessible. |
| Adjusted daily OHLC fields | **UNVERIFIED in this access attempt.** Earlier project research summarized Tiingo materials as describing adjusted Open/High/Low/Close and split/dividend adjustments, but the official page could not be independently inspected here; this is not sufficient evidence for selection. |
| Split adjustments | **UNVERIFIED** for this plan and these ETFs. |
| Ordinary distributions | **UNVERIFIED** for the applicable endpoint and ETFs. |
| Special, capital-gain, return-of-capital, or other ETF distributions | **UNVERIFIED.** No accessible source resolved classification or adjustment treatment. |
| ^GSPC and ^VIX | **UNVERIFIED.** No symbol support or plan entitlement was accessible. |
| API request limits | **UNVERIFIED.** No current plan limits were verifiable. |
| Local retention and university internal research | **UNVERIFIED.** A free API credential or personal plan would not establish data-retention rights. |
| Model training and faculty review | **UNVERIFIED.** No written license terms were accessible. Do not assume internal use includes sharing results with faculty. |
| Dashboard demonstration or public display | **UNVERIFIED.** No permission was established for private classroom demonstration, public dashboard access, screenshots, charts, or derived performance output. |

### Assessment

Tiingo cannot currently be called feasible under the ₹0 constraint because neither free-plan access nor entitlement to the complete universe and period could be established. Adjusted OHLC and a documented adjustment methodology would be useful only if confirmed to apply to all six ETFs and relevant ETF distributions. Even then, adjusted open is a synthetic adjusted-price coordinate, not an actual historical executable price.

**Status: NO-GO pending verification and written permission.** This is an evidence limitation, not a conclusion that Tiingo does not offer these features.

## 4. Questions for Tiingo Support

No message has been sent. The following questions are prepared for GLA University to send to Tiingo support and request written answers:

1. Does Tiingo currently offer a **free Starter plan**? Please provide the current plan page or written feature/limit schedule, including any student, academic, or educational exception.
2. Is the plan free for a GLA University final-year project, or does university affiliation require a paid/institutional license? Does an academic exception cover API use, storage, and subsequent display?
3. Are daily EOD records available for **each** of SPY, QQQ, IEF, GLD, XLU, and XLV? Please confirm each ticker individually and provide the first available date, latest available date, and any material gaps, including coverage from 2021-11-13 through 2026-11-13.
4. Can the same plan provide **^GSPC and ^VIX** daily data? Please identify the exact symbols and whether either requires another dataset, plan, or exchange entitlement.
5. Which exact returned fields are raw and which are adjusted? Does the daily response return adjusted Open, High, Low, and Close for each ETF, or only adjusted close and corporate-action fields?
6. What is the adjustment method? Are splits and ordinary cash distributions reflected? How are ETF capital-gain distributions, return of capital, special distributions, and other non-ordinary distributions treated? Are action records and dates supplied?
7. Does adjusted volume exist, and if so, is it raw, split-adjusted, or adjusted in another way?
8. What are the free plan's API request limits per minute, hour, and day; concurrency limits; burst limits; and any symbol, endpoint, or historical-depth restrictions?
9. May a student retain retrieved history locally for a non-commercial university project? For how long, and what must be deleted upon account closure or plan termination?
10. May the project use the data to train and evaluate a reinforcement-learning/PPO model and retain model weights, portfolio histories, and derived performance metrics?
11. May data-derived charts, screenshots, and metrics be shown privately to the faculty supervisor and examination panel at GLA University? Does that count as display or redistribution under the license?
12. May the project demonstrate a dashboard to faculty or evaluators? If public access or a public URL is planned, is that allowed under the free plan? If not, are there restrictions on local-only/private demos?
13. May the project publish source code on GitHub if no raw data or API key is included? May it publish derived returns, selected weights, trades, model weights, or other outputs?
14. Are raw data, transformed price files, or substantially similar data prohibited from being redistributed, even to faculty? What attribution, display limits, or exchange/vendor terms apply?
15. Are historical values revised after corporate actions or corrections? Is there a dated snapshot, as-of query, revision history, or supported way to reproduce the same dataset later?
16. Please confirm in writing whether the complete proposed use—local retention, academic model training, faculty review, dashboard demonstration, and publication of code/derived results—is permitted at no cost under a specific named plan.

## 5. OHLC.io Free Plan Assessment

### Official sources

Access date: 2026-11-13.

- [OHLC.io pricing and plan comparison](https://ohlc.io/pricing)
- [OHLC.io documentation](https://ohlc.io/docs)
- [OHLC.io Terms of Service](https://ohlc.io/terms)

The official pricing page presents a **Free $0** plan and states that it is “Exclusively for testing on 7 predefined tickers.” It names: **AAPL, NVDA, JNJ, PEP, MSFT, SPY, and QQQ**. It states five years of history (comparison table: “Back to Nov 2021”), EOD and 15-minute delayed quotes, 12,500 credits/day, 30 requests/minute, two concurrent requests, no credit card required, and **personal use only**. It explicitly says “No other ETFs” and excludes research datasets. These are provider-published plan claims, not evidence that an API query was made or that returned data were validated.

### Required-symbol check against the published free allowlist

| Required symbol | Asset type/role | On listed free-plan symbols? | Conclusion |
|---|---|---:|---|
| SPY | ETF | Yes | Included in stated free ticker allowlist |
| QQQ | ETF | Yes | Included in stated free ticker allowlist |
| IEF | ETF | No | Not included; free plan is restricted to the listed seven symbols |
| GLD | ETF | No | Not included; free plan says no other ETFs |
| XLU | ETF | No | Not included; free plan says no other ETFs |
| XLV | ETF | No | Not included; free plan says no other ETFs |
| ^GSPC | Benchmark index | No | Not on the seven-symbol allowlist; exact index access is not stated for Free |
| ^VIX | Volatility index | No | Not on the seven-symbol allowlist; exact index access is not stated for Free |

The free plan therefore covers **2 of 6 required ETFs** and **0 of 2 required benchmarks**. It cannot satisfy the mandatory universe. Its stated five-year history would nominally reach approximately November 2021, but the ticker restriction is independently disqualifying. The page labels its Free tier “personal use only”; the terms say Free and Developer plans are for personal use. That does not grant university/faculty sharing or public dashboard display permission.

OHLC.io's documentation page describes a broad API catalog, including stocks, funds/ETFs, and indices, but the catalog does not override the pricing page's explicit Free ticker allowlist or confer plan entitlements.

**Status: NO-GO** for this project under the current free plan. Do not attempt aliases, batch calls, or other means to bypass its symbol restriction.

## 6. Stooq Assessment

### Official sources attempted

Access date: 2026-11-13.

- [Stooq homepage](https://stooq.com/)
- [Stooq historical database area](https://stooq.com/db/h/)
- [Stooq historical-data/download page](https://stooq.com/t/?i=573)
- [Stooq daily CSV route (documentation/access check only)](https://stooq.com/q/d/l/?s=spy.us&i=d)
- [Stooq terms location attempted](https://stooq.com/terms)

The page retriever blocked `stooq.com` during URL resolution (blocked-address error). No Stooq download was performed. No official page contents for documentation or terms could be verified.

| Requirement | Result |
|---|---|
| SPY support | **UNVERIFIED** from accessible official evidence |
| QQQ support | **UNVERIFIED** |
| IEF support | **UNVERIFIED** |
| GLD support | **UNVERIFIED** |
| XLU support | **UNVERIFIED** |
| XLV support | **UNVERIFIED** |
| ^GSPC support | **UNVERIFIED** |
| ^VIX support | **UNVERIFIED** |
| Five-year daily OHLC history for every required symbol | **UNVERIFIED** |
| Raw versus adjusted OHLC semantics | **UNVERIFIED** |
| Split adjustment method | **UNVERIFIED** |
| Dividend/ETF distribution adjustment | **UNVERIFIED** |
| Automated download permission and request limits | **UNVERIFIED** |
| Local retention, academic model training, faculty review | **UNVERIFIED** |
| Private/public dashboard demonstration and data display | **UNVERIFIED** |
| Raw/derived redistribution and attribution | **UNVERIFIED** |

The presence of a conventional CSV URL pattern in a proposed link is not proof of legal permission, current endpoint availability, complete data, or adjustment policy. No automated request should be made until Stooq's official terms and allowed access methods are reviewed. The task is not to treat an accessible download route as an implied license.

**Status: NO-GO pending official verification and explicit rights clarification.**

## 7. Comparative Decision

| Provider/plan | Symbol coverage evidenced | Five-year daily coverage | Adjusted OHLC evidence | Request limits | Academic and dashboard rights | Decision |
|---|---|---|---|---|---|---|
| Tiingo free Starter (plan itself unverified) | None verified ticker-by-ticker | Unverified | Not independently verified from accessible docs | Unverified | Unverified; request written permission | **NO-GO pending confirmation** |
| OHLC.io Free $0 | SPY and QQQ only; IEF, GLD, XLU, XLV, ^GSPC, ^VIX excluded/not listed | Plan page states five years, back to Nov 2021 | This research pass did not establish the Free endpoint's adjusted OHLC semantics; not material to disqualification | 12,500 credits/day; 30 requests/min; 2 concurrent | Pricing says personal use only; no faculty/public display right established | **NO-GO** |
| Stooq free/access path (not verified) | All eight unverified | Unverified | Unverified | Unverified | Unverified | **NO-GO pending confirmation** |

**Preferred provider:** None can be selected on current evidence.  
**Most concrete finding:** OHLC.io Free is definitively incompatible with the required ticker list.  
**Potential next research candidate:** Tiingo only if support responds in writing with a qualifying free academic plan and confirms each symbol, history, adjustment method, and rights. Stooq should remain excluded until its official terms and exact symbol coverage can be verified.

## 8. Required Follow-up Before Data Acquisition

1. Send the prepared questions in Section 4 to Tiingo support manually; do not include or send credentials.
2. Request a written academic exception that explicitly covers GLA University, local retention, model training, faculty review, dashboard demonstration, and publication of code/derived outputs.
3. Ask Tiingo to confirm each ETF and both indices individually, with first/last dates, daily frequency, fields, distribution adjustments, and current free-tier rate limits.
4. For Stooq, obtain an official terms/documentation URL or written response covering permitted automated downloads, retention, symbol coverage, adjustment methodology, and academic/public-display uses before making any data request.
5. Do not use OHLC.io Free for this project unless the project scope changes; the Free ticker restriction conflicts with mandatory instruments.
6. Keep PPO training **NO-GO** until complete data coverage, financial accounting, reproducible snapshots, and use permissions have all been established.

## 9. Scope and Safeguards

- Provider documentation, pricing, and terms pages only were accessed.
- No Tiingo or Stooq API/data endpoint was called.
- No full or partial historical data download was performed.
- No email or support request was sent.
- No purchase, credential access, PPO training, or trade execution occurred.
- No production code, existing project configuration, or data files were modified.

