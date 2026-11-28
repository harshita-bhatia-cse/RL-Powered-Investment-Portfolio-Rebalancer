
<div align="center">

# Investment Portfolio Rebalancer

### An engineering project for portfolio research and paper simulation

*A final-year project with separate India-focused portfolio APIs, historical CSV research, and synthetic-only PPO experiments.*

[Quick Start](#quick-start) | [Dashboard](#investment-portfolio-rebalancer-dashboard) | [Implemented Research and Portfolio Functions](#implemented-research-and-portfolio-functions) | [Results](#results)

</div>

---

## About

**Investment Portfolio Rebalancer** is an academic engineering project. Its implemented components include an India-focused account and portfolio API, stored end-of-day price valuation, portfolio monitoring and rebalance previews, a historical CSV backtest API, and a separate synthetic PPO research experiment.

These components are separate: the historical backtest reads its configured CSV dataset; account valuation and previews use stored India EOD prices; and Phase 7.2 PPO experiments use synthetic data only. Synthetic experiment metrics are not NSE/BSE or real-market investment performance.

> This project is for education and research. It is not financial advice, does not guarantee returns, and does not execute real-money trades from the dashboard.

> **Scope:** account records are limited to exchange-qualified NSE/BSE
> listings. Valuation uses already-stored EOD prices. Monitoring reports and
> rebalance previews are simulations; they do not execute or automate trades
> and are not investment recommendations. The India-focused portfolio API
> and teammate-owned dashboard do not connect to a brokerage; the dashboard is
> not connected to the account API. Separately, the repository retains a
> legacy Alpaca CLI trading path that can submit broker orders when configured.
> That path is not part of the India-focused first-version portfolio workflow.

The historical research code includes a weight-based allocation interface:

$$w_t = \mathcal{R}_t\bigl(\mathcal{T}_t\bigl(\mathcal{A}_t\bigl(\mathcal{S}_t(\mathcal{X}_{\le t})\bigr)\bigr)\bigr)$$

where $\mathcal{S}$ denotes stock selection, $\mathcal{A}$ portfolio allocation, $\mathcal{T}$ timing adjustment, and $\mathcal{R}$ portfolio-level risk overlay. This describes retained research code, not a live execution or brokerage interface.

---

## Legacy Research Architecture

The repository retains older research modules and examples. They are not
claims that the current application provides every listed data source,
strategy, or execution capability.

| Layer | Legacy role | Components |
|:------|:-----|:-----------|
| **Data** | Unified market data pipeline | FMP, Yahoo Finance, WRDS; LLM sentiment preprocessing; SQLite cache |
| **Strategy** | Weight-centric signal generation | Stock selection, portfolio allocation, timing adjustment, risk overlay |
| **Backtest** | Offline evaluation | `bt`-powered engine with multi-benchmark comparison and transaction costs |
| **Execution** | Legacy CLI integration, separate from the India-focused portfolio API | Alpaca order submission when configured; not used by the account API or dashboard |

```
finrl-trading/
├── src/
│   ├── config/                     # ⚙️  Centralized configuration management
│   │   └── settings.py             #     Pydantic-based settings + environment variables
│   ├── data/                       # 🗄️  Data acquisition and processing
│   │   ├── data_fetcher.py         #     Multi-source integration (Yahoo / FMP / WRDS)
│   │   ├── data_processor.py       #     Feature engineering & data cleaning
│   │   └── data_store.py           #     SQLite persistence with caching
│   ├── backtest/                   # 📊  Backtesting engine
│   │   └── backtest_engine.py      #     bt-powered engine with benchmark comparison
│   ├── strategies/                 # 🤖  Trading strategies
│   │   ├── base_strategy.py        #     Abstract strategy framework
│   │   └── ml_strategy.py          #     Random Forest stock selection
│   ├── trading/                    # Legacy repository code; not part of this API
│   │   ├── alpaca_manager.py       # Historical source retained in repository
│   │   └── trade_executor.py       # Not used by account or research APIs
│   └── main.py                     # 🚀  CLI entry point
├── examples/
│   └── README.md
├── data/                           # Runtime data storage (gitignored)
├── logs/                           # Application logs (gitignored)
├── requirements.txt
└── setup.py
```

---

## Implemented Research and Portfolio Functions

The current implementation has distinct workflows:

- **India-focused account API:** stores owner-scoped account, portfolio,
  holding, and transaction records. See the account sections below.
- **Historical CSV backtesting:** runs the supported deterministic strategies
  on the configured CSV dataset and fixed train, validation, or test
  partition. It does not connect to account portfolios or brokers.
- **Synthetic PPO research:** Phase 7.2 is the only implemented Phase 7
  sub-phase. It evaluates PPO using synthetic data only; it does not evaluate
  NSE/BSE market performance.
- **Phase 8A research API:** exposes authenticated, read-only views of the
  frozen Phase 7.2 results. It does not run an experiment or load a model.

## Results

Phase 7.2 completed a synthetic PPO experiment with 15 trained runs, 300
validation results, one frozen configuration selection, and 100 held-out
synthetic test results. These are outputs of the specified synthetic
environment only. They are not NSE/BSE or real-market performance, and they
do not establish investment value, profitability, or future results. The
results are available through the Phase 8A API when the required frozen
artifacts are provisioned; numeric metrics are intentionally not reproduced
here.

---

## Quick Start

### Setup

```bash
# From the repository root
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install project dependencies
pip install -r requirements.txt
```

### Investment Portfolio Rebalancer dashboard

The existing dashboard is served by the FastAPI application. Its UI is
teammate-owned, has not been changed for this work, and is not connected to the
India-focused account API.

```powershell
# Windows PowerShell
python src/main.py dashboard
```

The dashboard remains a separate research UI; it is not the interface for
account records, stored-price portfolio valuation, monitoring, or rebalance
previews.

#### Authenticated portfolio API foundation

The account API stores users, India-listed portfolio snapshots, holdings, and
transaction records in a separate SQLAlchemy database. The default local URL
is `sqlite:///./data/portfolio_platform.db`; set `PORTFOLIO_DATABASE_URL` to a
PostgreSQL SQLAlchemy URL for deployment. This database does not replace or
migrate the SQLite research `DataStore`.

Set `AUTH_JWT_SECRET` to a unique random value of at least 32 bytes before
using login-protected routes. Generate one locally with
`python -c "import secrets; print(secrets.token_urlsafe(48))"` and store it in
your private `.env` file; never commit the generated secret.

The account schema is initialized at FastAPI startup. If account database
initialization fails, the application can start, but account routes and the
Phase 8A research routes that share account authentication return HTTP 503.
The historical backtest routes are separate from that account database.
Pending schema revisions can also be applied with
`python -m src.database.migrations`. Existing databases at revision
`0001_account_foundation` are checked for invalid transaction records before
revision 0002 adds stricter enforcement; migration stops without changing
those records if invalid rows are found.
Revision 0003 adds the ledger opening-balance snapshot and a nullable posting
marker. Pre-existing transaction rows remain historical and are not replayed.
If legacy cash transactions contain non-zero fees, the migration stops and
identifies the affected transaction IDs rather than rewriting them.
Revision 0004 adds a nullable client idempotency key and enforces uniqueness
per portfolio. Existing transaction records remain unchanged and keyless
requests remain supported. Submit `idempotency_key` in the transaction JSON:
an identical retry returns the original transaction with HTTP 200 without
reposting balances; reusing the key with a different payload returns HTTP 409.
Database uniqueness protects this behavior under concurrent requests.

The SQLite database is the zero-cost local-development option. For a
PostgreSQL SQLAlchemy URL, install the optional driver extra with
`pip install -e ".[database]"`, then set `PORTFOLIO_DATABASE_URL` in `.env`.

Available routes include `POST /api/auth/register`, `POST /api/auth/login`,
`GET /api/auth/me`, and authenticated `/api/portfolios` operations for
portfolios, holdings, and transaction history. Portfolio resources are scoped
to the authenticated owner. Holdings identify each listing by the required
`exchange` (`NSE` or `BSE`) plus symbol; the application does not infer that
two exchange listings represent the same issuer. Transactions are stored
in an immutable ledger and the authenticated transaction POST endpoint applies
BUY, SELL, cash deposit, and cash withdrawal entries to the portfolio. A BUY
debits quantity multiplied by unit price plus fees and adds fees to weighted
average cost; a SELL credits gross proceeds less fees and retains the remaining
shares' average cost. Cash movements use `cash_amount` and do not accept fees.
Amounts posted to cash are rounded to INR paise (two decimals, half-up), and
share quantities and average cost are rounded to six decimal places (half-up).
This is manual paper bookkeeping only; no order is submitted by the ledger.

Before the first posted transaction, the service snapshots the portfolio's
current cash and holdings once. Earlier transaction rows are historical and
are excluded from ledger replay. `GET /api/portfolios/{portfolio_id}/reconciliation`
compares that immutable snapshot plus subsequently posted transactions with
current cash and holdings; manual edits remain allowed but appear as
discrepancies. There is no automatic baseline reset or rebase operation.
PostgreSQL posting locks the portfolio row. SQLite uses a conditional cash
balance update to detect competing writes, but SQLite has no row-level locking
and concurrent writers may receive a retryable conflict or database-busy
error. This local account ledger is not a brokerage statement or tax ledger.

To import holdings into an existing owned portfolio, use the authenticated
`POST /api/portfolios/{portfolio_id}/holdings/import` endpoint with a
multipart field named `file`. CSV files must be UTF-8, no larger than 1 MiB,
and contain exactly these columns:

```csv
exchange,symbol,quantity,average_price
NSE,RELIANCE,5,1250.50
NSE,TCS,2,3400.00
BSE,500325,3,1500.00
```

Rows are validated before insertion. Duplicate listings, malformed values,
and conflicts with existing holdings reject the entire upload; no holdings
are overwritten. The endpoint reports invalid row numbers. The portfolio
summary endpoint reports holdings count, exchange counts, cash balance, and
quantity multiplied by average purchase price as invested amount. It does not
estimate current market value or profit/loss. Install project requirements
with `pip install -r requirements.txt` to include the free multipart parser.

#### Indian EOD market-data storage and valuation

The repository already includes `YahooFinanceIndiaProvider` (through
yfinance), an Alpha Vantage adapter, and an India daily-feed writer. Those
legacy collection paths are separate from account valuation: the feed can
write to the research `DataStore`, Yahoo/yfinance licensing and redistribution
rights are not verified here, and Alpha Vantage availability/limits depend on
its plan. The project does not scrape NSE/BSE. None of those providers is
activated by the account valuation API.

Phase 4 introduces a provider contract and dedicated `eod_prices` database,
defaulting to `sqlite:///./data/market_data.db` and configurable with
`MARKET_DATA_DATABASE_URL`. It is independent of both account persistence and
the research DataStore. Its additive migration is `0001_eod_prices`; it creates
the price table and migration ledger without changing existing databases.
Validated price ingestion and historical retrieval are available through the
`src.market_data.service` functions and an injected provider. No HTTP endpoint
or background job fetches third-party prices. The included mock provider is
for tests only.

The EOD record stores exchange, symbol, market date, INR OHLC, optional volume,
provider/source, fetch time, and an explicit valuation-permission flag.
Adjusted close is intentionally not stored: no source-specific adjustment
semantics have been verified for this account product. Ingestion rejects
future dates, non-positive/non-finite prices, invalid OHLC relationships,
invalid volume/currency, and duplicate listing dates.

Authenticated `GET /api/portfolios/{portfolio_id}/valuation` reads stored
prices only and is owner-scoped. It calculates invested cost basis separately
from market value and unrealized profit/loss. A price can be used only when
its stored source is explicitly marked permitted for valuation and its market
date is within `MARKET_DATA_MAX_AGE_DAYS` (default 4 elapsed calendar days).
This conservative elapsed-calendar threshold does not infer a trading session
from weekends or holidays. Stale, missing, and non-permitted prices are
reported separately. Market value and unrealized P/L are identified as
partial when not all holdings can be priced; total portfolio value is `null`
unless every holding has a sufficiently fresh permitted price. Cash and
holdings are not modified by valuation.

#### Read-only portfolio monitoring (Phase 6.1)

Authenticated `POST /api/portfolios/{portfolio_id}/monitor` evaluates one
explicit `session_date` (`YYYY-MM-DD`). The service reads the portfolio and
holdings, reuses stored-price valuation and ledger reconciliation, and writes
only a separate monitoring report. It does not ingest prices, create trades,
or modify portfolio cash, holdings, or transactions. A price later than the
requested session is rejected for that report to prevent look-ahead on
historical reruns.

Reports are stored in `portfolio_monitoring_reports` by additive account
migration `0005_portfolio_monitoring_reports`. The portfolio/session-date key
is unique: repeating a request returns the original persisted result rather
than reevaluating or overwriting it. Owners can list reports with
`GET /api/portfolios/{portfolio_id}/monitor/reports` and retrieve an owned
report with `GET /api/portfolios/{portfolio_id}/monitor/reports/{report_id}`.
The one-shot runner is:

```powershell
.\.venv\Scripts\python.exe -m src.services.portfolio_monitor `
  --portfolio-id <portfolio-id> --session-date YYYY-MM-DD
```

Statuses are:

| Status | Meaning |
| --- | --- |
| `NO_ACTION_REQUIRED` | The configured data-quality and ledger checks completed, valuation is complete, and reconciliation found no discrepancy. This is not an investment recommendation and does not mean the portfolio is optimal or that no trade could be appropriate. |
| `ACTION_REQUIRED` | Evaluation completed, but ledger reconciliation is unavailable or found a discrepancy that needs data-integrity follow-up. |
| `DATA_UNAVAILABLE` | At least one required holding lacks a fresh permitted price on or before the requested session. No complete NAV is reported. |
| `EVALUATION_FAILED` | An unexpected evaluation error occurred; the report contains safe generic error details and diagnostic details are logged server-side. |

Phase 6.1 requires the caller to provide the session date; it adds no
scheduler or automatic exchange-calendar logic. Only prices already present
in the dedicated market-data database are read. Live ingestion, news and
company-announcement intelligence, scheduling, notifications, and financial
BUY/SELL recommendations are deferred. No provider is activated by monitoring.

#### Exposure and concentration risk assessment (Phase 6.2A)

Each newly evaluated Phase 6.1 monitoring report includes a separate
`exposure_risk_assessment` using policy `exposure-risk-v1`. It uses the same
complete, fresh, explicitly `valuation_permitted` stored EOD valuation as the
monitor; it performs no second price lookup and never requests or activates a
provider. The existing valuation freshness rule (elapsed calendar days) and
the requested session-date cutoff apply. Prices later than the session,
missing/stale/unpermitted prices, invalid NAV, or inconsistent valuation
totals suppress every derived exposure metric and alert.

The initial research thresholds are centralized in the exposure-risk service
and use strict comparisons:

| Metric | Alert condition |
| --- | --- |
| One exchange-qualified listing's portfolio exposure | Greater than 25% of total NAV |
| Five largest listings' combined exposure | Greater than 70% of total NAV |
| Cash allocation | Less than 5% of total NAV |
| Equity HHI | Greater than 0.18 |

These are configurable research-policy thresholds, not universal investment
recommendations. Exactly 25%, 70%, 5%, or 0.18 does not trigger its respective
alert. Alert severity is a policy attention label, not a prediction of loss.

For each NSE/BSE listing, portfolio weight is listing market value divided by
total portfolio NAV; the top-five measure sums the five largest listing market
values before dividing by NAV. Ties are resolved by exchange and symbol.
NSE:TCS and BSE:TCS remain separate because the account data has no verified
issuer/ISIN mapping. Cash allocation is cash divided by total NAV. Equity HHI
is the sum of squared listing weights normalized by total equity market value
(cash is excluded); larger values indicate more concentrated equity
allocations. HHI is not applicable (`null`) when there are no equity holdings.
A positive-cash-only portfolio is evaluated with 100% cash allocation and no
equity alerts. Since the existing valuation service rounds holding market
values to paise, a complete valuation can contain a valid `₹0.00` holding
when its unrounded value is below half a paise. Such a positive-quantity
listing remains in the assessment with zero weight and does not create an
exposure alert. Negative values or invalid quantities are rejected. A zero
or invalid NAV is `NOT_EVALUATED`.

The risk assessment has `EVALUATED` and `NOT_EVALUATED` states, independent of
the four Phase 6.1 monitoring statuses. An evaluated result reports NAV,
equity and cash values, listing weights, top-five exposure, cash allocation,
equity HHI, and deterministic alerts with a stable code, severity, observed
value, threshold, policy version, and explanation. An unevaluated result
contains reason codes and unavailable exchange-qualified listings, but no
partial metrics or alerts. It is not equivalent to a successful assessment
with no alerts.

Migration `0006_portfolio_exposure_risk` adds a nullable JSON field to the
existing monitoring report. Older reports remain unchanged and expose a null
risk assessment; same-session retries continue to return the originally
persisted report. The API allowlists and validates risk data before response
model validation and does not rewrite legacy report data.

These alerts describe exposure only. They are not financial advice, price
forecasts, or BUY/SELL/HOLD instructions and do not assess issuer identity,
sector concentration, corporate actions, or price-history returns. No
brokerage orders or automatic trades are created.

#### Verified issuer concentration (Phase 6.2B)

Issuer concentration is a separate, metrics-only research assessment. Its
identity chain is exchange-qualified listing (`exchange` + `symbol`) to a
verified instrument identifier (currently `ISIN` or explicitly sourced
`OTHER_VERIFIED`) to a canonical issuer identifier (`CIN`, `LEI`, or
`OTHER_VERIFIED`). A CIN is not assumed to exist for every issuer. Company
names and matching ticker text are never used to resolve or consolidate
issuers. NSE and BSE listings remain separate traceable listing records, and
are aggregated only when their locally curated records identify the same
canonical issuer.

The local reference file is
[`src/services/issuer_reference_v1.json`](./src/services/issuer_reference_v1.json).
It currently contains no mappings; it does not claim issuer coverage. Each
record must provide exchange, symbol, instrument identifier and type, canonical
issuer ID and type, inclusive `effective_from`/`effective_to` dates (`null`
means no end date), `source`, `source_reference`, `verified_on`, and the matching
`reference_data_version`. Dates are ISO `YYYY-MM-DD`; records must not overlap
for one exchange-qualified listing, and an instrument identifier cannot map
to different issuers over overlapping dates. Duplicate records, unsupported
fields, malformed identifiers, and conflicting intervals make the assessment
unavailable rather than being guessed through.

The loader checks ISIN structure and its Luhn check digit, LEI structure and
mod-97 check digits, and the 21-character CIN structure. CIN structure has no
checksum in this validation; a matching shape does not establish that an
identifier was issued or is current. `OTHER_VERIFIED` identifiers are opaque:
they must be non-empty and have source, source-reference, and verification-date
provenance, but neither the application nor these fields establish source
authority, identifier ownership, or data-use licensing. These checks are
format validation, not authoritative verification.

To add a mapping, obtain an authoritative identifier and source whose use is
permitted for this research, verify both the listing-to-instrument and
instrument-to-issuer relationships, then add a record to the reference file.
Use a new, never-reused `reference_data_version` for any content change and
set the record's version to match. Do not infer identity from company names,
similar symbols, or a remembered ISIN/CIN. Keep prior versions available in
version control; monitoring reports store the version and computed listing
and issuer result, and same-date replay returns the persisted report without
recalculating against a changed reference file. No external provider,
scraping, or API is used.

Every positive-value holding must have exactly one verified mapping effective
on the monitoring session date. If any positive holding is unmapped, outside
its effective dates, or the reference data is invalid, the issuer assessment
is `NOT_EVALUATED` and exposes no partial listing or issuer weights. An
unmapped zero-value holding does not block assessment but remains visible as
an unresolved listing trace. A complete cash-only portfolio is valid and has
no issuer exposures. Issuer concentration is evaluated only when valuation
is complete and ledger reconciliation is available and clean; an unavailable
or discrepant reconciliation produces `NOT_EVALUATED` with no partial metrics.
Incomplete valuation never produces partial issuer metrics. Issuer weights use
total portfolio NAV; the assessment retains exchange-qualified listings and
the mapping provenance used for traceability. No issuer concentration alert
threshold has been approved, so no issuer alerts are generated. Existing
Phase 6.2A listing thresholds and the four monitoring statuses are unchanged.
Sector classification and sector concentration are deferred to Phase 6.2C.

Migration `0007_portfolio_issuer_concentration` adds a nullable JSON assessment
to monitoring reports. Existing reports are not backfilled. The new assessment
is independent from Phase 6.2A's listing-level `exposure_risk_assessment`.

#### Sector concentration (Phase 6.2C)

Sector concentration is a separate descriptive assessment built only from the
validated Phase 6.2B issuer assessment. It groups canonical issuers, not
exchange listings, so NSE and BSE listings already consolidated by Phase 6.2B
are not counted twice. It uses the same complete valuation snapshot and clean
ledger reconciliation gate. It performs no price lookup, provider call,
brokerage integration, or automatic trading.

The local reference file is
[`src/services/sector_reference_v1.json`](./src/services/sector_reference_v1.json).
It declares the project-defined taxonomy `project-india-equity-sectors` version
`1.0`, but its issuer mapping records are intentionally empty. The taxonomy is
not represented as an NSE/BSE taxonomy. Do not copy exchange taxonomy codes or
labels into this project without explicit reuse permission. The existing
symbol-level sector metadata elsewhere in the repository is unverified and is
not used for issuer sector assessment.

Every future issuer mapping must identify the verified canonical issuer, one
primary project sector, effective-from/effective-to dates, source publisher,
source document title and URL with the relevant section/page, source
publication date, verification date, and reference-data version. Use primary
source annual reports or official issuer filings; keep evidence sufficient for
another reviewer to verify the classification. A source supports a session
date only when its publication date is on or before that date and the
effective interval includes it. Do not infer identity or classification from
company names, ticker text, similar names, or unverified legacy metadata.
`DIVERSIFIED` is appropriate only where primary-source evidence supports that
classification; it is not a fallback for missing or ambiguous evidence.

If a positive-value issuer has no unique eligible mapping, sector assessment is
`NOT_EVALUATED` and exposes no partial sector weights or metrics. Cash-only
portfolios are valid and have no sector rows; zero-value listings remain
traceable but do not require sector mappings to calculate exposure. Historical
reports retain their persisted sector assessment and reference version on
same-session replay, even after the local reference changes. Each assessment
stores a canonical validated reference snapshot and SHA-256 fingerprint, and
each issuer contribution is checked against its stored mapping evidence.
Fingerprints detect inconsistency only while the original fingerprint/evidence
remains protected; a hash alone does not prove authenticity if an attacker can
replace both the stored snapshot and its fingerprint. Existing reports are not
backfilled.

Sector exposure percentages use total portfolio NAV. Largest-sector and
top-three concentration are the largest one and three sector values divided by
NAV. Sector HHI is the sum of squared sector shares of total equity value
(cash excluded), and is null when equity value is zero. Sector weights against
NAV may sum to less than 100% because cash is excluded. Metrics are
descriptive; no sector alert thresholds or alerts are approved. Migration
`0008_portfolio_sector_concentration` adds a separate nullable JSON assessment
and does not change Phase 6.2A/6.2B fields, monitoring statuses, or report
uniqueness.

#### Portfolio-aware rebalance preview

Authenticated `POST /api/portfolios/{portfolio_id}/rebalance-preview` creates
a paper-only proposal using the account's current cash and NSE/BSE holdings,
the existing stored-price valuation service, and the existing deterministic
allocation engine. The default equal-weight baseline distributes `1 -
cash_reserve_pct` equally across selected current holdings and leaves the
configured reserve in cash. `selected_holdings` accepts exchange-qualified
identifiers such as `NSE:TCS`; omitting it selects every current holding.
Current holdings excluded from a selection are held at their current
market-value weights. In custom allocations, current holdings omitted from
`target_weights` are also preserved; a zero target must be explicit. If those
fixed positions conflict with the requested reserve or asset caps, the request
is rejected rather than silently proposing their sale. With no holdings and
positive preview cash, the service returns a cash-only no-trade preview. A
zero-NAV portfolio returns an unavailable preview.

To customize targets, pass `target_weights` keyed by exchange-qualified
current listings, and optionally pass `target_cash_weight`. Target weights
and cash are checked and adjusted by the allocation engine's existing cash
reserve and asset-cap constraints. The request can also set simulated cost,
slippage, turnover, and fractional-share assumptions. `additional_investment`
is included only in the preview cash and NAV; it is not deposited or recorded.

The response preserves custom user-supplied weights separately from the full
requested allocation (which includes default-preserved positions), then
distinguishes that allocation from constraint-adjusted targets and final
simulated weights and cash allocation. Allocation-adjustment events,
listing-specific constraint events, and portfolio-level constraint events are
reported separately. Monetary aggregate fees, slippage, and gross turnover
are sums of the displayed paise-rounded trade amounts; the turnover ratio is
calculated from the displayed turnover and initial NAV. Market-price dates and
sources, proposed simulated trades,
estimated fees/slippage, final simulated holdings and cash, aggregate gross
turnover and turnover ratio, and a per-holding suggestion are also returned.
`allocation_status` is `COMPLETE` only when the derived allocation is based on
a complete permitted valuation. Otherwise it is `UNAVAILABLE`, derived
requested/adjusted/final allocation fields are `null`, and
`allocation_unavailable_reason` explains that preserved weights cannot be
calculated. Explicit user-entered weights remain available in
`user_target_weights`, but are not a complete portfolio allocation.
Each simulated holding result reports an intended BUY/SELL/HOLD
or UNAVAILABLE action separately from execution status, along with target and
simulated values, relevant constraint messages, and estimated listing-level
fees/slippage. HOLD means current quantity matches the constrained target
within the larger of 1e-10 shares and half the configured fractional share
step (whole-share mode uses 1e-10). A constrained intended action remains BUY or
SELL even when no trade or only a partial trade is simulated. If any current holding lacks
a sufficiently fresh permitted EOD price, the endpoint returns
`preview_available: false`, categorized price warnings, and no proposed
trades; affected holdings are marked UNAVAILABLE rather than HOLD. A rejected
plan likewise has UNAVAILABLE per-holding simulation results. Custom and selected listings must already be holdings; the endpoint
does not yet accept new securities. A preview never changes account cash,
holdings, or transaction history and never submits brokerage orders. These
simulated suggestions are not financial advice and are not executable trade
instructions.
Monetary response values are rounded to INR paise (two decimal places).
Simulated values are estimates based on stored EOD closes and configured
execution assumptions; they are not executable quotes or exact settlement
calculations.

### Historical CSV backtesting API

The FastAPI application also exposes a separate historical research path
that loads `data/historical_market_csv/` through `load_market_data` and the
CSV-backed `PortfolioTradingEnv`. It does not use SQLite, fetch market data,
load the synthetic PPO model, or submit trades.

Supported strategies are `equal_weight` (rebalanced to equal ETF-and-cash
weights each session) and `buy_and_hold` (equal initial target followed by
no-trade transitions). Requests select the environment's fixed `train`,
`validation`, or `test` partition; validation and test reuse the scaler fitted
on the 2018–2021 training interval. Endpoints are:

- `POST /api/backtest/historical`
- `POST /api/backtest/compare`
- `GET /api/backtest/health`

Minimum variance is not exposed: a fixed-universe implementation using only
training-period covariance estimates and the environment's allocation
constraints has not been validated.

The response includes realized NAV, daily returns, environment-reported
turnover and transaction costs, and performance/risk metrics. Annualized
volatility and Sharpe/Sortino ratios use 252 sessions per year; CAGR uses
elapsed calendar days with 365.25 days per year. Historical corporate-action
coverage is unverified by default, so results must not be described as
corporate-action-complete or total-return accurate. The existing allocation
configuration defaults transaction fees and slippage to zero; response totals
reflect those settings rather than estimated costs.

Historical CSV results are paper simulations, not investment recommendations
or evidence of NSE/BSE or other real-market performance. Provider provenance
and redistribution permissions have not been established, and
corporate-action coverage remains unverified. These CSV results are separate
from the synthetic PPO experiment.

### Phase 8A synthetic research API

Phase 8A exposes read-only views of the completed, frozen Phase 7.2 synthetic
experiment through these authenticated GET endpoints:

- `/api/research/phase7-2/status`
- `/api/research/phase7-2/selection`
- `/api/research/phase7-2/validation`
- `/api/research/phase7-2/final-test`
- `/api/research/phase7-2/final-test/seeds`
- `/api/research/phase7-2/integrity`

Every request uses the shared account `get_current_user` authentication
dependency. Authentication requires the account database to be ready; if that
database is unavailable, these research routes also return HTTP 503.
Artifact verification or read failures produce a separate generic research
HTTP 503 response.

The API constructs a fresh research-store instance and verifies the stored
artifacts before reading a response. It expects the complete frozen Phase 7.2
artifact set under the fixed project path `.artifacts/phase7_2`; no
caller-supplied artifact path is accepted. The frozen artifact set must be
provisioned at that path. If it is absent or fails verification, the API does
not serve partial results.

The research API does not load PPO models, train models, run inference, or
execute backtests. It only returns verified stored synthetic experiment
summaries. These metrics must not be described as NSE/BSE or real-market
performance. The Phase 8A API is separate from the account, market-data, and
historical CSV backtest functionality. The dashboard remains teammate-owned
and is not modified or connected to this API.

---

## Contributing

```bash
git checkout -b feature/your-feature
pip install -r requirements.txt
# make changes, add tests
git commit -m "Add: your feature"
git push origin feature/your-feature
# open a Pull Request
```

Adding a custom strategy:

```python
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult

class MyStrategy(BaseStrategy):
    def generate_weights(self, data, **kwargs) -> StrategyResult:
        # your alpha logic — return portfolio weights
        pass
```

---

## Citation

```bibtex
@inproceedings{yang2026finrlx,
  title     = {FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading},
  author    = {Yang, Hongyang and Zhang, Boyu and She, Yang and Liao, Xinyu and Zhang, Xiaoli},
  booktitle = {Proceedings of the 2nd International Workshop on Decision Making and Optimization in Financial Technologies (DMO-FinTech)},
  year      = {2026},
  note      = {Workshop at PAKDD 2026}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Disclaimer

This software is for **educational and research purposes only**. Not financial advice. Always consult qualified professionals before making investment decisions. Past performance does not guarantee future results.

---

<div align="center">

**[AI4Finance Foundation](https://github.com/AI4Finance-Foundation)**

</div>
