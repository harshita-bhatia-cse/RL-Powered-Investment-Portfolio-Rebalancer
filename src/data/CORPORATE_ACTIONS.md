# Historical corporate-action acquisition

The acquisition module is separate from the FMP price fetcher and the
portfolio ledger. It uses Massive's documented endpoints:

- Dividends: `GET /stocks/v1/dividends`
- Splits: `GET /stocks/v1/splits`

The client uses per-symbol inclusive ex-date/execution-date filters, consumes
all `next_url` pages, retries rate limits and transient server/network errors
with a bounded policy, and distinguishes successful empty results from denied,
unsupported, malformed, and failed requests. It does not infer a no-event result
from an HTTP/API failure. All symbols and both endpoints must complete before
the CLI writes a dataset.

## Fields and normalization

`corporate_actions.csv` uses these columns in order:

```text
symbol,event_type,source_id,currency,announcement_date,ex_date,payment_date,
effective_date,distribution_per_share,split_ratio
```

The physical CSV header has no line break. Dates are ISO `YYYY-MM-DD`; amount
and ratio fields are decimal numbers. Massive dividend `cash_amount` is used
for `distribution_per_share`, not `split_adjusted_cash_amount`. Dividend
records are normalized as `etf_distribution` for this fixed ETF universe.
Dividend currency must be explicitly `USD`; ex-date and payment date are
required. Split `split_to / split_from` becomes the ledger's new-shares per
old-share ratio. Massive's split records do not supply a currency; USD is set
as the ledger/account currency for these U.S.-listed symbols, not represented
as a source-reported split currency.

`corporate_actions_raw.jsonl` stores each original provider record, wrapped
with its ticker, endpoint category, and endpoint path. API keys are never
included in these files or status messages. The manifest contains per-symbol
dividend/split counts, action-date ranges, endpoint outcomes, page counts,
duplicate and validation summaries, retrieval time, and SHA-256 checksums for
the event CSV and raw-record file.

## Validation and coverage

Export requires all endpoint queries to finish successfully, including all
pages. Action dates must be within the requested inclusive interval and must
align with a supplied common market session. Event IDs must be stable; exact
duplicates are counted and deduplicated, while conflicting IDs, invalid dates,
unsupported symbols/currencies, missing payment dates, and non-positive or
non-finite amounts/ratios fail validation.

A successful empty response is retained as `empty` in query outcomes. It is
not proof that no event occurred: verify symbol support and provider
completeness independently. A completed query and a checksummed manifest do
not certify that the provider's historical event coverage is correct or
complete. The manifest intentionally reports
`queries_complete; provider event coverage not independently verified`.

The Massive account must permit the endpoints and requested historical range.
ETF coverage for SPY, QQQ, IEF, GLD, XLU, and XLV is not presumed by this
implementation; any denied or unsupported result blocks export. Review provider
access and licensing before acquisition and redistribution. Generated artifacts
default to `data/corporate_actions/`, covered by the repository's existing
`/data/*` ignore rule; do not commit them without explicit redistribution
permission.

## Run acquisition (opt-in)

Use an isolated project environment and set `MASSIVE_API_KEY` in the process
environment without putting it in a command-line argument, source file, or
manifest. The command refuses to make requests unless licensing is explicitly
acknowledged:

```powershell
$env:MASSIVE_API_KEY = "<set securely in your local environment>"
python scripts\acquire_corporate_actions.py --confirm-licensed-use
```

The script uses existing `data/fmp_daily/<SYMBOL>_daily.csv` files for all six
assets plus `^GSPC` and `^VIX` to check action-date alignment against the
environment's common sessions. It does not retrieve prices, contact a broker,
or make trades. It refuses to write partial results.

## Loading and ledger use

Use `load_normalized_events` to read and revalidate the CSV, then pass the
returned events to `PortfolioTradingEnv` as `corporate_action_events`. Supply
the same validated market-session dates and requested bounds. The environment
processes events chronologically. For the historical acquisition workflow,
initialize with all cash and no inherited holdings or receivables.

Do not set `corporate_actions_verified=True` merely because acquisition
completed. That flag is caller-supplied metadata. First independently verify
coverage for each symbol and date range and reconcile representative events
and adjusted-price behavior. The ledger does not model FX, withholding/taxes,
cash-in-lieu, or distribution tax classifications.

## Yahoo Finance candidate acquisition (Phase 5.1)

The separate `YahooFinanceCorporateActionProvider` uses yfinance's
`Ticker.history(..., actions=True, auto_adjust=False)` for the fixed ETF
universe. It collects Yahoo's `Dividends`, `Capital Gains`, and `Stock Splits`
action columns alongside raw close and adjusted close. It does not silently
convert failed or empty price-history responses into zero-event results. A
successful history response containing no action rows means only that Yahoo
returned no action rows for that query; it does not prove event completeness.

Yahoo's current [Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)
(Section 2.4.9) restrict automated collection from its services without
express, prior permission. No such permission has been verified for this
project, so no live acquisition was performed. Do not run the CLI unless you
have express prior permission for automated access and local retention; it
requires an explicit confirmation flag and keeps outputs under the ignored
`data/` directory:

```powershell
python scripts\acquire_yahoo_corporate_actions.py --confirm-express-prior-permission
```

The script uses local `data/fmp_daily/<SYMBOL>_daily.csv` files to calculate
common trading sessions and compare Yahoo raw/adjusted closes against local
`close` and `adj_close`. It does not download prices separately or access a
broker. The output includes:

- `yahoo_action_raw.jsonl`: action-bearing Yahoo rows and source fields.
- `yahoo_action_candidates.csv`: normalized candidates, with unavailable
  source IDs and payment dates left blank.
- `issuer_verification_template.csv`: a review template containing the Yahoo
  event key and reported amount/ratio.
- `price_comparison.json`: per-symbol date/price comparison and potential
  adjusted/raw factor discontinuities. These are investigation leads only.
- `manifest.json`: retrieval timestamp, range, counts/date coverage, failed
  symbols, validation summary, endpoint/library details, and SHA-256 checksums.

yfinance does not expose a stable event identifier or a distribution payment
date in the sampled history interface. Consequently, Yahoo candidates are not
passed directly to `CorporateActionLedger`. Fill the verification template
only from a traceable issuer document or another authoritative source, with a
genuine unique source ID, currency, distribution amount, payment date (for
distributions), or split ratio (for splits), plus verification source and
reference. A distribution amount must match the Yahoo per-share amount; no
split-adjusted reinterpretation is made. Then optionally export the verified
ledger schema:

```powershell
python scripts\acquire_yahoo_corporate_actions.py `
  --confirm-express-prior-permission `
  --issuer-verification-csv data\yahoo_corporate_actions\issuer_verification_template.csv
```

The CLI validates every verification row and refuses incomplete or
inconsistent issuer evidence. It does not establish that Yahoo's source
coverage is complete, and issuer verification of returned rows does not prove
that no actions are missing. Review issuer records and the price comparison
before deciding whether the history is appropriate for simulation. Do not
redistribute Yahoo data without confirming applicable rights.

## Official issuer verification (Phase 5.2)

`scripts/build_issuer_verified_corporate_actions.py` consumes the unchanged
Yahoo candidate CSV, a locally obtained State Street historical-distributions
workbook, and existing `data/fmp_daily` date columns. It performs no network
requests. It checks issuer ex-dates against local per-symbol market sessions and
writes only under the ignored `data/issuer_verified_corporate_actions/` tree.

The output separates:

- `verified_normalized_events.csv`: exact `EVENT_COLUMNS` schema accepted by
  `load_normalized_events`.
- `verified_events_evidence.csv`: adds record date, source worksheet/row,
  issuer reference, and candidate-comparison status.
- `unverified_candidates.csv`: every Yahoo row not fully corroborated remains
  present with a reason.
- `issuer_source_records.jsonl`, the original issuer workbook, `issuer_research.json`,
  and `manifest.json`: parsed references, preserved source, access limitations,
  and SHA-256 checksums.

State Street's workbook provides SPY, XLU, and XLV distribution dates and
payable dates, but no announcement dates, event IDs, or complete split history.
Where it has no event ID, the normalized CSV uses a deterministic
`local-ssga-sha256-...` identifier derived from event fields and the issuer
worksheet/row reference. It is explicitly a local ID, not an issuer ID.
Amounts in the official workbook are retained at source precision. Yahoo's
three-decimal values are compared at that precision; an amount-basis mismatch
does not convert a Yahoo candidate into a verified event.

The accessible official QQQ history and the attempted iShares IEF export did
not produce parseable event-level data. A GLD prospectus says shareholders do
not receive dividends, but this alone is not a complete period-wide action
history. The XLU 2025-12-05 2:1 candidate remains unverified: the local raw and
adjusted closes do not show a half-price step, while XLU's Yahoo distribution
amount basis changes by approximately 2x around the reported date. Neither
price continuity nor a distribution-basis change is an official split notice.

Example, after obtaining the State Street workbook from the official source
and confirming local retention rights:

```powershell
python scripts\build_issuer_verified_corporate_actions.py `
  --issuer-workbook path\to\spdr-etf-historical-distributions.xlsx
```

The current partial issuer dataset must not be used to claim complete
total-return accuracy or to train historical PPO. Confirm rights before
redistribution; generated data remains ignored by Git.
