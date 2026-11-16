# Phase 5.5 — Alpha Vantage Initial SPY Endpoint Test

## Test details

- **Test date:** 2026-11-13
- **Symbol tested:** SPY only
- **Endpoint:** `https://www.alphavantage.co/query`
- **Function:** `TIME_SERIES_DAILY_ADJUSTED`
- **Output size requested:** `compact` (the request was not for full history)
- **Credential handling:** Loaded `ALPHA_VANTAGE_API_KEY` from the repository `.env` using `python-dotenv`. The key and credential-bearing request URL were not printed, stored in the report, or hardcoded.
- **HTTP status:** 200
- **API access result:** **PREMIUM RESTRICTION**

## Result

The request reached Alpha Vantage, but the API response stated that
`TIME_SERIES_DAILY_ADJUSTED` is a premium endpoint and that a premium plan is
needed. No daily time-series records were returned.

| Requested observation | Result |
|---|---|
| Returned ticker | Not returned |
| Available records | 0; endpoint returned a premium-access message instead of a time series |
| Latest available date | Not available |
| Earliest available date | Not available |
| OHLCV fields in this response | None; no records were returned |
| Adjusted-close field in this response | None; no records were returned |
| Adjusted open/high/low | Not available from this test |

The official Alpha Vantage documentation describes the Daily Adjusted response
as **raw daily OHLCV plus adjusted close and historical split/dividend events**.
That documented response shape does not establish adjusted open, high, and low.
The support FAQ says Alpha Vantage adjusts OHLCV for splits and cash-dividend
events in supported adjusted-data APIs, but this test did not access a response
with those fields. Adjusted OHLC must not be inferred from the presence of
adjusted close.

## Free-tier feasibility

The tested endpoint was not accessible with the configured free key. The HTTP
200 status is not a successful data result: the API returned an application
message identifying a premium restriction. No subscription was purchased and
no restriction was bypassed.

The project is targeting a ₹0 budget, so this endpoint is **not feasible under
the current free-key entitlement**. The requested latest-five-year SPY
coverage and field availability therefore remain unverified. Do not treat
this as a successful data download or use it to start historical training.

## Recommendation

**NO-GO** for using `TIME_SERIES_DAILY_ADJUSTED` with the current free
Alpha Vantage entitlement.

Further investigation may consider only a documented no-cost endpoint or
provider that supports the exact required fields and use rights. This result
does not establish access for any other ticker or endpoint. No request was
made for QQQ, IEF, GLD, XLU, XLV, `^GSPC`, or `^VIX`; stop here pending user
approval.

## Validation and scope

- Verification script: `tests/test_alpha_vantage.py`
- The script made one request for SPY with `outputsize=compact`, printed only a
  sanitized summary, and did not output the key or full API response.
- `.env` is excluded by the repository `.gitignore`.
- No full historical dataset was downloaded.
- No production fetching, portfolio environment, training, broker, or trading
  code was changed.
- No subscription was purchased.
