# Phase 4: India Daily Market Data Integration

## Provider interface

`src/data/india_market_provider.py` provides a provider interface for India
ETF daily OHLCV data. The first implementation uses the repository-declared
`yfinance` dependency and Yahoo Chart History.

NSE symbols use the `.NS` suffix and BSE symbols use `.BO`. Symbols may be
configured with or without the suffix.

The provider:

- accepts a start/end date and configured India ETF symbols;
- requests completed daily history;
- excludes the current India date to avoid an incomplete current session;
- requires Open, High, Low, Close, Adj Close and Volume;
- rejects missing, duplicate, non-finite, negative-volume and inconsistent OHLCV;
- preserves gaps rather than filling or fabricating sessions;
- retries provider failures according to configuration;
- returns normalized Phase 1 `Date/Symbol/Open/High/Low/Close/Adjusted Close/Volume` rows.

## Download location

`download_india_etf_data()` writes to `data/india_daily/` by default:

```text
data/india_daily/
  raw/
    <SYMBOL>_daily.csv
    india_etf_provider_combined.csv
    india_etf_actions.csv
  validated/
    indian_etf_validated.csv
  india_market_data_manifest.json
  indian_etf_ingestion_manifest.json
```

Raw provider files are retained separately from validated CSV output.

## Provenance and licensing

The generated manifest records provider, source, exchange, symbols, UTC
retrieval timestamp, requested historical range, adjusted-price method,
license/permission status and corporate-action verification status.

Yahoo/yfinance public availability does not establish rights for ML training.
The default status is `unverified`. Set
`INDIA_DATA_LICENSE_STATUS=approved` only after written/provider permission is
obtained. Explicit approval is never inferred from successful download.

Without approval:

- data download and CSV validation may complete;
- `training_ready` remains false;
- PPO preparation rejects the dataset.

## Tests

Provider tests use mock `Ticker.history()` responses and do not access the live
Yahoo API or real market data.
