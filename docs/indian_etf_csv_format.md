# Indian ETF Historical CSV Format

Phase 1 validates local Indian ETF CSV files only. It does not download data,
fill missing sessions or dates, approve licensing, or prepare PPO training.

## Required columns

Every input CSV must contain these exact column names:

```text
Date,Symbol,Open,High,Low,Close,Adjusted Close,Volume
```

- `Date`: session date, preferably in `YYYY-MM-DD` format.
- `Symbol`: configured ETF symbol.
- `Open`, `High`, `Low`, `Close`, `Adjusted Close`: positive numeric values.
- `Volume`: numeric, finite, and non-negative.

The input may be one combined CSV or a directory containing multiple CSV files.
ETF symbols must be supplied through `IndianETFIngestionConfig`; the module does
not infer or hardcode an Indian ETF universe.

## Validation

The loader:

- rejects missing required columns;
- reports missing values and invalid dates;
- rejects non-positive prices;
- rejects negative or missing volume;
- rejects positive or negative infinity and missing volume;
- rejects inconsistent OHLC rows;
- detects duplicate dates per symbol;
- detects each symbol's chronological ordering violations;
- reports configured-symbol absence and unexpected symbols;
- reports missing dates relative to the observed union of validated symbol sessions;
- reports common-session coverage.

No calendar is assumed and no missing price is invented or forward-filled.
The missing-date report is descriptive; it does not claim NSE holiday status.

## Outputs

For an output directory:

```text
raw/                                      Exact source copies
validated/indian_etf_validated.csv        Clean validated rows
indian_etf_ingestion_manifest.json        Provenance and validation report
```

Raw files are copied separately from validated output. If a destination filename
collides with different source content, a 12-character SHA-256 suffix is added.

## Provenance and training readiness

The manifest contains provider, retrieval timestamp, adjustment methodology,
license/permission status, source filenames and hashes, date range, row counts,
quality metrics, symbol coverage and validation issues. It also contains a
top-level `symbols` list and separate raw-observed and validated date ranges.

Retrieval timestamps must be timezone-aware ISO 8601 values, for example
`2024-01-04T00:00:00Z` or `2024-01-04T05:30:00+05:30`. The supplied value is
preserved in `retrieval_timestamp`; `retrieval_timestamp_normalized_utc`
contains the normalized UTC form. Missing, `unknown`, naive and invalid timestamps
block training readiness.

`training_ready` is `false` unless:

- provider provenance is known;
- adjustment methodology is known;
- license/permission status is explicitly `approved`;
- every configured symbol is present; and
- the file has no validation issues.

Unknown or unverified licensing status always prevents training-ready status.
