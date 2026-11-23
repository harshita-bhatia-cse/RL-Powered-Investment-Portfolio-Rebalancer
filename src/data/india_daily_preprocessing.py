"""India-specific daily preprocessing for validated Phase 1 CSV output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as _pd


REQUIRED_COLUMNS = (
    "Date",
    "Symbol",
    "Open",
    "High",
    "Low",
    "Close",
    "Adjusted Close",
    "Volume",
)


@dataclass(frozen=True)
class IndiaChronologicalSplits:
    train: tuple[_pd.Timestamp, _pd.Timestamp]
    validation: tuple[_pd.Timestamp, _pd.Timestamp]
    test: tuple[_pd.Timestamp, _pd.Timestamp]


@dataclass(frozen=True)
class IndiaFeatureScaler:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        features: _pd.DataFrame,
    ) -> "IndiaFeatureScaler":
        if features.empty:
            raise ValueError("training features are empty")
        values = features.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("training features contain non-finite values")
        means = values.mean(axis=0)
        scales = values.std(axis=0, ddof=0)
        scales[scales == 0.0] = 1.0
        return cls(
            feature_names=tuple(features.columns),
            means=tuple(float(value) for value in means),
            scales=tuple(float(value) for value in scales),
        )

    def transform(self, features: _pd.DataFrame) -> np.ndarray:
        if tuple(features.columns) != self.feature_names:
            raise ValueError("feature columns do not match the fitted scaler")
        values = features.to_numpy(dtype=np.float64)
        means = np.asarray(self.means, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        transformed = (values - means) / scales
        if not np.isfinite(transformed).all():
            raise ValueError("scaled features contain non-finite values")
        return transformed.astype(np.float32)


@dataclass(frozen=True)
class IndiaDailyDataset:
    symbols: tuple[str, ...]
    dates: tuple[_pd.Timestamp, ...]
    open_prices: _pd.DataFrame
    close_prices: _pd.DataFrame
    adjusted_close_prices: _pd.DataFrame
    volume: _pd.DataFrame
    returns: _pd.DataFrame
    features: _pd.DataFrame
    splits: IndiaChronologicalSplits
    manifest: dict[str, object]


def load_indian_daily_dataset(
    validated_csv: str | Path,
    *,
    manifest_json: str | Path,
    symbols: Sequence[str],
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> IndiaDailyDataset:
    """Load validated Phase 1 output and make chronological train/validation/test splits."""
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("split fractions must be positive and less than 1")
    test_fraction = 1.0 - train_fraction - validation_fraction
    if test_fraction <= 0.0:
        raise ValueError("train and validation fractions must leave a positive test fraction")

    normalized_symbols = tuple(str(symbol).strip().upper() for symbol in symbols)
    if not normalized_symbols or len(set(normalized_symbols)) != len(normalized_symbols):
        raise ValueError("symbols must be a non-empty unique universe")

    manifest_path = Path(manifest_json)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read ingestion manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("ingestion manifest must be a JSON object")
    if manifest.get("training_ready") is not True:
        raise ValueError(
            "Phase 1 data is not training-ready; provenance, permission or validation failed"
        )
    manifest_symbols = tuple(str(symbol).upper() for symbol in manifest.get("symbols", ()))
    if set(manifest_symbols) != set(normalized_symbols):
        raise ValueError("configured symbols do not match the validated ingestion manifest")

    adjustment = str(manifest.get("adjustment_methodology", "")).strip().lower()
    if adjustment in ("", "unknown", "unverified", "raw"):
        raise ValueError(
            "adjusted-price methodology is ambiguous or unsuitable for preprocessing"
        )

    csv_path = Path(validated_csv)
    try:
        frame = _pd.read_csv(csv_path)
    except (OSError, _pd.errors.ParserError) as error:
        raise ValueError(f"unable to read validated CSV: {csv_path}") from error
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("missing required Phase 1 columns: " + ", ".join(missing))

    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["Symbol"] = frame["Symbol"].astype("string").str.strip().str.upper()
    frame["Date"] = _pd.to_datetime(frame["Date"], errors="coerce")
    if frame["Date"].isna().any():
        raise ValueError("validated CSV contains invalid dates")
    if frame.duplicated(subset=["Symbol", "Date"], keep=False).any():
        raise ValueError("validated CSV contains duplicate symbol-date records")
    unexpected = sorted(set(frame["Symbol"].dropna()) - set(normalized_symbols))
    if unexpected:
        raise ValueError("CSV contains symbols outside the configured universe")

    pivot_columns = ("Open", "High", "Low", "Close", "Adjusted Close", "Volume")
    pivots: dict[str, _pd.DataFrame] = {}
    for column in pivot_columns:
        numeric = _pd.to_numeric(frame[column], errors="coerce")
        frame[column] = numeric
        if frame[column].isna().any():
            raise ValueError(f"validated CSV contains missing numeric values in {column}")
        if column != "Volume":
            if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
                raise ValueError(f"validated CSV contains non-finite values in {column}")
            if (frame[column] <= 0).any():
                raise ValueError(f"validated CSV contains non-positive values in {column}")
        else:
            if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
                raise ValueError("validated CSV contains non-finite volume")
            if (frame[column] < 0).any():
                raise ValueError("validated CSV contains negative volume")
        pivots[column] = frame.pivot(index="Date", columns="Symbol", values=column)

    inconsistent = (
        (pivots["High"] < pivots["Open"])
        | (pivots["High"] < pivots["Low"])
        | (pivots["High"] < pivots["Close"])
        | (pivots["Low"] > pivots["Open"])
        | (pivots["Low"] > pivots["High"])
        | (pivots["Low"] > pivots["Close"])
    )
    if inconsistent.any(axis=None):
        raise ValueError("validated CSV contains inconsistent OHLC values")

    common_dates = pivots["Adjusted Close"].loc[:, list(normalized_symbols)].dropna().index
    for column in pivot_columns:
        frame_pivot = pivots[column].loc[:, list(normalized_symbols)]
        pivots[column] = frame_pivot.reindex(common_dates).dropna(how="any")
        if pivots[column].empty:
            raise ValueError("no common trading sessions remain after alignment")

    common_dates = sorted(
        set(pivots["Adjusted Close"].index).intersection(
            *(set(pivots[column].index) for column in pivot_columns)
        )
    )
    common_dates = _pd.DatetimeIndex(common_dates)
    if common_dates.empty:
        raise ValueError("no common trading sessions exist across configured symbols")

    open_prices = pivots["Open"].reindex(common_dates)
    close_prices = pivots["Close"].reindex(common_dates)
    adjusted_close = pivots["Adjusted Close"].reindex(common_dates)
    volumes = pivots["Volume"].reindex(common_dates)
    if any(frame.isna().any().any() for frame in (open_prices, close_prices, adjusted_close, volumes)):
        raise ValueError("common-session alignment contains missing values")

    returns = adjusted_close.pct_change(fill_method=None).iloc[1:].replace(
        [np.inf, -np.inf], np.nan
    )
    returns = returns.dropna(how="any")
    if returns.empty:
        raise ValueError("insufficient adjusted-close return observations")

    aligned_dates = tuple(_pd.DatetimeIndex(returns.index))
    open_prices = open_prices.reindex(aligned_dates).dropna(how="any")
    close_prices = close_prices.reindex(aligned_dates).dropna(how="any")
    adjusted_close = adjusted_close.reindex(aligned_dates).dropna(how="any")
    volumes = volumes.reindex(aligned_dates).dropna(how="any")
    if not (
        len(open_prices)
        == len(close_prices)
        == len(adjusted_close)
        == len(volumes)
        == len(returns)
    ):
        raise ValueError("aligned Indian market arrays have inconsistent lengths")

    feature_names = tuple(
        f"{symbol}_{feature}"
        for symbol in normalized_symbols
        for feature in ("return_1", "volatility_21")
    )
    feature_rows: list[list[float]] = []
    feature_dates: list[_pd.Timestamp] = []
    for position in range(21, len(returns)):
        window = returns.iloc[position - 21:position]
        row: list[float] = []
        for symbol in normalized_symbols:
            series = window[symbol]
            row.extend((float(series.iloc[-1]), float(series.std(ddof=0))))
        feature_rows.append(row)
        feature_dates.append(_pd.Timestamp(returns.index[position]))
    if not feature_rows:
        raise ValueError("insufficient observations for the 21-session feature lookback")
    features = _pd.DataFrame(feature_rows, index=_pd.DatetimeIndex(feature_dates), columns=feature_names)

    count = len(features)
    train_end = int(count * train_fraction)
    validation_end = train_end + int(count * validation_fraction)
    train_end = max(1, train_end)
    validation_end = max(train_end + 1, validation_end)
    if validation_end >= count:
        validation_end = count - 1
    if train_end >= validation_end or validation_end >= count:
        raise ValueError("insufficient observations for chronological train/validation/test splits")

    train_dates = (features.index[0], features.index[train_end - 1])
    validation_dates = (features.index[train_end], features.index[validation_end - 1])
    test_dates = (features.index[validation_end], features.index[-1])
    splits = IndiaChronologicalSplits(
        train=train_dates,
        validation=validation_dates,
        test=test_dates,
    )

    final_dates = tuple(_pd.Timestamp(value) for value in features.index)
    return IndiaDailyDataset(
        symbols=normalized_symbols,
        dates=final_dates,
        open_prices=open_prices.reindex(final_dates).dropna(how="any"),
        close_prices=close_prices.reindex(final_dates).dropna(how="any"),
        adjusted_close_prices=adjusted_close.reindex(final_dates).dropna(how="any"),
        volume=volumes.reindex(final_dates).dropna(how="any"),
        returns=returns.reindex(final_dates).dropna(how="any"),
        features=features,
        splits=splits,
        manifest=manifest,
    )
