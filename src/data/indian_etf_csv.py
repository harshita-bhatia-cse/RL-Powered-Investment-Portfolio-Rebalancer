"""Historical Indian ETF CSV ingestion and validation."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.data_quality import calculate_price_quality


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
class IndianETFIngestionConfig:
    """Provenance and universe configuration supplied by the caller."""

    symbols: Sequence[str]
    provider: str = "unknown"
    retrieval_timestamp: str = "unknown"
    adjustment_methodology: str = "unknown"
    license_permission_status: str = "unverified"

    def normalized_symbols(self) -> tuple[str, ...]:
        symbols = tuple(
            str(symbol).strip().upper()
            for symbol in self.symbols
            if str(symbol).strip()
        )
        if not symbols:
            raise ValueError("symbols must contain at least one non-empty ETF symbol")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique")
        return symbols


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_retrieval_timestamp(value: str) -> tuple[bool, str]:
    """Return whether a timestamp is timezone-aware ISO 8601 and its UTC form."""
    text = str(value).strip()
    if not text or text.lower() in ("unknown", "unverified", "none", "null"):
        return False, ""
    try:
        iso_text = text[:-1] + "+00:00" if text.upper().endswith("Z") else text
        parsed = datetime.fromisoformat(iso_text)
    except (TypeError, ValueError):
        return False, ""
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return False, ""
    normalized = parsed.astimezone(timezone.utc)
    return True, normalized.isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError("input file must have a .csv extension")
        return [path]
    if path.is_dir():
        files = sorted(candidate for candidate in path.rglob("*.csv") if candidate.is_file())
        if not files:
            raise FileNotFoundError("no CSV files found in input directory")
        return files
    raise FileNotFoundError(f"input path does not exist: {path}")


def _read_csvs(files: Iterable[Path]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    file_records: list[dict[str, object]] = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as error:
            raise ValueError(f"unable to read CSV: {path}") from error
        frames.append(frame)
        file_records.append({
            "source_file": str(path),
            "filename": path.name,
            "sha256": _sha256(path),
            "row_count": int(len(frame)),
        })
    if not frames:
        raise ValueError("at least one CSV is required")
    return pd.concat(frames, ignore_index=True, sort=False), file_records


def validate_indian_etf_csv(
    input_path: str | Path,
    config: IndianETFIngestionConfig,
) -> dict[str, object]:
    """Validate CSV data and return a report without downloading or filling data."""
    files = _csv_files(input_path)
    combined, source_files = _read_csvs(files)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in combined.columns
    ]
    if missing_columns:
        raise ValueError(
            "missing required CSV columns: " + ", ".join(missing_columns)
        )

    frame = combined.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["Symbol"] = frame["Symbol"].astype("string").str.strip().str.upper()

    issues: list[str] = []
    null_counts = {
        column: int(frame[column].isna().sum()) + int(
            frame[column].astype("string").str.strip().eq("").fillna(False).sum()
            if column in ("Date", "Symbol")
            else 0
        )
        for column in REQUIRED_COLUMNS
    }
    if any(value > 0 for value in null_counts.values()):
        issues.append("missing values detected")

    parsed_dates = pd.to_datetime(frame["Date"], errors="coerce")
    invalid_dates = parsed_dates.isna()
    if invalid_dates.any():
        issues.append("invalid or missing dates detected")

    numeric_columns = ("Open", "High", "Low", "Close", "Adjusted Close", "Volume")
    numeric = {
        column: pd.to_numeric(frame[column], errors="coerce")
        for column in numeric_columns
    }

    invalid_prices = pd.Series(False, index=frame.index)
    for column in ("Open", "High", "Low", "Close", "Adjusted Close"):
        values = numeric[column]
        invalid_prices |= (
            values.isna()
            | ~np.isfinite(values.to_numpy(dtype=np.float64))
            | (values <= 0)
        )
    if invalid_prices.any():
        issues.append("invalid or non-positive price values detected")

    invalid_volume = numeric["Volume"].isna() | (numeric["Volume"] < 0)
    if invalid_volume.any():
        issues.append("negative or missing volume values detected")
    non_finite_volume = ~np.isfinite(numeric["Volume"].to_numpy(dtype=np.float64))
    if non_finite_volume.any():
        issues.append("non-finite volume values detected")

    inconsistent_ohlc = pd.Series(False, index=frame.index)
    inconsistent_ohlc |= (
        (numeric["High"] < numeric["Open"])
        | (numeric["High"] < numeric["Low"])
        | (numeric["High"] < numeric["Close"])
        | (numeric["Low"] > numeric["Open"])
        | (numeric["Low"] > numeric["High"])
        | (numeric["Low"] > numeric["Close"])
    )
    if inconsistent_ohlc.any():
        issues.append("inconsistent OHLC values detected")

    duplicate_dates = frame.assign(
        _ParsedDate=parsed_dates
    ).duplicated(subset=["Symbol", "_ParsedDate"], keep=False)
    if duplicate_dates.any():
        issues.append("duplicate dates detected per symbol")

    chronological_violations: list[str] = []
    for symbol in config.normalized_symbols():
        symbol_rows = frame.loc[frame["Symbol"] == symbol]
        symbol_dates = pd.to_datetime(symbol_rows["Date"], errors="coerce").dropna()
        if symbol_dates.size > 1:
            previous = None
            for date in symbol_dates.tolist():
                if previous is not None and date < previous:
                    chronological_violations.append(symbol)
                    break
                previous = date
    if chronological_violations:
        issues.append("chronological ordering violations detected")

    symbols = config.normalized_symbols()
    observed_symbols = {
        symbol for symbol in frame["Symbol"].dropna().tolist() if str(symbol).strip()
    }
    missing_symbols = sorted(set(symbols) - observed_symbols)
    unexpected_symbols = sorted(observed_symbols - set(symbols))
    if missing_symbols:
        issues.append("configured symbols missing from input")
    if unexpected_symbols:
        issues.append("unexpected symbols detected")

    clean_mask = ~(
        invalid_dates
        | invalid_prices
        | invalid_volume
        | ~np.isfinite(numeric["Volume"].to_numpy(dtype=np.float64))
        | inconsistent_ohlc
        | duplicate_dates
    )
    clean_mask &= frame["Symbol"].isin(symbols)

    clean = frame.loc[clean_mask].copy()
    clean["Date"] = parsed_dates.loc[clean_mask].dt.strftime("%Y-%m-%d")
    for column in numeric_columns:
        clean[column] = numeric[column].loc[clean_mask].astype(np.float64)

    per_symbol: dict[str, dict[str, object]] = {}
    date_sets: dict[str, set[str]] = {}
    for symbol in symbols:
        symbol_dates = set(
            clean.loc[clean["Symbol"] == symbol, "Date"].astype(str).tolist()
        )
        date_sets[symbol] = symbol_dates
        per_symbol[symbol] = {
            "row_count": int(len(symbol_dates)),
            "start_date": min(symbol_dates) if symbol_dates else None,
            "end_date": max(symbol_dates) if symbol_dates else None,
        }

    all_observed_dates = set()
    for dates in date_sets.values():
        all_observed_dates |= dates
    common_dates = set.intersection(*date_sets.values()) if date_sets else set()
    missing_dates_by_symbol = {
        symbol: sorted(all_observed_dates - dates)
        for symbol, dates in date_sets.items()
    }

    quality = None
    if not clean.empty:
        quality_frame = clean.rename(columns={
            "Symbol": "ticker",
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adjusted Close": "adj_close",
            "Volume": "volume",
        })
        quality = calculate_price_quality(quality_frame)

    raw_parsed_dates = pd.to_datetime(combined["Date"], errors="coerce")
    raw_observed_dates = raw_parsed_dates.dropna()
    raw_observed_date_range = {
        "start_date": (
            raw_observed_dates.min().strftime("%Y-%m-%d")
            if not raw_observed_dates.empty
            else None
        ),
        "end_date": (
            raw_observed_dates.max().strftime("%Y-%m-%d")
            if not raw_observed_dates.empty
            else None
        ),
    }

    provider = str(config.provider).strip()
    permission = str(config.license_permission_status).strip().lower()
    adjustment = str(config.adjustment_methodology).strip()
    retrieval_timestamp_valid, normalized_retrieval_timestamp = (
        _validate_retrieval_timestamp(config.retrieval_timestamp)
    )

    provider_verified = provider.lower() not in ("", "unknown", "unverified")
    adjustment_verified = adjustment.lower() not in ("", "unknown", "unverified")
    permission_approved = permission == "approved"
    if not retrieval_timestamp_valid:
        issues.append("invalid, missing or unknown retrieval timestamp")
    valid = not issues

    training_ready = bool(
        valid
        and provider_verified
        and adjustment_verified
        and permission_approved
        and retrieval_timestamp_valid
        and not missing_symbols
    )

    return {
        "valid": bool(valid),
        "training_ready": training_ready,
        "symbols": list(symbols),
        "source_files": source_files,
        "issues": sorted(set(issues)),
        "missing_columns": missing_columns,
        "missing_symbols": missing_symbols,
        "unexpected_symbols": unexpected_symbols,
        "duplicate_rows": int(duplicate_dates.sum()),
        "chronological_violations": sorted(set(chronological_violations)),
        "null_counts": null_counts,
        "per_symbol": per_symbol,
        "common_sessions": {
            "count": len(common_dates),
            "start_date": min(common_dates) if common_dates else None,
            "end_date": max(common_dates) if common_dates else None,
        },
        "missing_dates_by_symbol": missing_dates_by_symbol,
        "raw_observed_date_range": raw_observed_date_range,
        "quality": quality,
        "provider": provider,
        "provider_verified": provider_verified,
        "retrieval_timestamp": config.retrieval_timestamp,
        "retrieval_timestamp_valid": retrieval_timestamp_valid,
        "retrieval_timestamp_normalized_utc": normalized_retrieval_timestamp,
        "adjustment_methodology": adjustment,
        "license_permission_status": permission,
        "provenance_status": (
            "approved"
            if provider_verified
            and permission_approved
            and adjustment_verified
            and retrieval_timestamp_valid
            else "unverified"
        ),
        "validated_row_count": int(len(clean)),
        "validated_rows_by_symbol": {
            symbol: int(len(clean.loc[clean["Symbol"] == symbol]))
            for symbol in symbols
        },
    }


def ingest_indian_etf_csv(
    input_path: str | Path,
    config: IndianETFIngestionConfig,
    *,
    output_dir: str | Path,
) -> dict[str, object]:
    """Preserve raw files and write validated CSV and a provenance manifest."""
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    validated_dir = output_dir / "validated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    validated_dir.mkdir(parents=True, exist_ok=True)

    preserved_files: list[dict[str, object]] = []
    for source in _csv_files(input_path):
        source_hash = _sha256(source)
        destination = raw_dir / source.name
        if destination.exists() and _sha256(destination) != source_hash:
            destination = raw_dir / f"{source.stem}-{source_hash[:12]}{source.suffix}"
        shutil.copy2(source, destination)
        preserved_files.append({
            "source_file": str(source),
            "preserved_file": str(destination),
            "sha256": _sha256(destination),
        })

    combined, source_files = _read_csvs(_csv_files(input_path))
    try:
        report = validate_indian_etf_csv(input_path, config)
    except ValueError as error:
        symbols = config.normalized_symbols()
        raw_dates = pd.to_datetime(
            combined["Date"] if "Date" in combined.columns else pd.Series(dtype="object"),
            errors="coerce",
        ).dropna()
        report = {
            "valid": False,
            "training_ready": False,
            "symbols": list(symbols),
            "source_files": source_files,
            "issues": [str(error)],
            "missing_columns": [
                column
                for column in REQUIRED_COLUMNS
                if column not in combined.columns
            ],
            "missing_symbols": sorted(
                set(symbols)
                - set(
                    combined["Symbol"].astype("string").str.strip().str.upper().dropna()
                    if "Symbol" in combined.columns
                    else []
                )
            ),
            "unexpected_symbols": (
                sorted(
                    set(
                        combined["Symbol"].astype("string").str.strip().str.upper().dropna()
                    )
                    - set(symbols)
                )
                if "Symbol" in combined.columns
                else []
            ),
            "duplicate_rows": 0,
            "chronological_violations": [],
            "null_counts": {},
            "per_symbol": {},
            "common_sessions": {
                "count": 0,
                "start_date": None,
                "end_date": None,
            },
            "missing_dates_by_symbol": {},
            "raw_observed_date_range": {
                "start_date": (
                    raw_dates.min().strftime("%Y-%m-%d")
                    if not raw_dates.empty
                    else None
                ),
                "end_date": (
                    raw_dates.max().strftime("%Y-%m-%d")
                    if not raw_dates.empty
                    else None
                ),
            },
            "quality": None,
            "provider": str(config.provider).strip(),
            "provider_verified": str(config.provider).strip().lower()
            not in ("", "unknown", "unverified"),
            "retrieval_timestamp": config.retrieval_timestamp,
            "adjustment_methodology": str(config.adjustment_methodology).strip(),
            "license_permission_status": str(
                config.license_permission_status
            ).strip().lower(),
            "provenance_status": "unverified",
            "validated_row_count": 0,
            "validated_rows_by_symbol": {},
        }
        timestamp_valid, timestamp_normalized = _validate_retrieval_timestamp(
            config.retrieval_timestamp
        )
        report["retrieval_timestamp_valid"] = timestamp_valid
        report["retrieval_timestamp_normalized_utc"] = timestamp_normalized

    combined = combined.loc[
        :, [column for column in REQUIRED_COLUMNS if column in combined.columns]
    ].copy()
    if list(combined.columns) != list(REQUIRED_COLUMNS):
        empty_validated = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
        validated_path = validated_dir / "indian_etf_validated.csv"
        empty_validated.to_csv(validated_path, index=False, encoding="utf-8")
        report["preserved_raw_files"] = preserved_files
        report["validated_output"] = str(validated_path)
        report["manifest_generated_at"] = _utc_timestamp()
        report["raw_observed_date_range"] = report["raw_observed_date_range"]
        report["date_range"] = {"start_date": None, "end_date": None}
        manifest_path = output_dir / "indian_etf_ingestion_manifest.json"
        manifest_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report["manifest"] = str(manifest_path)
        return report

    combined = combined.loc[:, list(REQUIRED_COLUMNS)].copy()
    combined["Symbol"] = combined["Symbol"].astype("string").str.strip().str.upper()
    parsed_dates = pd.to_datetime(combined["Date"], errors="coerce")

    numeric = {
        column: pd.to_numeric(combined[column], errors="coerce")
        for column in ("Open", "High", "Low", "Close", "Adjusted Close", "Volume")
    }
    clean_mask = parsed_dates.notna()
    for column in ("Open", "High", "Low", "Close", "Adjusted Close"):
        clean_mask &= (
            numeric[column].notna()
            & np.isfinite(numeric[column].to_numpy(dtype=np.float64))
            & (numeric[column] > 0)
        )
    clean_mask &= (
        numeric["Volume"].notna()
        & np.isfinite(numeric["Volume"].to_numpy(dtype=np.float64))
        & (numeric["Volume"] >= 0)
    )
    clean_mask &= ~combined.assign(_ParsedDate=parsed_dates).duplicated(
        subset=["Symbol", "_ParsedDate"], keep=False
    )

    inconsistent_ohlc = pd.Series(False, index=combined.index)
    inconsistent_ohlc |= (
        (numeric["High"] < numeric["Open"])
        | (numeric["High"] < numeric["Low"])
        | (numeric["High"] < numeric["Close"])
        | (numeric["Low"] > numeric["Open"])
        | (numeric["Low"] > numeric["High"])
        | (numeric["Low"] > numeric["Close"])
    )
    clean_mask &= ~inconsistent_ohlc
    clean_mask &= combined["Symbol"].isin(config.normalized_symbols())

    validated = combined.loc[clean_mask].copy()
    validated["Date"] = parsed_dates.loc[clean_mask].dt.strftime("%Y-%m-%d")
    validated = validated.sort_values(["Symbol", "Date"], kind="stable").reset_index(
        drop=True
    )

    validated_path = validated_dir / "indian_etf_validated.csv"
    validated.to_csv(validated_path, index=False, encoding="utf-8")

    report["preserved_raw_files"] = preserved_files
    report["validated_output"] = str(validated_path)
    report["manifest_generated_at"] = _utc_timestamp()
    raw_parsed_dates = pd.to_datetime(combined["Date"], errors="coerce").dropna()
    report["raw_observed_date_range"] = {
        "start_date": (
            raw_parsed_dates.min().strftime("%Y-%m-%d")
            if not raw_parsed_dates.empty
            else None
        ),
        "end_date": (
            raw_parsed_dates.max().strftime("%Y-%m-%d")
            if not raw_parsed_dates.empty
            else None
        ),
    }
    report["date_range"] = {
        "start_date": validated["Date"].min() if not validated.empty else None,
        "end_date": validated["Date"].max() if not validated.empty else None,
    }

    manifest_path = output_dir / "indian_etf_ingestion_manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report["manifest"] = str(manifest_path)
    return report
