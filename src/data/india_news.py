"""CSV-first ingestion and explainable classification for Indian financial news."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "india_news"

NORMALIZED_COLUMNS = (
    "Headline",
    "Article URL",
    "Source",
    "Publication Timestamp",
    "Publication Date",
    "Company Name",
    "Symbol",
    "Exchange",
    "NSE Symbol",
    "BSE Symbol",
    "Retrieval Timestamp",
    "Provenance Status",
    "Dataset Type",
    "Classification",
    "Classification Evidence",
    "Duplicate Status",
    "Duplicate Key",
)

FIELD_ALIASES = {
    "Headline": ("headline", "title", "news_headline", "article_title"),
    "Article URL": ("article_url", "url", "link"),
    "Source": ("source", "publisher", "news_source"),
    "Publication Timestamp": (
        "publication_timestamp",
        "published_at",
        "published_datetime",
        "publication_date",
        "published_date",
    ),
    "Company Name": ("company_name", "company", "name"),
    "Symbol": ("symbol", "ticker", "company_symbol"),
    "Exchange": ("exchange", "listing_exchange"),
    "NSE Symbol": ("nse_symbol", "nse"),
    "BSE Symbol": ("bse_symbol", "bse"),
    "Retrieval Timestamp": ("retrieval_timestamp", "retrieved_at"),
    "Provenance Status": ("provenance_status", "license_status"),
    "Dataset Type": ("dataset_type", "data_type"),
}


POSITIVE_TERMS = (
    "profit growth",
    "revenue growth",
    "record profit",
    "record revenue",
    "beats estimates",
    "raises guidance",
    "order win",
    "contract win",
    "expansion",
    "upgrade",
    "strong demand",
)

NEGATIVE_TERMS = (
    "profit decline",
    "revenue decline",
    "loss",
    "misses estimates",
    "downgrade",
    "layoffs",
    "recall",
    "fine",
    "penalty",
)

RISK_TERMS = (
    "regulatory",
    "investigation",
    "fraud",
    "default",
    "bankruptcy",
    "insolvency",
    "cyberattack",
    "data breach",
    "compliance",
    "litigation",
)


class IndiaNewsProvider(Protocol):
    """Future external providers implement fetch() and return raw articles."""

    def fetch(self) -> pd.DataFrame:
        ...


@dataclass(frozen=True)
class IndiaNewsCSVProvider:
    """Read-only CSV provider for the initial implementation."""

    input_path: str | Path

    def fetch(self) -> pd.DataFrame:
        path = Path(self.input_path)
        if not path.is_file():
            raise FileNotFoundError(f"India news CSV not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError("India news input must be a CSV file")
        return pd.read_csv(path)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_alias(frame: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    lower_columns = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        column = lower_columns.get(alias.strip().lower())
        if column is not None:
            return frame[column]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def classify_india_news_headline(headline: object) -> tuple[str, str]:
    """Classify one headline using documented transparent terms."""
    if pd.isna(headline) or not str(headline).strip():
        return "unclassified", "missing headline"

    text = " ".join(str(headline).lower().split())
    found_risk = [term for term in RISK_TERMS if term in text]
    found_negative = [term for term in NEGATIVE_TERMS if term in text]
    found_positive = [term for term in POSITIVE_TERMS if term in text]

    if found_risk:
        return "risk-related", "matched risk terms: " + ", ".join(found_risk)
    if found_negative:
        return "negative", "matched negative terms: " + ", ".join(found_negative)
    if found_positive:
        return "positive", "matched positive terms: " + ", ".join(found_positive)
    return "neutral", "no positive, negative or risk term matched"


def normalize_india_news(
    raw: pd.DataFrame,
    *,
    default_source: str = "CSV",
    retrieval_timestamp: str | None = None,
    default_dataset_type: str = "unknown",
) -> pd.DataFrame:
    """Normalize news records while preserving duplicate original records."""
    if raw.empty:
        raise ValueError("India news input contains no records")

    retrieved = retrieval_timestamp or _utc_timestamp()
    normalized = pd.DataFrame(index=raw.index)
    for target, aliases in FIELD_ALIASES.items():
        normalized[target] = _first_alias(raw, aliases)

    normalized["Headline"] = normalized["Headline"].apply(
        lambda value: str(value).strip()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )
    normalized["Article URL"] = normalized["Article URL"].apply(
        lambda value: str(value).strip()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )
    normalized["Source"] = normalized["Source"].apply(
        lambda value: str(value).strip()
        if pd.notna(value) and str(value).strip()
        else default_source
    )

    for column in ("Company Name", "Symbol", "Exchange", "NSE Symbol", "BSE Symbol", "ISIN"):
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized["Company Name"] = normalized["Company Name"].apply(
        lambda value: str(value).strip()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )
    normalized["Symbol"] = normalized["Symbol"].apply(
        lambda value: str(value).strip().upper()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )
    normalized["Exchange"] = normalized["Exchange"].apply(
        lambda value: str(value).strip().upper()
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )
    for column in ("NSE Symbol", "BSE Symbol"):
        normalized[column] = normalized[column].apply(
            lambda value: str(value).strip().upper()
            if pd.notna(value) and str(value).strip()
            else pd.NA
        )

    for index in normalized.index:
        symbol = normalized.at[index, "Symbol"]
        exchange = normalized.at[index, "Exchange"]
        if pd.isna(normalized.at[index, "NSE Symbol"]) and pd.notna(symbol) and str(exchange).upper() == "NSE":
            normalized.at[index, "NSE Symbol"] = symbol
        if pd.isna(normalized.at[index, "BSE Symbol"]) and pd.notna(symbol) and str(exchange).upper() == "BSE":
            normalized.at[index, "BSE Symbol"] = symbol

    parsed_publication = pd.to_datetime(
        normalized["Publication Timestamp"],
        errors="coerce",
        utc=True,
    )
    normalized["Publication Timestamp"] = parsed_publication.apply(
        lambda value: value.isoformat().replace("+00:00", "Z")
        if pd.notna(value)
        else pd.NA
    )
    normalized["Publication Date"] = parsed_publication.apply(
        lambda value: value.strftime("%Y-%m-%d")
        if pd.notna(value)
        else pd.NA
    )

    normalized["Retrieval Timestamp"] = normalized["Retrieval Timestamp"].apply(
        lambda value: str(value).strip()
        if pd.notna(value) and str(value).strip()
        else retrieved
    )
    normalized["Provenance Status"] = normalized["Provenance Status"].apply(
        lambda value: str(value).strip().lower()
        if pd.notna(value) and str(value).strip()
        else "unverified"
    )
    normalized["Dataset Type"] = normalized["Dataset Type"].apply(
        lambda value: str(value).strip().lower()
        if pd.notna(value) and str(value).strip()
        else default_dataset_type
    )

    classifications: list[tuple[str, str]] = []
    for headline in normalized["Headline"]:
        classifications.append(classify_india_news_headline(headline))
    normalized["Classification"] = [item[0] for item in classifications]
    normalized["Classification Evidence"] = [item[1] for item in classifications]

    duplicate_keys: list[str] = []
    for index in normalized.index:
        url = normalized.at[index, "Article URL"]
        headline = normalized.at[index, "Headline"]
        publication_date = normalized.at[index, "Publication Date"]
        symbol = normalized.at[index, "Symbol"]
        source = normalized.at[index, "Source"]
        if pd.notna(url) and str(url).strip():
            key = "url:" + str(url).strip().lower()
        else:
            key = "fallback:" + "|".join((
                str(headline).strip().lower() if pd.notna(headline) else "",
                str(publication_date).strip() if pd.notna(publication_date) else "",
                str(symbol).strip().lower() if pd.notna(symbol) else "",
                str(source).strip().lower(),
            ))
        duplicate_keys.append(key)
    normalized["Duplicate Key"] = duplicate_keys
    seen: set[str] = set()
    duplicate_statuses: list[str] = []
    for key in duplicate_keys:
        if key in seen:
            duplicate_statuses.append("duplicate")
        else:
            duplicate_statuses.append("original")
            seen.add(key)
    normalized["Duplicate Status"] = duplicate_statuses

    normalized = normalized.loc[:, list(NORMALIZED_COLUMNS)]
    normalized = normalized.sort_values(
        ["Publication Timestamp", "Symbol", "Headline"],
        ascending=[False, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    return normalized


def build_company_news_summary(normalized: pd.DataFrame) -> pd.DataFrame:
    """Summarize article counts and publication coverage per mapped company."""
    if "Symbol" not in normalized.columns:
        raise ValueError("Normalized news is missing the Symbol column")

    rows: list[dict[str, object]] = []
    for symbol, frame in normalized.groupby("Symbol", dropna=False, sort=True):
        display_symbol = "unknown" if pd.isna(symbol) else str(symbol)
        frame = frame.copy()
        company_names = (
            frame["Company Name"].dropna().astype(str).str.strip()
            if "Company Name" in frame.columns
            else pd.Series(dtype=str)
        )
        company_name = (
            company_names.value_counts().index[0]
            if not company_names.empty
            else "unknown"
        )
        publication_dates = frame["Publication Date"].dropna()
        classification_counts = frame["Classification"].fillna("unclassified").value_counts()

        rows.append({
            "Company Name": company_name,
            "Symbol": display_symbol,
            "Article Count": int(len(frame)),
            "Original Article Count": int(
                frame["Duplicate Status"].eq("original").sum()
            ),
            "Duplicate Article Count": int(
                frame["Duplicate Status"].eq("duplicate").sum()
            ),
            "First Publication Date": (
                str(publication_dates.min())
                if not publication_dates.empty
                else "unknown"
            ),
            "Last Publication Date": (
                str(publication_dates.max())
                if not publication_dates.empty
                else "unknown"
            ),
            "Missing Publication Timestamp Count": int(
                frame["Publication Timestamp"].isna().sum()
            ),
            "Positive Count": int(classification_counts.get("positive", 0)),
            "Negative Count": int(classification_counts.get("negative", 0)),
            "Neutral Count": int(classification_counts.get("neutral", 0)),
            "Risk-Related Count": int(classification_counts.get("risk-related", 0)),
            "Unclassified Count": int(classification_counts.get("unclassified", 0)),
            "Unknown Company": "yes" if pd.isna(symbol) else "no",
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Company Name",
            "Symbol",
            "Article Count",
            "Original Article Count",
            "Duplicate Article Count",
            "First Publication Date",
            "Last Publication Date",
            "Missing Publication Timestamp Count",
            "Positive Count",
            "Negative Count",
            "Neutral Count",
            "Risk-Related Count",
            "Unclassified Count",
            "Unknown Company",
        ])
    return pd.DataFrame(rows).sort_values("Symbol").reset_index(drop=True)


def ingest_india_news(
    provider: IndiaNewsProvider,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    source_name: str = "CSV",
    dataset_type: str = "unknown",
) -> dict[str, object]:
    """Preserve raw news and write normalized articles and company summaries."""
    raw = provider.fetch()
    if not isinstance(raw, pd.DataFrame):
        raise ValueError("News provider fetch() must return a DataFrame")

    output_path = Path(output_dir)
    raw_dir = output_path / "raw"
    normalized_dir = output_path / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    retrieval_timestamp = _utc_timestamp()
    raw_path: Path | None = None
    if isinstance(provider, IndiaNewsCSVProvider):
        source_path = Path(provider.input_path)
        raw_path = raw_dir / source_path.name
        if raw_path.exists() and _sha256(raw_path) != _sha256(source_path):
            raw_path = raw_dir / f"{source_path.stem}-{_sha256(source_path)[:12]}{source_path.suffix}"
        shutil.copy2(source_path, raw_path)

    normalized = normalize_india_news(
        raw,
        default_source=source_name,
        retrieval_timestamp=retrieval_timestamp,
        default_dataset_type=dataset_type,
    )
    summary = build_company_news_summary(normalized)

    normalized_path = normalized_dir / "india_news_normalized.csv"
    summary_path = normalized_dir / "india_company_news_summary.csv"
    normalized.to_csv(normalized_path, index=False)
    summary.to_csv(summary_path, index=False)

    missing_timestamps = int(normalized["Publication Timestamp"].isna().sum())
    unknown_companies = int(normalized["Symbol"].isna().sum())
    duplicates = int(normalized["Duplicate Status"].eq("duplicate").sum())
    classification_counts = normalized["Classification"].fillna("unclassified").value_counts().to_dict()

    manifest = {
        "generated_at_utc": retrieval_timestamp,
        "timezone": "Asia/Kolkata",
        "dataset_type": dataset_type,
        "source_name": source_name,
        "raw_input": str(raw_path) if raw_path is not None else None,
        "raw_sha256": _sha256(raw_path) if raw_path is not None else None,
        "normalized_output": str(normalized_path),
        "company_summary_output": str(summary_path),
        "record_count": int(len(normalized)),
        "original_article_count": int(normalized["Duplicate Status"].eq("original").sum()),
        "duplicate_article_count": duplicates,
        "missing_timestamp_count": missing_timestamps,
        "unknown_company_count": unknown_companies,
        "classification_counts": classification_counts,
        "raw_record_policy": (
            "Raw records are preserved and not deleted. Duplicates are marked "
            "but retained in the raw and normalized datasets."
        ),
        "classification_method": {
            "approach": "Determinable keyword matching",
            "priority": "risk-related, negative, positive, neutral",
            "risk_terms": list(RISK_TERMS),
            "negative_terms": list(NEGATIVE_TERMS),
            "positive_terms": list(POSITIVE_TERMS),
            "limitations": (
                "Keyword matching is explainable but cannot understand context, "
                "sarcasm or complex financial language."
            ),
        },
        "duplicate_method": (
            "Article URL is the primary key. Without a URL, headline, "
            "publication date, symbol and source are used."
        ),
        "limitations": [
            "CSV ingestion does not verify article authenticity or publication.",
            "Missing timestamps and unknown symbols are reported explicitly.",
            "No paid news API or external news provider is used.",
            "This module does not perform forecasting or produce trading advice.",
        ],
    }
    manifest_path = output_path / "india_news_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "record_count": int(len(normalized)),
        "duplicate_count": duplicates,
        "raw_output": str(raw_path) if raw_path is not None else None,
        "normalized_output": str(normalized_path),
        "company_summary_output": str(summary_path),
        "manifest": str(manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Indian financial news from CSV"
    )
    parser.add_argument("input_csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--source", default="CSV")
    parser.add_argument("--dataset-type", default="unknown")
    args = parser.parse_args(argv)
    result = ingest_india_news(
        IndiaNewsCSVProvider(args.input_csv),
        output_dir=args.output_dir,
        source_name=args.source,
        dataset_type=args.dataset_type,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
