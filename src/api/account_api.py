"""Authenticated account, portfolio, holding and transaction endpoints."""

from __future__ import annotations

import csv
import io
import re
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, localcontext
from enum import Enum
from typing import Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from src.auth.dependencies import get_current_user
from src.auth.security import (
    AuthenticationConfigurationError,
    create_access_token,
    hash_password,
    verify_password,
)
from src.database.models import (
    Holding,
    Portfolio,
    PortfolioMonitoringReport,
    Transaction,
    User,
)
from src.database.session import get_db
from src.services.portfolio_ledger import (
    LedgerError,
    find_idempotent_transaction,
    post_transaction,
    reconcile_portfolio,
)
from src.services.portfolio_rebalance import preview_portfolio_rebalance
from src.services.portfolio_monitor import monitor_portfolio
from src.services.issuer_concentration import (
    is_valid_instrument_identifier,
    is_valid_issuer_identifier,
)
from src.services.sector_concentration import (
    safe_sector_concentration_assessment,
)
from src.services.portfolio_exposure_risk import (
    EXPOSURE_RISK_POLICY_VERSION,
    EXPOSURE_RISK_THRESHOLDS,
)
from src.market_data.session import get_market_db
from src.market_data.service import (
    MarketDataStorageUnavailableError,
    PortfolioNotFoundError,
    value_portfolio,
)


router = APIRouter(tags=["accounts"])
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&.-]{0,31}$")
_MAX_CSV_BYTES = 1024 * 1024
_CSV_COLUMNS = ("exchange", "symbol", "quantity", "average_price")
_CSV_CONTENT_TYPES = {"text/csv", "application/csv", "application/vnd.ms-excel"}


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=1024)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _EMAIL_RE.fullmatch(normalized):
            raise ValueError("email must be a valid email address")
        return normalized


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _EMAIL_RE.fullmatch(normalized):
            raise ValueError("email must be a valid email address")
        return normalized


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class PortfolioCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    cash_balance: Decimal = Field(
        default=Decimal("0.00"), ge=0, max_digits=20, decimal_places=2
    )

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("portfolio name must not be blank")
        return normalized


class PortfolioUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    cash_balance: Decimal | None = Field(default=None, ge=0, max_digits=20, decimal_places=2)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("portfolio name must not be blank")
        return normalized

    @model_validator(mode="after")
    def require_update(self):
        if not self.model_fields_set:
            raise ValueError("at least one portfolio field must be provided")
        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("portfolio fields cannot be null")
        return self


class PortfolioResponse(BaseModel):
    id: str
    name: str
    base_currency: str
    cash_balance: Decimal
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"


class HoldingCreate(BaseModel):
    exchange: Exchange
    symbol: str = Field(min_length=1, max_length=32)
    quantity: Decimal = Field(gt=0, max_digits=20, decimal_places=6)
    average_purchase_price: Decimal = Field(gt=0, max_digits=20, decimal_places=6)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not _SYMBOL_RE.fullmatch(symbol):
            raise ValueError("symbol contains unsupported characters")
        return symbol


class HoldingUpdate(BaseModel):
    exchange: Exchange | None = None
    symbol: str | None = Field(default=None, min_length=1, max_length=32)
    quantity: Decimal | None = Field(default=None, gt=0, max_digits=20, decimal_places=6)
    average_purchase_price: Decimal | None = Field(
        default=None, gt=0, max_digits=20, decimal_places=6
    )

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        symbol = value.strip().upper()
        if not _SYMBOL_RE.fullmatch(symbol):
            raise ValueError("symbol contains unsupported characters")
        return symbol

    @model_validator(mode="after")
    def require_update(self):
        if not self.model_fields_set:
            raise ValueError("at least one holding field must be provided")
        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("holding fields cannot be null")
        return self


class HoldingResponse(BaseModel):
    id: str
    portfolio_id: str
    exchange: Exchange
    symbol: str
    quantity: Decimal
    average_purchase_price: Decimal
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PortfolioSummaryResponse(BaseModel):
    portfolio_id: str
    currency: str
    holdings_count: int
    total_invested_amount: Decimal
    cash_balance: Decimal
    holdings_by_exchange: dict[str, int]
    valuation_note: str


class HoldingImportResponse(BaseModel):
    imported_count: int
    holdings: list[HoldingResponse]


class TransactionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    CASH_DEPOSIT = "CASH_DEPOSIT"
    CASH_WITHDRAWAL = "CASH_WITHDRAWAL"


class TransactionCreate(BaseModel):
    transaction_type: TransactionType
    transaction_date: date
    exchange: Exchange | None = None
    symbol: str | None = Field(default=None, min_length=1, max_length=32)
    quantity: Decimal | None = Field(default=None, gt=0, max_digits=20, decimal_places=6)
    unit_price: Decimal | None = Field(default=None, gt=0, max_digits=20, decimal_places=6)
    cash_amount: Decimal | None = Field(default=None, gt=0, max_digits=20, decimal_places=2)
    fees: Decimal = Field(default=Decimal("0.00"), ge=0, max_digits=20, decimal_places=2)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        symbol = value.strip().upper()
        if not _SYMBOL_RE.fullmatch(symbol):
            raise ValueError("symbol contains unsupported characters")
        return symbol

    @field_validator("idempotency_key")
    @classmethod
    def normalize_idempotency_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        key = value.strip()
        if not key:
            raise ValueError("idempotency_key must not be blank")
        return key

    @model_validator(mode="after")
    def validate_transaction_fields(self):
        stock_trade = self.transaction_type in (
            TransactionType.BUY,
            TransactionType.SELL,
        )
        if stock_trade:
            if (
                self.exchange is None
                or self.symbol is None
                or self.quantity is None
                or self.unit_price is None
                or self.cash_amount is not None
            ):
                raise ValueError(
                    "BUY/SELL requires exchange, symbol, quantity and unit_price only"
                )
        elif (
            self.exchange is not None
            or self.symbol is not None
            or self.quantity is not None
            or self.unit_price is not None
            or self.cash_amount is None
        ):
            raise ValueError(
                "cash transactions require cash_amount and no stock fields"
            )
        if not stock_trade and self.fees != Decimal("0.00"):
            raise ValueError("cash transactions do not support fees")
        return self


class TransactionResponse(BaseModel):
    id: str
    portfolio_id: str
    transaction_type: TransactionType
    transaction_date: date
    exchange: Exchange | None
    symbol: str | None
    quantity: Decimal | None
    unit_price: Decimal | None
    cash_amount: Decimal | None
    fees: Decimal
    idempotency_key: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ReconciliationHolding(BaseModel):
    exchange: Exchange
    symbol: str
    quantity: Decimal
    average_purchase_price: Decimal


class HoldingDiscrepancy(BaseModel):
    exchange: Exchange
    symbol: str
    expected_quantity: Decimal
    actual_quantity: Decimal
    quantity_difference: Decimal
    expected_average_purchase_price: Decimal | None
    actual_average_purchase_price: Decimal | None
    average_purchase_price_difference: Decimal | None
    missing: bool
    unexpected: bool


class ReconciliationResponse(BaseModel):
    is_reconciled: bool
    baseline_timestamp: datetime
    expected_cash: Decimal
    actual_cash: Decimal
    cash_difference: Decimal
    expected_holdings: list[ReconciliationHolding]
    actual_holdings: list[ReconciliationHolding]
    holding_discrepancies: list[HoldingDiscrepancy]


class ValuedHoldingResponse(BaseModel):
    exchange: Exchange
    symbol: str
    quantity: Decimal
    cost_basis: Decimal
    market_date: date
    close: Decimal
    market_value: Decimal
    unrealized_profit_loss: Decimal
    provider: str
    source: str
    fetched_at: datetime


class ValuationIssueResponse(BaseModel):
    exchange: Exchange
    symbol: str
    reason: str | None = None
    market_date: date | None = None
    close: Decimal | None = None
    provider: str | None = None
    source: str | None = None
    fetched_at: datetime | None = None
    age_calendar_days: int | None = None


class PortfolioValuationResponse(BaseModel):
    portfolio_id: str
    currency: str
    valuation_date: date
    valuation_status: str
    valuation_complete: bool
    invested_cost_basis: Decimal
    market_value: Decimal
    market_value_is_partial: bool
    unrealized_profit_loss: Decimal
    unrealized_profit_loss_is_partial: bool
    cash_balance: Decimal
    total_portfolio_value: Decimal | None
    valued_holdings: list[ValuedHoldingResponse]
    stale_prices: list[ValuationIssueResponse]
    missing_prices: list[ValuationIssueResponse]
    unavailable_prices: list[ValuationIssueResponse]
    freshness_policy: dict[str, str | int]


class PortfolioMonitoringRequest(BaseModel):
    session_date: date


class ExposureRiskListingResponse(BaseModel):
    exchange: Literal["NSE", "BSE"]
    symbol: str
    listing: str
    quantity: Decimal
    market_value: Decimal
    portfolio_weight_pct: Decimal


class ExposureRiskUnavailableListingResponse(BaseModel):
    exchange: Literal["NSE", "BSE"]
    symbol: str
    listing: str
    reason: str


class PortfolioExposureRiskAlertResponse(BaseModel):
    code: Literal[
        "SINGLE_STOCK_EXPOSURE_HIGH",
        "TOP_FIVE_CONCENTRATION_HIGH",
        "CASH_ALLOCATION_LOW",
        "EQUITY_HHI_HIGH",
    ]
    severity: Literal["MEDIUM"]
    metric_name: Literal[
        "single_stock_exposure",
        "top_five_concentration",
        "cash_allocation",
        "equity_hhi",
    ]
    affected_listing: str | None
    observed_value: Decimal
    threshold: Decimal
    policy_version: str
    explanation: str


class PortfolioExposureRiskAssessmentResponse(BaseModel):
    assessment_status: Literal["EVALUATED", "NOT_EVALUATED"]
    policy_version: str
    evaluated_at: datetime
    session_date: date
    portfolio_nav: Decimal | None
    total_equity_market_value: Decimal | None
    cash_balance: Decimal | None
    listing_exposures: list[ExposureRiskListingResponse]
    top_five_weight_pct: Decimal | None
    cash_weight_pct: Decimal | None
    equity_hhi: Decimal | None
    alerts: list[PortfolioExposureRiskAlertResponse]
    reason_codes: list[str]
    unavailable_listings: list[ExposureRiskUnavailableListingResponse]


class IssuerListingExposureResponse(BaseModel):
    exchange: Literal["NSE", "BSE"]
    symbol: str
    listing: str
    market_value: Decimal
    instrument_identifier: str | None
    instrument_identifier_type: Literal["ISIN", "OTHER_VERIFIED"] | None
    issuer_id: str | None
    issuer_id_type: Literal["CIN", "LEI", "OTHER_VERIFIED"] | None
    source: str | None
    source_reference: str | None
    verified_on: date | None


class IssuerExposureResponse(BaseModel):
    issuer_id: str
    issuer_id_type: Literal["CIN", "LEI", "OTHER_VERIFIED"]
    market_value: Decimal
    portfolio_weight_pct: Decimal
    listings: list[str]


class UnresolvedIssuerListingResponse(BaseModel):
    exchange: Literal["NSE", "BSE"]
    symbol: str
    listing: str
    reason: str


class PortfolioIssuerConcentrationAssessmentResponse(BaseModel):
    assessment_status: Literal["EVALUATED", "NOT_EVALUATED"]
    policy_version: str
    reference_data_version: str
    evaluated_at: datetime
    session_date: date
    portfolio_nav: Decimal | None
    total_equity_market_value: Decimal | None
    listing_exposures: list[IssuerListingExposureResponse]
    issuer_exposures: list[IssuerExposureResponse]
    alerts: list[dict]
    reason_codes: list[str]
    unresolved_listings: list[UnresolvedIssuerListingResponse]


class SectorIssuerReferenceResponse(BaseModel):
    issuer_id_type: Literal["CIN", "LEI", "OTHER_VERIFIED"]
    issuer_id: str


class SectorMappingProvenanceResponse(BaseModel):
    effective_from: date
    effective_to: date | None
    source: str
    source_reference: str
    source_published_on: date
    verified_on: date
    reference_data_version: str
    mapping_sha256: str


class SectorIssuerContributionResponse(BaseModel):
    issuer_id_type: Literal["CIN", "LEI", "OTHER_VERIFIED"]
    issuer_id: str
    market_value: Decimal
    sector_code: str
    sector_name: str
    mapping: SectorMappingProvenanceResponse


class SectorExposureResponse(BaseModel):
    sector_code: str
    sector_name: str
    market_value: Decimal
    portfolio_weight_pct: Decimal
    issuers: list[SectorIssuerReferenceResponse]


class SectorZeroValueListingResponse(BaseModel):
    exchange: Literal["NSE", "BSE"]
    symbol: str
    listing: str
    market_value: Decimal
    issuer_id_type: Literal["CIN", "LEI", "OTHER_VERIFIED"] | None
    issuer_id: str | None


class UnresolvedSectorIssuerResponse(BaseModel):
    issuer_id_type: Literal["CIN", "LEI", "OTHER_VERIFIED"]
    issuer_id: str
    reason: str


class PortfolioSectorConcentrationAssessmentResponse(BaseModel):
    assessment_status: Literal["EVALUATED", "NOT_EVALUATED"]
    policy_version: str
    reference_data_version: str
    reference_snapshot: dict | None
    reference_snapshot_sha256: str | None
    taxonomy_name: str
    taxonomy_version: str
    evaluated_at: datetime
    session_date: date
    portfolio_nav: Decimal | None
    total_equity_market_value: Decimal | None
    sector_exposures: list[SectorExposureResponse]
    largest_sector_weight_pct: Decimal | None
    top_three_sector_weight_pct: Decimal | None
    sector_hhi: Decimal | None
    issuer_contributions: list[SectorIssuerContributionResponse]
    zero_value_listings: list[SectorZeroValueListingResponse]
    alerts: list[dict]
    reason_codes: list[str]
    unresolved_issuers: list[UnresolvedSectorIssuerResponse]


class PortfolioMonitoringReportResponse(BaseModel):
    id: str
    portfolio_id: str
    session_date: date
    evaluated_at: datetime
    status: Literal[
        "NO_ACTION_REQUIRED",
        "ACTION_REQUIRED",
        "DATA_UNAVAILABLE",
        "EVALUATION_FAILED",
    ]
    valuation_complete: bool
    portfolio_nav: Decimal | None
    evaluated_holdings_count: int
    evaluated_holdings: list[dict]
    data_quality_warnings: list[dict]
    ledger_reconciliation: dict | None
    reason_codes: list[str]
    explanation: str
    policy_version: str
    source_metadata: list[dict]
    error_details: dict[str, str] | None
    exposure_risk_assessment: PortfolioExposureRiskAssessmentResponse | None
    issuer_concentration_assessment: (
        PortfolioIssuerConcentrationAssessmentResponse | None
    )
    sector_concentration_assessment: (
        PortfolioSectorConcentrationAssessmentResponse | None
    )


_MONITORING_ERROR_FALLBACK = {
    "code": "monitoring_evaluation_failed",
    "message": "Monitoring evaluation failed. Please try again later.",
}
_SAFE_MONITORING_ERROR_MESSAGES = {
    "Monitoring evaluation failed unexpectedly.",
    _MONITORING_ERROR_FALLBACK["message"],
}
_EXPOSURE_RISK_POLICY_VERSION = EXPOSURE_RISK_POLICY_VERSION
_EXPOSURE_RISK_ALERTS = {
    "SINGLE_STOCK_EXPOSURE_HIGH": (
        "single_stock_exposure",
        EXPOSURE_RISK_THRESHOLDS["single_stock_exposure"] * Decimal("100"),
        True,
    ),
    "TOP_FIVE_CONCENTRATION_HIGH": (
        "top_five_concentration",
        EXPOSURE_RISK_THRESHOLDS["top_five_concentration"] * Decimal("100"),
        False,
    ),
    "CASH_ALLOCATION_LOW": (
        "cash_allocation",
        EXPOSURE_RISK_THRESHOLDS["minimum_cash_allocation"] * Decimal("100"),
        False,
    ),
    "EQUITY_HHI_HIGH": (
        "equity_hhi",
        EXPOSURE_RISK_THRESHOLDS["equity_hhi"],
        False,
    ),
}
_EXPOSURE_RISK_REASONS = {
    "valuation_incomplete",
    "missing_price",
    "stale_price",
    "unpermitted_price",
    "price_after_session",
    "portfolio_nav_unavailable",
    "portfolio_nav_not_positive",
    "valuation_nav_mismatch",
}
_ISSUER_CONCENTRATION_POLICY_VERSION = "issuer-concentration-v1"
_ISSUER_ID_TYPES = {"CIN", "LEI", "OTHER_VERIFIED"}
_ISSUER_CONCENTRATION_REASONS = {
    "valuation_incomplete",
    "portfolio_nav_unavailable",
    "portfolio_nav_not_positive",
    "valuation_nav_mismatch",
    "issuer_reference_invalid",
    "issuer_mapping_unresolved",
    "ledger_reconciliation_unavailable",
    "ledger_discrepancy",
}
_SAFE_UNAVAILABLE_LISTING_REASON = (
    "No valid fresh, permitted stored EOD price is available."
)


def _monitoring_report_response(
    report: PortfolioMonitoringReport,
) -> PortfolioMonitoringReportResponse:
    """Return a report with only known-safe monitoring error details."""
    stored_details = report.error_details
    if stored_details is None and report.status != "EVALUATION_FAILED":
        safe_details = None
    elif (
        isinstance(stored_details, dict)
        and isinstance(stored_details.get("code"), str)
        and isinstance(stored_details.get("message"), str)
        and stored_details["code"] == "monitoring_evaluation_failed"
        and stored_details["message"] in _SAFE_MONITORING_ERROR_MESSAGES
    ):
        safe_details = {
            "code": "monitoring_evaluation_failed",
            "message": stored_details["message"],
        }
    else:
        safe_details = _MONITORING_ERROR_FALLBACK.copy()

    safe_risk_assessment = _safe_exposure_risk_assessment(
        report.exposure_risk_assessment
    )
    if (
        safe_risk_assessment is not None
        and safe_risk_assessment["session_date"] != report.session_date
    ):
        safe_risk_assessment = None
    safe_issuer_assessment = _safe_issuer_concentration_assessment(
        report.issuer_concentration_assessment
    )
    if (
        safe_issuer_assessment is not None
        and safe_issuer_assessment["session_date"] != report.session_date
    ):
        safe_issuer_assessment = None
    safe_sector_assessment = safe_sector_concentration_assessment(
        report.sector_concentration_assessment,
        safe_issuer_assessment,
    )
    if (
        safe_sector_assessment is not None
        and safe_sector_assessment["session_date"] != report.session_date
    ):
        safe_sector_assessment = None
    report_data = {
        field_name: getattr(report, field_name)
        for field_name in PortfolioMonitoringReportResponse.model_fields
        if field_name
        not in {
            "error_details",
            "exposure_risk_assessment",
            "issuer_concentration_assessment",
            "sector_concentration_assessment",
        }
    }
    report_data["error_details"] = safe_details
    report_data["exposure_risk_assessment"] = safe_risk_assessment
    report_data["issuer_concentration_assessment"] = safe_issuer_assessment
    report_data["sector_concentration_assessment"] = safe_sector_assessment
    return PortfolioMonitoringReportResponse.model_validate(report_data)


def _safe_decimal_string(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 80
        or any(character not in "0123456789.-" for character in value)
    ):
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    if (
        not parsed.is_finite()
        or parsed < 0
        or parsed.adjusted() > 30
        or parsed.as_tuple().exponent < -60
        or len(parsed.as_tuple().digits) > 50
    ):
        return None
    return value


def _is_paise_value(value: str) -> bool:
    parsed = Decimal(value)
    numerator, denominator = parsed.as_integer_ratio()
    return numerator * 100 % denominator == 0


def _safe_exposure_risk_assessment(
    stored: object,
) -> dict | None:
    """Allowlist persisted risk data before nested response-model validation."""
    if not isinstance(stored, dict):
        return None
    status_value = stored.get("assessment_status")
    if (
        not isinstance(status_value, str)
        or status_value not in {"EVALUATED", "NOT_EVALUATED"}
        or stored.get("policy_version") != _EXPOSURE_RISK_POLICY_VERSION
        or not isinstance(stored.get("session_date"), str)
        or not isinstance(stored.get("evaluated_at"), str)
        or len(stored["session_date"]) > 10
        or len(stored["evaluated_at"]) > 40
    ):
        return None

    try:
        risk_session_date = date.fromisoformat(stored["session_date"])
        risk_evaluated_at = datetime.fromisoformat(stored["evaluated_at"])
    except ValueError:
        return None
    if risk_evaluated_at.tzinfo is None:
        return None

    reason_codes = stored.get("reason_codes")
    if (
        not isinstance(reason_codes, list)
        or any(
            not isinstance(code, str) or code not in _EXPOSURE_RISK_REASONS
            for code in reason_codes
        )
    ):
        return None
    unavailable = stored.get("unavailable_listings")
    if not isinstance(unavailable, list):
        return None
    safe_unavailable = []
    for item in unavailable:
        if not isinstance(item, dict):
            return None
        exchange = item.get("exchange")
        symbol = item.get("symbol")
        if (
            not isinstance(exchange, str)
            or exchange not in {"NSE", "BSE"}
            or not isinstance(symbol, str)
            or not _SYMBOL_RE.fullmatch(symbol)
        ):
            return None
        safe_unavailable.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "listing": f"{exchange}:{symbol}",
                "reason": _SAFE_UNAVAILABLE_LISTING_REASON,
            }
        )

    raw_listings = stored.get("listing_exposures")
    raw_alerts = stored.get("alerts")
    if not isinstance(raw_listings, list) or not isinstance(raw_alerts, list):
        return None
    safe_listings = []
    seen_listings = set()
    for item in raw_listings:
        if not isinstance(item, dict):
            return None
        exchange = item.get("exchange")
        symbol = item.get("symbol")
        if (
            not isinstance(exchange, str)
            or exchange not in {"NSE", "BSE"}
            or not isinstance(symbol, str)
            or not _SYMBOL_RE.fullmatch(symbol)
        ):
            return None
        if (exchange, symbol) in seen_listings:
            return None
        seen_listings.add((exchange, symbol))
        decimal_fields = {
            key: _safe_decimal_string(item.get(key))
            for key in ("quantity", "market_value", "portfolio_weight_pct")
        }
        if (
            any(value is None for value in decimal_fields.values())
            or Decimal(decimal_fields["quantity"]) <= 0
            or Decimal(decimal_fields["market_value"]) < 0
            or Decimal(decimal_fields["portfolio_weight_pct"]) > 100
            or not _is_paise_value(decimal_fields["market_value"])
        ):
            return None
        safe_listings.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "listing": f"{exchange}:{symbol}",
                **decimal_fields,
            }
        )
    if safe_listings != sorted(
        safe_listings,
        key=lambda item: (
            -Decimal(item["market_value"]),
            item["exchange"],
            item["symbol"],
        ),
    ):
        return None

    safe_alerts = []
    for item in raw_alerts:
        if not isinstance(item, dict):
            return None
        code = item.get("code")
        if not isinstance(code, str):
            return None
        alert_definition = _EXPOSURE_RISK_ALERTS.get(code)
        if alert_definition is None:
            return None
        metric_name, expected_threshold, listing_required = alert_definition
        if (
            item.get("metric_name") != metric_name
            or item.get("severity") != "MEDIUM"
            or item.get("policy_version") != _EXPOSURE_RISK_POLICY_VERSION
        ):
            return None
        observed = _safe_decimal_string(item.get("observed_value"))
        threshold = _safe_decimal_string(item.get("threshold"))
        if (
            observed is None
            or threshold is None
            or Decimal(threshold) != expected_threshold
        ):
            return None
        listing = item.get("affected_listing")
        if listing_required:
            if (
                not isinstance(listing, str)
                or ":" not in listing
                or listing.split(":", 1)[0] not in {"NSE", "BSE"}
                or not _SYMBOL_RE.fullmatch(listing.split(":", 1)[1])
            ):
                return None
        elif listing is not None:
            return None
        shown_value = Decimal(observed)
        shown_threshold = Decimal(threshold)
        if (
            code
            in {
                "SINGLE_STOCK_EXPOSURE_HIGH",
                "TOP_FIVE_CONCENTRATION_HIGH",
                "EQUITY_HHI_HIGH",
            }
            and shown_value <= shown_threshold
            or code == "CASH_ALLOCATION_LOW"
            and shown_value >= shown_threshold
            or code != "EQUITY_HHI_HIGH"
            and shown_value > Decimal("100")
            or code == "EQUITY_HHI_HIGH"
            and shown_value > Decimal("1")
        ):
            return None
        if listing_required and listing not in {
            entry["listing"] for entry in safe_listings
        }:
            return None
        explanation = {
            "SINGLE_STOCK_EXPOSURE_HIGH": (
                f"{listing} represents {shown_value}% of total portfolio NAV, "
                f"exceeding the configured {shown_threshold}% exposure threshold."
            ),
            "TOP_FIVE_CONCENTRATION_HIGH": (
                f"The five largest listings represent {shown_value}% of total "
                f"portfolio NAV, exceeding the configured {shown_threshold}% "
                "concentration threshold."
            ),
            "CASH_ALLOCATION_LOW": (
                f"Cash represents {shown_value}% of total portfolio NAV, below "
                f"the configured {shown_threshold}% minimum."
            ),
            "EQUITY_HHI_HIGH": (
                f"Equity HHI is {shown_value}, exceeding the configured "
                f"{shown_threshold} concentration threshold."
            ),
        }[code]
        safe_alerts.append(
            {
                "code": code,
                "severity": "MEDIUM",
                "metric_name": metric_name,
                "affected_listing": listing,
                "observed_value": observed,
                "threshold": threshold,
                "policy_version": _EXPOSURE_RISK_POLICY_VERSION,
                "explanation": explanation,
            }
        )

    is_evaluated = status_value == "EVALUATED"
    metric_fields = (
        "portfolio_nav",
        "total_equity_market_value",
        "cash_balance",
        "top_five_weight_pct",
        "cash_weight_pct",
        "equity_hhi",
    )
    safe_metrics = {}
    for field in metric_fields:
        value = stored.get(field)
        if value is None:
            safe_metrics[field] = None
        else:
            safe_value = _safe_decimal_string(value)
            if safe_value is None:
                return None
            safe_metrics[field] = safe_value

    if is_evaluated:
        if (
            any(safe_metrics[field] is None for field in metric_fields[:-1])
            or reason_codes
            or unavailable
        ):
            return None
        if any(
            not _is_paise_value(safe_metrics[field])
            for field in (
                "portfolio_nav",
                "total_equity_market_value",
                "cash_balance",
            )
        ):
            return None
        with localcontext() as context:
            context.prec = 50
            nav_value = Decimal(safe_metrics["portfolio_nav"])
            cash_value = Decimal(safe_metrics["cash_balance"])
            equity_value = Decimal(safe_metrics["total_equity_market_value"])
            equity_hhi_value = (
                Decimal(safe_metrics["equity_hhi"])
                if safe_metrics["equity_hhi"] is not None
                else None
            )
            listing_value = sum(
                (Decimal(item["market_value"]) for item in safe_listings),
                Decimal("0"),
            )
            expected_hhi = (
                sum(
                    (
                        (Decimal(item["market_value"]) / equity_value) ** 2
                        for item in safe_listings
                    ),
                    Decimal("0"),
                )
                if equity_value
                else None
            )
            expected_top_five = (
                sum(
                    (Decimal(item["market_value"]) for item in safe_listings[:5]),
                    Decimal("0"),
                )
                / nav_value
                * Decimal("100")
            )
            expected_cash_weight = cash_value / nav_value * Decimal("100")
            metrics_invalid = (
                nav_value <= 0
                or cash_value + equity_value != nav_value
                or listing_value != equity_value
                or (equity_value == 0) != (equity_hhi_value is None)
                or (
                    equity_hhi_value is not None
                    and (
                        equity_hhi_value != expected_hhi
                        or equity_hhi_value > 1
                    )
                )
                or Decimal(safe_metrics["cash_weight_pct"]) != expected_cash_weight
                or Decimal(safe_metrics["top_five_weight_pct"]) != expected_top_five
                or any(
                    Decimal(item["portfolio_weight_pct"])
                    != Decimal(item["market_value"]) / nav_value * Decimal("100")
                    for item in safe_listings
                )
            )
            if metrics_invalid:
                return None

            expected_alerts = {}
            single_limit = (
                EXPOSURE_RISK_THRESHOLDS["single_stock_exposure"] * Decimal("100")
            )
            for item in safe_listings:
                listing_weight = Decimal(item["portfolio_weight_pct"])
                if listing_weight > single_limit:
                    expected_alerts[
                        ("SINGLE_STOCK_EXPOSURE_HIGH", item["listing"])
                    ] = listing_weight
            top_limit = (
                EXPOSURE_RISK_THRESHOLDS["top_five_concentration"]
                * Decimal("100")
            )
            if expected_top_five > top_limit:
                expected_alerts[("TOP_FIVE_CONCENTRATION_HIGH", None)] = (
                    expected_top_five
                )
            cash_limit = (
                EXPOSURE_RISK_THRESHOLDS["minimum_cash_allocation"]
                * Decimal("100")
            )
            if expected_cash_weight < cash_limit:
                expected_alerts[("CASH_ALLOCATION_LOW", None)] = (
                    expected_cash_weight
                )
            if (
                expected_hhi is not None
                and expected_hhi > EXPOSURE_RISK_THRESHOLDS["equity_hhi"]
            ):
                expected_alerts[("EQUITY_HHI_HIGH", None)] = expected_hhi

            if len(expected_alerts) != len(safe_alerts):
                return None
            for alert in safe_alerts:
                key = (alert["code"], alert["affected_listing"])
                if expected_alerts.get(key) != Decimal(alert["observed_value"]):
                    return None
    elif (
        not reason_codes
        or any(value is not None for value in safe_metrics.values())
        or safe_listings
        or safe_alerts
    ):
        return None

    return {
        "assessment_status": status_value,
        "policy_version": _EXPOSURE_RISK_POLICY_VERSION,
        "evaluated_at": risk_evaluated_at,
        "session_date": risk_session_date,
        **safe_metrics,
        "listing_exposures": safe_listings,
        "alerts": safe_alerts,
        "reason_codes": reason_codes,
        "unavailable_listings": safe_unavailable,
    }


def _safe_issuer_concentration_assessment(stored: object) -> dict | None:
    """Allowlist persisted issuer-risk data without rewriting legacy reports."""
    try:
        return _validate_issuer_concentration_assessment(stored)
    except (ArithmeticError, TypeError, ValueError):
        return None


def _validate_issuer_concentration_assessment(stored: object) -> dict | None:
    if not isinstance(stored, dict):
        return None
    status_value = stored.get("assessment_status")
    reference_version = stored.get("reference_data_version")
    if (
        not isinstance(status_value, str)
        or status_value not in {"EVALUATED", "NOT_EVALUATED"}
        or stored.get("policy_version") != _ISSUER_CONCENTRATION_POLICY_VERSION
        or not isinstance(reference_version, str)
        or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", reference_version)
        or not isinstance(stored.get("session_date"), str)
        or not isinstance(stored.get("evaluated_at"), str)
        or len(stored["session_date"]) > 10
        or len(stored["evaluated_at"]) > 40
    ):
        return None
    try:
        assessment_date = date.fromisoformat(stored["session_date"])
        evaluated_at = datetime.fromisoformat(stored["evaluated_at"])
    except ValueError:
        return None
    if assessment_date.isoformat() != stored["session_date"] or evaluated_at.tzinfo is None:
        return None

    raw_reasons = stored.get("reason_codes")
    raw_listings = stored.get("listing_exposures")
    raw_issuers = stored.get("issuer_exposures")
    raw_alerts = stored.get("alerts")
    raw_unresolved = stored.get("unresolved_listings")
    if (
        not isinstance(raw_reasons, list)
        or any(
            not isinstance(code, str)
            or code not in _ISSUER_CONCENTRATION_REASONS
            for code in raw_reasons
        )
        or not isinstance(raw_listings, list)
        or not isinstance(raw_issuers, list)
        or raw_alerts != []
        or not isinstance(raw_unresolved, list)
    ):
        return None

    safe_listings = []
    seen_listings = set()
    for item in raw_listings:
        if not isinstance(item, dict):
            return None
        exchange = item.get("exchange")
        symbol = item.get("symbol")
        instrument_id = item.get("instrument_identifier")
        instrument_type = item.get("instrument_identifier_type")
        issuer_id = item.get("issuer_id")
        issuer_type = item.get("issuer_id_type")
        source = item.get("source")
        source_reference = item.get("source_reference")
        verified_on_raw = item.get("verified_on")
        market_value = _safe_decimal_string(item.get("market_value"))
        try:
            verified_on = (
                date.fromisoformat(verified_on_raw)
                if isinstance(verified_on_raw, str)
                else None
            )
        except ValueError:
            return None
        mapped = issuer_id is not None
        if (
            not isinstance(exchange, str)
            or exchange not in {"NSE", "BSE"}
            or not isinstance(symbol, str)
            or not _SYMBOL_RE.fullmatch(symbol)
            or market_value is None
            or not _is_paise_value(market_value)
            or mapped != (issuer_type is not None)
            or mapped != (instrument_id is not None)
            or mapped != (instrument_type is not None)
            or mapped != (source is not None)
            or mapped != (source_reference is not None)
            or mapped != (verified_on is not None)
            or mapped
            and (
                not is_valid_issuer_identifier(issuer_type, issuer_id)
                or not is_valid_instrument_identifier(instrument_type, instrument_id)
                or not isinstance(source, str)
                or not source.strip()
                or len(source) > 512
                or not isinstance(source_reference, str)
                or not source_reference.strip()
                or len(source_reference) > 512
                or verified_on.isoformat() != verified_on_raw
            )
        ):
            return None
        listing = f"{exchange}:{symbol}"
        if listing in seen_listings or item.get("listing") != listing:
            return None
        seen_listings.add(listing)
        safe_listings.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "listing": listing,
                "market_value": market_value,
                "instrument_identifier": instrument_id,
                "instrument_identifier_type": instrument_type,
                "issuer_id": issuer_id,
                "issuer_id_type": issuer_type,
                "source": source,
                "source_reference": source_reference,
                "verified_on": verified_on,
            }
        )
    if safe_listings != sorted(
        safe_listings,
        key=lambda item: (
            -Decimal(item["market_value"]),
            item["exchange"],
            item["symbol"],
        ),
    ):
        return None

    safe_unresolved = []
    seen_unresolved = set()
    for item in raw_unresolved:
        if not isinstance(item, dict):
            return None
        exchange = item.get("exchange")
        symbol = item.get("symbol")
        if (
            not isinstance(exchange, str)
            or exchange not in {"NSE", "BSE"}
            or not isinstance(symbol, str)
            or not _SYMBOL_RE.fullmatch(symbol)
        ):
            return None
        listing = f"{exchange}:{symbol}"
        if (
            listing in seen_unresolved
            or item.get("reason")
            != "No verified issuer mapping applies to this session."
        ):
            return None
        seen_unresolved.add(listing)
        safe_unresolved.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "listing": listing,
                "reason": "No verified issuer mapping applies to this session.",
            }
        )

    nav_raw = stored.get("portfolio_nav")
    equity_raw = stored.get("total_equity_market_value")
    nav = _safe_decimal_string(nav_raw) if nav_raw is not None else None
    equity = _safe_decimal_string(equity_raw) if equity_raw is not None else None
    if nav is not None and Decimal(nav) <= 0:
        return None
    safe_issuers = []
    expected_issuer_values: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    expected_issuer_listings: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in safe_listings:
        if Decimal(item["market_value"]) > 0:
            if item["issuer_id"] is None:
                return None
            key = (item["issuer_id_type"], item["issuer_id"])
            expected_issuer_values[key] += Decimal(item["market_value"])
            expected_issuer_listings[key].append(item["listing"])

    serialized_issuer_keys = set()
    for item in raw_issuers:
        if not isinstance(item, dict):
            return None
        issuer_id = item.get("issuer_id")
        issuer_type = item.get("issuer_id_type")
        market_value = _safe_decimal_string(item.get("market_value"))
        weight = _safe_decimal_string(item.get("portfolio_weight_pct"))
        raw_issuer_listings = item.get("listings")
        if (
            not isinstance(issuer_id, str)
            or not issuer_id
            or len(issuer_id) > 128
            or not isinstance(issuer_type, str)
            or issuer_type not in _ISSUER_ID_TYPES
            or market_value is None
            or not _is_paise_value(market_value)
            or Decimal(market_value) <= 0
            or weight is None
            or Decimal(weight) > 100
            or not isinstance(raw_issuer_listings, list)
            or any(not isinstance(listing, str) for listing in raw_issuer_listings)
        ):
            return None
        key = (issuer_type, issuer_id)
        if key in serialized_issuer_keys:
            return None
        serialized_issuer_keys.add(key)
        expected_listings = sorted(expected_issuer_listings.get(key, []))
        with localcontext() as context:
            context.prec = 50
            expected_weight = (
                Decimal(market_value) / Decimal(nav) * Decimal("100")
                if nav is not None
                else None
            )
        if (
            key not in expected_issuer_values
            or Decimal(market_value) != expected_issuer_values[key]
            or raw_issuer_listings != expected_listings
            or expected_weight is None
            or Decimal(weight) != expected_weight
        ):
            return None
        safe_issuers.append(
            {
                "issuer_id": issuer_id,
                "issuer_id_type": issuer_type,
                "market_value": market_value,
                "portfolio_weight_pct": weight,
                "listings": expected_listings,
            }
        )
    if (
        serialized_issuer_keys != set(expected_issuer_values)
        or
        safe_issuers
        != sorted(
            safe_issuers,
            key=lambda item: (
                -Decimal(item["market_value"]),
                item["issuer_id_type"],
                item["issuer_id"],
            ),
        )
    ):
        return None

    if status_value == "EVALUATED":
        if (
            raw_reasons
            or safe_unresolved
            or nav is None
            or equity is None
            or not _is_paise_value(nav)
            or not _is_paise_value(equity)
            or Decimal(nav) <= 0
            or Decimal(equity) > Decimal(nav)
            or sum(
                (Decimal(item["market_value"]) for item in safe_listings),
                Decimal("0"),
            )
            != Decimal(equity)
            or sum(expected_issuer_values.values(), Decimal("0"))
            != Decimal(equity)
        ):
            return None
    elif (
        not raw_reasons
        or any(value is not None for value in (nav, equity))
        or safe_listings
        or safe_issuers
    ):
        return None

    return {
        "assessment_status": status_value,
        "policy_version": _ISSUER_CONCENTRATION_POLICY_VERSION,
        "reference_data_version": reference_version,
        "evaluated_at": evaluated_at,
        "session_date": assessment_date,
        "portfolio_nav": nav,
        "total_equity_market_value": equity,
        "listing_exposures": safe_listings,
        "issuer_exposures": safe_issuers,
        "alerts": [],
        "reason_codes": raw_reasons,
        "unresolved_listings": safe_unresolved,
    }


class PortfolioRebalancePreviewRequest(BaseModel):
    target_weights: dict[str, Decimal] | None = None
    target_cash_weight: Decimal | None = Field(default=None, ge=0, le=1)
    selected_holdings: list[str] | None = None
    additional_investment: Decimal = Field(
        default=Decimal("0.00"), ge=0, max_digits=20, decimal_places=2
    )
    cash_reserve_pct: Decimal = Field(
        default=Decimal("0.00"), ge=0, le=1, max_digits=8, decimal_places=6
    )
    max_weight_per_asset: Decimal = Field(
        default=Decimal("1.00"), ge=0, le=1, max_digits=8, decimal_places=6
    )
    transaction_cost_rate: Decimal = Field(
        default=Decimal("0.00"), ge=0, lt=1, max_digits=8, decimal_places=6
    )
    slippage_bps: Decimal = Field(
        default=Decimal("0.00"), ge=0, lt=10000, max_digits=10, decimal_places=4
    )
    allow_fractional_shares: bool = False
    fractional_share_precision: int = Field(default=6, ge=0, le=12)
    max_turnover: Decimal = Field(
        default=Decimal("1.00"), ge=0, max_digits=12, decimal_places=6
    )

    @field_validator("selected_holdings")
    @classmethod
    def normalize_selected_holdings(
        cls, values: list[str] | None
    ) -> list[str] | None:
        if values is None:
            return None
        normalized = []
        for value in values:
            listing = value.strip().upper()
            if not re.fullmatch(r"(NSE|BSE):[A-Z0-9][A-Z0-9&.-]{0,31}", listing):
                raise ValueError(
                    "selected_holdings entries must use NSE:SYMBOL or BSE:SYMBOL"
                )
            normalized.append(listing)
        if len(set(normalized)) != len(normalized):
            raise ValueError("selected_holdings must not contain duplicates")
        return normalized

    @field_validator("target_weights")
    @classmethod
    def normalize_target_weights(
        cls, values: dict[str, Decimal] | None
    ) -> dict[str, Decimal] | None:
        if values is None:
            return None
        normalized: dict[str, Decimal] = {}
        for raw_listing, weight in values.items():
            listing = raw_listing.strip().upper()
            if not re.fullmatch(r"(NSE|BSE):[A-Z0-9][A-Z0-9&.-]{0,31}", listing):
                raise ValueError(
                    "target_weights keys must use NSE:SYMBOL or BSE:SYMBOL"
                )
            if listing in normalized:
                raise ValueError(f"Duplicate target listing {listing}")
            normalized[listing] = weight
        return normalized

    @model_validator(mode="after")
    def require_custom_weights_for_cash_target(self):
        if self.target_cash_weight is not None and self.target_weights is None:
            raise ValueError("target_cash_weight requires custom target_weights")
        return self


class PortfolioRebalanceTradeResponse(BaseModel):
    listing: str
    exchange: Exchange
    symbol: str
    side: str
    quantity: float
    reference_price: Decimal
    execution_price: Decimal
    reference_notional: Decimal
    execution_notional: Decimal
    fee: Decimal
    slippage_cost: Decimal
    reason: str


class RecommendationExplanation(BaseModel):
    code: str
    message: str
    requested_value: float | None = None
    applied_value: float | None = None


class PortfolioConstraintEvent(BaseModel):
    code: str
    message: str
    listing: str | None = None
    requested_value: float | None = None
    applied_value: float | None = None


class PortfolioHoldingRecommendationResponse(BaseModel):
    listing: str
    exchange: Exchange
    symbol: str
    current_quantity: float
    current_market_value: Decimal | None
    requested_target_weight: float | None
    target_weight: float | None
    target_value: Decimal | None
    simulated_final_quantity: float | None
    simulated_final_value: Decimal | None
    action: Literal["BUY", "SELL", "HOLD", "UNAVAILABLE"]
    execution_status: Literal[
        "EXECUTED",
        "PARTIALLY_EXECUTED",
        "NOT_EXECUTED",
        "NOT_APPLICABLE",
    ]
    reason_codes: list[str]
    explanations: list[RecommendationExplanation]
    estimated_fees: Decimal
    estimated_slippage: Decimal


class PortfolioRebalancePreviewResponse(BaseModel):
    portfolio_id: str
    currency: str
    preview_available: bool
    allocation_source: str
    allocation_status: Literal["COMPLETE", "UNAVAILABLE"]
    allocation_unavailable_reason: str | None
    selected_holdings: list[str]
    user_target_weights: dict[str, float] | None
    user_target_cash_weight: float | None
    requested_target_weights: dict[str, float] | None
    requested_target_cash_weight: float | None
    adjusted_target_weights: dict[str, float] | None
    adjusted_target_cash_weight: float | None
    target_cash_weight: float | None
    portfolio_cash_balance: Decimal
    additional_investment: Decimal
    cash_available_for_preview: Decimal
    initial_nav: Decimal | None
    final_nav: Decimal | None
    status: str | None
    rejection_reasons: list[str]
    market_prices: list[dict]
    price_warnings: list[dict]
    proposed_trades: list[PortfolioRebalanceTradeResponse]
    recommendations: list[PortfolioHoldingRecommendationResponse]
    estimated_transaction_costs: dict[str, Decimal]
    final_simulated_holdings: dict[str, float]
    final_simulated_weights: dict[str, float] | None
    final_simulated_cash_weight: float | None
    final_simulated_cash_balance: Decimal | None
    constraint_audit_events: list[PortfolioConstraintEvent]
    portfolio_constraint_events: list[PortfolioConstraintEvent]
    allocation_adjustment_events: list[PortfolioConstraintEvent]
    gross_turnover: Decimal | None
    turnover_ratio: float | None
    freshness_policy: dict[str, str | int]


def _owned_portfolio(db: Session, user: User, portfolio_id: str) -> Portfolio:
    portfolio = db.scalar(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user.id,
        )
    )
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found",
        )
    return portfolio


def _owned_holding(
    db: Session,
    user: User,
    portfolio_id: str,
    holding_id: str,
) -> Holding:
    holding = db.scalar(
        select(Holding)
        .join(Portfolio, Holding.portfolio_id == Portfolio.id)
        .where(
            Holding.id == holding_id,
            Holding.portfolio_id == portfolio_id,
            Portfolio.user_id == user.id,
        )
    )
    if holding is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Holding not found",
        )
    return holding


def _owned_transaction(
    db: Session,
    user: User,
    portfolio_id: str,
    transaction_id: str,
) -> Transaction:
    transaction = db.scalar(
        select(Transaction)
        .join(Portfolio, Transaction.portfolio_id == Portfolio.id)
        .where(
            Transaction.id == transaction_id,
            Transaction.portfolio_id == portfolio_id,
            Portfolio.user_id == user.id,
        )
    )
    if transaction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found",
        )
    return transaction


@router.post(
    "/api/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_user(request: RegisterRequest, db: Session = Depends(get_db)):
    user = User(email=request.email, password_hash=hash_password(request.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        ) from error
    db.refresh(user)
    return user


@router.post("/api/auth/login", response_model=TokenResponse)
def login_user(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == request.email))
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        token, expires_in = create_access_token(user.id)
    except AuthenticationConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        ) from error
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse.model_validate(user),
    )


@router.get("/api/auth/me", response_model=UserResponse)
def read_current_user(user: User = Depends(get_current_user)):
    return user


@router.post(
    "/api/portfolios",
    response_model=PortfolioResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_portfolio(
    request: PortfolioCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    portfolio = Portfolio(
        user_id=user.id,
        name=request.name,
        cash_balance=request.cash_balance,
    )
    db.add(portfolio)
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A portfolio with this name already exists",
        ) from error
    db.refresh(portfolio)
    return portfolio


@router.get("/api/portfolios", response_model=list[PortfolioResponse])
def list_portfolios(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return list(
        db.scalars(
            select(Portfolio)
            .where(Portfolio.user_id == user.id)
            .order_by(Portfolio.created_at, Portfolio.id)
        )
    )


@router.get("/api/portfolios/{portfolio_id}", response_model=PortfolioResponse)
def read_portfolio(
    portfolio_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return _owned_portfolio(db, user, portfolio_id)


@router.get(
    "/api/portfolios/{portfolio_id}/summary",
    response_model=PortfolioSummaryResponse,
)
def portfolio_summary(
    portfolio_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    portfolio = _owned_portfolio(db, user, portfolio_id)
    holdings = list(
        db.scalars(
            select(Holding)
            .where(Holding.portfolio_id == portfolio_id)
            .order_by(Holding.exchange, Holding.symbol)
        )
    )
    total_invested = sum(
        (
            holding.quantity * holding.average_purchase_price
            for holding in holdings
        ),
        Decimal("0"),
    )
    exchange_counts = {"NSE": 0, "BSE": 0}
    for holding in holdings:
        exchange_counts[holding.exchange] += 1
    return PortfolioSummaryResponse(
        portfolio_id=portfolio.id,
        currency=portfolio.base_currency,
        holdings_count=len(holdings),
        total_invested_amount=total_invested,
        cash_balance=portfolio.cash_balance,
        holdings_by_exchange=exchange_counts,
        valuation_note=(
            "Invested amount is quantity multiplied by average purchase price; "
            "current market value and profit/loss are not calculated."
        ),
    )


@router.patch("/api/portfolios/{portfolio_id}", response_model=PortfolioResponse)
def update_portfolio(
    portfolio_id: str,
    request: PortfolioUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    portfolio = _owned_portfolio(db, user, portfolio_id)
    for field, value in request.model_dump(exclude_unset=True).items():
        setattr(portfolio, field, value)
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A portfolio with this name already exists",
        ) from error
    db.refresh(portfolio)
    return portfolio


@router.delete("/api/portfolios/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_portfolio(
    portfolio_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    portfolio = _owned_portfolio(db, user, portfolio_id)
    db.delete(portfolio)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/api/portfolios/{portfolio_id}/holdings",
    response_model=HoldingResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_holding(
    portfolio_id: str,
    request: HoldingCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _owned_portfolio(db, user, portfolio_id)
    holding = Holding(
        portfolio_id=portfolio_id,
        exchange=request.exchange.value,
        symbol=request.symbol,
        quantity=request.quantity,
        average_purchase_price=request.average_purchase_price,
    )
    db.add(holding)
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This exchange and symbol already exists in the portfolio",
        ) from error
    db.refresh(holding)
    return holding


@router.get(
    "/api/portfolios/{portfolio_id}/holdings",
    response_model=list[HoldingResponse],
)
def list_holdings(
    portfolio_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _owned_portfolio(db, user, portfolio_id)
    return list(
        db.scalars(
            select(Holding)
            .where(Holding.portfolio_id == portfolio_id)
            .order_by(Holding.exchange, Holding.symbol)
        )
    )


@router.post(
    "/api/portfolios/{portfolio_id}/holdings/import",
    response_model=HoldingImportResponse,
    status_code=status.HTTP_201_CREATED,
)
def import_holdings_csv(
    portfolio_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _owned_portfolio(db, user, portfolio_id)
    filename = file.filename or ""
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if (
        not filename.lower().endswith(".csv")
        or content_type not in _CSV_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a .csv file with a CSV content type",
        )

    content = file.file.read(_MAX_CSV_BYTES + 1)
    if len(content) > _MAX_CSV_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"CSV file exceeds the {_MAX_CSV_BYTES}-byte limit",
        )
    try:
        csv_text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="CSV file must use UTF-8 encoding",
        ) from error

    reader = csv.DictReader(io.StringIO(csv_text, newline=""))
    if reader.fieldnames is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="CSV header is missing",
        )
    normalized_headers = [header.strip().lower() for header in reader.fieldnames]
    if len(normalized_headers) != len(set(normalized_headers)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="CSV contains duplicate column names",
        )
    missing_columns = sorted(set(_CSV_COLUMNS) - set(normalized_headers))
    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "CSV is missing required columns",
                "missing_columns": missing_columns,
                "required_columns": list(_CSV_COLUMNS),
            },
        )
    if set(normalized_headers) != set(_CSV_COLUMNS):
        extra_columns = sorted(set(normalized_headers) - set(_CSV_COLUMNS))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "CSV contains unsupported columns",
                "extra_columns": extra_columns,
                "required_columns": list(_CSV_COLUMNS),
            },
        )
    reader.fieldnames = normalized_headers

    parsed_holdings: list[Holding] = []
    seen_listings: dict[tuple[str, str], int] = {}
    row_errors: list[dict[str, int | str]] = []
    try:
        for row_number, row in enumerate(reader, start=2):
            if row is None:
                continue
            if None in row:
                row_errors.append(
                    {
                        "row": row_number,
                        "error": "Row has a different number of values than the header",
                    }
                )
                continue
            if all(value is None or not value.strip() for value in row.values()):
                continue
            if any(row.get(column) is None for column in _CSV_COLUMNS):
                row_errors.append(
                    {
                        "row": row_number,
                        "error": "Row has a different number of values than the header",
                    }
                )
                continue
            try:
                validated = HoldingCreate(
                    exchange=row["exchange"],
                    symbol=row["symbol"],
                    quantity=row["quantity"],
                    average_purchase_price=row["average_price"],
                )
            except ValidationError as error:
                row_errors.append(
                    {
                        "row": row_number,
                        "error": "; ".join(
                            issue["msg"] for issue in error.errors()
                        ),
                    }
                )
                continue

            listing = (validated.exchange.value, validated.symbol)
            if listing in seen_listings:
                row_errors.append(
                    {
                        "row": row_number,
                        "error": (
                            "Duplicate listing in CSV; first occurrence is "
                            f"row {seen_listings[listing]}"
                        ),
                    }
                )
                continue
            seen_listings[listing] = row_number
            parsed_holdings.append(
                Holding(
                    portfolio_id=portfolio_id,
                    exchange=validated.exchange.value,
                    symbol=validated.symbol,
                    quantity=validated.quantity,
                    average_purchase_price=validated.average_purchase_price,
                )
            )
    except csv.Error as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Malformed CSV near row {reader.line_num + 1}: {error}",
        ) from error

    if not parsed_holdings and not row_errors:
        row_errors.append({"row": 0, "error": "CSV contains no holdings"})

    existing_listings = set(
        db.execute(
            select(Holding.exchange, Holding.symbol).where(
                Holding.portfolio_id == portfolio_id
            )
        ).all()
    )
    for listing, row_number in seen_listings.items():
        if listing in existing_listings:
            row_errors.append(
                {
                    "row": row_number,
                    "error": (
                        f"{listing[0]}:{listing[1]} already exists in this portfolio"
                    ),
                }
            )

    if row_errors:
        conflict = any("already exists in this portfolio" in str(item["error"]) for item in row_errors)
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
                if conflict
                else status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail={"message": "No holdings were imported", "errors": row_errors},
        )

    db.add_all(parsed_holdings)
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A holding conflicts with an existing portfolio listing",
        ) from error
    for holding in parsed_holdings:
        db.refresh(holding)
    return HoldingImportResponse(
        imported_count=len(parsed_holdings),
        holdings=parsed_holdings,
    )


@router.patch(
    "/api/portfolios/{portfolio_id}/holdings/{holding_id}",
    response_model=HoldingResponse,
)
def update_holding(
    portfolio_id: str,
    holding_id: str,
    request: HoldingUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    holding = _owned_holding(db, user, portfolio_id, holding_id)
    for field, value in request.model_dump(exclude_unset=True).items():
        setattr(holding, field, value.value if isinstance(value, Enum) else value)
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This exchange and symbol already exists in the portfolio",
        ) from error
    db.refresh(holding)
    return holding


@router.delete(
    "/api/portfolios/{portfolio_id}/holdings/{holding_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_holding(
    portfolio_id: str,
    holding_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    holding = _owned_holding(db, user, portfolio_id, holding_id)
    db.delete(holding)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/api/portfolios/{portfolio_id}/transactions",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_transaction(
    portfolio_id: str,
    request: TransactionCreate,
    response: Response,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _owned_portfolio(db, user, portfolio_id)
    try:
        result = post_transaction(
            db,
            portfolio_id,
            transaction_type=request.transaction_type.value,
            transaction_date=request.transaction_date,
            exchange=request.exchange.value if request.exchange is not None else None,
            symbol=request.symbol,
            quantity=request.quantity,
            unit_price=request.unit_price,
            cash_amount=request.cash_amount,
            fees=request.fees,
            idempotency_key=request.idempotency_key,
        )
        db.commit()
        transaction = result.transaction
        if result.replayed:
            response.status_code = status.HTTP_200_OK
    except LedgerError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except IntegrityError as error:
        db.rollback()
        if request.idempotency_key is not None:
            try:
                transaction = find_idempotent_transaction(
                    db,
                    portfolio_id,
                    request.idempotency_key,
                    transaction_type=request.transaction_type.value,
                    transaction_date=request.transaction_date,
                    exchange=request.exchange.value if request.exchange is not None else None,
                    symbol=request.symbol,
                    quantity=request.quantity,
                    unit_price=request.unit_price,
                    cash_amount=request.cash_amount,
                    fees=request.fees,
                )
            except LedgerError as conflict:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(conflict),
                ) from error
            if transaction is not None:
                response.status_code = status.HTTP_200_OK
                db.refresh(transaction)
                return transaction
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Transaction conflicts with current portfolio data",
        ) from error
    except SQLAlchemyError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction could not be persisted; retry later",
        ) from error
    db.refresh(transaction)
    return transaction


@router.get(
    "/api/portfolios/{portfolio_id}/reconciliation",
    response_model=ReconciliationResponse,
)
def get_portfolio_reconciliation(
    portfolio_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _owned_portfolio(db, user, portfolio_id)
    try:
        return reconcile_portfolio(db, portfolio_id)
    except LedgerError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error


@router.get(
    "/api/portfolios/{portfolio_id}/valuation",
    response_model=PortfolioValuationResponse,
)
def get_portfolio_valuation(
    portfolio_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    market_db: Session = Depends(get_market_db),
):
    _owned_portfolio(db, user, portfolio_id)
    try:
        return value_portfolio(
            db,
            market_db,
            user.id,
            portfolio_id,
        )
    except PortfolioNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except MarketDataStorageUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@router.post(
    "/api/portfolios/{portfolio_id}/rebalance-preview",
    response_model=PortfolioRebalancePreviewResponse,
)
def create_portfolio_rebalance_preview(
    portfolio_id: str,
    request: PortfolioRebalancePreviewRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    market_db: Session = Depends(get_market_db),
):
    _owned_portfolio(db, user, portfolio_id)
    try:
        return preview_portfolio_rebalance(
            db,
            market_db,
            user.id,
            portfolio_id,
            **request.model_dump(),
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except MarketDataStorageUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@router.post(
    "/api/portfolios/{portfolio_id}/monitor",
    response_model=PortfolioMonitoringReportResponse,
)
def create_portfolio_monitoring_report(
    portfolio_id: str,
    request: PortfolioMonitoringRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    market_db: Session = Depends(get_market_db),
):
    _owned_portfolio(db, user, portfolio_id)
    try:
        report = monitor_portfolio(
            db,
            market_db,
            user.id,
            portfolio_id,
            request.session_date,
        )
        return _monitoring_report_response(report)
    except PortfolioNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error


@router.get(
    "/api/portfolios/{portfolio_id}/monitor/reports",
    response_model=list[PortfolioMonitoringReportResponse],
)
def list_portfolio_monitoring_reports(
    portfolio_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _owned_portfolio(db, user, portfolio_id)
    reports = list(
        db.scalars(
            select(PortfolioMonitoringReport)
            .where(PortfolioMonitoringReport.portfolio_id == portfolio_id)
            .order_by(
                PortfolioMonitoringReport.session_date.desc(),
                PortfolioMonitoringReport.evaluated_at.desc(),
            )
            .limit(limit)
        )
    )
    return [_monitoring_report_response(report) for report in reports]


@router.get(
    "/api/portfolios/{portfolio_id}/monitor/reports/{report_id}",
    response_model=PortfolioMonitoringReportResponse,
)
def read_portfolio_monitoring_report(
    portfolio_id: str,
    report_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _owned_portfolio(db, user, portfolio_id)
    report = db.scalar(
        select(PortfolioMonitoringReport).where(
            PortfolioMonitoringReport.id == report_id,
            PortfolioMonitoringReport.portfolio_id == portfolio_id,
        )
    )
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Monitoring report not found",
        )
    return _monitoring_report_response(report)


@router.get(
    "/api/portfolios/{portfolio_id}/transactions",
    response_model=list[TransactionResponse],
)
def list_transactions(
    portfolio_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _owned_portfolio(db, user, portfolio_id)
    return list(
        db.scalars(
            select(Transaction)
            .where(Transaction.portfolio_id == portfolio_id)
            .order_by(Transaction.transaction_date.desc(), Transaction.created_at.desc())
        )
    )


@router.get(
    "/api/portfolios/{portfolio_id}/transactions/{transaction_id}",
    response_model=TransactionResponse,
)
def read_transaction(
    portfolio_id: str,
    transaction_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return _owned_transaction(db, user, portfolio_id, transaction_id)
