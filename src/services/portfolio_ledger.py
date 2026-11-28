"""Atomic transaction posting and read-only portfolio reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy import select, update
from sqlalchemy.orm import Session, selectinload

from src.database.models import (
    Holding,
    Portfolio,
    PortfolioLedgerBaseline,
    PortfolioLedgerBaselineHolding,
    Transaction,
)

_CENT = Decimal("0.01")
_SHARE = Decimal("0.000001")
_ZERO = Decimal("0")


class LedgerError(ValueError):
    """A transaction is validly formed but cannot be posted to current state."""


@dataclass(frozen=True)
class ReconciliationReport:
    is_reconciled: bool
    baseline_timestamp: datetime
    expected_cash: Decimal
    actual_cash: Decimal
    cash_difference: Decimal
    expected_holdings: list[dict[str, Decimal | str]]
    actual_holdings: list[dict[str, Decimal | str]]
    holding_discrepancies: list[dict[str, Decimal | str | bool]]


@dataclass(frozen=True)
class TransactionPostResult:
    transaction: Transaction
    replayed: bool


def _money(amount: Decimal) -> Decimal:
    return amount.quantize(_CENT, rounding=ROUND_HALF_UP)


def _shares(amount: Decimal) -> Decimal:
    return amount.quantize(_SHARE, rounding=ROUND_HALF_UP)


def post_transaction(
    db: Session,
    portfolio_id: str,
    *,
    transaction_type: str,
    transaction_date: date,
    exchange: str | None,
    symbol: str | None,
    quantity: Decimal | None,
    unit_price: Decimal | None,
    cash_amount: Decimal | None,
    fees: Decimal,
    idempotency_key: str | None = None,
) -> TransactionPostResult:
    """Apply one ledger transaction; the caller commits or rolls back."""
    statement = (
        select(Portfolio)
        .where(Portfolio.id == portfolio_id)
        .options(
            selectinload(Portfolio.holdings),
            selectinload(Portfolio.ledger_baseline),
        )
    )
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        statement = statement.with_for_update()
    portfolio = db.scalar(statement)
    if portfolio is None:
        raise LedgerError("Portfolio is not available")

    if idempotency_key is not None:
        existing_transaction = db.scalar(
            select(Transaction).where(
                Transaction.portfolio_id == portfolio_id,
                Transaction.idempotency_key == idempotency_key,
            )
        )
        if existing_transaction is not None:
            if not transaction_matches_payload(
                existing_transaction,
                transaction_type=transaction_type,
                transaction_date=transaction_date,
                exchange=exchange,
                symbol=symbol,
                quantity=quantity,
                unit_price=unit_price,
                cash_amount=cash_amount,
                fees=fees,
            ):
                raise LedgerError(
                    "Idempotency key was already used with a different transaction"
                )
            return TransactionPostResult(existing_transaction, replayed=True)

    now = datetime.now(timezone.utc)
    latest_posted_at = db.scalar(
        select(Transaction.posted_at)
        .where(
            Transaction.portfolio_id == portfolio_id,
            Transaction.posted_at.is_not(None),
        )
        .order_by(Transaction.posted_at.desc())
        .limit(1)
    )
    if latest_posted_at is not None:
        if latest_posted_at.tzinfo is None:
            latest_posted_at = latest_posted_at.replace(tzinfo=timezone.utc)
        now = max(now, latest_posted_at + timedelta(microseconds=1))

    current_cash = portfolio.cash_balance
    updated_cash = current_cash
    existing_holding = None

    if transaction_type == "BUY":
        assert exchange is not None and symbol is not None
        assert quantity is not None and unit_price is not None
        gross_cost = quantity * unit_price
        debit = _money(gross_cost + fees)
        if debit > current_cash:
            raise LedgerError("Insufficient cash for this BUY transaction")
        existing_holding = next(
            (
                holding
                for holding in portfolio.holdings
                if holding.exchange == exchange and holding.symbol == symbol
            ),
            None,
        )
        prior_quantity = existing_holding.quantity if existing_holding else _ZERO
        prior_cost = (
            prior_quantity * existing_holding.average_purchase_price
            if existing_holding
            else _ZERO
        )
        new_quantity = _shares(prior_quantity + quantity)
        new_average = _shares((prior_cost + gross_cost + fees) / new_quantity)
        updated_cash = _money(current_cash - debit)
    elif transaction_type == "SELL":
        assert exchange is not None and symbol is not None
        assert quantity is not None and unit_price is not None
        existing_holding = next(
            (
                holding
                for holding in portfolio.holdings
                if holding.exchange == exchange and holding.symbol == symbol
            ),
            None,
        )
        if existing_holding is None or quantity > existing_holding.quantity:
            raise LedgerError("Insufficient shares for this SELL transaction")
        net_proceeds = _money(quantity * unit_price - fees)
        if net_proceeds < _ZERO:
            raise LedgerError("SELL fees cannot exceed gross proceeds")
        updated_cash = _money(current_cash + net_proceeds)
    elif transaction_type == "CASH_DEPOSIT":
        assert cash_amount is not None
        updated_cash = _money(current_cash + cash_amount)
    elif transaction_type == "CASH_WITHDRAWAL":
        assert cash_amount is not None
        withdrawal = _money(cash_amount)
        if withdrawal > current_cash:
            raise LedgerError("Insufficient cash for this withdrawal")
        updated_cash = _money(current_cash - withdrawal)
    else:
        raise LedgerError("Unsupported transaction type")

    transaction = Transaction(
        portfolio_id=portfolio.id,
        transaction_type=transaction_type,
        transaction_date=transaction_date,
        exchange=exchange,
        symbol=symbol,
        quantity=quantity,
        unit_price=unit_price,
        cash_amount=cash_amount,
        fees=fees,
        idempotency_key=idempotency_key,
        posted_at=now,
    )
    db.add(transaction)
    db.flush()

    if portfolio.ledger_baseline is None:
        baseline = PortfolioLedgerBaseline(
            portfolio_id=portfolio.id,
            captured_at=now,
            cash_balance=current_cash,
            holdings=[
                PortfolioLedgerBaselineHolding(
                    exchange=holding.exchange,
                    symbol=holding.symbol,
                    quantity=holding.quantity,
                    average_purchase_price=holding.average_purchase_price,
                )
                for holding in portfolio.holdings
            ],
        )
        db.add(baseline)
        db.flush()

    # This compare-and-set serializes posting on SQLite as well as databases
    # that support row locks, and detects a stale balance before writing shares.
    changed = db.execute(
        update(Portfolio)
        .where(
            Portfolio.id == portfolio.id,
            Portfolio.cash_balance == current_cash,
        )
        .values(cash_balance=updated_cash)
    ).rowcount
    if changed != 1:
        raise LedgerError("Portfolio changed during posting; retry the transaction")
    portfolio.cash_balance = updated_cash

    if transaction_type == "BUY":
        if existing_holding is None:
            db.add(
                Holding(
                    portfolio_id=portfolio.id,
                    exchange=exchange,
                    symbol=symbol,
                    quantity=new_quantity,
                    average_purchase_price=new_average,
                )
            )
        else:
            existing_holding.quantity = new_quantity
            existing_holding.average_purchase_price = new_average
    elif transaction_type == "SELL":
        remaining_quantity = _shares(existing_holding.quantity - quantity)
        if remaining_quantity == _ZERO:
            db.delete(existing_holding)
        else:
            existing_holding.quantity = remaining_quantity

    db.flush()
    return TransactionPostResult(transaction, replayed=False)


def transaction_matches_payload(
    transaction: Transaction,
    *,
    transaction_type: str,
    transaction_date: date,
    exchange: str | None,
    symbol: str | None,
    quantity: Decimal | None,
    unit_price: Decimal | None,
    cash_amount: Decimal | None,
    fees: Decimal,
) -> bool:
    return (
        transaction.transaction_type == transaction_type
        and transaction.transaction_date == transaction_date
        and transaction.exchange == exchange
        and transaction.symbol == symbol
        and transaction.quantity == quantity
        and transaction.unit_price == unit_price
        and transaction.cash_amount == cash_amount
        and transaction.fees == fees
    )


def find_idempotent_transaction(
    db: Session,
    portfolio_id: str,
    idempotency_key: str,
    *,
    transaction_type: str,
    transaction_date: date,
    exchange: str | None,
    symbol: str | None,
    quantity: Decimal | None,
    unit_price: Decimal | None,
    cash_amount: Decimal | None,
    fees: Decimal,
) -> Transaction | None:
    transaction = db.scalar(
        select(Transaction).where(
            Transaction.portfolio_id == portfolio_id,
            Transaction.idempotency_key == idempotency_key,
        )
    )
    if transaction is None:
        return None
    if not transaction_matches_payload(
        transaction,
        transaction_type=transaction_type,
        transaction_date=transaction_date,
        exchange=exchange,
        symbol=symbol,
        quantity=quantity,
        unit_price=unit_price,
        cash_amount=cash_amount,
        fees=fees,
    ):
        raise LedgerError(
            "Idempotency key was already used with a different transaction"
        )
    return transaction


def reconcile_portfolio(db: Session, portfolio_id: str) -> ReconciliationReport:
    """Compare the immutable opening baseline plus posted rows with current state."""
    baseline = db.scalar(
        select(PortfolioLedgerBaseline)
        .where(PortfolioLedgerBaseline.portfolio_id == portfolio_id)
        .options(selectinload(PortfolioLedgerBaseline.holdings))
    )
    portfolio = db.get(Portfolio, portfolio_id)
    if portfolio is None:
        raise LedgerError("Portfolio is not available")
    if baseline is None:
        raise LedgerError(
            "No opening-balance baseline exists; it is captured when the first "
            "transaction is posted"
        )

    expected_cash = baseline.cash_balance
    expected: dict[tuple[str, str], tuple[Decimal, Decimal]] = {
        (holding.exchange, holding.symbol): (
            holding.quantity,
            holding.average_purchase_price,
        )
        for holding in baseline.holdings
    }
    posted_transactions = db.scalars(
        select(Transaction)
        .where(
            Transaction.portfolio_id == portfolio_id,
            Transaction.posted_at.is_not(None),
            Transaction.posted_at >= baseline.captured_at,
        )
        .order_by(Transaction.posted_at, Transaction.id)
    ).all()

    for transaction in posted_transactions:
        if transaction.transaction_type == "BUY":
            key = (transaction.exchange, transaction.symbol)
            prior_quantity, prior_average = expected.get(key, (_ZERO, _ZERO))
            gross_cost = transaction.quantity * transaction.unit_price
            new_quantity = _shares(prior_quantity + transaction.quantity)
            expected[key] = (
                new_quantity,
                _shares(
                    (
                        prior_quantity * prior_average
                        + gross_cost
                        + transaction.fees
                    )
                    / new_quantity
                ),
            )
            expected_cash = _money(
                expected_cash - _money(gross_cost + transaction.fees)
            )
        elif transaction.transaction_type == "SELL":
            key = (transaction.exchange, transaction.symbol)
            prior_quantity, prior_average = expected[key]
            remaining = _shares(prior_quantity - transaction.quantity)
            if remaining == _ZERO:
                del expected[key]
            else:
                expected[key] = (remaining, prior_average)
            expected_cash = _money(
                expected_cash
                + _money(transaction.quantity * transaction.unit_price - transaction.fees)
            )
        elif transaction.transaction_type == "CASH_DEPOSIT":
            expected_cash = _money(expected_cash + transaction.cash_amount)
        elif transaction.transaction_type == "CASH_WITHDRAWAL":
            expected_cash = _money(expected_cash - transaction.cash_amount)

    actual = {
        (holding.exchange, holding.symbol): (
            holding.quantity,
            holding.average_purchase_price,
        )
        for holding in db.scalars(
            select(Holding)
            .where(Holding.portfolio_id == portfolio_id)
            .order_by(Holding.exchange, Holding.symbol)
        ).all()
    }
    expected_rows = [
        {
            "exchange": exchange,
            "symbol": symbol,
            "quantity": quantity,
            "average_purchase_price": average,
        }
        for (exchange, symbol), (quantity, average) in sorted(expected.items())
    ]
    actual_rows = [
        {
            "exchange": exchange,
            "symbol": symbol,
            "quantity": quantity,
            "average_purchase_price": average,
        }
        for (exchange, symbol), (quantity, average) in sorted(actual.items())
    ]

    discrepancies = []
    for key in sorted(expected.keys() | actual.keys()):
        expected_position = expected.get(key)
        actual_position = actual.get(key)
        quantity_difference = (
            (actual_position[0] if actual_position else _ZERO)
            - (expected_position[0] if expected_position else _ZERO)
        )
        average_difference = (
            (actual_position[1] - expected_position[1])
            if actual_position is not None and expected_position is not None
            else None
        )
        if quantity_difference != _ZERO or (
            average_difference is not None and average_difference != _ZERO
        ):
            discrepancies.append(
                {
                    "exchange": key[0],
                    "symbol": key[1],
                    "expected_quantity": (
                        expected_position[0] if expected_position else _ZERO
                    ),
                    "actual_quantity": actual_position[0] if actual_position else _ZERO,
                    "quantity_difference": quantity_difference,
                    "expected_average_purchase_price": (
                        expected_position[1] if expected_position else None
                    ),
                    "actual_average_purchase_price": (
                        actual_position[1] if actual_position else None
                    ),
                    "average_purchase_price_difference": average_difference,
                    "missing": expected_position is not None and actual_position is None,
                    "unexpected": expected_position is None and actual_position is not None,
                }
            )

    cash_difference = _money(portfolio.cash_balance - expected_cash)
    return ReconciliationReport(
        is_reconciled=cash_difference == _ZERO and not discrepancies,
        baseline_timestamp=baseline.captured_at,
        expected_cash=expected_cash,
        actual_cash=portfolio.cash_balance,
        cash_difference=cash_difference,
        expected_holdings=expected_rows,
        actual_holdings=actual_rows,
        holding_discrepancies=discrepancies,
    )
