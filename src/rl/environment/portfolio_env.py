"""Custom Gymnasium environment for long-only portfolio rebalancing research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.strategies.allocation import (
    CorporateActionEvent,
    CorporateActionLedger,
    PlanStatus,
    RebalanceConfig,
    RebalanceRequest,
    rebalance,
)

ASSET_UNIVERSE = ("SPY", "QQQ", "IEF", "GLD", "XLU", "XLV")
BENCHMARK_SYMBOL = "^GSPC"
VIX_SYMBOL = "^VIX"
MARKET_SYMBOLS = ASSET_UNIVERSE + (BENCHMARK_SYMBOL, VIX_SYMBOL)

WARMUP_START = pd.Timestamp("2017-01-03")
WARMUP_END = pd.Timestamp("2018-01-02")
TRAIN_START = pd.Timestamp("2018-01-03")
TRAIN_END = pd.Timestamp("2021-12-31")
VALIDATION_START = pd.Timestamp("2022-01-03")
VALIDATION_END = pd.Timestamp("2022-12-30")
TEST_START = pd.Timestamp("2023-01-03")
TEST_END = pd.Timestamp("2026-11-11")

PARTITION_BOUNDS = {
    "train": (TRAIN_START, TRAIN_END),
    "validation": (VALIDATION_START, VALIDATION_END),
    "test": (TEST_START, TEST_END),
}

ASSET_FEATURES = (
    ("return_1", 1),
    ("return_5", 5),
    ("return_21", 21),
    ("return_63", 63),
    ("volatility_21", 21),
    ("volatility_63", 63),
    ("distance_sma_21", 21),
    ("distance_sma_63", 63),
)
MARKET_FEATURE_NAMES = tuple(
    f"{symbol}_{feature}"
    for symbol in ASSET_UNIVERSE
    for feature, _ in ASSET_FEATURES
) + ("benchmark_return_1", "benchmark_volatility_21", "vix_level_over_100")
OBSERVATION_SIZE = len(MARKET_FEATURE_NAMES) + len(ASSET_UNIVERSE) + 1 + 1 + 2
REQUIRED_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


@dataclass(frozen=True)
class TrainOnlyFeatureScaler:
    """Standard scaler fitted only on the fixed training date interval."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    fitted_start: str
    fitted_end: str

    @classmethod
    def fit(cls, features: pd.DataFrame) -> "TrainOnlyFeatureScaler":
        if tuple(features.columns) != MARKET_FEATURE_NAMES:
            raise ValueError("features do not match the fixed market feature order")
        training = features.loc[
            (features.index >= TRAIN_START) & (features.index <= TRAIN_END)
        ].dropna()
        if training.empty:
            raise ValueError("No complete feature rows in the training interval")
        values = training.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Training features contain non-finite values")
        means = values.mean(axis=0)
        scales = values.std(axis=0, ddof=0)
        scales[scales == 0.0] = 1.0
        return cls(
            feature_names=MARKET_FEATURE_NAMES,
            means=tuple(float(value) for value in means),
            scales=tuple(float(value) for value in scales),
            fitted_start=TRAIN_START.date().isoformat(),
            fitted_end=TRAIN_END.date().isoformat(),
        )

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if tuple(features.columns) != self.feature_names:
            raise ValueError("feature order does not match the fitted scaler")
        values = features.to_numpy(dtype=np.float64)
        normalized = (values - np.asarray(self.means)) / np.asarray(self.scales)
        if not np.isfinite(normalized).all():
            raise ValueError("Normalized features contain non-finite values")
        return normalized.astype(np.float32)


def load_market_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the fixed universe, benchmark and VIX CSV files from a directory."""
    root = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in MARKET_SYMBOLS:
        path = root / f"{symbol}_daily.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required market-data file not found: {path}")
        frame = pd.read_csv(path, parse_dates=["date"])
        frames[symbol] = frame.set_index("date")
    return frames


def fit_training_scaler(
    market_data: Mapping[str, pd.DataFrame],
) -> TrainOnlyFeatureScaler:
    """Fit the observation scaler on training rows only, never validation/test."""
    frames, dates = _normalize_market_data(market_data)
    features = _build_market_features(frames, dates)
    return TrainOnlyFeatureScaler.fit(features)


def project_to_simplex(action: np.ndarray) -> np.ndarray:
    """Project a finite one-dimensional vector onto the unit probability simplex."""
    vector = np.asarray(action, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("action must be a non-empty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise ValueError("action must contain only finite values")

    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    divisors = np.arange(1, vector.size + 1, dtype=np.float64)
    active = sorted_values - cumulative / divisors > 0.0
    rho = int(np.flatnonzero(active)[-1])
    threshold = cumulative[rho] / (rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    projected /= projected.sum()
    return projected


class PortfolioTradingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Periodic allocation environment with simulated next-session execution.

    The observation at session ``t`` uses market information through that
    session's close. An action is projected to asset-plus-cash weights, then
    the allocation engine simulates fills at session ``t+1`` open. Holdings are
    marked at the following close. Corporate actions are applied only from
    supplied normalized event records.
    """

    metadata = {"render_modes": [], "corporate_action_data_verified": False}

    def __init__(
        self,
        market_data: Mapping[str, pd.DataFrame],
        *,
        partition: str = "train",
        initial_cash: float = 100_000.0,
        initial_holdings: Optional[Mapping[str, float]] = None,
        allocation_config: Optional[RebalanceConfig] = None,
        corporate_action_events: Iterable[CorporateActionEvent] = (),
        corporate_actions_verified: bool = False,
        corporate_action_source: Optional[str] = None,
        scaler: Optional[TrainOnlyFeatureScaler] = None,
        random_start: bool = False,
    ) -> None:
        super().__init__()
        if partition not in PARTITION_BOUNDS:
            raise ValueError(f"partition must be one of {tuple(PARTITION_BOUNDS)}")
        if not math.isfinite(float(initial_cash)) or initial_cash < 0.0:
            raise ValueError("initial_cash must be non-negative and finite")
        if not isinstance(random_start, bool):
            raise ValueError("random_start must be a bool")
        if not isinstance(corporate_actions_verified, bool):
            raise ValueError("corporate_actions_verified must be a bool")
        if corporate_actions_verified and not (
            isinstance(corporate_action_source, str)
            and corporate_action_source.strip()
        ):
            raise ValueError(
                "corporate_action_source is required when events are verified"
            )

        self.partition = partition
        self.initial_cash = float(initial_cash)
        self.initial_holdings = _validate_initial_holdings(initial_holdings or {})
        unknown_holdings = sorted(set(self.initial_holdings) - set(ASSET_UNIVERSE))
        if unknown_holdings:
            raise ValueError(
                "Initial holdings are outside the fixed universe: "
                + ", ".join(unknown_holdings)
            )
        self.allocation_config = allocation_config or RebalanceConfig()
        self.allocation_config.validate()
        self.random_start = random_start
        self.corporate_actions_verified = corporate_actions_verified
        self.corporate_action_source = corporate_action_source

        self.market_data, self.dates = _normalize_market_data(market_data)
        self.market_features = _build_market_features(self.market_data, self.dates)
        if scaler is None:
            if partition != "train":
                raise ValueError(
                    "validation and test environments require a training-fitted scaler"
                )
            scaler = TrainOnlyFeatureScaler.fit(self.market_features)
        if (
            scaler.fitted_start != TRAIN_START.date().isoformat()
            or scaler.fitted_end != TRAIN_END.date().isoformat()
            or scaler.feature_names != MARKET_FEATURE_NAMES
        ):
            raise ValueError("scaler must be fitted on the fixed training interval")
        self.scaler = scaler

        self.scaled_market_features = np.full(
            (len(self.dates), len(MARKET_FEATURE_NAMES)), np.nan, dtype=np.float32
        )
        valid_features = self.market_features.notna().all(axis=1)
        if valid_features.any():
            self.scaled_market_features[valid_features.to_numpy()] = scaler.transform(
                self.market_features.loc[valid_features]
            )

        start, end = PARTITION_BOUNDS[partition]
        self.partition_dates = self.dates[(self.dates >= start) & (self.dates <= end)]
        if len(self.partition_dates) < 2:
            raise ValueError(f"Insufficient sessions in {partition} partition")
        positions = self.dates.get_indexer(self.partition_dates)
        if not np.isfinite(self.scaled_market_features[positions]).all():
            first_bad = self.partition_dates[
                ~np.isfinite(self.scaled_market_features[positions]).all(axis=1)
            ][0]
            raise ValueError(
                f"Missing or invalid market feature at {first_bad.date()} in {partition}"
            )

        self._date_positions = {day: index for index, day in enumerate(self.dates)}
        self._partition_positions = [
            self._date_positions[day] for day in self.partition_dates
        ]
        self._events = tuple(corporate_action_events)
        for event in self._events:
            if not isinstance(event, CorporateActionEvent):
                raise ValueError(
                    "corporate_action_events must contain CorporateActionEvent records"
                )
            if event.symbol not in ASSET_UNIVERSE:
                raise ValueError(
                    f"Corporate action symbol {event.symbol!r} is outside the fixed universe"
                )
            if event.currency != "USD":
                raise ValueError("the environment currently supports USD events only")

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(ASSET_UNIVERSE) + 1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._current_position = 0
        self._current_date: Optional[pd.Timestamp] = None
        self._current_nav = self.initial_cash
        self._initial_capital = self.initial_cash
        self._peak_nav = self.initial_cash
        self._ledger: Optional[CorporateActionLedger] = None
        self._active_events: tuple[CorporateActionEvent, ...] = ()
        self._episode_done = True

        self.metadata = {
            "render_modes": [],
            "corporate_action_data_verified": self.corporate_actions_verified,
            "corporate_action_source": self.corporate_action_source,
            "corporate_action_limitation": (
                None
                if self.corporate_actions_verified
                else "No verified historical corporate-action provider is configured."
            ),
            "starting_state_assumption": (
                "Initial holdings and settled cash are balances at the reset-date "
                "close; corporate actions dated on or before that date are not replayed."
            ),
            "return_accuracy": (
                "event-adjusted only for supplied, verified events"
                if self.corporate_actions_verified
                else "not verified as corporate-action-complete or total-return accurate"
            ),
            "asset_universe": ASSET_UNIVERSE,
            "partition": partition,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        options = options or {}
        unknown = set(options) - {"start_date"}
        if unknown:
            raise ValueError(f"Unsupported reset options: {sorted(unknown)}")

        if "start_date" in options:
            start_date = pd.Timestamp(options["start_date"]).normalize()
            if start_date not in self.partition_dates:
                raise ValueError("start_date must be a session in this partition")
            position = int(self.partition_dates.get_loc(start_date))
            if position >= len(self.partition_dates) - 1:
                raise ValueError("start_date has no next session in this partition")
        elif self.random_start:
            position = int(self.np_random.integers(0, len(self.partition_dates) - 1))
        else:
            position = 0

        self._current_position = position
        self._current_date = self.partition_dates[position]
        start_date = self._current_date.date()
        self._active_events = tuple(
            event for event in self._events if event.action_date > start_date
        )
        self._ledger = CorporateActionLedger(
            holdings=self.initial_holdings,
            settled_cash=self.initial_cash,
            allow_fractional_shares=self.allocation_config.allow_fractional_shares,
        )
        marks = self._prices_at(self._current_date, "close")
        snapshot = self._ledger.snapshot(marks)
        if not math.isfinite(snapshot.total_nav) or snapshot.total_nav <= 0.0:
            raise ValueError("Initial portfolio NAV must be positive and finite")
        self._current_nav = snapshot.total_nav
        self._initial_capital = snapshot.total_nav
        self._peak_nav = self._current_nav
        self._episode_done = False

        observation = self._observation(self._current_date, snapshot)
        info = self._state_info(self._current_date, snapshot)
        info["seed"] = seed
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        return self._step_transition(action, hold=False)

    def step_hold(self) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one session without changing holdings or submitting trades."""
        return self._step_transition(None, hold=True)

    def _step_transition(
        self, action: Optional[np.ndarray], *, hold: bool
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._episode_done or self._ledger is None or self._current_date is None:
            raise RuntimeError("reset() must be called before step() and after termination")

        action_was_clipped = False
        if hold:
            target_weights: dict[str, float] = {}
            target_cash_weight = None
        else:
            if action is None:
                raise ValueError("action must be a finite asset-plus-cash vector")
            raw_action = np.asarray(action, dtype=np.float64)
            expected_shape = (len(ASSET_UNIVERSE) + 1,)
            if raw_action.shape != expected_shape:
                raise ValueError(f"action must have shape {expected_shape}")
            if not np.isfinite(raw_action).all():
                raise ValueError("action must contain only finite values")
            clipped_action = np.clip(
                raw_action, self.action_space.low, self.action_space.high
            )
            action_was_clipped = not np.array_equal(clipped_action, raw_action)
            target = project_to_simplex(clipped_action)
            target_weights = {
                symbol: float(target[index])
                for index, symbol in enumerate(ASSET_UNIVERSE)
            }
            target_cash_weight = float(target[-1])

        next_position = self._current_position + 1
        next_date = self.partition_dates[next_position]
        current_date = self._current_date
        dates_to_process = {
            event_date
            for event in self._active_events
            for event_date in (event.action_date, event.payment_date)
            if event_date is not None
            and current_date.date() < event_date <= next_date.date()
        }
        dates_to_process.add(next_date.date())
        execution_prices = self._prices_at(next_date, "open")
        for event_date in sorted(dates_to_process):
            self._ledger.process_date(event_date, self._active_events)

        allocator_available_cash = self._ledger.settled_cash
        split_factors = {
            symbol: math.prod(
                float(event.split_ratio)
                for event in self._active_events
                if event.symbol == symbol
                and event.event_type.value == "split"
                and current_date.date() < event.action_date <= next_date.date()
            )
            for symbol in ASSET_UNIVERSE
        }
        initial_mark_prices = {
            symbol: float(self.market_data[symbol].loc[current_date, "close"])
            / split_factors[symbol]
            for symbol in ASSET_UNIVERSE
        }
        mark_prices = self._prices_at(next_date, "close")
        plan = None
        if not hold:
            plan = rebalance(
                RebalanceRequest(
                    current_holdings=self._ledger.holdings,
                    available_cash=self._ledger.settled_cash,
                    prices={},
                    target_weights=target_weights,
                    target_cash_weight=target_cash_weight,
                    config=self.allocation_config,
                    execution_prices=execution_prices,
                    initial_mark_prices=initial_mark_prices,
                    mark_prices=mark_prices,
                )
            )
        if plan is not None and plan.status is not PlanStatus.REJECTED:
            self._ledger.update_investable_balances(
                plan.final_holdings, plan.final_cash
            )

        snapshot = self._ledger.snapshot(mark_prices)
        if snapshot.total_nav <= 0.0 or not math.isfinite(snapshot.total_nav):
            raise RuntimeError("Portfolio NAV became non-positive or non-finite")
        reward = float(math.log(snapshot.total_nav / self._current_nav))
        self._peak_nav = max(self._peak_nav, snapshot.total_nav)
        self._current_nav = snapshot.total_nav
        self._current_position = next_position
        self._current_date = next_date

        terminated = next_position == len(self.partition_dates) - 1
        self._episode_done = terminated
        observation = self._observation(next_date, snapshot)
        info = self._state_info(next_date, snapshot)
        info.update(
            {
                "previous_date": current_date.date().isoformat(),
                "target_weights": target_weights,
                "target_cash_weight": target_cash_weight,
                "action_was_clipped": action_was_clipped,
                "allocation_status": (
                    plan.status.value if plan is not None else "held"
                ),
                "allocation_rejected": (
                    plan.status is PlanStatus.REJECTED if plan is not None else False
                ),
                "rejection_reasons": plan.rejection_reasons if plan is not None else (),
                "trades": plan.trades if plan is not None else (),
                "transaction_fees": plan.total_fees if plan is not None else 0.0,
                "slippage_cost": (
                    plan.total_slippage_cost if plan is not None else 0.0
                ),
                "turnover_ratio": plan.turnover_ratio if plan is not None else 0.0,
                "allocation_final_nav": (
                    plan.final_nav if plan is not None else snapshot.total_nav
                ),
                "allocator_available_cash": allocator_available_cash,
                "episode_terminated": terminated,
            }
        )
        return observation, reward, terminated, False, info

    def _prices_at(self, day: pd.Timestamp, column: str) -> dict[str, float]:
        return {
            symbol: float(self.market_data[symbol].loc[day, column])
            for symbol in ASSET_UNIVERSE
        }

    def _observation(self, day, snapshot) -> np.ndarray:
        index = self._date_positions[day]
        market = self.scaled_market_features[index]
        nav = snapshot.total_nav
        asset_weights = [
            snapshot.asset_quantities.get(symbol, 0.0)
            * float(self.market_data[symbol].loc[day, "close"])
            / nav
            for symbol in ASSET_UNIVERSE
        ]
        cash_weight = snapshot.settled_cash / nav
        receivable_weight = sum(snapshot.dividend_receivables.values()) / nav
        drawdown = self._current_nav / self._peak_nav - 1.0
        state = np.asarray(
            [
                *asset_weights,
                cash_weight,
                receivable_weight,
                drawdown,
                nav / self._initial_capital,
            ],
            dtype=np.float32,
        )
        observation = np.concatenate((market, state)).astype(np.float32, copy=False)
        if observation.shape != self.observation_space.shape:
            raise RuntimeError("Internal observation shape mismatch")
        if not np.isfinite(observation).all():
            raise RuntimeError("Observation contains non-finite values")
        return observation

    def _state_info(self, day, snapshot) -> dict:
        return {
            "date": day.date().isoformat(),
            "nav": snapshot.total_nav,
            "holdings": dict(snapshot.asset_quantities),
            "settled_cash": snapshot.settled_cash,
            "dividend_receivables": dict(snapshot.dividend_receivables),
            "corporate_actions_verified": self.corporate_actions_verified,
            "corporate_action_source": self.corporate_action_source,
            "corporate_action_limitation": self.metadata[
                "corporate_action_limitation"
            ],
            "starting_state_assumption": self.metadata["starting_state_assumption"],
            "return_accuracy": self.metadata["return_accuracy"],
        }


def _normalize_market_data(
    market_data: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    if not isinstance(market_data, Mapping):
        raise ValueError("market_data must map symbols to DataFrames")
    missing_symbols = sorted(set(MARKET_SYMBOLS) - set(market_data))
    if missing_symbols:
        raise ValueError("Missing required market symbols: " + ", ".join(missing_symbols))

    frames: dict[str, pd.DataFrame] = {}
    for symbol in MARKET_SYMBOLS:
        raw = market_data[symbol]
        if not isinstance(raw, pd.DataFrame):
            raise ValueError(f"market_data[{symbol!r}] must be a DataFrame")
        frame = raw.copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.set_index("date")
        try:
            dates = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid date index for {symbol}") from exc
        if dates.isna().any():
            raise ValueError(f"Invalid dates in {symbol} data")
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        dates = dates.normalize()
        if dates.has_duplicates:
            raise ValueError(f"Duplicate dates in {symbol} data")
        frame.index = dates
        frame = frame.sort_index()
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing_columns:
            raise ValueError(
                f"Missing required columns for {symbol}: {', '.join(missing_columns)}"
            )
        frame = frame.loc[:, list(REQUIRED_COLUMNS)]
        for column in REQUIRED_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Missing or non-finite OHLCV values in {symbol}")
        if (frame[["open", "high", "low", "close", "adj_close"]] <= 0.0).any().any():
            raise ValueError(f"Prices must be positive for {symbol}")
        if (frame["volume"] < 0.0).any():
            raise ValueError(f"Volume must be non-negative for {symbol}")
        invalid_ohlc = (
            (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
            | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        )
        if invalid_ohlc.any():
            raise ValueError(f"Inconsistent OHLC values for {symbol}")
        frames[symbol] = frame

    benchmark_dates = frames[BENCHMARK_SYMBOL].index
    common_start = max(frame.index.min() for frame in frames.values())
    common_end = min(frame.index.max() for frame in frames.values())
    dates = benchmark_dates[
        (benchmark_dates >= common_start) & (benchmark_dates <= common_end)
    ]
    if dates.empty:
        raise ValueError("Required market files have no common trading sessions")
    for symbol, frame in frames.items():
        missing = dates.difference(frame.index)
        if not missing.empty:
            raise ValueError(
                f"Missing executable/feature rows for {symbol} on "
                f"{missing[0].date()} (and {len(missing) - 1} more sessions)"
            )
        frames[symbol] = frame.loc[dates].copy()
    return frames, dates


def _build_market_features(
    frames: Mapping[str, pd.DataFrame], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    features = pd.DataFrame(index=dates)
    for symbol in ASSET_UNIVERSE:
        adjusted_close = frames[symbol]["adj_close"]
        daily_returns = adjusted_close.pct_change(fill_method=None)
        for name, period in ASSET_FEATURES:
            if name.startswith("return_"):
                features[f"{symbol}_{name}"] = adjusted_close.pct_change(
                    periods=period, fill_method=None
                )
            elif name.startswith("volatility_"):
                features[f"{symbol}_{name}"] = daily_returns.rolling(
                    window=period, min_periods=period
                ).std(ddof=0)
            else:
                moving_average = adjusted_close.rolling(
                    window=period, min_periods=period
                ).mean()
                features[f"{symbol}_{name}"] = adjusted_close / moving_average - 1.0

    benchmark_close = frames[BENCHMARK_SYMBOL]["adj_close"]
    benchmark_returns = benchmark_close.pct_change(fill_method=None)
    features["benchmark_return_1"] = benchmark_returns
    features["benchmark_volatility_21"] = benchmark_returns.rolling(
        window=21, min_periods=21
    ).std(ddof=0)
    features["vix_level_over_100"] = frames[VIX_SYMBOL]["close"] / 100.0
    return features.loc[:, list(MARKET_FEATURE_NAMES)]


def _validate_initial_holdings(holdings: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(holdings, Mapping):
        raise ValueError("initial_holdings must be a mapping")
    result: dict[str, float] = {}
    for symbol, quantity in holdings.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("initial_holdings keys must be non-empty strings")
        if isinstance(quantity, bool) or not math.isfinite(float(quantity)):
            raise ValueError(f"initial_holdings[{symbol!r}] must be finite")
        quantity = float(quantity)
        if quantity < 0.0:
            raise ValueError(f"initial_holdings[{symbol!r}] must be non-negative")
        result[symbol] = quantity
    return result
