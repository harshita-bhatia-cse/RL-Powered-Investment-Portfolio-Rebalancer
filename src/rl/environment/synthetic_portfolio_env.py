"""Gymnasium environment for synthetic-only portfolio rebalancing tests."""

from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .portfolio_env import ASSET_UNIVERSE, project_to_simplex

OBSERVATION_SCHEMA_VERSION = 1
OBSERVATION_SCHEMA_WITH_TIME = "synthetic_portfolio_v1_with_time"
OBSERVATION_SCHEMA_WITHOUT_TIME = "synthetic_portfolio_v1_without_time"
DEFAULT_OBSERVATION_SCHEMA = OBSERVATION_SCHEMA_WITH_TIME
OBSERVATION_SIZE = 43
OBSERVATION_SIZE_WITHOUT_TIME = 42
TRADING_DAYS_PER_YEAR = 252
OBSERVATION_WARMUP_SESSIONS = 63
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_PERIOD_SESSIONS = 21
DEFAULT_MAX_ASSET_WEIGHT = 0.35
DEFAULT_MIN_CASH_WEIGHT = 0.02
DEFAULT_MAX_TURNOVER = 1.0
DEFAULT_TRANSACTION_COST_RATE = 0.001
DEFAULT_SLIPPAGE_BPS = 5.0


def observation_size_for_schema(observation_schema: str) -> int:
    if observation_schema == OBSERVATION_SCHEMA_WITH_TIME:
        return OBSERVATION_SIZE
    if observation_schema == OBSERVATION_SCHEMA_WITHOUT_TIME:
        return OBSERVATION_SIZE_WITHOUT_TIME
    raise ValueError(f"unsupported synthetic observation schema: {observation_schema!r}")


class SyntheticPortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Rebalance six synthetic ETF return series at monthly decision points.

    The action is a seven-value vector (six assets followed by cash), clipped
    to ``[-1, 1]`` and projected onto the unit simplex. Asset caps, a cash
    reserve, and a gross-turnover limit are then enforced. Inputs must be
    explicitly marked synthetic; this environment does not load files.

    The asset and cash constraints apply to projected rebalance targets.
    Returns may drift marked holdings outside those bounds between decisions;
    if the turnover limit prevents full restoration, the applied trade must
    move toward feasibility and reports ``constraint_recovery``.

    At decision index ``t``, observations contain only returns ending at
    ``t - 1``. The action is applied before the next ``period_sessions``
    returns are consumed. The action therefore cannot observe its interval's
    returns.
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "synthetic": True,
        "source_type": "synthetic",
    }

    def __init__(
        self,
        asset_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        vix_levels: np.ndarray,
        *,
        data_metadata: Mapping[str, Any],
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        period_sessions: int = DEFAULT_PERIOD_SESSIONS,
        max_asset_weight: float | Sequence[float] = DEFAULT_MAX_ASSET_WEIGHT,
        min_cash_weight: float = DEFAULT_MIN_CASH_WEIGHT,
        max_turnover: float = DEFAULT_MAX_TURNOVER,
        transaction_cost_rate: float = DEFAULT_TRANSACTION_COST_RATE,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        vix_level_scale: float = 100.0,
        observation_schema: str = DEFAULT_OBSERVATION_SCHEMA,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.asset_returns = _validated_returns(
            "asset_returns", asset_returns, expected_columns=len(ASSET_UNIVERSE)
        )
        self.benchmark_returns = _validated_returns(
            "benchmark_returns", benchmark_returns, expected_columns=1
        ).reshape(-1)
        self.vix_levels = _validated_positive_series("vix_levels", vix_levels)
        if not (
            len(self.asset_returns)
            == len(self.benchmark_returns)
            == len(self.vix_levels)
        ):
            raise ValueError("asset_returns, benchmark_returns, and vix_levels must align")
        if len(self.asset_returns) <= OBSERVATION_WARMUP_SESSIONS:
            raise ValueError(
                "at least one return session is required after the 63-session warm-up"
            )

        if not isinstance(data_metadata, Mapping):
            raise ValueError("data_metadata must identify synthetic input")
        self.data_metadata = deepcopy(dict(data_metadata))
        if (
            self.data_metadata.get("synthetic") is not True
            or self.data_metadata.get("source_type") != "synthetic"
        ):
            raise ValueError("SyntheticPortfolioEnv accepts synthetic data only")

        self.initial_capital = _finite_positive("initial_capital", initial_capital)
        self.period_sessions = _positive_integer("period_sessions", period_sessions)
        self.max_asset_weights = _asset_caps(max_asset_weight)
        self.min_cash_weight = _finite_nonnegative(
            "min_cash_weight", min_cash_weight
        )
        self.max_turnover = _finite_nonnegative("max_turnover", max_turnover)
        self.transaction_cost_rate = _finite_nonnegative(
            "transaction_cost_rate", transaction_cost_rate
        )
        self.slippage_bps = _finite_nonnegative("slippage_bps", slippage_bps)
        self.vix_level_scale = _finite_positive("vix_level_scale", vix_level_scale)
        self.observation_size = observation_size_for_schema(observation_schema)
        self.observation_schema = observation_schema
        if self.min_cash_weight > 1.0:
            raise ValueError("minimum cash reserve cannot exceed 100%")
        if self.max_turnover > 2.0:
            raise ValueError("max_turnover cannot exceed the maximum gross turnover of 2")
        if self.transaction_cost_rate >= 1.0 or self.slippage_bps >= 10_000.0:
            raise ValueError("transaction costs must be less than 100%")
        if render_mode not in (None, "human", "ansi"):
            raise ValueError("render_mode must be None, 'human', or 'ansi'")
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(ASSET_UNIVERSE) + 1,),
            dtype=np.float32,
        )
        observation_limit = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-observation_limit,
            high=observation_limit,
            shape=(self.observation_size,),
            dtype=np.float32,
        )

        self._current_index = OBSERVATION_WARMUP_SESSIONS
        self._nav = self.initial_capital
        self._peak_nav = self.initial_capital
        self._weights = np.zeros(len(ASSET_UNIVERSE) + 1, dtype=np.float64)
        self._weights[-1] = 1.0
        self._done = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to all cash and return the first post-warm-up observation."""
        super().reset(seed=seed)
        if options:
            raise ValueError("reset options are not supported")
        self._current_index = OBSERVATION_WARMUP_SESSIONS
        self._nav = self.initial_capital
        self._peak_nav = self.initial_capital
        self._weights = np.zeros(len(ASSET_UNIVERSE) + 1, dtype=np.float64)
        self._weights[-1] = 1.0
        self._done = False
        return self._observation(), {
            "synthetic": True,
            "source_type": "synthetic",
            "data_metadata": deepcopy(self.data_metadata),
            "nav": self._nav,
            "weights": self._weights_by_name(),
            "session_index": self._current_index,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an action, charge costs, and consume the next return interval."""
        if self._done:
            raise RuntimeError("step() called after episode termination; call reset()")
        requested = _validated_action(action, self.action_space.shape)
        clipped_action = np.clip(requested, self.action_space.low, self.action_space.high)
        was_clipped = not np.array_equal(requested, clipped_action)
        simplex_weights = project_to_simplex(clipped_action)
        constrained_weights = _project_to_capped_simplex(
            simplex_weights,
            self.max_asset_weights,
            self.min_cash_weight,
        )
        if not _is_valid_weight_vector(
            constrained_weights, self.max_asset_weights, self.min_cash_weight
        ):
            raise RuntimeError("portfolio constraint projection produced invalid target")

        pre_trade_nav = self._nav
        pre_trade_weights = self._weights.copy()
        gross_turnover = float(np.abs(constrained_weights - pre_trade_weights).sum())
        turnover_scale = (
            min(1.0, self.max_turnover / gross_turnover)
            if gross_turnover > self.max_turnover and gross_turnover > 0.0
            else 1.0
        )
        applied_weights = pre_trade_weights + turnover_scale * (
            constrained_weights - pre_trade_weights
        )
        applied_weights[-1] = 1.0 - float(applied_weights[:-1].sum())
        gross_turnover = float(np.abs(applied_weights - pre_trade_weights).sum())
        if not _is_valid_weight_transition(
            applied_weights,
            pre_trade_weights,
            self.max_asset_weights,
            self.min_cash_weight,
        ):
            raise RuntimeError("portfolio constraint projection produced invalid transition")
        constraint_recovery = not _is_valid_weight_vector(
            applied_weights, self.max_asset_weights, self.min_cash_weight
        )

        one_way_turnover = gross_turnover / 2.0
        traded_notional = pre_trade_nav * gross_turnover
        transaction_fees = traded_notional * self.transaction_cost_rate
        slippage_cost = traded_notional * (self.slippage_bps / 10_000.0)
        total_cost = transaction_fees + slippage_cost
        post_cost_nav = pre_trade_nav - total_cost
        if not math.isfinite(post_cost_nav) or post_cost_nav <= 0.0:
            raise FloatingPointError("transaction costs reduced portfolio NAV to zero")

        interval_start = self._current_index
        interval_end = min(
            interval_start + self.period_sessions, len(self.asset_returns)
        )
        interval_asset_returns = self.asset_returns[interval_start:interval_end]
        interval_factors = np.prod(1.0 + interval_asset_returns, axis=0)
        asset_growth_factors = np.append(interval_factors, 1.0)
        ending_values = post_cost_nav * applied_weights * asset_growth_factors
        ending_nav = float(ending_values.sum())
        if not math.isfinite(ending_nav) or ending_nav <= 0.0:
            raise FloatingPointError("synthetic returns reduced portfolio NAV to zero")

        net_return = ending_nav / pre_trade_nav - 1.0
        reward = math.log(ending_nav / pre_trade_nav)
        if not math.isfinite(reward):
            raise FloatingPointError("portfolio reward is not finite")

        self._nav = ending_nav
        self._peak_nav = max(self._peak_nav, ending_nav)
        self._weights = ending_values / ending_nav
        self._current_index = interval_end
        terminated = self._current_index >= len(self.asset_returns)
        truncated = False
        self._done = terminated

        info = {
            "synthetic": True,
            "source_type": "synthetic",
            "data_metadata": deepcopy(self.data_metadata),
            "session_start": interval_start,
            "session_end": interval_end,
            "sessions_in_interval": interval_end - interval_start,
            "pre_trade_nav": pre_trade_nav,
            "post_cost_nav": post_cost_nav,
            "nav": ending_nav,
            "gross_return": float(
                np.dot(applied_weights, asset_growth_factors) - 1.0
            ),
            "net_return": net_return,
            "reward_log_nav_return": reward,
            "requested_weights": self._weights_by_name(simplex_weights),
            "constraint_adjusted_weights": self._weights_by_name(constrained_weights),
            "applied_weights": self._weights_by_name(applied_weights),
            "weights": self._weights_by_name(),
            "action_clipped": was_clipped,
            "gross_turnover": gross_turnover,
            "one_way_turnover": one_way_turnover,
            "turnover_limited": turnover_scale < 1.0,
            "constraint_recovery": constraint_recovery,
            "traded_notional": traded_notional,
            "transaction_fees": transaction_fees,
            "slippage_cost": slippage_cost,
            "transaction_cost": total_cost,
            "peak_nav": self._peak_nav,
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> str | None:
        """Render a concise synthetic portfolio status when requested."""
        if self.render_mode is None:
            return None
        status = (
            f"Synthetic portfolio | session={self._current_index} "
            f"NAV={self._nav:.2f} weights={self._weights_by_name()}"
        )
        if self.render_mode == "human":
            print(status)
            return None
        return status

    def close(self) -> None:
        """Release environment resources (there are no external resources)."""

    def _observation(self) -> np.ndarray:
        end = self._current_index
        returns_63 = self.asset_returns[end - OBSERVATION_WARMUP_SESSIONS : end]
        asset_features: list[float] = []
        for column in range(len(ASSET_UNIVERSE)):
            series = returns_63[:, column]
            asset_features.extend(
                [
                    float(series[-1]),
                    _compounded_return(series[-5:]),
                    _compounded_return(series[-21:]),
                    _annualized_volatility(series[-21:]),
                    _annualized_volatility(series),
                ]
            )

        benchmark = self.benchmark_returns[end - OBSERVATION_WARMUP_SESSIONS : end]
        benchmark_features = [
            float(benchmark[-1]),
            _annualized_volatility(benchmark[-21:]),
        ]
        drawdown = self._nav / self._peak_nav - 1.0
        sessions_remaining = len(self.asset_returns) - end
        episode_sessions = len(self.asset_returns) - OBSERVATION_WARMUP_SESSIONS
        state_features = [
            *self._weights.tolist(),
            self._nav / self.initial_capital,
            drawdown,
        ]
        if self.observation_schema == OBSERVATION_SCHEMA_WITH_TIME:
            state_features.append(sessions_remaining / episode_sessions)
        observation = np.asarray(
            [
                *asset_features,
                *benchmark_features,
                self.vix_levels[end - 1] / self.vix_level_scale,
                *state_features,
            ],
            dtype=np.float32,
        )
        if (
            observation.shape != (self.observation_size,)
            or not np.isfinite(observation).all()
        ):
            raise RuntimeError("environment produced an invalid observation")
        return observation

    def _weights_by_name(self, weights: np.ndarray | None = None) -> dict[str, float]:
        values = self._weights if weights is None else weights
        return {
            **{
                symbol: float(values[index])
                for index, symbol in enumerate(ASSET_UNIVERSE)
            },
            "CASH": float(values[-1]),
        }


def _validated_returns(
    name: str, values: np.ndarray, *, expected_columns: int
) -> np.ndarray:
    try:
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if array.ndim == 1 and expected_columns == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[1] != expected_columns or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (sessions, {expected_columns})")
    if not np.isfinite(array).all() or np.any(array < -1.0):
        raise ValueError(f"{name} must contain finite simple returns >= -1")
    array.setflags(write=False)
    return array


def _validated_positive_series(name: str, values: np.ndarray) -> np.ndarray:
    try:
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric series") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional series")
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    array.setflags(write=False)
    return array


def _asset_caps(value: float | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, (Sequence, np.ndarray)) and not isinstance(
        value, (str, bytes)
    ):
        caps = tuple(_finite_nonnegative("max_asset_weight", cap) for cap in value)
        if len(caps) != len(ASSET_UNIVERSE):
            raise ValueError("max_asset_weight sequence must contain six values")
    else:
        cap = _finite_nonnegative("max_asset_weight", value)
        caps = (cap,) * len(ASSET_UNIVERSE)
    if any(cap > 1.0 for cap in caps):
        raise ValueError("asset caps must not exceed 1")
    return caps


def _project_to_capped_simplex(
    weights: np.ndarray,
    asset_caps: tuple[float, ...],
    min_cash_weight: float,
) -> np.ndarray:
    lower = np.zeros(len(ASSET_UNIVERSE) + 1, dtype=np.float64)
    lower[-1] = min_cash_weight
    upper = np.asarray((*asset_caps, 1.0), dtype=np.float64)
    if lower.sum() > 1.0 + 1e-12 or upper.sum() < 1.0 - 1e-12:
        raise ValueError("asset caps and cash reserve make full investment infeasible")

    low_shift = float(np.min(weights - upper)) - 1.0
    high_shift = float(np.max(weights - lower)) + 1.0
    for _ in range(100):
        shift = (low_shift + high_shift) / 2.0
        projected = np.clip(weights - shift, lower, upper)
        if projected.sum() > 1.0:
            low_shift = shift
        else:
            high_shift = shift
    projected = np.clip(weights - (low_shift + high_shift) / 2.0, lower, upper)
    residual = 1.0 - float(projected.sum())
    if abs(residual) > 1e-12:
        for index in range(len(projected)):
            capacity = (
                upper[index] - projected[index]
                if residual > 0.0
                else projected[index] - lower[index]
            )
            adjustment = math.copysign(min(abs(residual), capacity), residual)
            projected[index] += adjustment
            residual -= adjustment
            if abs(residual) <= 1e-12:
                break
    if abs(float(projected.sum()) - 1.0) > 1e-10:
        raise RuntimeError("could not normalize constrained portfolio weights")
    return projected


def _is_valid_weight_vector(
    weights: np.ndarray,
    asset_caps: tuple[float, ...],
    min_cash_weight: float,
) -> bool:
    return bool(
        np.isfinite(weights).all()
        and np.all(weights >= -1e-12)
        and np.all(weights[:-1] <= np.asarray(asset_caps) + 1e-10)
        and weights[-1] >= min_cash_weight - 1e-10
        and abs(float(weights.sum()) - 1.0) <= 1e-10
    )


def _is_valid_weight_transition(
    weights: np.ndarray,
    previous_weights: np.ndarray,
    asset_caps: tuple[float, ...],
    min_cash_weight: float,
) -> bool:
    if not (
        np.isfinite(weights).all()
        and np.all(weights >= -1e-12)
        and abs(float(weights.sum()) - 1.0) <= 1e-10
    ):
        return False
    tolerance = 1e-10
    caps = np.asarray(asset_caps)
    previously_overweight = previous_weights[:-1] > caps + tolerance
    if np.any(
        np.where(
            previously_overweight,
            weights[:-1] > previous_weights[:-1] + tolerance,
            weights[:-1] > caps + tolerance,
        )
    ):
        return False
    if previous_weights[-1] < min_cash_weight - tolerance:
        return bool(weights[-1] >= previous_weights[-1] - tolerance)
    return bool(weights[-1] >= min_cash_weight - tolerance)


def _validated_action(action: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    try:
        values = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("action must be a finite numeric vector") from error
    if values.shape != shape:
        raise ValueError(f"action must have shape {shape}")
    if not np.isfinite(values).all():
        raise ValueError("action must contain only finite values")
    return values


def _compounded_return(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) - 1.0)


def _annualized_volatility(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite non-negative number") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _finite_positive(name: str, value: float) -> float:
    number = _finite_nonnegative(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
