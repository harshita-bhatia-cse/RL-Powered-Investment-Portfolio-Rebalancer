"""SQLite-backed long-only historical portfolio environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.data.data_store import DataStore, get_data_store


DEFAULT_ASSETS = ("SPY", "QQQ", "GLD", "TLT", "BND")


@dataclass(frozen=True)
class RewardConfig:
    """Reward coefficients; costs are charged once through net portfolio value."""

    risk_penalty: float = 0.05
    drawdown_penalty: float = 0.10
    trading_cost_penalty: float = 1.0
    lookback: int = 20

    def validate(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if min(self.risk_penalty, self.drawdown_penalty, self.trading_cost_penalty) < 0:
            raise ValueError("reward coefficients must be non-negative")


class HistoricalPortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    """PPO-compatible environment backed by SQLite adjusted-close history.

    Actions are asset-plus-cash allocations. Orders are executed at the next
    session open and marked at that session's close. Prices are never
    forward-filled or generated.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        tickers: tuple[str, ...] = DEFAULT_ASSETS,
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 100_000.0,
        transaction_cost_rate: float = 0.001,
        slippage_bps: float = 0.0,
        reward_config: RewardConfig | None = None,
        data_store: DataStore | None = None,
    ) -> None:
        super().__init__()
        self.tickers = tuple(tickers)
        if self.tickers != DEFAULT_ASSETS:
            raise ValueError(f"tickers must be exactly {DEFAULT_ASSETS}")
        if initial_capital <= 0 or not np.isfinite(initial_capital):
            raise ValueError("initial_capital must be positive and finite")
        if transaction_cost_rate < 0 or slippage_bps < 0:
            raise ValueError("transaction costs and slippage must be non-negative")
        self.reward_config = reward_config or RewardConfig()
        self.reward_config.validate()
        self.initial_capital = float(initial_capital)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.slippage_rate = float(slippage_bps) / 10_000.0
        self.data_store = data_store or get_data_store()
        self.prices = self._load_prices(start_date, end_date)
        self.dates = self.prices.index
        self.returns = self.prices.pct_change(fill_method=None).dropna()
        self._fit_normalization()
        feature_size = len(self.tickers) * self.reward_config.lookback
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_size + len(self.tickers) + 2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self.tickers) + 1,), dtype=np.float32
        )
        self._reset_state()

    def _load_prices(self, start_date: str | None, end_date: str | None) -> pd.DataFrame:
        start = start_date or "1900-01-01"
        end = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        raw = self.data_store.get_price_data(list(self.tickers), start, end)
        if raw.empty:
            raise ValueError("No historical SQLite price data is available")
        frame = raw.assign(datadate=pd.to_datetime(raw["datadate"])).pivot(
            index="datadate", columns="tic", values="adj_close"
        ).sort_index()
        missing = [ticker for ticker in self.tickers if ticker not in frame.columns]
        if missing:
            raise ValueError("Missing historical data for: " + ", ".join(missing))
        frame = frame.loc[:, list(self.tickers)].dropna()
        if len(frame) < self.reward_config.lookback + 2:
            raise ValueError("Insufficient aligned historical observations")
        if (frame <= 0).any().any() or not np.isfinite(frame.to_numpy()).all():
            raise ValueError("Historical prices must be finite and positive")
        return frame

    def _fit_normalization(self) -> None:
        returns = self.returns.dropna()
        training_rows = max(self.reward_config.lookback, int(len(returns) * 0.6))
        training = returns.iloc[:training_rows]
        self.feature_means = training.mean()
        self.feature_scales = training.std(ddof=0).replace(0.0, 1.0)

    def _reset_state(self) -> None:
        self._position = self.reward_config.lookback
        self._cash = self.initial_capital
        self._holdings = np.zeros(len(self.tickers), dtype=float)
        self._nav = self.initial_capital
        self._peak_nav = self.initial_capital
        self._portfolio_returns: list[float] = []
        self._done = True

    def _weights(self) -> np.ndarray:
        values = self._holdings * self.prices.iloc[self._position].to_numpy()
        return np.append(values / self._nav, self._cash / self._nav)

    def _observation(self) -> np.ndarray:
        window = self.returns.iloc[self._position - self.reward_config.lookback:self._position]
        normalized = (window - self.feature_means) / self.feature_scales
        recent = normalized.to_numpy(dtype=np.float32).reshape(-1)
        weights = self._weights()[:-1].astype(np.float32)
        state = np.array([self._cash / self._nav, self._nav / self.initial_capital], dtype=np.float32)
        observation = np.concatenate((recent, weights, state)).astype(np.float32)
        if not np.isfinite(observation).all():
            raise RuntimeError("Observation contains non-finite values")
        return observation

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options:
            raise ValueError("HistoricalPortfolioEnv does not support reset options")
        self._reset_state()
        self._done = False
        return self._observation(), {"date": self.dates[self._position].date().isoformat(), "seed": seed}

    def step(self, action: np.ndarray):
        if self._done:
            raise RuntimeError("reset() must be called before step()")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("action must match the action space and be finite")
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        action_was_clipped = not np.array_equal(action, clipped_action)
        total = clipped_action.sum()
        if total <= 0:
            raise ValueError("action must contain a positive allocation")
        target = clipped_action / total
        current_date = self.dates[self._position]
        next_position = self._position + 1
        next_date = self.dates[next_position]
        current_prices = self.prices.iloc[self._position].to_numpy()
        execution_prices = self._open_prices(next_date)
        close_prices = self.prices.iloc[next_position].to_numpy()
        previous_nav = self._nav
        current_values = self._holdings * execution_prices
        cash = self._cash
        sell_values = np.maximum(current_values - target[:-1] * previous_nav, 0.0)
        sell_shares = np.divide(sell_values, execution_prices, out=np.zeros_like(sell_values), where=execution_prices > 0)
        self._holdings -= sell_shares
        sell_notional = float(sell_values.sum())
        sell_cost = sell_notional * (self.transaction_cost_rate + self.slippage_rate)
        cash += sell_notional - sell_cost
        desired_buy_values = np.maximum(target[:-1] * previous_nav - self._holdings * execution_prices, 0.0)
        buy_budget = max(0.0, cash)
        buy_total = float(desired_buy_values.sum())
        cost_multiplier = 1.0 + self.transaction_cost_rate + self.slippage_rate
        scale = min(1.0, buy_budget / (buy_total * cost_multiplier)) if buy_total > 0 else 1.0
        buy_values = desired_buy_values * scale
        buy_shares = np.divide(buy_values, execution_prices, out=np.zeros_like(buy_values), where=execution_prices > 0)
        self._holdings += buy_shares
        buy_notional = float(buy_values.sum())
        traded_notional = sell_notional + buy_notional
        buy_fees = buy_notional * self.transaction_cost_rate
        buy_slippage = buy_notional * self.slippage_rate
        fees = sell_notional * self.transaction_cost_rate + buy_fees
        slippage = sell_notional * self.slippage_rate + buy_slippage
        self._cash = cash - buy_notional - buy_fees - buy_slippage
        if self._cash < -1e-8:
            raise RuntimeError("portfolio simulation produced negative cash")
        gross_nav = float((self._holdings * close_prices).sum() + self._cash + fees + slippage)
        self._nav = float((self._holdings * close_prices).sum() + self._cash)
        net_return = self._nav / previous_nav - 1.0
        gross_return = gross_nav / previous_nav - 1.0
        turnover = traded_notional / previous_nav
        self._portfolio_returns.append(net_return)
        self._peak_nav = max(self._peak_nav, self._nav)
        drawdown = max(0.0, 1.0 - self._nav / self._peak_nav)
        volatility = float(np.std(self._portfolio_returns[-self.reward_config.lookback:], ddof=0))
        components = {
            "return_reward": gross_return,
            "risk_penalty": self.reward_config.risk_penalty * volatility,
            "drawdown_penalty": self.reward_config.drawdown_penalty * drawdown,
            "trading_cost_penalty": self.reward_config.trading_cost_penalty * (fees + slippage) / previous_nav,
        }
        reward = float(components["return_reward"] - components["risk_penalty"]
                       - components["drawdown_penalty"] - components["trading_cost_penalty"])
        self._position = next_position
        self._done = next_position == len(self.dates) - 1
        info = {
            "date": next_date.date().isoformat(),
            "previous_date": current_date.date().isoformat(),
            "nav": self._nav,
            "daily_return": net_return,
            "turnover": turnover,
            "transaction_fees": fees,
            "slippage_cost": slippage,
            "weights": self._weights(),
            "action_was_clipped": action_was_clipped,
            "drawdown": drawdown,
            "reward_components": components,
        }
        return self._observation(), reward, self._done, False, info

    def _open_prices(self, date: pd.Timestamp) -> np.ndarray:
        raw = self.data_store.get_price_data(list(self.tickers), date.date().isoformat(), date.date().isoformat())
        if raw.empty:
            return self.prices.loc[date].to_numpy()
        values = raw.pivot(index="datadate", columns="tic", values="prcod").reindex(columns=self.tickers)
        return values.iloc[0].to_numpy(dtype=float)

    def render(self):
        return {"date": self.dates[self._position].date().isoformat(), "nav": self._nav}

    def close(self):
        self._done = True

