"""Separate Gymnasium environment for validated Indian daily ETF data."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.data.india_daily_preprocessing import (
    IndiaDailyDataset,
    IndiaFeatureScaler,
)
from src.rl.environment.portfolio_env import project_to_simplex


class IndiaDailyPortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Daily completed-data India ETF allocation environment."""

    metadata = {
        "render_modes": [],
        "source_type": "india_historical_csv",
        "synthetic": False,
    }

    def __init__(
        self,
        dataset: IndiaDailyDataset,
        *,
        partition: str,
        scaler: IndiaFeatureScaler | None = None,
        initial_capital: float = 100_000.0,
        transaction_cost_rate: float = 0.001,
        slippage_bps: float = 5.0,
        min_cash_weight: float = 0.02,
        max_asset_weight: float = 0.35,
        max_turnover: float = 1.0,
        random_start: bool = False,
    ) -> None:
        super().__init__()
        if partition not in ("train", "validation", "test"):
            raise ValueError("partition must be train, validation or test")
        if not isinstance(random_start, bool):
            raise ValueError("random_start must be a boolean")
        if not np.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("initial_capital must be finite and positive")
        if not np.isfinite(transaction_cost_rate) or transaction_cost_rate < 0:
            raise ValueError("transaction_cost_rate must be finite and non-negative")
        if not np.isfinite(slippage_bps) or slippage_bps < 0:
            raise ValueError("slippage_bps must be finite and non-negative")
        if not 0 <= min_cash_weight <= 1:
            raise ValueError("min_cash_weight must be between zero and one")
        if not 0 < max_asset_weight <= 1:
            raise ValueError("max_asset_weight must be between zero and one")
        if not np.isfinite(max_turnover) or max_turnover < 0:
            raise ValueError("max_turnover must be finite and non-negative")

        self.dataset = dataset
        self.partition = partition
        self.initial_capital = float(initial_capital)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.slippage_rate = float(slippage_bps) / 10_000.0
        self.min_cash_weight = float(min_cash_weight)
        self.max_asset_weight = float(max_asset_weight)
        self.max_turnover = float(max_turnover)
        self.random_start = random_start

        if partition == "train":
            bounds = dataset.splits.train
            training_features = dataset.features.loc[
                (dataset.features.index >= bounds[0])
                & (dataset.features.index <= bounds[1])
            ]
            self.scaler = IndiaFeatureScaler.fit(training_features)
        else:
            if scaler is None:
                raise ValueError("validation and test environments require a training-fitted scaler")
            if tuple(scaler.feature_names) != tuple(dataset.features.columns):
                raise ValueError("scaler feature names do not match the India dataset")
            bounds = (
                dataset.splits.validation
                if partition == "validation"
                else dataset.splits.test
            )
            self.scaler = scaler

        partition_features = dataset.features.loc[
            (dataset.features.index >= bounds[0])
            & (dataset.features.index <= bounds[1])
        ]
        if len(partition_features) < 2:
            raise ValueError("partition requires at least two completed sessions")
        self.partition_dates = tuple(partition_features.index)
        if partition_features.index[0] not in dataset.features.index:
            raise ValueError("partition date is not in the feature index")

        self.scaled_features = pd.DataFrame(
            self.scaler.transform(partition_features),
            index=partition_features.index,
            columns=partition_features.columns,
        )
        feature_count = len(partition_features.columns)
        state_count = len(dataset.symbols) + 4
        self.observation_size = feature_count + state_count
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(dataset.symbols) + 1,),
            dtype=np.float32,
        )
        observation_limit = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-observation_limit,
            high=observation_limit,
            shape=(self.observation_size,),
            dtype=np.float32,
        )
        self.metadata = {
            "source_type": "india_historical_csv",
            "synthetic": False,
            "assets": dataset.symbols,
            "partition": partition,
            "corporate_action_limitation": (
                "Corporate-action completeness is not established. "
                "Only use adjusted close if its methodology is approved."
            ),
            "adjustment_methodology": dataset.manifest.get(
                "adjustment_methodology", "unknown"
            ),
        }
        self._reset_state()

    def _reset_state(self) -> None:
        self._position = 0
        self._nav = self.initial_capital
        self._peak_nav = self.initial_capital
        self._holdings = np.zeros(len(self.dataset.symbols), dtype=np.float64)
        self._cash = self.initial_capital
        self._done = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if options:
            unknown = set(options) - {"start_date"}
            if unknown:
                raise ValueError(f"unsupported reset options: {sorted(unknown)}")
            start_date = pd.Timestamp(options["start_date"])
            if start_date not in self.partition_dates:
                raise ValueError("start_date must be a partition session")
            position = self.partition_dates.index(start_date)
            if position >= len(self.partition_dates) - 1:
                raise ValueError("start_date must leave at least one next session")
        elif self.random_start:
            position = int(self.np_random.integers(0, len(self.partition_dates) - 1))
        else:
            position = 0

        self._position = position
        self._nav = self.initial_capital
        self._peak_nav = self.initial_capital
        self._holdings = np.zeros(len(self.dataset.symbols), dtype=np.float64)
        self._cash = self.initial_capital
        self._done = False
        date = self.partition_dates[position]
        observation = self._observation(date)
        info = self._state_info(date)
        info["seed"] = seed
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("reset() must be called before step()")
        raw_action = np.asarray(action, dtype=np.float64)
        if raw_action.shape != self.action_space.shape:
            raise ValueError("action must match the asset-plus-cash action shape")
        if not np.isfinite(raw_action).all():
            raise ValueError("action must contain finite values")
        clipped = np.clip(raw_action, self.action_space.low, self.action_space.high)
        action_was_clipped = not np.array_equal(clipped, raw_action)
        target = project_to_simplex(clipped)
        target = self._constrain_target(target)

        current_date = self.partition_dates[self._position]
        next_position = self._position + 1
        next_date = self.partition_dates[next_position]
        previous_nav = self._nav

        current_close = self.dataset.close_prices.loc[
            current_date
        ].to_numpy(dtype=np.float64)
        pre_weights = self._weights_at_close(current_close)
        desired_turnover = float(np.abs(target - pre_weights).sum())
        if desired_turnover > self.max_turnover:
            scale = self.max_turnover / desired_turnover
            target = pre_weights + scale * (target - pre_weights)
            target[-1] = max(
                self.min_cash_weight,
                1.0 - float(target[:-1].sum()),
            )
            target /= float(target.sum())

        open_prices = self.dataset.open_prices.loc[next_date].to_numpy(dtype=np.float64)
        close_prices = self.dataset.close_prices.loc[next_date].to_numpy(dtype=np.float64)
        current_values = self._holdings * open_prices

        target_asset_values = target[:-1] * previous_nav
        desired_delta = target_asset_values - current_values

        # Sell positions first at adverse open prices.
        sells = np.maximum(-desired_delta, 0.0)
        sell_shares = np.divide(
            sells,
            open_prices,
            out=np.zeros_like(sells),
            where=open_prices > 0,
        )
        sell_shares = np.minimum(sell_shares, self._holdings)
        sell_notional = float(
            np.sum(sell_shares * open_prices * (1.0 - self.slippage_rate))
        )
        sell_fees = sell_notional * self.transaction_cost_rate
        sell_slippage = float(
            np.sum(sell_shares * open_prices * self.slippage_rate)
        )
        self._holdings -= sell_shares
        self._cash += sell_notional - sell_fees

        desired_buys = np.maximum(desired_delta, 0.0)
        buy_gross = desired_buys * (1.0 + self.slippage_rate)
        buy_fees_factor = 1.0 + self.transaction_cost_rate
        buy_cost = buy_gross * buy_fees_factor
        maximum_cash = max(0.0, self._cash - self.min_cash_weight * previous_nav)
        buy_total = float(buy_cost.sum())
        buy_scale = min(1.0, maximum_cash / buy_total) if buy_total > 0 else 1.0
        buy_values = desired_buys * buy_scale
        buy_shares = np.divide(
            buy_values,
            open_prices,
            out=np.zeros_like(buy_values),
            where=open_prices > 0,
        )
        buy_notional = float(np.sum(buy_shares * open_prices))
        buy_fee_cost = buy_notional * (1.0 + self.slippage_rate) * self.transaction_cost_rate
        buy_slippage = float(np.sum(buy_shares * open_prices * self.slippage_rate))
        buy_cash_cost = buy_notional * (1.0 + self.slippage_rate) + buy_fee_cost
        self._holdings += buy_shares
        self._cash -= buy_cash_cost
        if self._cash < -1e-8:
            raise RuntimeError("portfolio simulation produced negative cash")
        self._cash = max(0.0, self._cash)

        fees = sell_fees + buy_fee_cost
        slippage = sell_slippage + buy_slippage
        self._nav = float(
            np.sum(self._holdings * close_prices) + self._cash
        )
        if not np.isfinite(self._nav) or self._nav <= 0:
            raise RuntimeError("portfolio NAV became non-positive or non-finite")
        net_return = self._nav / previous_nav - 1.0
        reward = float(math.log(self._nav / previous_nav))
        post_weights = self._weights_at_close(close_prices)
        gross_turnover = float(np.abs(post_weights - pre_weights).sum())
        turnover = gross_turnover / 2.0

        self._peak_nav = max(self._peak_nav, self._nav)
        self._position = next_position
        self._done = next_position == len(self.partition_dates) - 1
        observation = self._observation(next_date)
        info = self._state_info(next_date)
        info.update({
            "previous_date": current_date.strftime("%Y-%m-%d"),
            "date": next_date.strftime("%Y-%m-%d"),
            "daily_return": net_return,
            "reward_log_nav_return": reward,
            "requested_target_weights": self._weights_by_name(target),
            "applied_weights": self._weights_by_name(
                self._weights_at_close(close_prices)
            ),
            "action_was_clipped": action_was_clipped,
            "transaction_fees": fees,
            "slippage_cost": slippage,
            "total_transaction_cost": fees + slippage,
            "gross_turnover": gross_turnover,
            "turnover": turnover,
            "nav": self._nav,
            "episode_terminated": self._done,
        })
        return observation, reward, self._done, False, info

    def _constrain_target(self, target: np.ndarray) -> np.ndarray:
        asset_weights = np.minimum(target[:-1], self.max_asset_weight)
        excess = target[:-1].sum() - asset_weights.sum()
        eligible = asset_weights < self.max_asset_weight
        while excess > 1e-12 and eligible.any():
            capacity = self.max_asset_weight - asset_weights
            capacity[~eligible] = 0.0
            total_capacity = capacity.sum()
            if total_capacity <= 0:
                break
            addition = np.minimum(capacity, excess * capacity / total_capacity)
            asset_weights += addition
            redistributed = addition.sum()
            excess -= redistributed
            eligible = asset_weights < self.max_asset_weight - 1e-12
            if redistributed <= 1e-12:
                break
        cash = max(
            self.min_cash_weight,
            1.0 - float(asset_weights.sum()),
        )
        result = np.append(asset_weights, cash)
        total = float(result.sum())
        if total != 1.0:
            result /= total
        if not np.isfinite(result).all() or np.any(result < -1e-12):
            raise RuntimeError("target weight projection produced invalid weights")
        return result.astype(np.float64)

    def _weights_at_close(self, close_prices: np.ndarray) -> np.ndarray:
        asset_values = self._holdings * close_prices
        values = np.append(asset_values, self._cash)
        total = float(values.sum())
        if total <= 0:
            raise RuntimeError("portfolio value is not positive")
        return values / total

    def _weights_by_name(self, weights: np.ndarray) -> dict[str, float]:
        return {
            **{
                symbol: float(weights[index])
                for index, symbol in enumerate(self.dataset.symbols)
            },
            "CASH": float(weights[-1]),
        }

    def _observation(self, date: pd.Timestamp) -> np.ndarray:
        market = self.scaled_features.loc[date].to_numpy(dtype=np.float32)
        close_prices = self.dataset.close_prices.loc[date].to_numpy(dtype=np.float64)
        weights = self._weights_at_close(close_prices)
        drawdown = self._nav / self._peak_nav - 1.0
        position = self._position
        remaining = (
            len(self.partition_dates) - 1 - position
        ) / max(len(self.partition_dates) - 1, 1)
        state = np.asarray(
            [
                *weights.tolist(),
                self._nav / self.initial_capital,
                drawdown,
                remaining,
            ],
            dtype=np.float32,
        )
        observation = np.concatenate((market, state)).astype(np.float32)
        if not np.isfinite(observation).all():
            raise RuntimeError("observation contains non-finite values")
        return observation

    def _state_info(self, date: pd.Timestamp) -> dict:
        close_prices = self.dataset.close_prices.loc[date].to_numpy(dtype=np.float64)
        weights = self._weights_at_close(close_prices)
        return {
            "source_type": "india_historical_csv",
            "synthetic": False,
            "date": date.strftime("%Y-%m-%d"),
            "nav": self._nav,
            "cash": self._cash,
            "weights": self._weights_by_name(weights),
            "drawdown": self._nav / self._peak_nav - 1.0,
            "corporate_action_limitation": self.metadata[
                "corporate_action_limitation"
            ],
        }

    def render(self) -> dict:
        date = self.partition_dates[self._position]
        return {"date": date.strftime("%Y-%m-%d"), "nav": self._nav}

    def close(self) -> None:
        self._done = True
