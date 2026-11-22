"""Small integration helpers for synthetic scenarios and portfolio environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import gymnasium as gym
import numpy as np

from src.rl.environment.synthetic_portfolio_env import (
    DEFAULT_OBSERVATION_SCHEMA,
    OBSERVATION_WARMUP_SESSIONS,
    SyntheticPortfolioEnv,
    observation_size_for_schema,
)
from src.rl.synthetic_data import (
    SyntheticDataset,
    SyntheticMarketConfig,
    SyntheticScenarioSplits,
    generate_synthetic_splits,
)

ScenarioName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SyntheticPipeline:
    """Three independent generated scenarios and their fresh environments."""

    datasets: SyntheticScenarioSplits
    environments: dict[ScenarioName, SyntheticPortfolioEnv]


@dataclass(frozen=True)
class EnvironmentStep:
    """Immutable snapshot of one environment transition."""

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


@dataclass(frozen=True)
class ScenarioRun:
    """Initial observation and collected deterministic episode transitions."""

    scenario_id: str
    seed: int
    initial_observation: np.ndarray
    initial_info: dict[str, object]
    steps: tuple[EnvironmentStep, ...]


class RandomizedEpisodeHorizonEnv(gym.Wrapper[np.ndarray, np.ndarray]):
    """Randomize complete training episodes without changing the base env."""

    def __init__(
        self,
        env: SyntheticPortfolioEnv,
        *,
        min_episode_decisions: int = 11,
        max_episode_decisions: int | None = None,
        seed: int,
    ) -> None:
        if not isinstance(env, SyntheticPortfolioEnv):
            raise TypeError("randomized horizons require a SyntheticPortfolioEnv")
        if not str(env.data_metadata.get("scenario_id", "")).startswith("train-"):
            raise ValueError("randomized horizons may wrap training scenarios only")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("horizon seed must be a non-negative integer")

        available_decisions = (
            len(env.asset_returns) - OBSERVATION_WARMUP_SESSIONS
        ) // env.period_sessions
        if (
            isinstance(min_episode_decisions, bool)
            or not isinstance(min_episode_decisions, int)
            or min_episode_decisions <= 0
        ):
            raise ValueError("min_episode_decisions must be a positive integer")
        if max_episode_decisions is None:
            max_episode_decisions = available_decisions
        if (
            isinstance(max_episode_decisions, bool)
            or not isinstance(max_episode_decisions, int)
            or max_episode_decisions <= 0
        ):
            raise ValueError("max_episode_decisions must be a positive integer")
        if (
            min_episode_decisions > max_episode_decisions
            or max_episode_decisions > available_decisions
        ):
            raise ValueError(
                "episode decision bounds must fit complete periods in the training data"
            )

        super().__init__(env)
        self._template_env = env
        self._rng = np.random.default_rng(seed)
        self.horizon_seed = seed
        self.min_episode_decisions = min_episode_decisions
        self.max_episode_decisions = max_episode_decisions
        self.last_episode_horizon_decisions: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if options:
            raise ValueError("reset options are not supported")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("reset seed must be a non-negative integer")
            seed_words: list[int] = []
            # Length prefixes keep the configured and reset seeds unambiguous.
            for value in (self.horizon_seed, seed):
                words = []
                while value:
                    words.append(value & 0xFFFFFFFF)
                    value >>= 32
                words = words or [0]
                seed_words.extend((len(words), *words))
            self._rng = np.random.default_rng(np.random.SeedSequence(seed_words))

        decisions = int(
            self._rng.integers(
                self.min_episode_decisions,
                self.max_episode_decisions + 1,
            )
        )
        template = self._template_env
        endpoint = (
            OBSERVATION_WARMUP_SESSIONS + decisions * template.period_sessions
        )
        self.env = SyntheticPortfolioEnv(
            template.asset_returns[:endpoint],
            template.benchmark_returns[:endpoint],
            template.vix_levels[:endpoint],
            data_metadata=template.data_metadata,
            initial_capital=template.initial_capital,
            period_sessions=template.period_sessions,
            max_asset_weight=template.max_asset_weights,
            min_cash_weight=template.min_cash_weight,
            max_turnover=template.max_turnover,
            transaction_cost_rate=template.transaction_cost_rate,
            slippage_bps=template.slippage_bps,
            vix_level_scale=template.vix_level_scale,
            observation_schema=template.observation_schema,
            render_mode=template.render_mode,
        )
        observation, info = self.env.reset(seed=seed)
        info = dict(info)
        info.update(
            {
                "episode_horizon_decisions": decisions,
                "episode_horizon_sessions": decisions * template.period_sessions,
                "episode_endpoint_index": endpoint,
                "episode_horizon_seed": self.horizon_seed,
                "episode_reset_seed": seed,
            }
        )
        self.last_episode_horizon_decisions = decisions
        return observation, info


def build_synthetic_pipeline(
    config: SyntheticMarketConfig | None = None,
    *,
    seed: int,
    observation_schema: str = DEFAULT_OBSERVATION_SCHEMA,
) -> SyntheticPipeline:
    """Generate train/validation/test data and construct isolated environments."""
    observation_size_for_schema(observation_schema)
    datasets = generate_synthetic_splits(config, seed=seed)
    environments: dict[ScenarioName, SyntheticPortfolioEnv] = {
        "train": SyntheticPortfolioEnv(
            **datasets.train.environment_kwargs(),
            observation_schema=observation_schema,
        ),
        "validation": SyntheticPortfolioEnv(
            **datasets.validation.environment_kwargs(),
            observation_schema=observation_schema,
        ),
        "test": SyntheticPortfolioEnv(
            **datasets.test.environment_kwargs(),
            observation_schema=observation_schema,
        ),
    }
    return SyntheticPipeline(datasets=datasets, environments=environments)


def run_deterministic_actions(
    dataset: SyntheticDataset,
    actions: Iterable[np.ndarray],
) -> ScenarioRun:
    """Run the supplied action sequence on a fresh environment for one dataset.

    Actions are copied before each call. The run stops at natural termination
    or when the caller's action sequence is exhausted.
    """
    environment = SyntheticPortfolioEnv(**dataset.environment_kwargs())
    initial_observation, initial_info = environment.reset(seed=dataset.seed)
    steps: list[EnvironmentStep] = []
    for action in actions:
        observation, reward, terminated, truncated, info = environment.step(
            np.array(action, copy=True)
        )
        observation_copy = np.array(observation, copy=True)
        observation_copy.setflags(write=False)
        steps.append(
            EnvironmentStep(
                observation=observation_copy,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=dict(info),
            )
        )
        if terminated or truncated:
            break
    initial_observation = np.array(initial_observation, copy=True)
    initial_observation.setflags(write=False)
    return ScenarioRun(
        scenario_id=str(dataset.metadata["scenario_id"]),
        seed=dataset.seed,
        initial_observation=initial_observation,
        initial_info=dict(initial_info),
        steps=tuple(steps),
    )
