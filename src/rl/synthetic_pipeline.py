"""Small integration helpers for synthetic scenarios and portfolio environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from src.rl.environment.synthetic_portfolio_env import SyntheticPortfolioEnv
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


def build_synthetic_pipeline(
    config: SyntheticMarketConfig | None = None,
    *,
    seed: int,
) -> SyntheticPipeline:
    """Generate train/validation/test data and construct isolated environments."""
    datasets = generate_synthetic_splits(config, seed=seed)
    environments: dict[ScenarioName, SyntheticPortfolioEnv] = {
        "train": SyntheticPortfolioEnv(**datasets.train.environment_kwargs()),
        "validation": SyntheticPortfolioEnv(
            **datasets.validation.environment_kwargs()
        ),
        "test": SyntheticPortfolioEnv(**datasets.test.environment_kwargs()),
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
