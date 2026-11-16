"""Configuration models for the optional synthetic PPO workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    """Explicit PPO algorithm and policy configuration."""

    n_steps: int = 1024
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (128, 128)
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        """Reject invalid PPO settings before optional-library initialization."""
        for name in ("n_steps", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size:
            raise ValueError("batch_size must divide n_steps and not exceed it")
        for name in ("gamma", "gae_lambda"):
            value = _finite_number(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("learning_rate", "clip_range", "max_grad_norm"):
            if _finite_number(name, getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("ent_coef", "vf_coef"):
            if _finite_number(name, getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if (
            not self.net_arch
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in self.net_arch
            )
        ):
            raise ValueError("net_arch must contain positive integer layer sizes")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")


@dataclass(frozen=True)
class TrainingOptions:
    """Runtime-only training budget, logging, and checkpoint options."""

    total_timesteps: int
    seed: int | None = None
    tensorboard_log: str | None = None
    checkpoint_dir: str | None = None
    checkpoint_freq: int | None = None
    checkpoint_prefix: str = "synthetic_ppo"
    tb_log_name: str = "synthetic_ppo"
    log_interval: int = 1
    verbose: int = 1

    def validate(self) -> None:
        if (
            isinstance(self.total_timesteps, bool)
            or not isinstance(self.total_timesteps, int)
            or self.total_timesteps <= 0
        ):
            raise ValueError("total_timesteps must be a positive integer")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if (self.checkpoint_dir is None) != (self.checkpoint_freq is None):
            raise ValueError("checkpoint_dir and checkpoint_freq must be set together")
        if self.checkpoint_freq is not None and (
            isinstance(self.checkpoint_freq, bool)
            or not isinstance(self.checkpoint_freq, int)
            or self.checkpoint_freq <= 0
        ):
            raise ValueError("checkpoint_freq must be a positive integer")
        if (
            isinstance(self.log_interval, bool)
            or not isinstance(self.log_interval, int)
            or self.log_interval <= 0
        ):
            raise ValueError("log_interval must be a positive integer")
        if (
            isinstance(self.verbose, bool)
            or not isinstance(self.verbose, int)
            or self.verbose not in (0, 1, 2)
        ):
            raise ValueError("verbose must be 0, 1, or 2")
        if (
            not isinstance(self.checkpoint_prefix, str)
            or not self.checkpoint_prefix.strip()
            or not isinstance(self.tb_log_name, str)
            or not self.tb_log_name.strip()
        ):
            raise ValueError("checkpoint_prefix and tb_log_name must not be empty")


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number
