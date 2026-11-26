"""Evaluate one selected synthetic PPO model on the fixed held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.rl.phase4_evaluation import evaluate_phase4
from src.rl.ppo_agent import build_ppo_environments, load_ppo_model
from src.rl.ppo_config import PPOConfig


def _non_negative_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seed must be a non-negative integer"
        ) from error
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer")
    return seed


def evaluate_final_test(
    model_path: str | Path,
    *,
    scenario_seed: int = 42,
) -> dict[str, object]:
    """Evaluate one model on the test split without training or selecting models."""
    bundle = build_ppo_environments(seed=scenario_seed)
    model = load_ppo_model(
        model_path,
        env=bundle.test_env,
        expected_config=PPOConfig(seed=42, device="cpu"),
    )
    return {
        "evaluation_stage": "final_test",
        "source_type": "synthetic",
        "synthetic": True,
        "scenario_root_seed": scenario_seed,
        "scenario_id": bundle.pipeline.datasets.test.metadata["scenario_id"],
        "evaluation": evaluate_phase4(model, bundle.test_env),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--scenario-seed", type=_non_negative_seed, default=42)
    parser.add_argument(
        "--output-path",
        type=Path,
        help="JSON result path; defaults to final_test_evaluation.json beside the model.",
    )
    args = parser.parse_args(argv)

    model_path = args.model_path.expanduser().resolve()
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else model_path.parent / "final_test_evaluation.json"
    )
    repository_artifacts = Path(__file__).resolve().parents[1] / ".artifacts"
    protected_baseline = (repository_artifacts / "ppo").resolve()
    if output_path == protected_baseline or protected_baseline in output_path.parents:
        raise ValueError(
            "Final test results must not be written inside the protected "
            ".artifacts/ppo baseline directory"
        )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite test results: {output_path}")

    report = evaluate_final_test(
        model_path,
        scenario_seed=args.scenario_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Final test evaluation: {output_path}")


if __name__ == "__main__":
    main()
