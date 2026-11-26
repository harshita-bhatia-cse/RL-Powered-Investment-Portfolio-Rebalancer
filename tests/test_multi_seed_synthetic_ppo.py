import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts import evaluate_synthetic_ppo_test, train_synthetic_ppo_dashboard
from src.rl.ppo_agent import (
    build_ppo_environments,
    create_ppo_agent,
    load_ppo_model,
    save_ppo_model,
)
from src.rl.ppo_config import PPOConfig
from src.rl.synthetic_data import SyntheticMarketConfig, generate_synthetic_splits


class MultiSeedSyntheticPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import stable_baselines3
            import torch
        except ImportError as error:
            raise unittest.SkipTest(
                "Multi-seed model compatibility tests require Stable-Baselines3/Torch"
            ) from error

    def test_training_seeds_vary_while_scenario_root_and_validation_stay_fixed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scenario_metadata = {
                split: SimpleNamespace(
                    metadata={"scenario_id": f"{split}-scenario-42"},
                    seed=4200 + index,
                )
                for index, split in enumerate(("train", "validation", "test"))
            }
            scenario_splits_by_run = []
            for training_seed in (42, 7, 21, 123, 2026):
                with self.subTest(training_seed=training_seed):
                    train_env = object()
                    validation_env = object()
                    test_env = object()
                    environments = SimpleNamespace(
                        train_env=train_env,
                        validation_env=validation_env,
                        test_env=test_env,
                        pipeline=SimpleNamespace(
                            datasets=SimpleNamespace(**scenario_metadata)
                        ),
                    )
                    metadata = {}
                    model_path = root / f"seed_{training_seed}" / "model.zip"
                    manifest_path = model_path.with_suffix(".manifest.json")
                    orchestrator = SimpleNamespace(
                        train=Mock(
                            return_value=(
                                SimpleNamespace(model=object()),
                                metadata,
                                model_path,
                                manifest_path,
                            )
                        )
                    )
                    output_dir = root / f"seed_{training_seed}"
                    with patch(
                        "scripts.train_synthetic_ppo_dashboard.build_ppo_environments",
                        return_value=environments,
                    ) as build_environments, patch(
                        "scripts.train_synthetic_ppo_dashboard.TrainingOrchestrator",
                        return_value=orchestrator,
                    ) as orchestrator_factory, patch(
                        "scripts.train_synthetic_ppo_dashboard.evaluate_phase4",
                        return_value={"ppo": {}, "benchmarks": {}},
                    ) as evaluate, patch(
                        "scripts.train_synthetic_ppo_dashboard.save_phase4_results"
                    ), patch(
                        "scripts.train_synthetic_ppo_dashboard.save_phase4_diagnostics"
                    ), patch(
                        "scripts.train_synthetic_ppo_dashboard.save_phase4_observation_action_diagnostics"
                    ):
                        arguments = ["--output-dir", str(output_dir)]
                        if training_seed != 42:
                            arguments.extend(
                                [
                                    "--training-seed",
                                    str(training_seed),
                                    "--scenario-seed",
                                    "42",
                                ]
                            )
                        train_synthetic_ppo_dashboard.main(arguments)

                    self.assertEqual(build_environments.call_args.kwargs["seed"], 42)
                    config = orchestrator_factory.call_args.kwargs["config"]
                    self.assertEqual(config.seed, training_seed)
                    self.assertEqual(
                        orchestrator_factory.call_args.kwargs["seed"], training_seed
                    )
                    options = orchestrator.train.call_args.kwargs["options"]
                    self.assertEqual(options.seed, training_seed)
                    self.assertEqual(options.total_timesteps, 10_000)
                    self.assertEqual(
                        orchestrator.train.call_args.args[0],
                        train_env,
                    )
                    self.assertEqual(evaluate.call_args.args[1], validation_env)
                    self.assertIsNot(evaluate.call_args.args[1], test_env)
                    self.assertEqual(metadata["scenario_root_seed"], 42)
                    self.assertEqual(metadata["training_seed"], training_seed)
                    scenario_splits_by_run.append(metadata["scenario_splits"])
            self.assertEqual(scenario_splits_by_run[0], scenario_splits_by_run[1])

    def test_root_seed_reproduces_train_validation_and_test_data(self):
        config = SyntheticMarketConfig(sessions=130)
        first = generate_synthetic_splits(config, seed=42)
        second = generate_synthetic_splits(config, seed=42)

        for split_name in ("train", "validation", "test"):
            first_split = getattr(first, split_name)
            second_split = getattr(second, split_name)
            self.assertEqual(
                first_split.metadata["scenario_id"],
                second_split.metadata["scenario_id"],
            )
            np.testing.assert_array_equal(
                first_split.asset_returns,
                second_split.asset_returns,
            )
            np.testing.assert_array_equal(
                first_split.benchmark_returns,
                second_split.benchmark_returns,
            )

    def test_model_load_allows_training_seed_difference_but_checks_other_settings(self):
        train_config = PPOConfig(
            n_steps=8,
            batch_size=4,
            net_arch=(16, 16),
            seed=7,
            device="cpu",
        )
        bundle = build_ppo_environments(
            SyntheticMarketConfig(sessions=130),
            seed=42,
        )
        model = create_ppo_agent(bundle.train_env, train_config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path, _ = save_ppo_model(
                model,
                Path(temporary_directory) / "model",
                config=train_config,
                seed=7,
                provenance=bundle.pipeline.datasets.train.metadata,
            )
            expected_config = PPOConfig(
                n_steps=8,
                batch_size=4,
                net_arch=(16, 16),
                seed=42,
                device="cpu",
            )
            loaded = load_ppo_model(
                model_path,
                env=bundle.validation_env,
                expected_config=expected_config,
            )
            self.assertEqual(loaded.observation_schema, model.observation_schema)

            incompatible_config = PPOConfig(
                n_steps=8,
                batch_size=4,
                gamma=0.98,
                net_arch=(16, 16),
                seed=42,
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "does not match expected_config"):
                load_ppo_model(
                    model_path,
                    env=bundle.validation_env,
                    expected_config=incompatible_config,
                )

    def test_output_guard_preserves_existing_baseline_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            baseline = Path(temporary_directory) / ".artifacts" / "ppo"
            baseline.mkdir(parents=True)
            model_path = baseline / "model.zip"
            model_path.write_bytes(b"baseline-model")

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                train_synthetic_ppo_dashboard._validate_output_directory(
                    baseline,
                    legacy_output=baseline,
                )
            with self.assertRaisesRegex(ValueError, "inside the protected"):
                train_synthetic_ppo_dashboard._validate_output_directory(
                    baseline / "seed_7",
                    legacy_output=baseline,
                )
            self.assertEqual(model_path.read_bytes(), b"baseline-model")

    def test_output_guard_allows_dashboard_log_but_not_existing_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "dashboard-run"
            output_dir.mkdir()
            (output_dir / "training.log").write_text("", encoding="utf-8")

            accepted = train_synthetic_ppo_dashboard._validate_output_directory(
                output_dir,
                legacy_output=Path(temporary_directory) / ".artifacts" / "ppo",
            )
            self.assertEqual(accepted, output_dir.resolve())

            (output_dir / "model.zip").write_bytes(b"existing-model")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                train_synthetic_ppo_dashboard._validate_output_directory(
                    output_dir,
                    legacy_output=Path(temporary_directory) / ".artifacts" / "ppo",
                )

    def test_final_test_evaluation_uses_test_environment_only(self):
        test_env = object()
        bundle = SimpleNamespace(
            test_env=test_env,
            pipeline=SimpleNamespace(
                datasets=SimpleNamespace(
                    test=SimpleNamespace(
                        metadata={"scenario_id": "test-fixed-scenario"}
                    )
                )
            ),
        )
        model = object()
        evaluation = {"ppo": {"scenario_id": "test-fixed-scenario"}}
        with patch(
            "scripts.evaluate_synthetic_ppo_test.build_ppo_environments",
            return_value=bundle,
        ) as build_environments, patch(
            "scripts.evaluate_synthetic_ppo_test.load_ppo_model",
            return_value=model,
        ) as load_model, patch(
            "scripts.evaluate_synthetic_ppo_test.evaluate_phase4",
            return_value=evaluation,
        ) as evaluate:
            report = evaluate_synthetic_ppo_test.evaluate_final_test(
                "selected-model.zip",
                scenario_seed=42,
            )

        self.assertEqual(build_environments.call_args.kwargs["seed"], 42)
        self.assertIs(load_model.call_args.kwargs["env"], test_env)
        self.assertIs(evaluate.call_args.args[1], test_env)
        self.assertEqual(report["evaluation_stage"], "final_test")
        self.assertEqual(report["scenario_id"], "test-fixed-scenario")
        self.assertEqual(report["evaluation"], evaluation)


if __name__ == "__main__":
    unittest.main()
