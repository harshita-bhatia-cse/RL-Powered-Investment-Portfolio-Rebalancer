import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rl.ppo_agent import (
    build_ppo_environments,
    create_ppo_agent,
    evaluate_ppo,
    load_ppo_model,
    save_ppo_model,
    train_ppo,
)
from src.rl.ppo_config import PPOConfig, TrainingOptions
from src.rl.synthetic_data import SyntheticMarketConfig, generate_synthetic_market


class PPOAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import stable_baselines3
            import torch
        except ImportError as error:
            raise unittest.SkipTest(
                "PPO plumbing tests require the optional Stable-Baselines3/Torch dependencies"
            ) from error
        cls.sb3_version = stable_baselines3.__version__

    def setUp(self):
        self.config = PPOConfig(
            n_steps=8,
            batch_size=4,
            net_arch=(16, 16),
            seed=17,
            device="cpu",
        )
        self.bundle = build_ppo_environments(
            SyntheticMarketConfig(sessions=130), seed=111
        )

    def test_ppo_configuration_validation_and_constructor_values(self):
        self.config.validate()
        self.assertEqual(self.config.n_steps, 8)
        self.assertEqual(self.config.batch_size, 4)
        self.assertEqual(self.config.gamma, 0.99)
        self.assertEqual(self.config.gae_lambda, 0.95)
        self.assertEqual(self.config.net_arch, (16, 16))
        self.assertEqual(self.config.seed, 17)

        with self.assertRaisesRegex(ValueError, "divide n_steps"):
            PPOConfig(n_steps=8, batch_size=3).validate()
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            PPOConfig(gamma=1.1).validate()
        with self.assertRaisesRegex(ValueError, "positive"):
            PPOConfig(learning_rate=0.0).validate()

    def test_training_options_validate_without_starting_training(self):
        TrainingOptions(total_timesteps=50, seed=9).validate()
        TrainingOptions(total_timesteps=50, seed=9, verbose=2).validate()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            TrainingOptions(total_timesteps=0).validate()
        with self.assertRaisesRegex(ValueError, "verbose must be"):
            TrainingOptions(total_timesteps=50, verbose=3).validate()
        with self.assertRaisesRegex(ValueError, "set together"):
            TrainingOptions(
                total_timesteps=10,
                checkpoint_dir="checkpoints",
            ).validate()
        signature = inspect.signature(train_ppo)
        self.assertIn("total_timesteps", signature.parameters)
        self.assertIn("config", signature.parameters)
        self.assertIn("checkpoint_freq", signature.parameters)
        self.assertIn("tensorboard_log", signature.parameters)
        self.assertIn("verbose", signature.parameters)

    def test_agent_verbosity_can_enable_training_logs(self):
        model = create_ppo_agent(
            self.bundle.train_env, self.config, verbose=1
        )
        self.assertEqual(model.verbose, 1)

    def test_factory_wraps_only_training_scenario(self):
        from stable_baselines3.common.monitor import Monitor

        self.assertIsInstance(self.bundle.train_env, Monitor)
        self.assertNotIsInstance(self.bundle.validation_env, Monitor)
        self.assertNotIsInstance(self.bundle.test_env, Monitor)
        self.assertEqual(
            self.bundle.train_env.unwrapped.data_metadata["scenario_id"],
            self.bundle.pipeline.datasets.train.metadata["scenario_id"],
        )
        self.assertEqual(
            self.bundle.validation_env.data_metadata["scenario_id"],
            self.bundle.pipeline.datasets.validation.metadata["scenario_id"],
        )
        self.assertEqual(
            self.bundle.test_env.data_metadata["scenario_id"],
            self.bundle.pipeline.datasets.test.metadata["scenario_id"],
        )
        self.assertEqual(
            len(
                {
                    self.bundle.pipeline.datasets.train.seed,
                    self.bundle.pipeline.datasets.validation.seed,
                    self.bundle.pipeline.datasets.test.seed,
                }
            ),
            3,
        )

    def test_agent_dimensions_and_deterministic_prediction(self):
        model = create_ppo_agent(self.bundle.train_env, self.config)
        self.assertEqual(model.observation_space.shape, (43,))
        self.assertEqual(model.action_space.shape, (7,))

        observation, _ = self.bundle.validation_env.reset(seed=22)
        first_action, _ = model.predict(observation, deterministic=True)
        second_action, _ = model.predict(observation, deterministic=True)

        self.assertEqual(first_action.shape, (7,))
        self.assertTrue(np.isfinite(first_action).all())
        self.assertTrue(self.bundle.validation_env.action_space.contains(first_action))
        np.testing.assert_array_equal(first_action, second_action)

    def test_deterministic_evaluation_returns_structured_synthetic_results(self):
        model = create_ppo_agent(self.bundle.train_env, self.config)
        result = evaluate_ppo(model, self.bundle.validation_env, max_steps=2)

        self.assertEqual(result.source_type, "synthetic")
        self.assertTrue(result.synthetic)
        self.assertEqual(
            result.scenario_id,
            self.bundle.pipeline.datasets.validation.metadata["scenario_id"],
        )
        self.assertEqual(result.seed, self.bundle.pipeline.datasets.validation.seed)
        self.assertEqual(result.initial_observation.shape, (43,))
        self.assertEqual(len(result.steps), 2)
        for step in result.steps:
            self.assertEqual(step.observation.shape, (43,))
            self.assertEqual(step.action.shape, (7,))
            self.assertTrue(np.isfinite(step.action).all())
            self.assertEqual(step.requested_weights.keys(), step.applied_weights.keys())
            self.assertAlmostEqual(sum(step.requested_weights.values()), 1.0, places=8)
            self.assertAlmostEqual(sum(step.applied_weights.values()), 1.0, places=8)
            self.assertTrue(all(weight >= 0.0 for weight in step.applied_weights.values()))
            self.assertTrue(np.isfinite(step.reward))
            self.assertGreater(step.nav, 0)
            self.assertEqual(len(step.daily_navs), 21)
            self.assertTrue(np.isfinite(step.daily_navs).all())
            self.assertAlmostEqual(step.daily_navs[-1], step.nav, places=7)
            self.assertGreaterEqual(step.gross_turnover, 0.0)
            self.assertAlmostEqual(
                step.one_way_turnover, step.gross_turnover / 2.0
            )
            self.assertAlmostEqual(
                step.transaction_cost,
                step.transaction_fees + step.slippage_cost,
            )
            self.assertAlmostEqual(sum(step.weights.values()), 1.0, places=8)
            self.assertFalse(step.terminated)
            self.assertFalse(step.truncated)

    def test_model_save_load_round_trip_and_manifest_provenance(self):
        dataset = generate_synthetic_market(
            SyntheticMarketConfig(sessions=90), seed=222, scenario_id="ppo-test"
        )
        model = create_ppo_agent(self.bundle.train_env, self.config)
        observation, _ = self.bundle.validation_env.reset(seed=4)
        expected_action, _ = model.predict(observation, deterministic=True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path, manifest_path = save_ppo_model(
                model,
                Path(temporary_directory) / "synthetic_model",
                config=self.config,
                seed=self.config.seed,
                provenance=dataset.metadata,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["synthetic"])
            self.assertEqual(manifest["source_type"], "synthetic")
            self.assertEqual(manifest["seed"], self.config.seed)
            self.assertEqual(manifest["provenance"]["seed"], dataset.seed)
            self.assertEqual(manifest["asset_order"], list(dataset.asset_names))
            self.assertEqual(
                manifest["provenance"]["generation_config"],
                json.loads(json.dumps(dataset.metadata["generation_config"])),
            )
            self.assertIn("stable_baselines3", manifest["versions"])
            self.assertIn("torch", manifest["versions"])

            loaded = load_ppo_model(
                model_path,
                env=self.bundle.validation_env,
                expected_config=self.config,
            )
            actual_action, _ = loaded.predict(observation, deterministic=True)
            np.testing.assert_array_equal(expected_action, actual_action)

    def test_model_load_rejects_bad_manifest_and_missing_sidecar(self):
        model = create_ppo_agent(self.bundle.train_env, self.config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path, manifest_path = save_ppo_model(
                model,
                Path(temporary_directory) / "model.zip",
                config=self.config,
                seed=17,
                provenance={
                    "source_type": "synthetic",
                    "synthetic": True,
                    "seed": 17,
                    "scenario_id": "train-test",
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_type"] = "historical"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "synthetic-only"):
                load_ppo_model(model_path)

            manifest_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "manifest"):
                load_ppo_model(model_path)

    def test_persistence_requires_synthetic_provenance_and_matching_config(self):
        model = create_ppo_agent(self.bundle.train_env, self.config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "synthetic-only provenance"):
                save_ppo_model(
                    model,
                    Path(temporary_directory) / "bad_model",
                    config=self.config,
                    seed=17,
                    provenance={"source_type": "historical", "synthetic": False},
                )
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                save_ppo_model(
                    model,
                    Path(temporary_directory) / "bad_seed_model",
                    config=self.config,
                    seed=17,
                    provenance={
                        "source_type": "synthetic",
                        "synthetic": True,
                        "seed": "invalid",
                    },
                )

    def test_validation_and_test_environments_are_required_synthetic_inputs(self):
        invalid_env = self.bundle.validation_env
        invalid_env.data_metadata["source_type"] = "historical"
        with self.assertRaisesRegex(ValueError, "synthetic data"):
            create_ppo_agent(invalid_env, self.config)


if __name__ == "__main__":
    unittest.main()
