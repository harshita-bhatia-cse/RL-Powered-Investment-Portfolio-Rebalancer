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
from src.rl.environment.synthetic_portfolio_env import (
    DEFAULT_OBSERVATION_SCHEMA,
    OBSERVATION_SCHEMA_WITHOUT_TIME,
    SyntheticPortfolioEnv,
)
from src.rl.phase7_experiment import (
    build_phase72_run_metadata,
    load_phase72_protocol,
)
from src.rl.synthetic_data import (
    SyntheticMarketConfig,
    generate_synthetic_market,
    generate_synthetic_splits,
)
from src.rl.synthetic_pipeline import RandomizedEpisodeHorizonEnv


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

    def test_training_horizon_randomization_is_opt_in_and_training_only(self):
        fixed_bundle = build_ppo_environments(
            SyntheticMarketConfig(sessions=130), seed=411
        )
        self.assertIs(
            fixed_bundle.train_env.unwrapped,
            fixed_bundle.pipeline.environments["train"],
        )
        _, fixed_info = fixed_bundle.train_env.reset(seed=4)
        self.assertNotIn("episode_horizon_decisions", fixed_info)
        self.assertEqual(len(fixed_bundle.train_env.unwrapped.asset_returns), 130)

        randomized_bundle = build_ppo_environments(
            SyntheticMarketConfig(sessions=130),
            seed=412,
            randomize_training_horizons=True,
            min_episode_decisions=1,
            max_episode_decisions=3,
            horizon_seed=93,
        )
        self.assertIsInstance(
            randomized_bundle.train_env.env,
            RandomizedEpisodeHorizonEnv,
        )
        self.assertIsInstance(randomized_bundle.validation_env, SyntheticPortfolioEnv)
        self.assertIsInstance(randomized_bundle.test_env, SyntheticPortfolioEnv)
        self.assertEqual(
            len(randomized_bundle.validation_env.asset_returns),
            130,
        )
        self.assertEqual(len(randomized_bundle.test_env.asset_returns), 130)
        _, train_info = randomized_bundle.train_env.reset()
        self.assertIn(train_info["episode_horizon_decisions"], (1, 2, 3))
        self.assertNotIn(
            "episode_horizon_decisions",
            randomized_bundle.validation_env.reset(seed=5)[1],
        )
        self.assertNotIn(
            "episode_horizon_decisions",
            randomized_bundle.test_env.reset(seed=5)[1],
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
        for index, step in enumerate(result.steps):
            self.assertEqual(step.observation.shape, (43,))
            self.assertEqual(step.decision_observation.shape, (43,))
            self.assertTrue(step.decision_observation.flags.writeable is False)
            if index == 0:
                np.testing.assert_array_equal(
                    step.decision_observation,
                    result.initial_observation,
                )
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

    def test_no_time_schema_agent_manifest_load_and_mismatch_validation(self):
        no_time_bundle = build_ppo_environments(
            SyntheticMarketConfig(sessions=130),
            seed=112,
            observation_schema=OBSERVATION_SCHEMA_WITHOUT_TIME,
        )
        self.assertEqual(
            no_time_bundle.train_env.observation_space.shape,
            (42,),
        )
        self.assertEqual(
            no_time_bundle.validation_env.observation_schema,
            OBSERVATION_SCHEMA_WITHOUT_TIME,
        )
        model = create_ppo_agent(no_time_bundle.train_env, self.config)
        observation, _ = no_time_bundle.validation_env.reset(seed=6)
        self.assertEqual(model.observation_space.shape, (42,))
        self.assertEqual(model.observation_schema, OBSERVATION_SCHEMA_WITHOUT_TIME)

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path, manifest_path = save_ppo_model(
                model,
                Path(temporary_directory) / "no_time_model",
                config=self.config,
                seed=17,
                provenance=no_time_bundle.pipeline.datasets.train.metadata,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["observation_schema"],
                OBSERVATION_SCHEMA_WITHOUT_TIME,
            )
            self.assertEqual(manifest["observation_schema_version"], 1)
            self.assertEqual(manifest["observation_shape"], [42])

            loaded = load_ppo_model(model_path, env=no_time_bundle.validation_env)
            self.assertEqual(loaded.observation_schema, OBSERVATION_SCHEMA_WITHOUT_TIME)
            result = evaluate_ppo(loaded, no_time_bundle.validation_env, max_steps=1)
            self.assertEqual(result.observation_schema, OBSERVATION_SCHEMA_WITHOUT_TIME)
            with self.assertRaisesRegex(ValueError, "incompatible with environment"):
                load_ppo_model(model_path, env=self.bundle.validation_env)

        with self.assertRaisesRegex(
            ValueError,
            "must be run as separate configurations",
        ):
            build_ppo_environments(
                SyntheticMarketConfig(sessions=130),
                seed=112,
                randomize_training_horizons=True,
                observation_schema=OBSERVATION_SCHEMA_WITHOUT_TIME,
            )

    def test_schema_less_legacy_manifest_remains_43_feature_compatible(self):
        model = create_ppo_agent(self.bundle.train_env, self.config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path, manifest_path = save_ppo_model(
                model,
                Path(temporary_directory) / "legacy_model",
                config=self.config,
                seed=17,
                provenance=self.bundle.pipeline.datasets.train.metadata,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["observation_schema"],
                DEFAULT_OBSERVATION_SCHEMA,
            )
            manifest.pop("observation_schema")
            manifest.pop("observation_schema_version")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_ppo_model(model_path, env=self.bundle.validation_env)
            self.assertEqual(loaded.observation_schema, DEFAULT_OBSERVATION_SCHEMA)

    def test_phase72_metadata_is_saved_and_checked_with_model_manifest(self):
        protocol = load_phase72_protocol()
        dataset = generate_synthetic_splits(
            SyntheticMarketConfig(),
            seed=101,
        ).train
        protocol_ppo_settings = dict(
            protocol["shared_settings"]["ppo_settings"]
        )
        protocol_ppo_settings.pop("seed_role")
        protocol_ppo_settings["seed"] = 101
        protocol_ppo_settings["net_arch"] = tuple(
            protocol_ppo_settings["net_arch"]
        )
        config = PPOConfig(**protocol_ppo_settings)
        model = create_ppo_agent(self.bundle.train_env, config)
        run_metadata = build_phase72_run_metadata(
            protocol,
            configuration="A",
            training_seed=101,
            training_provenance=dataset.metadata,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path, manifest_path = save_ppo_model(
                model,
                Path(temporary_directory) / "phase72_model",
                config=config,
                seed=101,
                provenance=dataset.metadata,
                experiment_metadata=run_metadata,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["experiment_metadata"], run_metadata)
            loaded = load_ppo_model(model_path)
            self.assertEqual(loaded.experiment_metadata, run_metadata)
            model_bytes = model_path.read_bytes()
            manifest_bytes = manifest_path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                save_ppo_model(
                    model,
                    model_path,
                    config=config,
                    seed=101,
                    provenance=dataset.metadata,
                    experiment_metadata=run_metadata,
                )
            self.assertEqual(model_path.read_bytes(), model_bytes)
            self.assertEqual(manifest_path.read_bytes(), manifest_bytes)

            inconsistent_metadata = json.loads(json.dumps(run_metadata))
            inconsistent_metadata["ppo_settings"]["gamma"] = 0.9
            with self.assertRaisesRegex(ValueError, "PPO settings"):
                save_ppo_model(
                    model,
                    Path(temporary_directory) / "rejected_model",
                    config=config,
                    seed=101,
                    provenance=dataset.metadata,
                    experiment_metadata=inconsistent_metadata,
                )
            self.assertFalse(
                (Path(temporary_directory) / "rejected_model.zip").exists()
            )

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
