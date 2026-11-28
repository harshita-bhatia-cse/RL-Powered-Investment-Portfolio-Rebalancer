import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.rl.environment.synthetic_portfolio_env import SyntheticPortfolioEnv
from src.rl.phase4_training import TrainingOrchestrator
from src.rl.phase7_experiment import (
    CONFIGURATIONS,
    OBSERVATION_SCHEMA_WITHOUT_TIME,
    OBSERVATION_SCHEMA_VERSION,
    Phase72Run,
    build_phase72_run_matrix,
    build_phase72_run_metadata,
    load_phase72_protocol,
    reserve_phase72_output_directories,
    validate_phase72_protocol,
    validate_phase72_run_directory,
    validate_phase72_run_metadata,
)
from src.rl.ppo_config import PPOConfig, TrainingOptions
from src.rl.synthetic_data import SyntheticMarketConfig, generate_synthetic_splits
from src.rl.synthetic_pipeline import RandomizedEpisodeHorizonEnv


class Phase72ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_phase72_protocol()

    def test_protocol_records_disjoint_seed_roles_and_matching_configurations(self):
        validate_phase72_protocol(self.protocol)
        roles = self.protocol["seed_roles"]
        self.assertEqual(roles["training_seeds"], [101, 202, 303, 404, 505])
        self.assertEqual(
            roles["validation_scenario_roots"],
            list(range(7201, 7221)),
        )
        self.assertEqual(
            roles["reserved_test_scenario_roots"],
            list(range(9201, 9221)),
        )
        self.assertEqual(set(self.protocol["configurations"]), set(CONFIGURATIONS))
        shared = self.protocol["shared_settings"]
        self.assertEqual(
            shared["evaluation"]["evaluation_sessions"],
            shared["synthetic_generator"]["sessions"]
            - shared["environment"]["warmup_sessions"],
        )
        self.assertEqual(
            shared["evaluation"]["rebalance_decisions"],
            shared["evaluation"]["evaluation_sessions"]
            // shared["environment"]["period_sessions"],
        )

        overlapping = copy.deepcopy(self.protocol)
        overlapping["seed_roles"]["validation_scenario_roots"][0] = 101
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            validate_phase72_protocol(overlapping)

    def test_retry_run_directory_rejects_reparse_point_components(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            output_root = repository / ".artifacts" / "phase7_2"
            retry_root = output_root / "retries"
            specification = self.protocol["configurations"]["A"]
            run = Phase72Run(
                configuration="A",
                training_seed=101,
                output_directory=(
                    retry_root / "attempt_0002" / "config_A" / "seed_101"
                ),
                observation_schema=specification["observation_schema"],
                randomize_training_horizons=specification[
                    "randomize_training_horizons"
                ],
                horizon_seed=None,
                reservation_token="reservation",
                attempt_id="attempt_0002",
            )
            with patch(
                "src.rl.phase7_experiment._path_is_reparse_point",
                side_effect=lambda path: path == retry_root,
            ), self.assertRaisesRegex(ValueError, "reparse points"):
                validate_phase72_run_directory(
                    self.protocol,
                    run,
                    repository_root=repository,
                )

        configuration_override = copy.deepcopy(self.protocol)
        configuration_override["configurations"]["B"]["ppo_settings"] = {
            "gamma": 0.9
        }
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_phase72_protocol(configuration_override)

    def test_same_id_and_version_protocol_mutations_are_rejected(self):
        mutations = (
            lambda protocol: protocol["shared_settings"]["training_budget"].update(
                {"requested_total_timesteps": 10_241}
            ),
            lambda protocol: protocol["shared_settings"]["ppo_settings"].update(
                {"gamma": 0.9}
            ),
            lambda protocol: protocol["shared_settings"]["evaluation"].update(
                {"evaluation_sessions": 672}
            ),
            lambda protocol: protocol["seed_roles"]["validation_scenario_roots"].__setitem__(
                0, 7301
            ),
            lambda protocol: protocol["configurations"]["B"].update(
                {"randomize_training_horizons": False}
            ),
        )
        for mutate in mutations:
            changed = copy.deepcopy(self.protocol)
            mutate(changed)
            self.assertEqual(changed["protocol_id"], self.protocol["protocol_id"])
            self.assertEqual(
                changed["protocol_version"], self.protocol["protocol_version"]
            )
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_phase72_protocol(changed)

    def test_custom_protocol_path_cannot_bypass_approved_contents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "modified-protocol.json"
            changed = copy.deepcopy(self.protocol)
            changed["shared_settings"]["training_budget"][
                "requested_total_timesteps"
            ] += 1
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "approved Phase 7.2 protocol"):
                load_phase72_protocol(path)

    def test_output_matrix_is_unique_and_reserves_all_paths_only_after_preflight(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory) / "repo"
            output_root = repository / ".artifacts" / "phase7_2"
            plans = build_phase72_run_matrix(
                self.protocol,
                output_root=output_root,
                repository_root=repository,
            )
            self.assertEqual(len(plans), 15)
            self.assertEqual(len({plan.output_directory for plan in plans}), 15)
            self.assertEqual(
                len({(plan.configuration, plan.training_seed) for plan in plans}),
                15,
            )
            self.assertTrue(
                all(
                    plan.output_directory
                    != repository / ".artifacts" / "ppo"
                    and repository / ".artifacts" / "ppo"
                    not in plan.output_directory.parents
                    for plan in plans
                )
            )
            reserved = reserve_phase72_output_directories(
                self.protocol,
                output_root=output_root,
                repository_root=repository,
            )
            self.assertEqual(len(reserved), 15)
            self.assertTrue(all(plan.output_directory.is_dir() for plan in reserved))

    def test_concurrent_reservations_cannot_claim_the_same_tree(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory) / "repo"
            output_root = repository / ".artifacts" / "phase7_2"
            barrier = threading.Barrier(2)
            from src.rl import phase7_experiment

            original_builder = phase7_experiment.build_phase72_run_matrix

            def synchronized_builder(*args, **kwargs):
                runs = original_builder(*args, **kwargs)
                barrier.wait(timeout=5)
                return runs

            outcomes = []
            errors = []

            def reserve():
                try:
                    outcomes.append(
                        reserve_phase72_output_directories(
                            self.protocol,
                            output_root=output_root,
                            repository_root=repository,
                        )
                    )
                except Exception as error:
                    errors.append(error)

            with patch(
                "src.rl.phase7_experiment.build_phase72_run_matrix",
                side_effect=synchronized_builder,
            ):
                workers = [threading.Thread(target=reserve) for _ in range(2)]
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=10)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(len(errors), 1)
            error = errors[0]
            self.assertTrue(
                isinstance(error, FileExistsError)
                or (
                    isinstance(error, RuntimeError)
                    and "reservation failed" in str(error)
                ),
                repr(error),
            )
            winning_directories = {run.output_directory for run in outcomes[0]}
            actual_directories = set(output_root.glob("config_*/seed_*"))
            self.assertEqual(len(winning_directories), 15)
            self.assertEqual(actual_directories, winning_directories)
            self.assertTrue(all(path.is_dir() for path in winning_directories))

    def test_failed_partial_reservation_rolls_back_only_empty_created_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory) / "repo"
            output_root = repository / ".artifacts" / "phase7_2"
            original_mkdir = Path.mkdir

            def fail_on_second_config(path, *args, **kwargs):
                if path == output_root / "config_B":
                    raise OSError("injected reservation failure")
                return original_mkdir(path, *args, **kwargs)

            with patch.object(Path, "mkdir", fail_on_second_config):
                with self.assertRaisesRegex(OSError, "injected reservation failure"):
                    reserve_phase72_output_directories(
                        self.protocol,
                        output_root=output_root,
                        repository_root=repository,
                    )
            self.assertFalse(output_root.exists())
            self.assertFalse((repository / ".artifacts").exists())
            reserved = reserve_phase72_output_directories(
                self.protocol,
                output_root=output_root,
                repository_root=repository,
            )
            self.assertEqual(len(reserved), 15)

    def test_nonempty_run_directory_is_rejected_before_other_paths_are_created(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory) / "repo"
            output_root = repository / ".artifacts" / "phase7_2"
            collision = output_root / "config_A" / "seed_101"
            collision.mkdir(parents=True)
            sentinel = collision / "existing.json"
            sentinel.write_text('{"keep": true}', encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                reserve_phase72_output_directories(
                    self.protocol,
                    output_root=output_root,
                    repository_root=repository,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), '{"keep": true}')
            self.assertFalse((output_root / "config_B" / "seed_101").exists())

    def test_existing_ppo_model_directory_is_protected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory) / "repo"
            protected = repository / ".artifacts" / "ppo"
            protected.mkdir(parents=True)
            model = protected / "model.zip"
            model.write_bytes(b"original-model")

            with self.assertRaisesRegex(ValueError, r"never use.*\.artifacts/ppo"):
                reserve_phase72_output_directories(
                    self.protocol,
                    output_root=protected,
                    repository_root=repository,
                )

            self.assertEqual(model.read_bytes(), b"original-model")

    def test_all_run_metadata_has_correct_schema_and_b_records_horizons(self):
        training_data = generate_synthetic_splits(
            SyntheticMarketConfig(),
            seed=101,
        ).train.metadata
        expected_schemas = {
            "A": "synthetic_portfolio_v1_with_time",
            "B": "synthetic_portfolio_v1_with_time",
            "C": "synthetic_portfolio_v1_without_time",
        }
        metadata_by_configuration = {}
        for configuration in CONFIGURATIONS:
            metadata = build_phase72_run_metadata(
                self.protocol,
                configuration=configuration,
                training_seed=101,
                training_provenance=training_data,
            )
            metadata_by_configuration[configuration] = metadata
            self.assertEqual(
                metadata["observation_schema"],
                expected_schemas[configuration],
            )
            self.assertEqual(
                metadata["observation_schema_version"],
                OBSERVATION_SCHEMA_VERSION,
            )
            self.assertEqual(metadata["training_seed"], 101)
            self.assertEqual(
                metadata["training_budget"],
                self.protocol["shared_settings"]["training_budget"],
            )
            self.assertEqual(
                metadata["ppo_settings"]["seed"],
                metadata["training_seed"],
            )
            validate_phase72_run_metadata(
                self.protocol,
                metadata,
                configuration=configuration,
                training_seed=101,
            )

        horizon = metadata_by_configuration["B"]["horizon_settings"]
        self.assertEqual(horizon["minimum_decisions"], 11)
        self.assertEqual(horizon["maximum_decisions"], 33)
        self.assertEqual(horizon["sampler_seed"], 101)
        self.assertIsNone(metadata_by_configuration["A"]["horizon_settings"])
        self.assertIsNone(metadata_by_configuration["C"]["horizon_settings"])

    def test_missing_or_inconsistent_run_metadata_is_rejected(self):
        training_data = generate_synthetic_splits(
            SyntheticMarketConfig(),
            seed=202,
        ).train.metadata
        metadata = build_phase72_run_metadata(
            self.protocol,
            configuration="B",
            training_seed=202,
            training_provenance=training_data,
        )

        missing_schema = copy.deepcopy(metadata)
        del missing_schema["observation_schema"]
        with self.assertRaisesRegex(ValueError, "differs from the protocol"):
            validate_phase72_run_metadata(
                self.protocol,
                missing_schema,
                configuration="B",
                training_seed=202,
            )

        incorrect_budget = copy.deepcopy(metadata)
        incorrect_budget["training_budget"]["requested_total_timesteps"] += 1
        with self.assertRaisesRegex(ValueError, "differs from the protocol"):
            validate_phase72_run_metadata(
                self.protocol,
                incorrect_budget,
                configuration="B",
                training_seed=202,
            )

        missing_horizon = copy.deepcopy(metadata)
        missing_horizon["horizon_settings"] = None
        with self.assertRaisesRegex(ValueError, "differs from the protocol"):
            validate_phase72_run_metadata(
                self.protocol,
                missing_horizon,
                configuration="B",
                training_seed=202,
            )

    def test_training_orchestrator_validates_phase72_run_before_training(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory) / "repo"
            plans = reserve_phase72_output_directories(
                self.protocol,
                output_root=repository / ".artifacts" / "phase7_2",
                repository_root=repository,
            )
            run = next(
                plan
                for plan in plans
                if plan.configuration == "A" and plan.training_seed == 101
            )
            training_data = generate_synthetic_splits(
                SyntheticMarketConfig(),
                seed=101,
            ).train
            env = SyntheticPortfolioEnv(**training_data.environment_kwargs())
            orchestrator = TrainingOrchestrator(
                config=PPOConfig(seed=101, device="cpu"),
                seed=101,
                output_dir=run.output_directory,
                repository_root=repository,
            )
            options = TrainingOptions(total_timesteps=10000, seed=101, verbose=0)

            run_metadata = orchestrator._validate_phase72_run(
                env,
                base_env=env,
                options=options,
                protocol=self.protocol,
                run=run,
            )
            self.assertEqual(run_metadata["configuration"], "A")

            incorrect_options = TrainingOptions(
                total_timesteps=9999,
                seed=101,
                verbose=0,
            )
            with self.assertRaisesRegex(ValueError, "budget"):
                orchestrator._validate_phase72_run(
                    env,
                    base_env=env,
                    options=incorrect_options,
                    protocol=self.protocol,
                    run=run,
                )

    def test_training_orchestrator_accepts_protocol_b_and_c_environment_shapes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory) / "repo"
            plans = reserve_phase72_output_directories(
                self.protocol,
                output_root=repository / ".artifacts" / "phase7_2",
                repository_root=repository,
            )
            plan_by_key = {
                (plan.configuration, plan.training_seed): plan for plan in plans
            }

            b_data = generate_synthetic_splits(
                SyntheticMarketConfig(),
                seed=202,
            ).train
            b_base_env = SyntheticPortfolioEnv(**b_data.environment_kwargs())
            b_env = RandomizedEpisodeHorizonEnv(
                b_base_env,
                min_episode_decisions=11,
                max_episode_decisions=33,
                seed=202,
            )
            b_run = plan_by_key[("B", 202)]
            b_orchestrator = TrainingOrchestrator(
                config=PPOConfig(seed=202, device="cpu"),
                seed=202,
                output_dir=b_run.output_directory,
                repository_root=repository,
            )
            b_metadata = b_orchestrator._validate_phase72_run(
                b_env,
                base_env=b_base_env,
                options=TrainingOptions(
                    total_timesteps=10000,
                    seed=202,
                    verbose=0,
                ),
                protocol=self.protocol,
                run=b_run,
            )
            self.assertEqual(b_metadata["horizon_settings"]["sampler_seed"], 202)

            c_data = generate_synthetic_splits(
                SyntheticMarketConfig(),
                seed=303,
            ).train
            c_env = SyntheticPortfolioEnv(
                **c_data.environment_kwargs(),
                observation_schema=OBSERVATION_SCHEMA_WITHOUT_TIME,
            )
            c_run = plan_by_key[("C", 303)]
            c_orchestrator = TrainingOrchestrator(
                config=PPOConfig(seed=303, device="cpu"),
                seed=303,
                output_dir=c_run.output_directory,
                repository_root=repository,
            )
            c_metadata = c_orchestrator._validate_phase72_run(
                c_env,
                base_env=c_env,
                options=TrainingOptions(
                    total_timesteps=10000,
                    seed=303,
                    verbose=0,
                ),
                protocol=self.protocol,
                run=c_run,
            )
            self.assertEqual(c_metadata["observation_shape"], [42])


if __name__ == "__main__":
    unittest.main()
