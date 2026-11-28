import hashlib
import json
import multiprocessing
import os
import socket
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.rl.phase7_experiment import (
    build_phase72_run_metadata,
    load_phase72_protocol,
)
from src.rl.phase7_runner import (
    FINAL_TEST_REPORT_NAME,
    RUN_MANIFEST_NAME,
    SELECTION_MANIFEST_NAME,
    Phase72ExperimentRunner,
    _create_stage_lock,
    _delete_stage_lock_handle,
    _machine_identity,
    _ppo_config,
    _release_stage_lock,
    _require_local_lock_filesystem,
    canonical_json_bytes,
    sha256_file,
)
from src.rl.synthetic_data import SyntheticMarketConfig, synthetic_configuration_hash


def _stage_lock_process_worker(output_root, barrier, release, results, role):
    root = Path(output_root)
    runner = Phase72ExperimentRunner(
        output_root=root,
        repository_root=root.parents[1],
        training_adapter=lambda protocol, run: {},
        evaluation_adapter=lambda *args: {},
    )
    try:
        barrier.wait(timeout=60)
        with runner._stage_lock():
            results.put(("acquired", role, os.getpid()))
            release.wait(timeout=60)
    except Exception as error:
        results.put(("rejected", role, type(error).__name__, str(error)))


def _finish_lock_processes(processes, release):
    release.set()
    for process in processes:
        if process.pid is None:
            continue
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


class Phase72RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_phase72_protocol()

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository = Path(self.temporary_directory.name) / "repo"
        self.output_root = self.repository / ".artifacts" / "phase7_2"
        self.training_calls = []
        self.evaluation_calls = []
        self.evaluation_model_paths = []
        self.sharpes = {"A": 1.0, "B": 1.0, "C": 1.0}
        self.drawdowns = {"A": -0.2, "B": -0.1, "C": -0.1}
        self.turnovers = {"A": 3.0, "B": 2.0, "C": 1.0}
        self.invalid_metrics = False
        self.fail_evaluation_once = False
        self.runner = self._new_runner()

    def _new_runner(self):
        return Phase72ExperimentRunner(
            output_root=self.output_root,
            repository_root=self.repository,
            training_adapter=self._fake_train,
            evaluation_adapter=self._fake_evaluate,
        )

    @staticmethod
    def _volume_information_result(filesystem: str, success: bool = True):
        def set_filesystem_name(*args):
            args[6].value = filesystem
            return int(success)

        return set_filesystem_name

    @staticmethod
    def _fake_provenance(protocol, training_seed):
        data_seed = int(
            np.random.SeedSequence(training_seed, spawn_key=(0,)).generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        return {
            "scenario_id": f"train-{data_seed}",
            "seed": data_seed,
            "generator_version": protocol["shared_settings"]["synthetic_generator"][
                "generator_version"
            ],
            "configuration_hash": synthetic_configuration_hash(
                SyntheticMarketConfig()
            ),
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "generation_config": protocol["shared_settings"]["synthetic_generator"],
        }

    def _fake_train(self, protocol, run):
        self.training_calls.append((run.configuration, run.training_seed))
        provenance = self._fake_provenance(protocol, run.training_seed)
        run_metadata = build_phase72_run_metadata(
            protocol,
            configuration=run.configuration,
            training_seed=run.training_seed,
            training_provenance={
                "source_type": "synthetic",
                "synthetic": True,
                **provenance,
            },
        )
        model_path = run.output_directory / "model.zip"
        model_path.write_bytes(
            f"{run.configuration}:{run.training_seed}".encode("ascii")
        )
        model_manifest_path = run.output_directory / "model.manifest.json"
        model_manifest = {
            "manifest_version": "1.0",
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "provenance": provenance,
            "experiment_metadata": run_metadata,
            "ppo_config": asdict(_ppo_config(protocol, run.training_seed)),
        }
        model_manifest_path.write_text(
            json.dumps(model_manifest, sort_keys=True),
            encoding="utf-8",
        )
        experiment_path = run.output_directory / "experiment.json"
        experiment_path.write_text(
            json.dumps(
                {
                    "phase72_experiment": run_metadata,
                    "seed": run.training_seed,
                    "observation_schema": run.observation_schema,
                    "ppo_config": asdict(_ppo_config(protocol, run.training_seed)),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "model_path": str(model_path),
            "model_manifest_path": str(model_manifest_path),
            "experiment_path": str(experiment_path),
            "actual_timesteps": 10_240,
        }

    def _fake_evaluate(self, model_path, configuration, seed, stage, root, protocol):
        self.evaluation_calls.append((configuration, seed, stage, root))
        self.evaluation_model_paths.append(
            (Path(model_path), configuration, seed, stage, root)
        )
        if self.fail_evaluation_once:
            self.fail_evaluation_once = False
            raise RuntimeError("mock evaluation interruption")
        return self._metrics(configuration, stage, root)

    def _metrics(self, configuration, stage, root):
        metrics = {
            "initial_portfolio_value": 100_000.0,
            "final_portfolio_value": 101_000.0,
            "cumulative_return": 0.01,
            "annualized_return": 0.01,
            "annualized_volatility": 0.1,
            "sharpe_ratio": self.sharpes[configuration],
            "sortino_ratio": 0.2,
            "maximum_drawdown": self.drawdowns[configuration],
            "total_transaction_costs": 10.0,
            "portfolio_turnover": self.turnovers[configuration],
            "scenario_id": f"{stage}-{root}",
            "seed": root,
            "synthetic": True,
            "source_type": "synthetic",
            "steps": 33,
            "terminated": True,
        }
        if self.invalid_metrics:
            metrics["sharpe_ratio"] = float("nan")
        return metrics

    def _train(self):
        self.runner.plan()
        return self.runner.train()

    def test_planning_is_15_jobs_schema_aware_and_does_not_generate_data(self):
        from unittest.mock import patch

        with patch(
            "src.rl.phase7_runner.generate_synthetic_market",
            side_effect=AssertionError("planning must not generate data"),
        ), patch(
            "src.rl.phase7_runner._require_local_lock_filesystem",
            wraps=_require_local_lock_filesystem,
        ) as filesystem_guard, patch(
            "ctypes.windll.kernel32.GetDriveTypeW",
            return_value=3,
        ), patch(
            "ctypes.windll.kernel32.GetVolumeInformationW",
            side_effect=self._volume_information_result("NTFS"),
        ) as volume_information:
            manifest = self.runner.plan()
        filesystem_guard.assert_called_once_with(self.output_root)
        volume_information.assert_called_once()
        self.assertEqual(len(manifest["runs"]), 15)
        self.assertEqual(
            [(run["configuration"], run["training_seed"]) for run in manifest["runs"]],
            [
                (configuration, seed)
                for configuration in ("A", "B", "C")
                for seed in [101, 202, 303, 404, 505]
            ],
        )
        schemas = {run["configuration"]: run["observation_schema"] for run in manifest["runs"]}
        self.assertEqual(schemas["A"], schemas["B"])
        self.assertNotEqual(schemas["A"], schemas["C"])
        b_runs = [run for run in manifest["runs"] if run["configuration"] == "B"]
        self.assertTrue(all(run["randomize_training_horizons"] for run in b_runs))
        self.assertEqual([run["horizon_seed"] for run in b_runs], [101, 202, 303, 404, 505])
        self.assertTrue((self.output_root / RUN_MANIFEST_NAME).is_file())
        with self.assertRaises(FileExistsError):
            self.runner.plan()

    def test_planning_rejects_unsupported_platform_before_reservation(self):
        for system in ("Darwin", "Linux"):
            with (
                self.subTest(system=system),
                patch("src.rl.phase7_runner.platform.system", return_value=system),
                patch(
                    "src.rl.phase7_runner.reserve_phase72_output_directories"
                ) as reserve,
                self.assertRaisesRegex(
                    RuntimeError, "unsupported on this platform"
                ),
            ):
                self.runner.plan()

            reserve.assert_not_called()
            self.assertFalse(self.output_root.exists())
            self.assertFalse(self.runner.manifest_path.exists())

    def test_planning_rejects_onedrive_path_before_reservation(self):
        synchronized_root = Path.home() / "OneDrive - Test" / "phase72-plan"
        self.runner.output_root = synchronized_root / ".artifacts" / "phase7_2"
        self.runner.manifest_path = self.runner.output_root / RUN_MANIFEST_NAME
        environment = {
            "OneDrive": str(synchronized_root),
            "OneDriveCommercial": "",
            "OneDriveConsumer": "",
        }
        with (
            patch.dict(os.environ, environment),
            patch(
                "src.rl.phase7_runner.reserve_phase72_output_directories"
            ) as reserve,
            self.assertRaisesRegex(RuntimeError, "OneDrive-synchronized"),
        ):
            self.runner.plan()

        reserve.assert_not_called()
        self.assertFalse(self.runner.output_root.exists())
        self.assertFalse(self.runner.manifest_path.exists())

    def test_planning_rejects_network_drive_before_reservation(self):
        with (
            patch.dict(
                os.environ,
                {
                    "OneDrive": "",
                    "OneDriveCommercial": "",
                    "OneDriveConsumer": "",
                },
            ),
            patch(
                "ctypes.windll.kernel32.GetDriveTypeW",
                return_value=4,
            ) as get_drive_type,
            patch(
                "src.rl.phase7_runner.reserve_phase72_output_directories"
            ) as reserve,
            self.assertRaisesRegex(RuntimeError, "network, removable, and unknown"),
        ):
            self.runner.plan()

        get_drive_type.assert_called_once()
        reserve.assert_not_called()
        self.assertFalse(self.output_root.exists())
        self.assertFalse(self.runner.manifest_path.exists())

    def test_planning_rejects_unsupported_or_unknown_filesystem_before_reservation(self):
        cases = (
            ("ReFS", True, "only NTFS is approved"),
            ("", True, "empty.*filesystem type"),
            ("NTFS", False, "cannot query.*filesystem type"),
        )
        for filesystem, success, message in cases:
            with (
                self.subTest(filesystem=filesystem, success=success),
                patch.dict(
                    os.environ,
                    {
                        "OneDrive": "",
                        "OneDriveCommercial": "",
                        "OneDriveConsumer": "",
                    },
                ),
                patch(
                    "ctypes.windll.kernel32.GetDriveTypeW",
                    return_value=3,
                ),
                patch(
                    "ctypes.windll.kernel32.GetVolumeInformationW",
                    side_effect=self._volume_information_result(
                        filesystem, success
                    ),
                ) as volume_information,
                patch(
                    "src.rl.phase7_runner.reserve_phase72_output_directories"
                ) as reserve,
                self.assertRaisesRegex(RuntimeError, message),
            ):
                self.runner.plan()

            volume_information.assert_called_once()
            reserve.assert_not_called()
            self.assertFalse(self.output_root.exists())
            self.assertFalse(self.runner.manifest_path.exists())

    def test_failed_planning_guard_leaves_no_partial_reservation(self):
        with (
            patch(
                "src.rl.phase7_runner._require_local_lock_filesystem",
                side_effect=RuntimeError("uncertain filesystem"),
            ),
            patch(
                "src.rl.phase7_runner.reserve_phase72_output_directories"
            ) as reserve,
            self.assertRaisesRegex(RuntimeError, "uncertain filesystem"),
        ):
            self.runner.plan()

        reserve.assert_not_called()
        self.assertFalse(self.repository.exists())
        self.assertFalse(self.output_root.exists())
        self.assertFalse(self.runner.manifest_path.exists())

    def test_training_is_mocked_and_completed_artifacts_are_reused_safely(self):
        manifest = self._train()
        self.assertEqual(manifest["status"], "training_complete")
        self.assertEqual(len(self.training_calls), 15)
        self.assertEqual(len(set(self.training_calls)), 15)
        self.assertTrue(
            all(
                not any(root in (seed,) for root in self.protocol["seed_roles"]["validation_scenario_roots"])
                for _, seed in self.training_calls
            )
        )
        self.runner.train()
        self.assertEqual(len(self.training_calls), 15)

    def test_synthetic_training_provenance_is_recomputed_and_bound_to_run(self):
        provenance = self._fake_provenance(self.protocol, 101)
        metadata = build_phase72_run_metadata(
            self.protocol,
            configuration="A",
            training_seed=101,
            training_provenance={
                "source_type": "synthetic",
                "synthetic": True,
                **provenance,
            },
        )
        self.assertEqual(
            metadata["synthetic_data_provenance"]["configuration_hash"],
            provenance["configuration_hash"],
        )
        cases = []
        altered_hash = dict(provenance, configuration_hash="0" * 64)
        cases.append(altered_hash)
        altered_config = dict(provenance)
        altered_config["generation_config"] = dict(provenance["generation_config"])
        altered_config["generation_config"]["sessions"] += 1
        cases.append(altered_config)
        altered_seed = dict(provenance)
        altered_seed["seed"] += 1
        altered_seed["scenario_id"] = f"train-{altered_seed['seed']}"
        cases.append(altered_seed)
        altered_root = dict(provenance, scenario_id="train-999999")
        cases.append(altered_root)
        for invalid in cases:
            with self.subTest(provenance=invalid), self.assertRaises(ValueError):
                build_phase72_run_metadata(
                    self.protocol,
                    configuration="A",
                    training_seed=101,
                    training_provenance={
                        "source_type": "synthetic",
                        "synthetic": True,
                        **invalid,
                    },
                )

    def test_stage_lock_blocks_live_owner_and_releases_normally(self):
        lock_path = self.output_root / ".phase72_stage.lock"
        with self.runner._stage_lock():
            self.assertTrue(lock_path.is_file())
            with self.assertRaisesRegex(RuntimeError, "stage lock already exists"):
                with self.runner._stage_lock():
                    pass
        self.assertFalse(lock_path.exists())

    def test_stage_lock_replacement_is_blocked_between_verification_and_release(self):
        lock_path = self.output_root / ".phase72_stage.lock"
        replacement_path = self.output_root / "replacement.lock"
        replacement_contents = b"replacement owner"
        replacement_attempts = []

        def attempt_replacement(descriptor):
            with self.assertRaises(PermissionError):
                os.replace(replacement_path, lock_path)
            replacement_attempts.append(True)
            self.assertEqual(replacement_path.read_bytes(), replacement_contents)
            _delete_stage_lock_handle(descriptor)

        replacement_path.parent.mkdir(parents=True, exist_ok=True)
        replacement_path.write_bytes(replacement_contents)
        with patch(
            "src.rl.phase7_runner._delete_stage_lock_handle",
            side_effect=attempt_replacement,
        ):
            with self.runner._stage_lock():
                pass

        self.assertEqual(replacement_attempts, [True])
        self.assertFalse(lock_path.exists())
        self.assertEqual(replacement_path.read_bytes(), replacement_contents)

    def test_stage_lock_release_rejects_ownership_mismatch(self):
        lock_path = self.output_root / ".phase72_stage.lock"
        token = "release-mismatch-token"
        machine_id = _machine_identity()
        self.assertIsNotNone(machine_id)
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "machine_identity": machine_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "invocation_token": token,
                "process_start_identity": "test-process-start",
            }
        ).encode("utf-8")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, identity = _create_stage_lock(lock_path, payload)
        try:
            with self.assertRaisesRegex(RuntimeError, "ownership changed"):
                _release_stage_lock(
                    lock_path, descriptor, identity, "different-token", machine_id
                )
            self.assertTrue(lock_path.exists())
        finally:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)

    def test_stage_lock_release_rejects_missing_path(self):
        machine_id = _machine_identity()
        self.assertIsNotNone(machine_id)
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "machine_identity": machine_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "invocation_token": "missing-path-token",
                "process_start_identity": "test-process-start",
            }
        ).encode("utf-8")
        source_path = self.output_root / "source.lock"
        missing_path = self.output_root / "missing.lock"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(payload)
        descriptor = os.open(source_path, os.O_RDWR | getattr(os, "O_BINARY", 0))
        stat_result = os.fstat(descriptor)
        try:
            with self.assertRaisesRegex(RuntimeError, "missing or unverifiable"):
                _release_stage_lock(
                    missing_path,
                    descriptor,
                    (stat_result.st_dev, stat_result.st_ino),
                    "missing-path-token",
                    machine_id,
                )
            self.assertTrue(source_path.exists())
            self.assertFalse(missing_path.exists())
        finally:
            os.close(descriptor)

    def test_stage_lock_release_error_leaves_lock_for_manual_inspection(self):
        lock_path = self.output_root / ".phase72_stage.lock"
        with (
            patch(
                "src.rl.phase7_runner._delete_stage_lock_handle",
                side_effect=OSError("injected handle deletion failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "could not safely release"),
        ):
            with self.runner._stage_lock():
                pass
        self.assertTrue(lock_path.exists())
        lock_path.unlink()

    def test_stale_stage_lock_fails_closed_without_recovery(self):
        self.output_root.mkdir(parents=True)
        lock_path = self.output_root / ".phase72_stage.lock"
        payload = {
            "pid": 987654321,
            "host": socket.gethostname(),
            "machine_identity": "local-machine-id",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "invocation_token": "dead-owner",
            "process_start_identity": "old-process",
        }
        lock_path.write_text(json.dumps(payload), encoding="utf-8")
        original = lock_path.read_bytes()
        with (
            patch("src.rl.phase7_runner._machine_identity", return_value="local-machine-id"),
            patch("src.rl.phase7_runner._process_start_identity", return_value="current-process"),
            self.assertRaisesRegex(RuntimeError, "stale-lock recovery is disabled"),
        ):
            with self.runner._stage_lock():
                self.fail("stale lock must not be automatically recovered")
        self.assertEqual(lock_path.read_bytes(), original)

    def test_same_hostname_different_machine_identity_is_never_recovered(self):
        self.output_root.mkdir(parents=True)
        lock_path = self.output_root / ".phase72_stage.lock"
        payload = {
            "pid": 2468,
            "host": socket.gethostname(),
            "machine_identity": "remote-machine-id",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "invocation_token": "remote-owner",
            "process_start_identity": "remote-process-start",
        }
        lock_path.write_text(json.dumps(payload), encoding="utf-8")
        original = lock_path.read_bytes()
        with (
            patch("src.rl.phase7_runner._machine_identity", return_value="local-machine-id"),
            patch("src.rl.phase7_runner._process_start_identity", return_value="current-process"),
            self.assertRaisesRegex(RuntimeError, "different or unverifiable machine"),
        ):
            with self.runner._stage_lock():
                self.fail("remote lock must not be recovered")
        self.assertEqual(lock_path.read_bytes(), original)

    def test_unknown_machine_identity_fails_before_creating_a_lock(self):
        with patch("src.rl.phase7_runner._machine_identity", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "stable machine identity"):
                with self.runner._stage_lock():
                    self.fail("lock acquisition must fail closed")
        self.assertFalse((self.output_root / ".phase72_stage.lock").exists())

    def test_onedrive_synchronized_lock_path_fails_closed(self):
        synchronized_root = Path.home() / "OneDrive - Test" / "repo"
        with patch.dict(os.environ, {"OneDrive": str(synchronized_root)}):
            with self.assertRaisesRegex(RuntimeError, "OneDrive-synchronized"):
                _require_local_lock_filesystem(
                    synchronized_root / ".artifacts" / "phase7_2" / ".lock"
                )

    def test_multiprocess_stage_lock_races_allow_at_most_one_live_owner(self):
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(2)
        release = context.Event()
        results = context.Queue()
        processes = [
            context.Process(
                target=_stage_lock_process_worker,
                args=(str(self.output_root), barrier, release, results, "fresh-contender"),
            )
            for _ in range(2)
        ]
        try:
            for process in processes:
                process.start()
            observed = [results.get(timeout=120) for _ in processes]
        finally:
            _finish_lock_processes(processes, release)
            results.close()
            results.join_thread()
        acquired = [item for item in observed if item[0] == "acquired"]
        rejected = [item for item in observed if item[0] == "rejected"]
        self.assertEqual(len(acquired), 1, observed)
        self.assertEqual(len(rejected), 1, observed)
        for process in processes:
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 0)

    def test_multiprocess_stale_lock_contenders_all_fail_closed(self):
        self.output_root.mkdir(parents=True)
        lock_path = self.output_root / ".phase72_stage.lock"
        payload = {
            "pid": 987654321,
            "host": socket.gethostname(),
            "machine_identity": _machine_identity(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "invocation_token": "stale-owner",
            "process_start_identity": "old-process",
        }
        lock_path.write_text(json.dumps(payload), encoding="utf-8")
        original = lock_path.read_bytes()
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(3)
        release = context.Event()
        results = context.Queue()
        processes = [
            context.Process(
                target=_stage_lock_process_worker,
                args=(
                    str(self.output_root),
                    barrier,
                    release,
                    results,
                    "stale-recovery-contender" if index < 2 else "new-acquisition-contender",
                ),
            )
            for index in range(3)
        ]
        try:
            for process in processes:
                process.start()
            observed = [results.get(timeout=120) for _ in processes]
        finally:
            _finish_lock_processes(processes, release)
            results.close()
            results.join_thread()
        self.assertTrue(all(item[0] == "rejected" for item in observed), observed)
        self.assertTrue(
            all("stale-lock recovery is disabled" in item[3] for item in observed),
            observed,
        )
        self.assertEqual(
            {item[1] for item in observed},
            {"stale-recovery-contender", "new-acquisition-contender"},
        )
        self.assertEqual(
            sum(item[1] == "stale-recovery-contender" for item in observed),
            2,
        )
        self.assertEqual(
            sum(item[1] == "new-acquisition-contender" for item in observed),
            1,
        )
        for process in processes:
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(lock_path.read_bytes(), original)

    def test_stage_lock_fails_closed_for_malformed_or_unverifiable_lock(self):
        self.output_root.mkdir(parents=True)
        lock_path = self.output_root / ".phase72_stage.lock"
        lock_path.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "remove it manually"):
            with self.runner._stage_lock():
                pass
        self.assertTrue(lock_path.exists())

    def test_training_adapter_uses_protocol_provenance_and_b_wrapper_only_for_b(self):
        self.runner.plan()
        entries = json.loads(
            (self.output_root / RUN_MANIFEST_NAME).read_text(encoding="utf-8")
        )["runs"]
        run_a = self.runner._run_from_entry(next(
            entry for entry in entries
            if entry["configuration"] == "A" and entry["training_seed"] == 101
        ))
        run_b = self.runner._run_from_entry(next(
            entry for entry in entries
            if entry["configuration"] == "B" and entry["training_seed"] == 101
        ))
        dataset = SimpleNamespace(
            asset_returns=object(),
            benchmark_returns=object(),
            vix_levels=object(),
            metadata={"source_type": "synthetic", "synthetic": True},
            environment_kwargs=lambda: {
                "asset_returns": object(),
                "benchmark_returns": object(),
                "vix_levels": object(),
                "data_metadata": {"source_type": "synthetic", "synthetic": True},
            },
        )
        trainer_result = SimpleNamespace(model=SimpleNamespace(num_timesteps=10_240))
        trainer_instance = MagicMock()
        trainer_instance.train.return_value = (
            trainer_result,
            {},
            run_a.output_directory / "model.zip",
            run_a.output_directory / "model.manifest.json",
        )
        generated = []
        with (
            patch(
                "src.rl.phase7_runner.generate_synthetic_market",
                side_effect=lambda config, seed, scenario_id: (
                    generated.append((seed, scenario_id)) or dataset
                ),
            ),
            patch(
                "src.rl.phase7_runner.SyntheticPortfolioEnv",
                return_value=MagicMock(),
            ),
            patch(
                "src.rl.phase7_runner.RandomizedEpisodeHorizonEnv",
                side_effect=lambda env, **kwargs: env,
            ) as horizon_wrapper,
            patch(
                "src.rl.phase7_runner.TrainingOrchestrator",
                return_value=trainer_instance,
            ) as orchestrator,
        ):
            self.runner._train_one(self.protocol, run_a)
            self.runner._train_one(
                self.protocol,
                run_b,
            )
        self.assertEqual(len(generated), 2)
        self.assertTrue(all(scenario.startswith("train-") for _, scenario in generated))
        self.assertEqual(horizon_wrapper.call_count, 1)
        self.assertEqual(horizon_wrapper.call_args.kwargs["seed"], 101)
        first_call = orchestrator.call_args_list[0]
        self.assertEqual(first_call.kwargs["seed"], 101)
        self.assertEqual(trainer_instance.train.call_count, 2)
        self.assertEqual(
            trainer_instance.train.call_args_list[0].kwargs["phase72_run"],
            run_a,
        )
        self.assertEqual(
            trainer_instance.train.call_args_list[1].kwargs["phase72_run"],
            run_b,
        )

    def test_default_evaluation_adapter_preserves_schema_costs_and_benchmarks(self):
        root = self.protocol["seed_roles"]["validation_scenario_roots"][0]
        environment = MagicMock()
        metrics = self._metrics("C", "validation", root)
        benchmarks = {
            name: {"sharpe_ratio": 0.1, "maximum_drawdown": -0.2}
            for name in (
                "SPY buy-and-hold",
                "Equal-weight",
                "Monthly rebalancing",
                "Threshold rebalancing",
            )
        }
        evaluation = {
            "evaluation_type": "synthetic-only",
            "period": {
                "warmup_sessions": 63,
                "period_sessions": 21,
                "evaluation_sessions": 693,
            },
            "cost_assumptions": {
                "initial_capital": 100_000.0,
                "transaction_cost_rate": 0.001,
                "slippage_bps": 5.0,
            },
            "ppo": metrics,
            "benchmarks": benchmarks,
        }
        self.runner._active_roots = {root}
        with (
            patch(
                "src.rl.phase7_runner.generate_synthetic_market",
                return_value=SimpleNamespace(
                    environment_kwargs=lambda: {
                        "asset_returns": object(),
                        "benchmark_returns": object(),
                        "vix_levels": object(),
                        "data_metadata": {"source_type": "synthetic", "synthetic": True},
                    }
                ),
            ) as generate,
            patch(
                "src.rl.phase7_runner.SyntheticPortfolioEnv",
                return_value=environment,
            ) as create_environment,
            patch(
                "src.rl.phase7_runner.load_ppo_model",
                return_value=object(),
            ),
            patch(
                "src.rl.phase7_runner.evaluate_phase4",
                return_value=evaluation,
            ),
        ):
            result = self.runner._evaluate_one(
                Path("mock-model.zip"),
                "C",
                101,
                "validation",
                root,
                self.protocol,
            )
        self.assertEqual(result, evaluation)
        self.assertEqual(generate.call_args.kwargs["scenario_id"], f"validation-{root}")
        self.assertEqual(
            create_environment.call_args.kwargs["observation_schema"],
            self.protocol["configurations"]["C"]["observation_schema"],
        )
        self.runner._active_roots = {root}
        record = self.runner._evaluation_record(
            "validation",
            "C",
            101,
            root,
            result,
        )
        self.assertEqual(record["evaluation_output"]["benchmarks"], benchmarks)
        self.assertEqual(record["evaluation_parameters"]["evaluation_sessions"], 693)

    def test_training_failure_is_not_retried_and_preserves_partial_artifacts(self):
        self.runner.plan()
        calls = 0

        def partial_failure(protocol, run):
            nonlocal calls
            calls += 1
            (run.output_directory / "partial.tmp").write_text("partial", encoding="utf-8")
            raise RuntimeError("mock training failure")

        self.runner.training_adapter = partial_failure
        with self.assertRaisesRegex(RuntimeError, "mock training failure"):
            self.runner.train()
        with self.assertRaisesRegex(ValueError, "not retried automatically"):
            self.runner.train()
        self.assertEqual(calls, 1)

    def test_explicit_a101_retry_preserves_history_and_resolves_downstream_paths(self):
        self.runner.plan()

        def fail_initial_attempt(protocol, run):
            self.training_calls.append((run.configuration, run.training_seed))
            raise RuntimeError("initial PPO dependency failure")

        self.runner.training_adapter = fail_initial_attempt
        with self.assertRaisesRegex(RuntimeError, "dependency failure"):
            self.runner.train()
        original_manifest = self.runner._read_manifest()
        original_entry = next(
            item for item in original_manifest["runs"]
            if item["configuration"] == "A" and item["training_seed"] == 101
        )
        original_snapshot = json.loads(
            json.dumps(original_entry, sort_keys=True)
        )
        self.assertEqual(list(self.runner._canonical_run_directory(original_entry).iterdir()), [])

        self.runner.training_adapter = self._fake_train
        recovered = self.runner.retry_failed_run(
            "A", 101, "Install the approved PPO runtime in the workspace venv"
        )
        target = next(
            item for item in recovered["runs"]
            if item["configuration"] == "A" and item["training_seed"] == 101
        )
        self.assertEqual(recovered["format"], "phase72-run-manifest-v2")
        self.assertEqual(target["attempt_history"][0]["entry_snapshot"], original_snapshot)
        self.assertEqual(target["attempt_history"][0]["status"], "failed")
        self.assertEqual(target["attempt_history"][1]["attempt_id"], "attempt_0002")
        self.assertEqual(target["attempt_history"][1]["status"], "trained")
        self.assertEqual(target["active_attempt_id"], "attempt_0002")
        self.assertTrue(
            (self.runner.output_root / "recovery_events" / "retry_attempt_0002.json").is_file()
        )
        retry_run = self.runner._run_from_entry(target)
        expected_retry = (
            self.runner.output_root
            / "retries"
            / "attempt_0002"
            / "config_A"
            / "seed_101"
        )
        self.assertEqual(retry_run.output_directory, expected_retry)
        self.assertEqual(list(self.runner._canonical_run_directory(target).iterdir()), [])

        resumed = self.runner.train()
        self.assertEqual(resumed["status"], "training_complete")
        self.assertEqual(len(self.training_calls), 16)
        self.assertEqual(
            self.training_calls.count(("A", 101)),
            2,
        )
        self.assertEqual(
            sum(1 for entry in resumed["runs"] if entry["status"] == "trained"),
            15,
        )
        self.sharpes = {"A": 2.0, "B": 1.0, "C": 1.0}
        selection = self.runner.validate_and_select()
        self.assertEqual(selection["selected_configuration"], "A")
        self.assertEqual(len(selection["validation_result_references"]), 300)
        validation_paths = {
            entry["training_seed"]: Path(entry["output_directory"])
            for entry in resumed["runs"]
            if entry["configuration"] == "A"
        }
        a101_validation = [
            item
            for item in self.evaluation_model_paths
            if item[1:4] == ("A", 101, "validation")
        ]
        self.assertEqual(len(a101_validation), 20)
        self.assertTrue(
            all(item[0] == expected_retry / "model.zip" for item in a101_validation)
        )
        current_manifest = self.runner._read_manifest()
        for root in self.protocol["seed_roles"]["validation_scenario_roots"]:
            reference = current_manifest["validation_results"][f"A:101:{root}"]
            result_path = self.runner.output_root / reference["path"]
            self.assertEqual(result_path.parent, expected_retry)
            self.assertTrue(result_path.is_file())
        self.assertEqual(len(current_manifest["validation_results"]), 300)

        self.evaluation_calls.clear()
        self.evaluation_model_paths.clear()
        report = self.runner.final_test()
        self.assertEqual(report["scenario_count"], 100)
        self.assertEqual(len(self.evaluation_calls), 100)
        a101_test = [
            item
            for item in self.evaluation_model_paths
            if item[1:4] == ("A", 101, "test")
        ]
        self.assertEqual(len(a101_test), 20)
        self.assertTrue(all(item[0] == expected_retry / "model.zip" for item in a101_test))
        completed = self.runner._read_manifest()
        for root in self.protocol["seed_roles"]["reserved_test_scenario_roots"]:
            reference = completed["test_results"][f"A:101:{root}"]
            self.assertEqual(
                (self.runner.output_root / reference["path"]).parent,
                expected_retry,
            )
        self.assertEqual(len(completed["test_results"]), 100)
        self.assertTrue(
            all(
                item[0] != self.runner._canonical_run_directory(target) / "model.zip"
                for item in self.evaluation_model_paths
                if item[1:4] == ("A", 101, "test")
            )
        )

    def test_failed_retry_is_preserved_and_never_automatically_retried(self):
        self.runner.plan()

        def fail_initial_attempt(protocol, run):
            raise RuntimeError("initial failed attempt")

        self.runner.training_adapter = fail_initial_attempt
        with self.assertRaisesRegex(RuntimeError, "initial failed"):
            self.runner.train()
        retry_calls = []

        def fail_retry(protocol, run):
            retry_calls.append((run.configuration, run.training_seed, run.output_directory))
            (run.output_directory / "partial_retry.bin").write_bytes(b"partial")
            raise RuntimeError("retry failure")

        self.runner.training_adapter = fail_retry
        reason = "Approved explicit retry after installing the optional PPO package"
        with self.assertRaisesRegex(RuntimeError, "retry failure"):
            self.runner.retry_failed_run("A", 101, reason)
        failed = self.runner._read_manifest()
        target = next(
            item for item in failed["runs"]
            if item["configuration"] == "A" and item["training_seed"] == 101
        )
        self.assertEqual(target["attempt_history"][0]["status"], "failed")
        self.assertEqual(target["attempt_history"][1]["status"], "failed")
        self.assertEqual(target["attempt_history"][1]["error_type"], "RuntimeError")
        self.assertIsNone(target["active_attempt_id"])
        self.assertTrue((retry_calls[0][2] / "partial_retry.bin").is_file())
        with self.assertRaisesRegex(ValueError, "not eligible for training"):
            self.runner.retry_failed_run("A", 101, reason)
        self.assertEqual(len(retry_calls), 1)
        self.assertEqual(
            list(self.runner._canonical_run_directory(target).iterdir()),
            [],
        )

    def test_interrupted_retry_with_verified_result_is_reconciled_without_retraining(self):
        self.runner.plan()

        def fail_initial_attempt(protocol, run):
            raise RuntimeError("initial failed attempt")

        self.runner.training_adapter = fail_initial_attempt
        with self.assertRaisesRegex(RuntimeError, "initial failed"):
            self.runner.train()
        calls = []

        def successful_retry(protocol, run):
            calls.append((run.configuration, run.training_seed))
            return self._fake_train(protocol, run)

        self.runner.training_adapter = successful_retry
        persist_calls = 0
        original_persist = self.runner._persist_manifest

        def interrupt_final_persist(manifest):
            nonlocal persist_calls
            persist_calls += 1
            if persist_calls == 3:
                raise OSError("simulated crash after verified result write")
            original_persist(manifest)

        with (
            patch.object(
                self.runner,
                "_persist_manifest",
                side_effect=interrupt_final_persist,
            ),
            self.assertRaisesRegex(OSError, "simulated crash"),
        ):
            self.runner.retry_failed_run("A", 101, "crash reconciliation test")
        interrupted = self.runner._read_manifest()
        self.assertEqual(interrupted["status"], "training")
        self.assertEqual(
            self.runner._retry_attempt(
                next(
                    item for item in interrupted["runs"]
                    if item["configuration"] == "A" and item["training_seed"] == 101
                )
            )["status"],
            "training",
        )
        reconciled = self.runner.retry_failed_run(
            "A", 101, "crash reconciliation test"
        )
        self.assertEqual(reconciled["status"], "training_interrupted")
        self.assertEqual(calls, [("A", 101)])
        self.assertEqual(self.runner._read_manifest()["runs"][0]["status"], "trained")

    def test_retry_rejects_wrong_target_lock_conflict_and_orphan_event(self):
        with self.assertRaisesRegex(ValueError, "only authorized"):
            self.runner.retry_failed_run("B", 101, "not allowed")
        with self.assertRaisesRegex(ValueError, "only authorized"):
            self.runner.retry_failed_run("A", 202, "not allowed")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.runner.retry_failed_run("A", 101, " ")

        self.runner.plan()

        def fail_initial_attempt(protocol, run):
            raise RuntimeError("initial failed attempt")

        self.runner.training_adapter = fail_initial_attempt
        with self.assertRaisesRegex(RuntimeError, "initial failed"):
            self.runner.train()
        self.runner.training_adapter = self._fake_train
        for lock_error in ("lock contention", "ownership mismatch"):
            with (
                patch.object(
                    self.runner,
                    "_stage_lock",
                    side_effect=RuntimeError(lock_error),
                ),
                self.assertRaisesRegex(RuntimeError, lock_error),
            ):
                self.runner.retry_failed_run("A", 101, "authorized")
        self.assertFalse((self.runner.output_root / "retries").exists())
        self.assertEqual(
            self.runner._read_manifest()["format"],
            "phase72-run-manifest-v1",
        )

    def test_retry_refuses_nonempty_original_and_interrupted_authorization_is_fail_closed(self):
        self.runner.plan()

        def fail_with_original_artifact(protocol, run):
            (run.output_directory / "partial_initial.bin").write_bytes(b"partial")
            raise RuntimeError("initial attempt failed")

        self.runner.training_adapter = fail_with_original_artifact
        with self.assertRaisesRegex(RuntimeError, "initial attempt failed"):
            self.runner.train()
        self.runner.training_adapter = self._fake_train
        with self.assertRaisesRegex(FileExistsError, "must exist and remain empty"):
            self.runner.retry_failed_run("A", 101, "must not overwrite")
        self.assertFalse((self.runner.output_root / "recovery_events").exists())
        self.assertFalse((self.runner.output_root / "retries").exists())

        (self.runner.output_root / "config_A" / "seed_101" / "partial_initial.bin").unlink()
        manifest_before = self.runner._read_manifest()
        with patch(
            "src.rl.phase7_runner._atomic_replace_json",
            side_effect=OSError("interrupted manifest persistence"),
        ), self.assertRaisesRegex(OSError, "interrupted manifest persistence"):
            self.runner.retry_failed_run("A", 101, "recovery persistence test")
        self.assertEqual(
            self.runner._read_manifest()["manifest_sha256"],
            manifest_before["manifest_sha256"],
        )
        self.assertTrue(
            (self.runner.output_root / "recovery_events" / "retry_attempt_0002.json").is_file()
        )
        self.assertTrue(
            (
                self.runner.output_root
                / "retries"
                / "attempt_0002"
                / "config_A"
                / "seed_101"
            ).is_dir()
        )
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self.runner.retry_failed_run("A", 101, "recovery persistence test")
        self.assertEqual(
            self.runner._read_manifest()["format"],
            "phase72-run-manifest-v1",
        )

    def test_retry_refuses_symlinked_retry_namespace(self):
        self.runner.plan()

        def fail_initial_attempt(protocol, run):
            raise RuntimeError("initial failed attempt")

        self.runner.training_adapter = fail_initial_attempt
        with self.assertRaisesRegex(RuntimeError, "initial failed"):
            self.runner.train()
        outside = Path(self.temporary_directory.name) / "outside-retries"
        outside.mkdir()
        retry_root = self.runner.output_root / "retries"
        try:
            retry_root.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"directory symlinks are unavailable: {error}")
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self.runner.retry_failed_run("A", 101, "symlink rejection test")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((self.runner.output_root / "recovery_events").exists())
        self.assertEqual(
            self.runner._read_manifest()["format"],
            "phase72-run-manifest-v1",
        )

    def test_retry_refuses_windows_reparse_point_before_creating_artifacts(self):
        self.runner.plan()

        def fail_initial_attempt(protocol, run):
            raise RuntimeError("initial failed attempt")

        self.runner.training_adapter = fail_initial_attempt
        with self.assertRaisesRegex(RuntimeError, "initial failed"):
            self.runner.train()

        retries_root = self.runner.output_root / "retries"
        with (
            patch(
                "src.rl.phase7_runner._path_has_reparse_component",
                side_effect=lambda path: path == retries_root
                or retries_root in path.parents,
            ),
            self.assertRaisesRegex(ValueError, "reparse point"),
        ):
            self.runner.retry_failed_run("A", 101, "reparse point regression")

        self.assertFalse((self.runner.output_root / "recovery_events").exists())
        self.assertFalse(retries_root.exists())
        self.assertEqual(
            self.runner._read_manifest()["format"],
            "phase72-run-manifest-v1",
        )

    def test_retry_refuses_actual_windows_junction(self):
        if os.name != "nt":
            self.skipTest("Windows junction validation requires Windows")
        self.runner.plan()

        def fail_initial_attempt(protocol, run):
            raise RuntimeError("initial failed attempt")

        self.runner.training_adapter = fail_initial_attempt
        with self.assertRaisesRegex(RuntimeError, "initial failed"):
            self.runner.train()

        outside = Path(self.temporary_directory.name) / "outside-junction-target"
        outside.mkdir()
        retries_root = self.runner.output_root / "retries"
        result = subprocess.run(
            [
                os.environ.get("COMSPEC", "cmd.exe"),
                "/c",
                "mklink",
                "/J",
                str(retries_root),
                str(outside),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if retries_root.exists():
                subprocess.run(
                    [
                        os.environ.get("COMSPEC", "cmd.exe"),
                        "/c",
                        "rmdir",
                        str(retries_root),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            self.skipTest(
                f"Windows junction creation was unavailable: {result.stdout} {result.stderr}"
            )
        try:
            from src.rl.phase7_experiment import _path_is_reparse_point

            self.assertFalse(retries_root.is_symlink())
            self.assertTrue(_path_is_reparse_point(retries_root))
            self.assertTrue(
                getattr(retries_root.lstat(), "st_file_attributes", 0) & 0x400
            )
            with self.assertRaisesRegex(ValueError, "reparse point"):
                self.runner.retry_failed_run(
                    "A",
                    101,
                    "actual Windows junction rejection",
                )
            self.assertFalse((self.runner.output_root / "recovery_events").exists())
            self.assertEqual(list(outside.iterdir()), [])
            self.assertEqual(
                self.runner._read_manifest()["format"],
                "phase72-run-manifest-v1",
            )
        finally:
            subprocess.run(
                [
                    os.environ.get("COMSPEC", "cmd.exe"),
                    "/c",
                    "rmdir",
                    str(retries_root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

    def test_windows_reparse_point_attribute_is_detected(self):
        from src.rl.phase7_experiment import _path_is_reparse_point

        metadata = SimpleNamespace(st_mode=0, st_file_attributes=0x400)
        with patch.object(Path, "lstat", autospec=True, return_value=metadata):
            self.assertTrue(_path_is_reparse_point(Path("junction")))

    def test_retry_path_resolution_rejects_escape_from_output_root(self):
        candidate = (
            self.runner.output_root
            / "retries"
            / "attempt_0002"
            / "config_A"
            / "seed_101"
        )
        outside = Path(self.temporary_directory.name) / "outside"
        original_resolve = Path.resolve

        def resolve_with_redirect(path, strict=False):
            if path == candidate:
                return outside
            return original_resolve(path, strict=strict)

        with patch.object(
            Path,
            "resolve",
            autospec=True,
            side_effect=resolve_with_redirect,
        ):
            self.assertFalse(self.runner._path_is_within_output_root(candidate))

    def test_active_attempt_resolver_rejects_symlink_escape(self):
        self.runner.plan()

        def fail_initial_attempt(protocol, run):
            raise RuntimeError("initial failed attempt")

        self.runner.training_adapter = fail_initial_attempt
        with self.assertRaisesRegex(RuntimeError, "initial failed"):
            self.runner.train()
        self.runner.training_adapter = self._fake_train
        self.runner.retry_failed_run("A", 101, "symlink resolver test")
        target = self.runner.output_root / "retries" / "attempt_0002"

        entry = next(
            item for item in self.runner._read_manifest()["runs"]
            if item["configuration"] == "A" and item["training_seed"] == 101
        )
        with (
            patch(
                "src.rl.phase7_runner._path_has_reparse_component",
                side_effect=lambda path: path == target or target in path.parents,
            ),
            self.assertRaisesRegex(ValueError, "reparse point"),
        ):
            self.runner._run_from_entry(entry)

    def test_v2_reader_rejects_retry_path_traversal_and_event_tampering(self):
        self.runner.plan()

        def fail_initial_attempt(protocol, run):
            raise RuntimeError("initial failed attempt")

        self.runner.training_adapter = fail_initial_attempt
        with self.assertRaisesRegex(RuntimeError, "initial failed"):
            self.runner.train()
        self.runner.training_adapter = self._fake_train
        self.runner.retry_failed_run("A", 101, "authorized recovery")
        manifest_path = self.runner.manifest_path
        manifest = self.runner._read_manifest()
        target = next(
            item for item in manifest["runs"]
            if item["configuration"] == "A" and item["training_seed"] == 101
        )
        target["attempt_history"][1]["output_directory"] = (
            ".artifacts\\phase7_2\\retries\\..\\..\\config_A\\seed_101"
        )
        manifest["manifest_sha256"] = self.runner._manifest_digest(manifest)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "retry attempt record is invalid"):
            self.runner._read_manifest()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = next(
            item for item in manifest["runs"]
            if item["configuration"] == "A" and item["training_seed"] == 101
        )
        target["attempt_history"][1]["output_directory"] = str(
            (
                self.runner.output_root
                / "retries"
                / "attempt_0002"
                / "config_A"
                / "seed_101"
            ).relative_to(self.repository)
        )
        manifest["manifest_sha256"] = self.runner._manifest_digest(manifest)
        event_path = self.runner.output_root / "recovery_events" / "retry_attempt_0002.json"
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event["operator_reason"] = "tampered"
        event_path.write_bytes(canonical_json_bytes(event))
        manifest["recovery_event"]["sha256"] = sha256_file(event_path)
        manifest["manifest_sha256"] = self.runner._manifest_digest(manifest)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "retry attempt record is invalid"):
            self.runner._read_manifest()

    def test_validation_uses_exact_roots_consistent_settings_and_no_test_roots(self):
        self._train()
        self.fail_evaluation_once = True
        with self.assertRaisesRegex(RuntimeError, "mock evaluation interruption"):
            self.runner.validate_and_select()
        calls_before_resume = len(self.evaluation_calls)
        selection = self.runner.validate_and_select()
        self.assertEqual(selection["selected_configuration"], "C")
        self.assertEqual(len(self.evaluation_calls), calls_before_resume + 15 * 20)
        self.assertEqual(
            {call[3] for call in self.evaluation_calls},
            set(self.protocol["seed_roles"]["validation_scenario_roots"]),
        )
        self.assertTrue(all(call[2] == "validation" for call in self.evaluation_calls))
        self.assertTrue(
            all(
                call[3] not in self.protocol["seed_roles"]["reserved_test_scenario_roots"]
                for call in self.evaluation_calls
            )
        )
        self.assertEqual(selection["training_seeds_retained"], [101, 202, 303, 404, 505])
        self.assertEqual(len(selection["validation_result_references"]), 300)
        self.assertEqual(selection["reserved_test_roots_accessed"], [])
        for configuration in ("A", "B", "C"):
            self.assertEqual(
                selection["configuration_metrics"][configuration][
                    "mean_validation_sharpe"
                ],
                1.0,
            )
            self.assertEqual(
                len(selection["configuration_metrics"][configuration]["seed_aggregates"]),
                5,
            )

    def test_selection_uses_sharpe_then_drawdown_then_turnover_then_lexical(self):
        self.assertEqual(
            self.runner._select_configuration(
                {
                    name: {
                        "mean_validation_sharpe": 1.0,
                        "mean_maximum_drawdown_magnitude": 0.1,
                        "mean_turnover": 1.0,
                    }
                    for name in ("A", "B", "C")
                }
            )[0],
            "A",
        )
        self.assertEqual(
            self.runner._select_configuration(
                {
                    "A": {
                        "mean_validation_sharpe": 0.9,
                        "mean_maximum_drawdown_magnitude": 0.01,
                        "mean_turnover": 0.1,
                    },
                    "B": {
                        "mean_validation_sharpe": 1.0,
                        "mean_maximum_drawdown_magnitude": 0.9,
                        "mean_turnover": 9.0,
                    },
                    "C": {
                        "mean_validation_sharpe": 1.0,
                        "mean_maximum_drawdown_magnitude": 0.1,
                        "mean_turnover": 0.2,
                    },
                }
            )[0],
            "C",
        )
        self.assertEqual(
            self.runner._select_configuration(
                {
                    name: {
                        "mean_validation_sharpe": 1.0,
                        "mean_maximum_drawdown_magnitude": 0.1,
                        "mean_turnover": turnover,
                    }
                    for name, turnover in {"A": 3.0, "B": 1.0, "C": 2.0}.items()
                }
            )[0],
            "B",
        )

    def test_invalid_metrics_block_selection_and_final_test_before_selection(self):
        self._train()
        self.invalid_metrics = True
        with self.assertRaisesRegex(ValueError, "finite"):
            self.runner.validate_and_select()
        with self.assertRaisesRegex(ValueError, "blocked"):
            self.runner.final_test()
        self.assertFalse((self.output_root / SELECTION_MANIFEST_NAME).exists())
        self.assertFalse((self.output_root / FINAL_TEST_REPORT_NAME).exists())

    def test_final_test_runs_only_selected_configuration_100_times(self):
        self._train()
        self.runner.validate_and_select()
        self.evaluation_calls.clear()
        report = self.runner.final_test()
        self.assertEqual(report["evaluation_stage"], "one_time_final_held_out_test")
        self.assertEqual(report["selected_configuration"], "C")
        self.assertEqual(report["scenario_count"], 100)
        self.assertEqual(report["overall_aggregate"]["scenario_count"], 100)
        self.assertEqual(len(self.evaluation_calls), 100)
        self.assertTrue(all(call[0] == "C" and call[2] == "test" for call in self.evaluation_calls))
        self.assertEqual(
            {call[3] for call in self.evaluation_calls},
            set(self.protocol["seed_roles"]["reserved_test_scenario_roots"]),
        )
        self.assertFalse(report["test_results_used_for_selection"])
        self.assertEqual(self.runner.final_test(), report)
        selection_path = self.output_root / SELECTION_MANIFEST_NAME
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["selected_configuration"] = "A"
        selection_path.write_text(json.dumps(selection), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.runner.final_test()

    def test_completed_test_replay_rejects_tampered_result_even_if_manifest_changes(self):
        self._train()
        self.runner.validate_and_select()
        self.runner.final_test()
        manifest_path = self.output_root / RUN_MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = next(iter(manifest["test_results"]))
        reference = manifest["test_results"][key]
        result_path = self.output_root / reference["path"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["metrics"]["sharpe_ratio"] += 1.0
        result["evaluation_output"]["ppo"]["sharpe_ratio"] += 1.0
        result_path.write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        reference["sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        manifest["manifest_sha256"] = self.runner._manifest_digest(manifest)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "checksum does not match"):
            self.runner.final_test()

    def test_completed_test_replay_rejects_missing_result_file_and_sidecar(self):
        self._train()
        self.runner.validate_and_select()
        self.runner.final_test()
        manifest = json.loads(
            (self.output_root / RUN_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        key = next(iter(manifest["test_results"]))
        result_path = self.output_root / manifest["test_results"][key]["path"]
        result_path.unlink()
        result_path.with_name(f"{result_path.name}.sha256").unlink()
        with self.assertRaises(FileNotFoundError):
            self.runner.final_test()

    def test_completed_test_replay_rejects_missing_checksum_sidecar(self):
        self._train()
        self.runner.validate_and_select()
        self.runner.final_test()
        manifest = json.loads(
            (self.output_root / RUN_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        key = next(iter(manifest["test_results"]))
        result_path = self.output_root / manifest["test_results"][key]["path"]
        result_path.with_name(f"{result_path.name}.sha256").unlink()
        with self.assertRaisesRegex(ValueError, "sidecar is missing"):
            self.runner.final_test()

    def test_tampered_result_is_rejected_even_when_manifest_reference_is_updated(self):
        self._train()
        self.runner.validate_and_select()
        manifest_path = self.output_root / RUN_MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = next(iter(manifest["validation_results"]))
        reference = manifest["validation_results"][key]
        result_path = self.output_root / reference["path"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["metrics"]["sharpe_ratio"] += 1.0
        result["evaluation_output"]["ppo"]["sharpe_ratio"] += 1.0
        serialized = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
        result_path.write_text(serialized, encoding="utf-8")
        reference["sha256"] = hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest()
        manifest["manifest_sha256"] = self.runner._manifest_digest(manifest)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "immutable sidecar"):
            self.runner.final_test()

    def test_model_checksum_mismatch_blocks_validation_reuse(self):
        self._train()
        run = self.output_root / "config_A" / "seed_101"
        (run / "model.zip").write_bytes(b"altered model")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.runner.validate_and_select()

    def test_resume_recomputes_training_provenance_after_checksums_are_updated(self):
        self._train()
        run = self.output_root / "config_A" / "seed_101"
        sidecar_path = run / "model.manifest.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["provenance"]["configuration_hash"] = "0" * 64
        sidecar_path.write_text(
            json.dumps(sidecar, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        training_path = run / "training_result.json"
        training = json.loads(training_path.read_text(encoding="utf-8"))
        training["model_manifest_sha256"] = hashlib.sha256(
            sidecar_path.read_bytes()
        ).hexdigest()
        training_path.write_text(
            json.dumps(training, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        manifest_path = self.output_root / RUN_MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next(
            item
            for item in manifest["runs"]
            if item["configuration"] == "A" and item["training_seed"] == 101
        )
        entry["training"] = training
        manifest["manifest_sha256"] = self.runner._manifest_digest(manifest)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "configuration_hash is invalid"):
            self.runner.train()


if __name__ == "__main__":
    unittest.main()
