"""Staged, resumable orchestration for the Phase 7.2 synthetic PPO experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generator, Mapping

import numpy as np

from src.rl.environment.synthetic_portfolio_env import SyntheticPortfolioEnv
from src.rl.phase4_evaluation import evaluate_phase4
from src.rl.phase4_training import TrainingOrchestrator
from src.rl.phase7_experiment import (
    CONFIGURATIONS,
    PHASE72_SELECTION_TOLERANCE,
    Phase72Run,
    build_phase72_run_metadata,
    load_phase72_protocol,
    _path_has_reparse_component,
    _path_is_reparse_point,
    phase72_synthetic_config,
    reserve_phase72_output_directories,
    validate_experiment_manifest_metadata,
    validate_phase72_run_metadata,
    validate_phase72_run_directory,
)
from src.rl.ppo_agent import load_ppo_model
from src.rl.ppo_config import PPOConfig, TrainingOptions
from src.rl.synthetic_data import generate_synthetic_market
from src.rl.synthetic_pipeline import RandomizedEpisodeHorizonEnv

RUN_MANIFEST_NAME = "run_manifest.json"
SELECTION_MANIFEST_NAME = "selection.json"
TRAINING_RESULT_NAME = "training_result.json"
FINAL_TEST_REPORT_NAME = "final_test_report.json"
RETRY_ATTEMPT_ID = "attempt_0002"
RETRY_EVENT_PATH = Path("recovery_events") / "retry_attempt_0002.json"
V1_MANIFEST_FORMAT = "phase72-run-manifest-v1"
V2_MANIFEST_FORMAT = "phase72-run-manifest-v2"
_EVALUATION_METRICS = (
    "initial_portfolio_value",
    "final_portfolio_value",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "maximum_drawdown",
    "total_transaction_costs",
    "portfolio_turnover",
)

TrainingAdapter = Callable[
    [Mapping[str, Any], Phase72Run],
    Mapping[str, Any],
]
EvaluationAdapter = Callable[
    [Path, str, int, str, int, Mapping[str, Any]],
    Mapping[str, Any],
]


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON data deterministically and reject non-finite numbers."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_start_identity(pid: int) -> str | None:
    """Return a PID's creation identity, or None only when it is confirmed absent."""
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error in (87, 1168):
                return None
            raise RuntimeError(f"cannot verify process {pid} ownership (Windows error {error})")
        try:
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel_time = wintypes.FILETIME()
            user_time = wintypes.FILETIME()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                raise RuntimeError(
                    f"cannot verify process {pid} creation time "
                    f"(Windows error {ctypes.get_last_error()})"
                )
            return f"{creation.dwHighDateTime:08x}{creation.dwLowDateTime:08x}"
        finally:
            kernel32.CloseHandle(handle)
    if platform.system() == "Linux":
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        except FileNotFoundError:
            return None
        except OSError as error:
            raise RuntimeError(f"cannot verify process {pid} ownership") from error
        try:
            fields = stat[stat.rfind(")") + 2 :].split()
            return f"linux-start-ticks:{fields[19]}"
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"cannot verify process {pid} creation time") from error
    raise RuntimeError("this platform cannot safely verify stale Phase 7.2 locks")


def _machine_identity() -> str | None:
    """Return a stable local machine identity, never falling back to hostname."""
    try:
        if platform.system() == "Windows":
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            ) as key:
                machine_guid, _ = winreg.QueryValueEx(key, "MachineGuid")
            identity = str(machine_guid).strip().lower()
        elif platform.system() == "Linux":
            identity = Path("/etc/machine-id").read_text(
                encoding="ascii"
            ).strip().lower()
        else:
            return None
    except (OSError, ImportError):
        return None
    if not identity:
        return None
    return hashlib.sha256(identity.encode("ascii")).hexdigest()


def _require_local_lock_filesystem(path: Path) -> None:
    """Fail closed unless the target is on a supported local filesystem."""
    if platform.system() == "Windows":
        import ctypes

        resolved = path.resolve()
        sync_roots = []
        for name in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
            value = os.environ.get(name)
            if value:
                sync_roots.append(Path(value).resolve())
        sync_roots.extend(
            candidate.resolve()
            for candidate in Path.home().glob("OneDrive*")
        )
        for root in sync_roots:
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            raise RuntimeError(
                "Phase 7.2 stage locks are unsupported inside OneDrive-synchronized "
                "folders; use an unsynchronized local workspace"
            )
        if any(
            part.casefold() == "onedrive"
            or part.casefold().startswith("onedrive -")
            for part in resolved.parts
        ):
            raise RuntimeError(
                "Phase 7.2 stage locks are unsupported inside OneDrive-synchronized "
                "folders; use an unsynchronized local workspace"
            )
        get_drive_type = ctypes.windll.kernel32.GetDriveTypeW
        get_drive_type.argtypes = [ctypes.c_wchar_p]
        get_drive_type.restype = ctypes.c_uint
        drive_root = Path(resolved.anchor)
        if not drive_root.anchor or get_drive_type(str(drive_root)) != 3:
            raise RuntimeError(
                "Phase 7.2 stage locks require a local fixed Windows drive; "
                "network, removable, and unknown drives are unsupported"
            )
        try:
            get_volume_information = ctypes.windll.kernel32.GetVolumeInformationW
        except AttributeError as error:
            raise RuntimeError(
                "cannot query the Phase 7.2 stage-lock filesystem type; "
                "refusing to proceed"
            ) from error
        from ctypes import wintypes

        get_volume_information.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        get_volume_information.restype = wintypes.BOOL
        filesystem_name = ctypes.create_unicode_buffer(256)
        if not get_volume_information(
            str(drive_root),
            None,
            0,
            None,
            None,
            None,
            filesystem_name,
            len(filesystem_name),
        ):
            error_code = ctypes.get_last_error()
            raise RuntimeError(
                "cannot query the Phase 7.2 stage-lock filesystem type "
                f"(Windows error {error_code}); refusing to proceed"
            )
        filesystem = filesystem_name.value.strip()
        if not filesystem:
            raise RuntimeError(
                "Windows returned an empty Phase 7.2 stage-lock filesystem type; "
                "refusing to proceed"
            )
        if filesystem.casefold() != "ntfs":
            raise RuntimeError(
                f"Phase 7.2 stage locks are unsupported on {filesystem!r}; "
                "only NTFS is approved"
            )
        return
    if platform.system() != "Windows":
        raise RuntimeError(
            "Phase 7.2 stage locks are unsupported on this platform because "
            "race-free handle-based release is unavailable"
        )


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _create_stage_lock(path: Path, payload: bytes) -> tuple[int, tuple[int, int]]:
    """Create and retain a Windows handle that prevents lock-path replacement."""
    _require_local_lock_filesystem(path)
    if os.name != "nt":
        raise RuntimeError(
            "race-free Phase 7.2 stage-lock release is unavailable on this platform; "
            "refusing to acquire a lock"
        )

    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    handle = kernel32.CreateFileW(
        str(path),
        0x80000000 | 0x40000000 | 0x00010000,
        0x00000001,
        None,
        1,
        0x00000080,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        error_code = ctypes.get_last_error()
        if error_code in (80, 183):
            raise FileExistsError(error_code, "stage lock already exists", str(path))
        raise OSError(error_code, "could not create stage lock", str(path))

    try:
        descriptor = msvcrt.open_osfhandle(
            int(handle),
            os.O_RDWR
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOINHERIT", 0),
        )
    except Exception:
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(handle)
        raise
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write while recording Phase 7.2 lock ownership")
            remaining = remaining[written:]
        os.fsync(descriptor)
        stat_result = os.fstat(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, (stat_result.st_dev, stat_result.st_ino)


def _delete_stage_lock_handle(descriptor: int) -> None:
    """Mark the file referenced by its retained Windows handle for deletion."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("DeleteFile", wintypes.BOOLEAN)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
    disposition = FileDispositionInfo(True)
    if not kernel32.SetFileInformationByHandle(
        msvcrt.get_osfhandle(descriptor),
        4,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        error_code = ctypes.get_last_error()
        raise OSError(error_code, "could not release stage lock by handle")


def _read_existing_stage_lock(path: Path) -> bytes:
    """Read lock metadata with sharing compatible with its retained owner handle."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    handle = kernel32.CreateFileW(
        str(path),
        0x80000000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x00000080,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        error_code = ctypes.get_last_error()
        raise OSError(error_code, "could not read existing stage lock", str(path))
    try:
        descriptor = msvcrt.open_osfhandle(
            int(handle),
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0),
        )
    except Exception:
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(handle)
        raise
    try:
        chunks = bytearray()
        while chunk := os.read(descriptor, 4096):
            chunks.extend(chunk)
        return bytes(chunks)
    finally:
        os.close(descriptor)


def _release_stage_lock(
    path: Path,
    descriptor: int,
    identity: tuple[int, int],
    token: str,
    machine_identity: str,
) -> None:
    """Verify ownership and delete only through the replacement-protected handle."""
    if os.name != "nt":
        raise RuntimeError(
            "race-free Phase 7.2 stage-lock release is unavailable on this platform"
        )
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        payload = bytearray()
        while chunk := os.read(descriptor, 4096):
            payload.extend(chunk)
        current = json.loads(payload)
        _validate_stage_lock_metadata(
            current,
            local_host=socket.gethostname(),
            local_machine_id=machine_identity,
        )
        descriptor_info = os.fstat(descriptor)
        path_info = os.stat(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            "owned Phase 7.2 stage lock is missing or unverifiable; inspect it manually"
        ) from error

    if (
        identity != (descriptor_info.st_dev, descriptor_info.st_ino)
        or identity != (path_info.st_dev, path_info.st_ino)
        or current.get("invocation_token") != token
    ):
        raise RuntimeError(
            "Phase 7.2 stage lock ownership changed before release; "
            "refusing to remove another owner's lock"
        )

    try:
        _delete_stage_lock_handle(descriptor)
    except OSError as error:
        raise RuntimeError(
            "could not safely release the owned Phase 7.2 stage lock; "
            "inspect it manually"
        ) from error


def _validate_stage_lock_metadata(
    metadata: Any,
    *,
    local_host: str,
    local_machine_id: str,
) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("lock metadata must be an object")
    required = {
        "pid",
        "host",
        "machine_identity",
        "created_at_utc",
        "invocation_token",
        "process_start_identity",
    }
    if set(metadata) != required:
        raise ValueError("lock metadata fields are incomplete or unexpected")
    pid = metadata["pid"]
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("lock PID is invalid")
    if (
        not isinstance(metadata["host"], str)
        or not metadata["host"]
        or not isinstance(metadata["machine_identity"], str)
        or not metadata["machine_identity"]
        or not isinstance(metadata["invocation_token"], str)
        or not metadata["invocation_token"]
        or not isinstance(metadata["process_start_identity"], str)
        or not metadata["process_start_identity"]
    ):
        raise ValueError("lock ownership metadata is invalid")
    created_at = metadata["created_at_utc"]
    if not isinstance(created_at, str):
        raise ValueError("lock creation timestamp is invalid")
    try:
        timestamp = datetime.fromisoformat(created_at)
    except ValueError as error:
        raise ValueError("lock creation timestamp is invalid") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("lock creation timestamp must include a timezone")
    if (
        metadata["host"] != local_host
        or metadata["machine_identity"] != local_machine_id
    ):
        raise ValueError(
            "lock belongs to a different or unverifiable machine; automatic "
            "recovery is disabled"
        )


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create one immutable result without replacing any pre-existing file."""
    _write_new_bytes(path, canonical_json_bytes(value))


def _write_new_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _software_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gymnasium": _version("gymnasium"),
        "stable_baselines3": _version("stable-baselines3"),
        "torch": _version("torch"),
    }


def _ppo_config(protocol: Mapping[str, Any], seed: int) -> PPOConfig:
    values = dict(protocol["shared_settings"]["ppo_settings"])
    values.pop("seed_role")
    values["net_arch"] = tuple(values["net_arch"])
    values["seed"] = seed
    config = PPOConfig(**values)
    config.validate()
    return config


def _environment_kwargs(
    protocol: Mapping[str, Any],
    dataset: Any,
    observation_schema: str,
) -> dict[str, Any]:
    settings = protocol["shared_settings"]["environment"]
    return {
        **dataset.environment_kwargs(),
        "initial_capital": settings["initial_capital"],
        "period_sessions": settings["period_sessions"],
        "max_asset_weight": settings["max_asset_weight"],
        "min_cash_weight": settings["min_cash_weight"],
        "max_turnover": settings["max_turnover"],
        "transaction_cost_rate": settings["transaction_cost_rate"],
        "slippage_bps": settings["slippage_bps"],
        "vix_level_scale": settings["vix_level_scale"],
        "observation_schema": observation_schema,
    }


class Phase72ExperimentRunner:
    """Own stage ordering, immutable outputs, resume checks, and test isolation."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        repository_root: str | Path,
        protocol_path: str | Path | None = None,
        training_adapter: TrainingAdapter | None = None,
        evaluation_adapter: EvaluationAdapter | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.protocol_path = (
            Path(protocol_path).resolve()
            if protocol_path is not None
            else None
        )
        self.protocol = load_phase72_protocol(self.protocol_path)
        actual_protocol_path = (
            self.protocol_path
            if self.protocol_path is not None
            else Path(__file__).resolve().parents[2]
            / "configs"
            / "phase7_2_experiment_protocol.json"
        )
        self.protocol_sha256 = sha256_file(actual_protocol_path)
        self.training_adapter = training_adapter or self._train_one
        self.evaluation_adapter = evaluation_adapter or self._evaluate_one
        self.manifest_path = self.output_root / RUN_MANIFEST_NAME
        self._active_roots: set[int] = set()

    def plan(self) -> dict[str, Any]:
        """Validate and reserve the 15-run matrix without generating any data."""
        _require_local_lock_filesystem(self.output_root)
        if self.output_root.exists():
            raise FileExistsError(
                f"refusing to reuse an existing Phase 7.2 output root: {self.output_root}"
            )
        runs = reserve_phase72_output_directories(
            self.protocol,
            output_root=self.output_root,
            repository_root=self.repository_root,
        )
        planned_runs = []
        for run in runs:
            planned_runs.append(
                {
                    "configuration": run.configuration,
                    "training_seed": run.training_seed,
                    "training_scenario_root_seed": run.training_seed,
                    "output_directory": str(
                        run.output_directory.relative_to(self.repository_root)
                    ),
                    "observation_schema": run.observation_schema,
                    "randomize_training_horizons": run.randomize_training_horizons,
                    "horizon_seed": run.horizon_seed,
                    "reservation_token": run.reservation_token,
                    "status": "planned",
                    "training": None,
                }
            )
        manifest: dict[str, Any] = {
            "format": "phase72-run-manifest-v1",
            "protocol_id": self.protocol["protocol_id"],
            "protocol_version": self.protocol["protocol_version"],
            "protocol_sha256": self.protocol_sha256,
            "selection_tolerance": PHASE72_SELECTION_TOLERANCE,
            "created_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
            "revision": 0,
            "status": "planned",
            "planned_stages": ["training", "validation_selection", "final_test"],
            "training_seeds": list(self.protocol["seed_roles"]["training_seeds"]),
            "validation_roots": list(
                self.protocol["seed_roles"]["validation_scenario_roots"]
            ),
            "reserved_test_roots": list(
                self.protocol["seed_roles"]["reserved_test_scenario_roots"]
            ),
            "validation_results": {},
            "test_results": {},
            "runs": planned_runs,
            "software_versions": _software_versions(),
        }
        manifest["manifest_sha256"] = self._manifest_digest(manifest)
        _write_new_json(self.manifest_path, manifest)
        return manifest

    def train(self) -> dict[str, Any]:
        """Train all candidates, preserving completed artifacts on failure."""
        with self._stage_lock():
            manifest = self._read_manifest()
            if manifest["status"] in ("selected", "test_complete"):
                return manifest
            if manifest["status"] == "training_complete":
                for entry in manifest["runs"]:
                    self._verify_trained_run(entry, self._run_from_entry(entry))
                return manifest
            if manifest["status"] == "planned":
                manifest["status"] = "training"
                self._persist_manifest(manifest)
            if manifest["status"] not in ("training", "training_interrupted"):
                raise ValueError(f"training is not allowed from status {manifest['status']!r}")

            for entry in manifest["runs"]:
                run = self._run_from_entry(entry)
                if entry["status"] == "trained":
                    self._verify_trained_run(entry, run)
                    continue
                if entry["status"] == "training":
                    recovered = self._recover_training(entry, run)
                    if recovered:
                        self._persist_manifest(manifest)
                        continue
                    raise ValueError(
                        "incomplete training artifacts cannot be retried automatically; "
                        f"inspect {run.output_directory}"
                    )
                if entry["status"] == "failed":
                    raise ValueError(
                        "failed Phase 7.2 runs are not retried automatically; "
                        f"inspect {run.output_directory}"
                    )
                if entry["status"] != "planned":
                    raise ValueError("run has an unsupported training status")
                if any(run.output_directory.iterdir()):
                    raise FileExistsError(
                        f"refusing to train into a non-empty run directory: "
                        f"{run.output_directory}"
                    )

                entry["status"] = "training"
                entry["training_started_at_utc"] = _utc_now()
                self._persist_manifest(manifest)
                try:
                    outcome = self.training_adapter(self.protocol, run)
                    summary = json.loads(
                        canonical_json_bytes(self._training_summary(run, outcome))
                    )
                    _write_new_json(
                        run.output_directory / TRAINING_RESULT_NAME,
                        summary,
                    )
                    entry["training"] = summary
                    entry["status"] = "trained"
                    entry["training_completed_at_utc"] = _utc_now()
                    self._verify_trained_run(entry, run)
                    self._persist_manifest(manifest)
                except Exception:
                    entry["status"] = "failed"
                    entry["training_failed_at_utc"] = _utc_now()
                    manifest["status"] = "training_interrupted"
                    self._persist_manifest(manifest)
                    raise

            manifest["status"] = "training_complete"
            manifest["training_completed_at_utc"] = _utc_now()
            self._persist_manifest(manifest)
            return manifest

    def validate_and_select(self) -> dict[str, Any]:
        """Evaluate every candidate on all validation roots, then freeze selection."""
        with self._stage_lock():
            manifest = self._read_manifest()
            if manifest["status"] in ("selected", "test_complete"):
                return self._verified_selection(manifest)
            if manifest["status"] not in (
                "training_complete",
                "validation_in_progress",
                "validation_interrupted",
            ):
                raise ValueError("validation requires all 15 trained models")
            for entry in manifest["runs"]:
                if entry["status"] != "trained":
                    raise ValueError("validation requires every training run to be complete")
                self._verify_trained_run(entry, self._run_from_entry(entry))

            manifest["status"] = "validation_in_progress"
            self._persist_manifest(manifest)
            roots = tuple(self.protocol["seed_roles"]["validation_scenario_roots"])
            reserved = set(self.protocol["seed_roles"]["reserved_test_scenario_roots"])
            if set(roots).intersection(reserved):
                raise ValueError("validation and reserved-test roots overlap")
            self._active_roots = set(roots)
            try:
                for entry in manifest["runs"]:
                    run = self._run_from_entry(entry)
                    model_path = run.output_directory / "model.zip"
                    for root in roots:
                        if root not in self._active_roots or root in reserved:
                            raise ValueError("validation attempted to access a reserved-test root")
                        result_path = self._validation_result_path(run, root)
                        key = self._result_key(run, root)
                        if result_path.exists():
                            result = self._load_evaluation_result(
                                result_path,
                                stage="validation",
                                configuration=run.configuration,
                                seed=run.training_seed,
                                root=root,
                            )
                        else:
                            metrics = self.evaluation_adapter(
                                model_path,
                                run.configuration,
                                run.training_seed,
                                "validation",
                                root,
                                self.protocol,
                            )
                            result = self._evaluation_record(
                                "validation",
                                run.configuration,
                                run.training_seed,
                                root,
                                metrics,
                            )
                            self._write_evaluation_result(result_path, result)
                        digest = sha256_file(result_path)
                        manifest["validation_results"][key] = {
                            "path": str(result_path.relative_to(self.output_root)),
                            "sha256": digest,
                        }
                    self._persist_manifest(manifest)
            except Exception:
                manifest["status"] = "validation_interrupted"
                self._persist_manifest(manifest)
                raise
            finally:
                self._active_roots.clear()

            selection = self._build_selection(manifest)
            selection_path = self.output_root / SELECTION_MANIFEST_NAME
            if selection_path.exists():
                saved_selection = _json_object(selection_path, "interrupted selection manifest")
                if not self._selection_matches(saved_selection, selection):
                    raise ValueError("existing selection output conflicts with validation results")
                selection = saved_selection
            else:
                _write_new_json(selection_path, selection)
            manifest["selection"] = {
                "path": SELECTION_MANIFEST_NAME,
                "sha256": sha256_file(selection_path),
                "configuration": selection["selected_configuration"],
            }
            manifest["status"] = "selected"
            manifest["selection_completed_at_utc"] = selection[
                "selection_timestamp_utc"
            ]
            self._persist_manifest(manifest)
            return selection

    def final_test(self) -> dict[str, Any]:
        """Evaluate only the selected configuration on the reserved roots."""
        with self._stage_lock():
            manifest = self._read_manifest()
            if manifest["status"] == "test_complete":
                report_path = self.output_root / FINAL_TEST_REPORT_NAME
                if manifest.get("final_test_report_sha256") != sha256_file(report_path):
                    raise ValueError("final test report checksum mismatch")
                selection = self._verified_selection(manifest)
                self._verify_completed_test_results(manifest, selection)
                report = _json_object(report_path, "final test report")
                if report.get("selected_configuration") != selection.get(
                    "selected_configuration"
                ):
                    raise ValueError("final test report does not match frozen selection")
                expected_report = self._build_final_test_report(manifest, selection)
                if not self._final_test_reports_match(report, expected_report):
                    raise ValueError(
                        "final test report does not match its verified result records"
                    )
                return report
            if manifest["status"] not in (
                "selected",
                "test_in_progress",
                "test_interrupted",
            ):
                raise ValueError("final test is blocked until validation selection is frozen")
            selection = self._verified_selection(manifest)
            selected = selection["selected_configuration"]
            selected_runs = [
                entry for entry in manifest["runs"]
                if entry["configuration"] == selected
            ]
            if len(selected_runs) != len(self.protocol["seed_roles"]["training_seeds"]):
                raise ValueError("selection does not retain all five training seeds")
            for entry in selected_runs:
                self._verify_trained_run(entry, self._run_from_entry(entry))

            manifest["status"] = "test_in_progress"
            manifest["final_test_started_at_utc"] = _utc_now()
            self._persist_manifest(manifest)
            roots = tuple(self.protocol["seed_roles"]["reserved_test_scenario_roots"])
            validation = set(self.protocol["seed_roles"]["validation_scenario_roots"])
            if set(roots).intersection(validation):
                raise ValueError("reserved-test and validation roots overlap")
            self._active_roots = set(roots)
            try:
                for entry in selected_runs:
                    run = self._run_from_entry(entry)
                    for root in roots:
                        if root not in self._active_roots or root in validation:
                            raise ValueError("final test attempted to access a validation root")
                        result_path = self._test_result_path(run, root)
                        key = self._result_key(run, root)
                        if result_path.exists():
                            result = self._load_evaluation_result(
                                result_path,
                                stage="test",
                                configuration=run.configuration,
                                seed=run.training_seed,
                                root=root,
                            )
                        else:
                            metrics = self.evaluation_adapter(
                                run.output_directory / "model.zip",
                                run.configuration,
                                run.training_seed,
                                "test",
                                root,
                                self.protocol,
                            )
                            result = self._evaluation_record(
                                "test",
                                run.configuration,
                                run.training_seed,
                                root,
                                metrics,
                            )
                            self._write_evaluation_result(result_path, result)
                        manifest["test_results"][key] = {
                            "path": str(result_path.relative_to(self.output_root)),
                            "sha256": sha256_file(result_path),
                        }
                    self._persist_manifest(manifest)
            except Exception:
                manifest["status"] = "test_interrupted"
                self._persist_manifest(manifest)
                raise
            finally:
                self._active_roots.clear()

            expected_result_count = len(selected_runs) * len(roots)
            if len(manifest["test_results"]) != expected_result_count:
                raise ValueError("final test did not produce exactly 100 evaluations")
            report = self._build_final_test_report(manifest, selection)
            report_path = self.output_root / FINAL_TEST_REPORT_NAME
            if report_path.exists():
                saved_report = _json_object(report_path, "interrupted final test report")
                if not self._final_test_reports_match(saved_report, report):
                    raise ValueError("existing final test report conflicts with test results")
                report = saved_report
            else:
                _write_new_json(report_path, report)
            manifest["status"] = "test_complete"
            manifest["final_test_completed_at_utc"] = _utc_now()
            manifest["final_test_report_sha256"] = sha256_file(
                self.output_root / FINAL_TEST_REPORT_NAME
            )
            self._persist_manifest(manifest)
            return report

    @contextmanager
    def _stage_lock(self) -> Generator[None, None, None]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_root / ".phase72_stage.lock"
        token = uuid.uuid4().hex
        machine_identity = _machine_identity()
        if machine_identity is None:
            raise RuntimeError(
                "stable machine identity is unavailable; refusing to acquire stage lock"
            )
        process_identity = _process_start_identity(os.getpid())
        if process_identity is None:
            raise RuntimeError(
                "cannot safely identify the current process; refusing to acquire stage lock"
            )
        owner = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "machine_identity": machine_identity,
            "created_at_utc": _utc_now(),
            "invocation_token": token,
            "process_start_identity": process_identity,
        }
        payload = canonical_json_bytes(owner)
        try:
            lock_descriptor, lock_identity = _create_stage_lock(lock_path, payload)
        except FileExistsError as error:
            try:
                existing = json.loads(_read_existing_stage_lock(lock_path))
                _validate_stage_lock_metadata(
                    existing,
                    local_host=socket.gethostname(),
                    local_machine_id=machine_identity,
                )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ) as metadata_error:
                raise RuntimeError(
                    "Phase 7.2 stage lock is malformed or unverifiable "
                    f"({metadata_error}); inspect "
                    f"{lock_path} and remove it manually only after verifying its owner is gone"
                ) from metadata_error
            raise RuntimeError(
                "Phase 7.2 stage lock already exists; stale-lock recovery is disabled. "
                f"Inspect {lock_path} and remove it manually only after verifying its owner is gone"
            ) from error
        try:
            yield
        finally:
            try:
                _release_stage_lock(
                    lock_path,
                    lock_descriptor,
                    lock_identity,
                    token,
                    machine_identity,
                )
            finally:
                os.close(lock_descriptor)

    def _manifest_digest(self, manifest: Mapping[str, Any]) -> str:
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        return sha256_bytes(canonical_json_bytes(payload))

    def _read_manifest(self) -> dict[str, Any]:
        manifest = _json_object(self.manifest_path, "Phase 7.2 run manifest")
        if manifest.get("format") not in (V1_MANIFEST_FORMAT, V2_MANIFEST_FORMAT):
            raise ValueError("unsupported Phase 7.2 run manifest format")
        if manifest.get("manifest_sha256") != self._manifest_digest(manifest):
            raise ValueError("Phase 7.2 run manifest integrity check failed")
        if (
            manifest.get("protocol_id") != self.protocol["protocol_id"]
            or manifest.get("protocol_version") != self.protocol["protocol_version"]
            or manifest.get("protocol_sha256") != self.protocol_sha256
        ):
            raise ValueError("the protocol differs from the planned experiment")
        expected_pairs = [
            (configuration, seed)
            for configuration in CONFIGURATIONS
            for seed in self.protocol["seed_roles"]["training_seeds"]
        ]
        actual_pairs = [
            (entry.get("configuration"), entry.get("training_seed"))
            for entry in manifest.get("runs", [])
            if isinstance(entry, Mapping)
        ]
        if actual_pairs != expected_pairs:
            raise ValueError("run manifest does not contain all 15 planned jobs")
        if (
            manifest.get("validation_roots")
            != self.protocol["seed_roles"]["validation_scenario_roots"]
            or manifest.get("reserved_test_roots")
            != self.protocol["seed_roles"]["reserved_test_scenario_roots"]
            or manifest.get("training_seeds")
            != self.protocol["seed_roles"]["training_seeds"]
        ):
            raise ValueError("run manifest scenario roots differ from the protocol")
        if manifest["format"] == V2_MANIFEST_FORMAT:
            self._validate_v2_manifest(manifest)
        return manifest

    def _validate_v2_manifest(self, manifest: Mapping[str, Any]) -> None:
        required_manifest_fields = {
            "format",
            "protocol_id",
            "protocol_version",
            "protocol_sha256",
            "selection_tolerance",
            "created_at_utc",
            "updated_at_utc",
            "revision",
            "status",
            "planned_stages",
            "training_seeds",
            "validation_roots",
            "reserved_test_roots",
            "validation_results",
            "test_results",
            "runs",
            "software_versions",
            "manifest_sha256",
            "recovery_event",
        }
        optional_manifest_fields = {
            "training_completed_at_utc",
            "selection",
            "selection_completed_at_utc",
            "final_test_started_at_utc",
            "final_test_completed_at_utc",
            "final_test_report_sha256",
        }
        if (
            not required_manifest_fields.issubset(manifest)
            or set(manifest) - required_manifest_fields - optional_manifest_fields
        ):
            raise ValueError("v2 manifest fields are incomplete or unexpected")
        if manifest.get("status") not in {
            "training_interrupted",
            "training",
            "training_complete",
            "validation_in_progress",
            "validation_interrupted",
            "selected",
            "test_in_progress",
            "test_interrupted",
            "test_complete",
        }:
            raise ValueError("v2 manifest stage status is invalid")
        event_reference = manifest.get("recovery_event")
        expected_reference_path = str(RETRY_EVENT_PATH).replace("\\", "/")
        if (
            not isinstance(event_reference, Mapping)
            or set(event_reference) != {"path", "sha256"}
            or event_reference.get("path") != expected_reference_path
            or not isinstance(event_reference.get("sha256"), str)
        ):
            raise ValueError("v2 manifest recovery-event reference is malformed")
        event_path = self.output_root / RETRY_EVENT_PATH
        recovery_events = self.output_root / "recovery_events"
        if _path_has_reparse_component(event_path):
            raise ValueError("recovery-event directory may not contain reparse points")
        if (
            not event_path.is_file()
            or not self._path_is_within_output_root(event_path)
            or {item.name for item in recovery_events.iterdir()} != {event_path.name}
            or event_reference["sha256"] != sha256_file(event_path)
        ):
            raise ValueError("v2 recovery event is missing or fails its checksum")
        event = _json_object(event_path, "Phase 7.2 recovery event")
        event_fields = {
            "format",
            "event_id",
            "created_at_utc",
            "protocol_id",
            "protocol_version",
            "protocol_sha256",
            "original_manifest_sha256",
            "configuration",
            "training_seed",
            "attempt_id",
            "output_directory",
            "reservation_token",
            "operator_reason",
            "original_failed_entry",
        }
        if set(event) != event_fields:
            raise ValueError("recovery event fields are incomplete or unexpected")
        try:
            event_timestamp = datetime.fromisoformat(event["created_at_utc"])
        except (TypeError, ValueError) as error:
            raise ValueError("recovery event timestamp is invalid") from error
        if (
            event["format"] != "phase72-recovery-event-v1"
            or event["event_id"] != RETRY_ATTEMPT_ID
            or event["protocol_id"] != self.protocol["protocol_id"]
            or event["protocol_version"] != self.protocol["protocol_version"]
            or event["protocol_sha256"] != self.protocol_sha256
            or event["configuration"] != "A"
            or event["training_seed"] != 101
            or event["attempt_id"] != RETRY_ATTEMPT_ID
            or not isinstance(event["operator_reason"], str)
            or not event["operator_reason"].strip()
            or event_timestamp.tzinfo is None
            or event_timestamp.utcoffset() is None
            or not isinstance(event["reservation_token"], str)
            or not event["reservation_token"]
            or not isinstance(event["original_manifest_sha256"], str)
            or len(event["original_manifest_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in event["original_manifest_sha256"]
            )
        ):
            raise ValueError("recovery event identity or authorization is invalid")
        entries = {
            (entry["configuration"], entry["training_seed"]): entry
            for entry in manifest["runs"]
        }
        if any(
            "attempt_history" in entry or "active_attempt_id" in entry
            for key, entry in entries.items()
            if key != ("A", 101)
        ):
            raise ValueError("retry attempt metadata is not allowed on other run identities")
        target = entries[("A", 101)]
        required_run_fields = {
            "configuration",
            "training_seed",
            "training_scenario_root_seed",
            "output_directory",
            "observation_schema",
            "randomize_training_horizons",
            "horizon_seed",
            "reservation_token",
            "status",
            "training",
            "training_started_at_utc",
            "training_failed_at_utc",
            "attempt_history",
            "active_attempt_id",
        }
        if set(target) - {
            "configuration", "training_seed", "training_scenario_root_seed",
            "output_directory", "observation_schema", "randomize_training_horizons",
            "horizon_seed", "reservation_token", "status", "training",
            "training_started_at_utc", "training_failed_at_utc",
            "training_completed_at_utc", "training_recovered_at_utc",
            "attempt_history", "active_attempt_id",
        }:
            raise ValueError("v2 A/101 run contains unexpected fields")
        if not required_run_fields.issubset(target):
            raise ValueError("v2 A/101 run fields are incomplete")
        all_reservation_tokens = [
            item.get("reservation_token")
            for item in manifest["runs"]
        ]
        if event["reservation_token"] in all_reservation_tokens:
            raise ValueError("retry reservation token conflicts with another run")
        attempts = target.get("attempt_history")
        if not isinstance(attempts, list) or len(attempts) != 2:
            raise ValueError("v2 manifest must contain exactly two A/101 attempts")
        first, retry = attempts
        if (
            not isinstance(first, Mapping)
            or set(first)
            != {"attempt_id", "status", "output_directory", "reservation_token", "entry_snapshot"}
            or first.get("attempt_id") != "attempt_0001"
            or first.get("status") != "failed"
            or first.get("entry_snapshot") != event["original_failed_entry"]
            or first.get("output_directory") != target.get("output_directory")
            or first.get("reservation_token") != target.get("reservation_token")
        ):
            raise ValueError("historical failed A/101 attempt was changed")
        snapshot = first["entry_snapshot"]
        if (
            not isinstance(snapshot, Mapping)
            or set(snapshot)
            != {
                "configuration", "training_seed", "training_scenario_root_seed",
                "output_directory", "observation_schema",
                "randomize_training_horizons", "horizon_seed",
                "reservation_token", "status", "training",
                "training_started_at_utc", "training_failed_at_utc",
            }
            or snapshot.get("configuration") != "A"
            or snapshot.get("training_seed") != 101
            or snapshot.get("status") != "failed"
            or snapshot.get("training") is not None
        ):
            raise ValueError("original A/101 failure snapshot is invalid")
        expected_retry_path = self._retry_directory(RETRY_ATTEMPT_ID, "A", 101)
        expected_retry_relative = str(expected_retry_path.relative_to(self.repository_root))
        required_retry_fields = {
            "attempt_id",
            "status",
            "output_directory",
            "reservation_token",
            "operator_reason",
            "training",
        }
        optional_retry_fields = {
            "training_started_at_utc",
            "training_completed_at_utc",
            "training_failed_at_utc",
            "error_type",
            "error_message",
        }
        if (
            not isinstance(retry, Mapping)
            or not required_retry_fields.issubset(retry)
            or set(retry) - required_retry_fields - optional_retry_fields
            or retry.get("attempt_id") != RETRY_ATTEMPT_ID
            or retry.get("status") not in ("planned", "training", "failed", "trained")
            or retry.get("output_directory") != expected_retry_relative
            or retry.get("output_directory") != event["output_directory"]
            or retry.get("reservation_token") != event["reservation_token"]
            or retry.get("reservation_token") == target.get("reservation_token")
            or retry.get("operator_reason") != event["operator_reason"]
        ):
            raise ValueError("retry attempt record is invalid")
        retry_status = retry["status"]
        if (
            (retry_status == "planned" and set(retry) != required_retry_fields)
            or (
                retry_status == "training"
                and (
                    set(retry) != required_retry_fields | {"training_started_at_utc"}
                    or retry.get("training") is not None
                )
            )
            or (
                retry_status == "failed"
                and set(retry)
                != required_retry_fields
                | {"training_started_at_utc", "training_failed_at_utc", "error_type", "error_message"}
            )
            or (
                retry_status == "failed"
                and retry.get("training") is not None
            )
            or (
                retry_status == "trained"
                and set(retry)
                != required_retry_fields
                | {"training_started_at_utc", "training_completed_at_utc"}
            )
            or (
                retry_status == "trained"
                and not isinstance(retry.get("training"), Mapping)
            )
        ):
            raise ValueError("retry attempt fields do not match its state")
        active_id = target.get("active_attempt_id")
        if (
            (retry_status == "trained" and (active_id != RETRY_ATTEMPT_ID or target["status"] != "trained"))
            or (retry_status != "trained" and active_id is not None)
            or (retry_status == "failed" and target["status"] != "failed")
            or (retry_status == "training" and target["status"] != "training")
            or (retry_status == "planned" and target["status"] != "failed")
            or (retry_status == "trained" and retry.get("training") != target.get("training"))
            or (retry_status == "failed" and not retry.get("error_type"))
            or (retry_status == "failed" and not retry.get("error_message"))
        ):
            raise ValueError("retry attempt state conflicts with its canonical run entry")
        original_directory = self._canonical_run_directory(target)
        if (
            _path_is_reparse_point(original_directory)
            or not original_directory.is_dir()
            or any(original_directory.iterdir())
        ):
            raise ValueError("original failed A/101 directory changed after recovery authorization")
        if _path_has_reparse_component(expected_retry_path):
            raise ValueError("retry attempt path contains a reparse point")
        if not self._path_is_within_output_root(expected_retry_path):
            raise ValueError("retry attempt path escapes its approved output root")
        retries_root = self.output_root / "retries"
        attempt_root = retries_root / RETRY_ATTEMPT_ID
        configuration_root = attempt_root / "config_A"
        if (
            {item.name for item in retries_root.iterdir()} != {RETRY_ATTEMPT_ID}
            or {item.name for item in attempt_root.iterdir()} != {"config_A"}
            or {item.name for item in configuration_root.iterdir()} != {"seed_101"}
        ):
            raise ValueError("retry namespace contains unexpected paths")
        if expected_retry_path.exists() and not expected_retry_path.is_dir():
            raise ValueError("retry attempt output is not a directory")
        if not expected_retry_path.is_dir():
            raise ValueError("retry attempt output directory is missing")
        if retry_status == "planned" and any(expected_retry_path.iterdir()):
            raise ValueError("planned retry attempt directory is not empty")

    def _persist_manifest(self, manifest: dict[str, Any]) -> None:
        current = self._read_manifest()
        if current["manifest_sha256"] != manifest.get("manifest_sha256"):
            raise ValueError("run manifest changed while a stage was active")
        manifest["revision"] = current["revision"] + 1
        manifest["updated_at_utc"] = _utc_now()
        manifest["manifest_sha256"] = self._manifest_digest(manifest)
        _atomic_replace_json(self.manifest_path, manifest)

    def retry_failed_run(
        self,
        configuration: str,
        seed: int,
        reason: str,
    ) -> dict[str, Any]:
        """Authorize and run the sole supported explicit recovery: A/101."""
        if (configuration, seed) != ("A", 101):
            raise ValueError(
                "the only authorized Phase 7.2 retry is configuration A, seed 101"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("retry reason must be non-empty")
        with self._stage_lock():
            manifest = self._read_manifest()
            if manifest.get("status") not in ("training_interrupted", "training"):
                raise ValueError("retry requires an interrupted training manifest")
            entry = next(
                item for item in manifest["runs"]
                if item["configuration"] == "A" and item["training_seed"] == 101
            )
            if manifest["format"] == V1_MANIFEST_FORMAT:
                self._authorize_retry_v1(manifest, entry, reason.strip())
                manifest = self._read_manifest()
                entry = next(
                    item for item in manifest["runs"]
                    if item["configuration"] == "A" and item["training_seed"] == 101
                )
            attempt = self._retry_attempt(entry)
            if attempt.get("operator_reason") != reason.strip():
                raise ValueError(
                    "retry reason differs from the authorized recovery event"
                )
            run = self._run_from_entry(entry, attempt_id=RETRY_ATTEMPT_ID)
            if attempt["status"] == "training":
                if self._recover_training(entry, run):
                    manifest["status"] = "training_interrupted"
                    self._persist_manifest(manifest)
                    return manifest
                raise ValueError(
                    "interrupted retry has no verifiable completed result; "
                    "automatic retry is disabled"
                )
            if attempt["status"] != "planned":
                raise ValueError(
                    "the authorized retry is not in the safely startable planned state; "
                    "automatic retry and recovery are disabled"
                )
            if any(run.output_directory.iterdir()):
                raise FileExistsError(
                    f"refusing to retry into a non-empty attempt directory: "
                    f"{run.output_directory}"
                )
            attempt["status"] = "training"
            attempt["training_started_at_utc"] = _utc_now()
            entry["status"] = "training"
            entry["training_started_at_utc"] = attempt["training_started_at_utc"]
            manifest["status"] = "training"
            self._persist_manifest(manifest)
            try:
                outcome = self.training_adapter(self.protocol, run)
                summary = json.loads(
                    canonical_json_bytes(self._training_summary(run, outcome))
                )
                _write_new_json(run.output_directory / TRAINING_RESULT_NAME, summary)
                entry["training"] = summary
                entry["status"] = "trained"
                entry["training_completed_at_utc"] = _utc_now()
                attempt["training"] = summary
                attempt["status"] = "trained"
                attempt["training_completed_at_utc"] = entry[
                    "training_completed_at_utc"
                ]
                entry["active_attempt_id"] = RETRY_ATTEMPT_ID
                self._verify_trained_run(entry, run)
            except Exception as error:
                attempt["status"] = "failed"
                attempt["training"] = None
                attempt["training_failed_at_utc"] = _utc_now()
                attempt["error_type"] = type(error).__name__
                attempt["error_message"] = str(error)
                entry["training"] = None
                entry["status"] = "failed"
                entry["training_failed_at_utc"] = attempt["training_failed_at_utc"]
                entry["active_attempt_id"] = None
                manifest["status"] = "training_interrupted"
                self._persist_manifest(manifest)
                raise
            manifest["status"] = "training"
            self._persist_manifest(manifest)
            return manifest

    def _authorize_retry_v1(
        self,
        manifest: dict[str, Any],
        entry: dict[str, Any],
        reason: str,
    ) -> None:
        if entry.get("status") != "failed":
            raise ValueError("A/101 must be the failed run before retry authorization")
        if any(
            item.get("status") != "planned"
            for item in manifest["runs"]
            if item is not entry
        ):
            raise ValueError(
                "retry authorization requires the other 14 runs to remain planned"
            )
        if entry.get("training") is not None:
            raise ValueError("failed A/101 has unexpected training result metadata")
        original_directory = self._canonical_run_directory(entry)
        if not original_directory.is_dir() or any(original_directory.iterdir()):
            raise FileExistsError(
                "original failed A/101 directory must exist and remain empty"
            )
        attempt_directory = self._retry_directory(RETRY_ATTEMPT_ID, "A", 101)
        event_path = self.output_root / RETRY_EVENT_PATH
        retries_root = self.output_root / "retries"
        events_root = event_path.parent
        self._validate_retry_path(attempt_directory)
        if (
            attempt_directory.exists()
            or _path_is_reparse_point(attempt_directory)
            or event_path.exists()
            or _path_is_reparse_point(retries_root)
            or (retries_root.exists() and any(retries_root.iterdir()))
            or _path_is_reparse_point(events_root)
            or not self._path_is_within_output_root(events_root)
            or events_root.exists()
        ):
            raise FileExistsError(
                "retry event, namespace, or attempt output already exists; refusing reuse"
            )

        original_entry = json.loads(canonical_json_bytes(entry))
        token = uuid.uuid4().hex
        if token in {
            run_entry.get("reservation_token")
            for run_entry in manifest["runs"]
        }:
            raise RuntimeError("generated retry reservation token conflicts with a planned run")
        event = {
            "format": "phase72-recovery-event-v1",
            "event_id": RETRY_ATTEMPT_ID,
            "created_at_utc": _utc_now(),
            "protocol_id": self.protocol["protocol_id"],
            "protocol_version": self.protocol["protocol_version"],
            "protocol_sha256": self.protocol_sha256,
            "original_manifest_sha256": manifest["manifest_sha256"],
            "configuration": "A",
            "training_seed": 101,
            "attempt_id": RETRY_ATTEMPT_ID,
            "output_directory": str(
                attempt_directory.relative_to(self.repository_root)
            ),
            "reservation_token": token,
            "operator_reason": reason,
            "original_failed_entry": original_entry,
        }
        events_root.mkdir(exist_ok=False)
        if _path_is_reparse_point(events_root) or not self._path_is_within_output_root(
            events_root
        ):
            raise ValueError(
                "recovery-event path is redirected outside the output root"
            )
        _write_new_json(event_path, event)
        self._reserve_retry_directory(attempt_directory)

        entry["attempt_history"] = [
            {
                "attempt_id": "attempt_0001",
                "status": "failed",
                "output_directory": entry["output_directory"],
                "reservation_token": entry["reservation_token"],
                "entry_snapshot": original_entry,
            },
            {
                "attempt_id": RETRY_ATTEMPT_ID,
                "status": "planned",
                "output_directory": event["output_directory"],
                "reservation_token": token,
                "operator_reason": reason,
                "training": None,
            },
        ]
        entry["active_attempt_id"] = None
        manifest["format"] = V2_MANIFEST_FORMAT
        manifest["recovery_event"] = {
            "path": str(RETRY_EVENT_PATH).replace("\\", "/"),
            "sha256": sha256_file(event_path),
        }
        self._persist_manifest(manifest)

    def _reserve_retry_directory(self, directory: Path) -> None:
        expected = self._retry_directory(RETRY_ATTEMPT_ID, "A", 101)
        if directory != expected:
            raise ValueError("retry directory does not match the authorized attempt path")
        self._validate_retry_path(directory)
        current = self.output_root
        if _path_is_reparse_point(current):
            raise ValueError("retry output root is a reparse point")
        for component in ("retries", RETRY_ATTEMPT_ID, "config_A"):
            current = current / component
            if current.exists():
                if _path_is_reparse_point(current) or not current.is_dir():
                    raise FileExistsError(
                        f"retry path component is not a safe directory: {current}"
                    )
            else:
                current.mkdir()
            self._validate_retry_path(directory)
        directory.mkdir(exist_ok=False)
        self._validate_retry_path(directory)

    def _path_is_within_output_root(self, path: Path) -> bool:
        resolved_root = self.output_root.resolve(strict=False)
        if resolved_root != self.output_root:
            return False
        resolved_path = path.resolve(strict=False)
        try:
            relative = resolved_path.relative_to(resolved_root)
        except ValueError:
            return False
        return bool(relative.parts)

    def _validate_retry_path(self, directory: Path) -> None:
        expected = self._retry_directory(RETRY_ATTEMPT_ID, "A", 101)
        if directory != expected:
            raise ValueError("retry path does not match the authorized A/101 attempt")
        if _path_has_reparse_component(directory):
            raise ValueError("retry path contains a reparse point")
        if not self._path_is_within_output_root(directory):
            raise ValueError("retry path escapes its approved output root")

    def _canonical_run_directory(self, entry: Mapping[str, Any]) -> Path:
        recorded_path = self.repository_root / entry["output_directory"]
        if _path_has_reparse_component(recorded_path):
            raise ValueError("manifest run path may not contain reparse points")
        expected = (
            self.output_root
            / f"config_{entry['configuration']}"
            / f"seed_{entry['training_seed']}"
        ).resolve()
        recorded = recorded_path.resolve()
        if recorded != expected:
            raise ValueError(
                "manifest run path differs from its canonical reserved location"
            )
        return expected

    def _retry_directory(self, attempt_id: str, configuration: str, seed: int) -> Path:
        if attempt_id != RETRY_ATTEMPT_ID or (configuration, seed) != ("A", 101):
            raise ValueError("unknown or unauthorized Phase 7.2 retry attempt")
        return (
            self.output_root
            / "retries"
            / attempt_id
            / f"config_{configuration}"
            / f"seed_{seed}"
        )

    @staticmethod
    def _retry_attempt(entry: Mapping[str, Any]) -> dict[str, Any]:
        attempts = entry.get("attempt_history")
        if not isinstance(attempts, list) or len(attempts) != 2:
            raise ValueError("v2 A/101 manifest has no unique retry attempt history")
        attempt = attempts[1]
        if not isinstance(attempt, dict) or attempt.get("attempt_id") != RETRY_ATTEMPT_ID:
            raise ValueError("v2 A/101 retry attempt identity is invalid")
        return attempt

    def _resolve_run_directory(
        self,
        entry: Mapping[str, Any],
        *,
        attempt_id: str | None = None,
    ) -> tuple[Path, str | None, str | None]:
        canonical = self._canonical_run_directory(entry)
        if entry.get("attempt_history") is None:
            if attempt_id is not None:
                raise ValueError("v1 runs cannot resolve retry attempts")
            return canonical, entry["reservation_token"], None
        if (entry.get("configuration"), entry.get("training_seed")) != ("A", 101):
            raise ValueError("retry attempt history is only valid for A/101")
        original_attempt, retry_attempt = entry["attempt_history"]
        if original_attempt["status"] != "failed":
            raise ValueError("historical A/101 attempt must remain failed")
        if (
            original_attempt["entry_snapshot"].get("output_directory")
            != entry["output_directory"]
        ):
            raise ValueError("historical A/101 attempt path changed")
        if (
            original_attempt["entry_snapshot"].get("reservation_token")
            != entry["reservation_token"]
        ):
            raise ValueError("historical A/101 reservation changed")
        if not canonical.is_dir() or any(canonical.iterdir()):
            raise ValueError("original failed A/101 directory is no longer empty")
        selected_id = attempt_id or entry.get("active_attempt_id")
        if selected_id is None:
            raise ValueError("A/101 has no verified active retry attempt")
        if selected_id != RETRY_ATTEMPT_ID:
            raise ValueError("unknown A/101 attempt id")
        if attempt_id is None and retry_attempt["status"] != "trained":
            raise ValueError("A/101 retry is not a verified trained active attempt")
        if attempt_id is not None and retry_attempt["status"] not in ("planned", "training"):
            raise ValueError("retry attempt is not eligible for training")
        directory = self._retry_directory(selected_id, "A", 101)
        recorded = self.repository_root / retry_attempt["output_directory"]
        expected_relative = str(directory.relative_to(self.repository_root))
        if (
            retry_attempt["output_directory"] != expected_relative
            or recorded.resolve() != directory.resolve()
            or retry_attempt["reservation_token"] == entry["reservation_token"]
        ):
            raise ValueError(
                "retry path or reservation token conflicts with the original run"
            )
        self._validate_retry_path(directory)
        if not directory.is_dir():
            raise ValueError("retry attempt directory is missing or redirected")
        return directory, retry_attempt["reservation_token"], selected_id

    def _run_from_entry(
        self,
        entry: Mapping[str, Any],
        *,
        attempt_id: str | None = None,
    ) -> Phase72Run:
        directory, reservation_token, resolved_attempt_id = self._resolve_run_directory(
            entry,
            attempt_id=attempt_id,
        )
        run = Phase72Run(
            configuration=entry["configuration"],
            training_seed=entry["training_seed"],
            output_directory=directory,
            observation_schema=entry["observation_schema"],
            randomize_training_horizons=entry["randomize_training_horizons"],
            horizon_seed=entry["horizon_seed"],
            reservation_token=reservation_token,
            attempt_id=resolved_attempt_id,
        )
        validate_phase72_run_directory(
            self.protocol,
            run,
            repository_root=self.repository_root,
        )
        return run

    def _train_one(
        self,
        protocol: Mapping[str, Any],
        run: Phase72Run,
    ) -> Mapping[str, Any]:
        generator_config = phase72_synthetic_config(protocol)
        data_seed = int(
            np.random.SeedSequence(run.training_seed, spawn_key=(0,)).generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        dataset = generate_synthetic_market(
            generator_config,
            seed=data_seed,
            scenario_id=f"train-{data_seed}",
        )
        base_env = SyntheticPortfolioEnv(
            **_environment_kwargs(protocol, dataset, run.observation_schema)
        )
        train_env: Any = base_env
        if run.randomize_training_horizons:
            horizon = protocol["configurations"][run.configuration]["horizon_settings"]
            train_env = RandomizedEpisodeHorizonEnv(
                base_env,
                min_episode_decisions=horizon["minimum_decisions"],
                max_episode_decisions=horizon["maximum_decisions"],
                seed=run.horizon_seed,
            )
        config = _ppo_config(protocol, run.training_seed)
        orchestrator = TrainingOrchestrator(
            config=config,
            seed=run.training_seed,
            output_dir=run.output_directory,
            repository_root=self.repository_root,
        )
        try:
            result, metadata, model_path, manifest_path = orchestrator.train(
                train_env,
                options=TrainingOptions(
                    total_timesteps=protocol["shared_settings"]["training_budget"][
                        "requested_total_timesteps"
                    ],
                    seed=run.training_seed,
                    verbose=0,
                ),
                experiment_name=f"phase72-config-{run.configuration}-seed{run.training_seed}",
                phase72_protocol=protocol,
                phase72_run=run,
            )
        finally:
            base_env.close()
        actual_timesteps = getattr(result.model, "num_timesteps", None)
        if (
            isinstance(actual_timesteps, bool)
            or not isinstance(actual_timesteps, int)
            or actual_timesteps <= 0
        ):
            raise ValueError("trained PPO model did not report actual timesteps")
        return {
            "model_path": str(model_path),
            "model_manifest_path": str(manifest_path),
            "experiment_path": str(run.output_directory / "experiment.json"),
            "actual_timesteps": actual_timesteps,
            "metadata": metadata,
        }

    def _training_summary(
        self,
        run: Phase72Run,
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        model_path = run.output_directory / "model.zip"
        model_manifest_path = run.output_directory / "model.manifest.json"
        experiment_path = run.output_directory / "experiment.json"
        expected_paths = {
            Path(str(outcome.get("model_path", ""))).resolve(): model_path.resolve(),
            Path(str(outcome.get("model_manifest_path", ""))).resolve():
                model_manifest_path.resolve(),
            Path(str(outcome.get("experiment_path", ""))).resolve():
                experiment_path.resolve(),
        }
        if any(actual != expected for actual, expected in expected_paths.items()):
            raise ValueError("training adapter returned artifact paths outside the planned run")
        actual_timesteps = outcome.get("actual_timesteps")
        if (
            isinstance(actual_timesteps, bool)
            or not isinstance(actual_timesteps, int)
            or actual_timesteps < self.protocol["shared_settings"]["training_budget"][
                "requested_total_timesteps"
            ]
        ):
            raise ValueError("training adapter reported invalid actual timesteps")
        model_digest = sha256_file(model_path)
        return {
            "configuration": run.configuration,
            "training_seed": run.training_seed,
            "training_scenario_root_seed": run.training_seed,
            "protocol_id": self.protocol["protocol_id"],
            "protocol_version": self.protocol["protocol_version"],
            "protocol_sha256": self.protocol_sha256,
            "requested_timesteps": self.protocol["shared_settings"]["training_budget"][
                "requested_total_timesteps"
            ],
            "actual_timesteps": actual_timesteps,
            "ppo_settings": asdict(_ppo_config(self.protocol, run.training_seed)),
            "observation_schema": run.observation_schema,
            "model_path": str(model_path.relative_to(self.output_root)),
            "model_sha256": model_digest,
            "model_manifest_sha256": sha256_file(model_manifest_path),
            "experiment_sha256": sha256_file(experiment_path),
            "training_completed_at_utc": _utc_now(),
            "training_status": "complete",
        }

    def _verify_trained_run(self, entry: Mapping[str, Any], run: Phase72Run) -> None:
        record_path = run.output_directory / TRAINING_RESULT_NAME
        record = _json_object(record_path, "training result")
        model_path = run.output_directory / "model.zip"
        model_manifest_path = run.output_directory / "model.manifest.json"
        experiment_path = run.output_directory / "experiment.json"
        if record.get("model_path") != str(model_path.relative_to(self.output_root)):
            raise ValueError("training result points outside its active run attempt")
        if (
            record.get("model_sha256") != sha256_file(model_path)
            or record.get("model_manifest_sha256") != sha256_file(model_manifest_path)
            or record.get("experiment_sha256") != sha256_file(experiment_path)
        ):
            raise ValueError("training artifact checksum mismatch; refusing reuse")
        if entry.get("training") != record:
            raise ValueError("run manifest training record does not match its saved result")
        sidecar = _json_object(model_manifest_path, "PPO model manifest")
        experiment = _json_object(experiment_path, "training experiment metadata")
        config = _ppo_config(self.protocol, run.training_seed)
        provenance = sidecar.get("provenance", {})
        training_provenance = {
            "source_type": "synthetic",
            "synthetic": True,
            **provenance,
        }
        expected_run_metadata = build_phase72_run_metadata(
            self.protocol,
            configuration=run.configuration,
            training_seed=run.training_seed,
            training_provenance=training_provenance,
        )
        validate_phase72_run_metadata(
            self.protocol,
            sidecar.get("experiment_metadata", {}),
            configuration=run.configuration,
            training_seed=run.training_seed,
            expected_training_provenance=training_provenance,
        )
        if sidecar.get("experiment_metadata") != expected_run_metadata:
            raise ValueError("PPO model metadata does not match the Phase 7.2 run")
        validate_experiment_manifest_metadata(
            sidecar["experiment_metadata"],
            seed=run.training_seed,
            observation_schema=run.observation_schema,
            ppo_settings=sidecar.get("ppo_config", {}),
            manifest_provenance=sidecar.get("provenance", {}),
        )
        if (
            experiment.get("phase72_experiment") != expected_run_metadata
            or experiment.get("seed") != run.training_seed
            or experiment.get("observation_schema") != run.observation_schema
            or experiment.get("ppo_config") != json.loads(
                canonical_json_bytes(config.__dict__)
            )
        ):
            raise ValueError("experiment metadata does not match its trained model")
        if sidecar.get("model_sha256") != record["model_sha256"]:
            raise ValueError("PPO model sidecar checksum mismatch")
        if (
            record.get("configuration") != run.configuration
            or record.get("training_seed") != run.training_seed
            or record.get("observation_schema") != run.observation_schema
            or record.get("requested_timesteps")
            != self.protocol["shared_settings"]["training_budget"][
                "requested_total_timesteps"
            ]
            or record.get("actual_timesteps", 0) < record["requested_timesteps"]
        ):
            raise ValueError("training result metadata is inconsistent")

    def _recover_training(self, entry: dict[str, Any], run: Phase72Run) -> bool:
        result_path = run.output_directory / TRAINING_RESULT_NAME
        if not result_path.is_file():
            return False
        result = _json_object(result_path, "interrupted training result")
        entry["training"] = result
        self._verify_trained_run(entry, run)
        entry["status"] = "trained"
        entry["training_recovered_at_utc"] = _utc_now()
        if run.attempt_id is not None:
            attempt = self._retry_attempt(entry)
            if attempt["status"] != "training":
                raise ValueError("only a recorded in-progress retry can be recovered")
            attempt["status"] = "trained"
            attempt["training"] = result
            attempt["training_completed_at_utc"] = entry["training_recovered_at_utc"]
            entry["active_attempt_id"] = run.attempt_id
        return True

    def _evaluate_one(
        self,
        model_path: Path,
        configuration: str,
        training_seed: int,
        stage: str,
        root: int,
        protocol: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        roots = (
            protocol["seed_roles"]["validation_scenario_roots"]
            if stage == "validation"
            else protocol["seed_roles"]["reserved_test_scenario_roots"]
        )
        if len(roots) == 0:
            raise ValueError(f"{stage} has no configured scenario roots")
        if root not in roots:
            raise ValueError(f"evaluation root is not configured for {stage}")
        allowed_roots = (
            self.protocol["seed_roles"]["validation_scenario_roots"]
            if stage == "validation"
            else self.protocol["seed_roles"]["reserved_test_scenario_roots"]
        )
        if root not in self._active_roots or root not in allowed_roots:
            raise ValueError(f"evaluation requested a root outside the active {stage} stage")
        schema = protocol["configurations"][configuration]["observation_schema"]
        dataset = generate_synthetic_market(
            phase72_synthetic_config(protocol),
            seed=root,
            scenario_id=f"{stage}-{root}",
        )
        environment = SyntheticPortfolioEnv(
            **_environment_kwargs(protocol, dataset, schema)
        )
        try:
            model = load_ppo_model(
                model_path,
                env=environment,
                expected_config=_ppo_config(protocol, training_seed),
            )
            evaluation = evaluate_phase4(model, environment)
            if evaluation.get("evaluation_type") != "synthetic-only":
                raise ValueError("Phase 7.2 evaluation must remain synthetic-only")
            if not isinstance(evaluation.get("ppo"), Mapping):
                raise ValueError("PPO evaluation did not return a metrics object")
            return evaluation
        finally:
            environment.close()

    def _evaluation_record(
        self,
        stage: str,
        configuration: str,
        seed: int,
        root: int,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        if root not in self._active_roots:
            raise ValueError("evaluation root is outside the active stage")
        scenario_id = f"{stage}-{root}"
        supplied = dict(metrics)
        if supplied.get("evaluation_type") == "synthetic-only":
            ppo_metrics = supplied.get("ppo")
            benchmarks = supplied.get("benchmarks")
            environment = self.protocol["shared_settings"]["environment"]
            evaluation_settings = self.protocol["shared_settings"]["evaluation"]
            expected_period = {
                "warmup_sessions": environment["warmup_sessions"],
                "period_sessions": environment["period_sessions"],
                "evaluation_sessions": evaluation_settings["evaluation_sessions"],
            }
            expected_costs = {
                "initial_capital": environment["initial_capital"],
                "transaction_cost_rate": environment["transaction_cost_rate"],
                "slippage_bps": environment["slippage_bps"],
            }
            if (
                supplied.get("period") != expected_period
                or supplied.get("cost_assumptions") != expected_costs
            ):
                raise ValueError("synthetic evaluation period or costs differ from protocol")
            if not isinstance(benchmarks, Mapping):
                raise ValueError("synthetic evaluation is missing benchmark results")
            required_benchmarks = {
                "SPY buy-and-hold",
                "Equal-weight",
                "Monthly rebalancing",
                "Threshold rebalancing",
            }
            if set(benchmarks) != required_benchmarks:
                raise ValueError("synthetic evaluation benchmark set is incomplete")
            for name, result in benchmarks.items():
                if not isinstance(result, Mapping) or not {
                    "sharpe_ratio",
                    "maximum_drawdown",
                }.issubset(result):
                    raise ValueError(f"benchmark metrics are incomplete: {name}")
        else:
            ppo_metrics = supplied
            benchmarks = {}
            supplied = {
                "evaluation_type": "synthetic-only",
                "ppo": ppo_metrics,
                "benchmarks": {},
            }
        if not isinstance(ppo_metrics, Mapping):
            raise ValueError("PPO evaluation did not return a metrics object")
        self._validate_metrics(ppo_metrics, scenario_id, expected_seed=root)
        normalized = json.loads(canonical_json_bytes(dict(ppo_metrics)))
        evaluation_output = json.loads(canonical_json_bytes(supplied))
        if evaluation_output.get("evaluation_type") != "synthetic-only":
            raise ValueError("evaluation output must remain synthetic-only")
        for name, result in benchmarks.items():
            for key, value in result.items():
                _finite_number(value, f"benchmark {name} metric {key}")
        expected_sessions = self.protocol["shared_settings"]["evaluation"][
            "evaluation_sessions"
        ]
        expected_decisions = self.protocol["shared_settings"]["evaluation"][
            "rebalance_decisions"
        ]
        if normalized.get("steps") != expected_decisions or normalized.get(
            "terminated"
        ) is not True:
            raise ValueError("evaluation did not complete the configured horizon")
        return {
            "format": "phase72-evaluation-v1",
            "stage": stage,
            "configuration": configuration,
            "training_seed": seed,
            "scenario_root": root,
            "scenario_id": scenario_id,
            "protocol_id": self.protocol["protocol_id"],
            "protocol_version": self.protocol["protocol_version"],
            "protocol_sha256": self.protocol_sha256,
            "evaluation_parameters": {
                "deterministic": self.protocol["shared_settings"]["evaluation"][
                    "deterministic"
                ],
                "evaluation_sessions": expected_sessions,
                "rebalance_decisions": expected_decisions,
                **self.protocol["shared_settings"]["environment"],
            },
            "evaluation_output": evaluation_output,
            "metrics": normalized,
            "recorded_at_utc": _utc_now(),
        }

    def _validate_metrics(
        self,
        metrics: Mapping[str, Any],
        scenario_id: str,
        *,
        expected_seed: int,
    ) -> None:
        missing = [key for key in _EVALUATION_METRICS if key not in metrics]
        if missing:
            raise ValueError(f"evaluation metrics are incomplete: {missing}")
        if metrics.get("scenario_id") != scenario_id:
            raise ValueError("evaluation scenario id does not match its configured root")
        if metrics.get("seed") != expected_seed:
            raise ValueError("evaluation data seed does not match its configured root")
        if metrics.get("synthetic") is not True or metrics.get(
            "source_type"
        ) != "synthetic":
            raise ValueError("evaluation metrics are not explicitly synthetic")
        if isinstance(metrics.get("steps"), bool) or not isinstance(
            metrics.get("steps"), int
        ):
            raise ValueError("evaluation step count is invalid")
        for key in _EVALUATION_METRICS:
            _finite_number(metrics[key], f"evaluation metric {key}")

    def _load_evaluation_result(
        self,
        path: Path,
        *,
        stage: str,
        configuration: str,
        seed: int,
        root: int,
    ) -> dict[str, Any]:
        digest_path = path.with_name(f"{path.name}.sha256")
        try:
            recorded_digest = digest_path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"{stage} result integrity sidecar is missing or unreadable") from error
        actual_digest = sha256_file(path)
        if (
            len(recorded_digest) != 64
            or any(character not in "0123456789abcdef" for character in recorded_digest)
            or recorded_digest != actual_digest
        ):
            raise ValueError(f"{stage} result checksum does not match its immutable sidecar")
        result = _json_object(path, f"{stage} result")
        if (
            result.get("format") != "phase72-evaluation-v1"
            or result.get("stage") != stage
            or result.get("configuration") != configuration
            or result.get("training_seed") != seed
            or result.get("scenario_root") != root
            or result.get("scenario_id") != f"{stage}-{root}"
            or result.get("protocol_sha256") != self.protocol_sha256
        ):
            raise ValueError(f"{stage} result metadata is inconsistent: {path}")
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{stage} result metrics are malformed: {path}")
        evaluation_output = result.get("evaluation_output")
        if (
            not isinstance(evaluation_output, Mapping)
            or evaluation_output.get("evaluation_type") != "synthetic-only"
            or evaluation_output.get("ppo") != metrics
        ):
            raise ValueError(f"{stage} result evaluation output is inconsistent: {path}")
        expected_parameters = {
            "deterministic": self.protocol["shared_settings"]["evaluation"][
                "deterministic"
            ],
            "evaluation_sessions": self.protocol["shared_settings"]["evaluation"][
                "evaluation_sessions"
            ],
            "rebalance_decisions": self.protocol["shared_settings"]["evaluation"][
                "rebalance_decisions"
            ],
            **self.protocol["shared_settings"]["environment"],
        }
        if result.get("evaluation_parameters") != expected_parameters:
            raise ValueError(f"{stage} result evaluation parameters are inconsistent")
        self._validate_metrics(metrics, f"{stage}-{root}", expected_seed=root)
        if (
            metrics.get("steps")
            != self.protocol["shared_settings"]["evaluation"]["rebalance_decisions"]
            or metrics.get("terminated") is not True
        ):
            raise ValueError(f"{stage} result is incomplete: {path}")
        return result

    def _write_evaluation_result(
        self,
        path: Path,
        result: Mapping[str, Any],
    ) -> None:
        _write_new_json(path, result)
        _write_new_bytes(
            path.with_name(f"{path.name}.sha256"),
            f"{sha256_file(path)}\n".encode("ascii"),
        )

    def _result_key(self, run: Phase72Run, root: int) -> str:
        return f"{run.configuration}:{run.training_seed}:{root}"

    def _validation_result_path(self, run: Phase72Run, root: int) -> Path:
        return run.output_directory / f"validation_{root}.json"

    def _test_result_path(self, run: Phase72Run, root: int) -> Path:
        return run.output_directory / f"test_{root}.json"

    def _all_validation_results(
        self,
        manifest: Mapping[str, Any],
    ) -> dict[tuple[str, int, int], dict[str, Any]]:
        expected = len(CONFIGURATIONS) * len(self.protocol["seed_roles"]["training_seeds"]) * len(
            self.protocol["seed_roles"]["validation_scenario_roots"]
        )
        results: dict[tuple[str, int, int], dict[str, Any]] = {}
        references = manifest.get("validation_results", {})
        if len(references) != expected:
            raise ValueError("validation results are missing; selection is blocked")
        for entry in manifest["runs"]:
            run = self._run_from_entry(entry)
            for root in self.protocol["seed_roles"]["validation_scenario_roots"]:
                key = self._result_key(run, root)
                reference = references.get(key)
                path = self._validation_result_path(run, root)
                if (
                    not isinstance(reference, Mapping)
                    or reference.get("path") != str(path.relative_to(self.output_root))
                    or reference.get("sha256") != sha256_file(path)
                ):
                    raise ValueError("validation result integrity check failed")
                results[(run.configuration, run.training_seed, root)] = (
                    self._load_evaluation_result(
                        path,
                        stage="validation",
                        configuration=run.configuration,
                        seed=run.training_seed,
                        root=root,
                    )
                )
        if len(results) != expected:
            raise ValueError("validation result matrix is incomplete")
        return results

    def _build_selection(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        results = self._all_validation_results(manifest)
        seeds = self.protocol["seed_roles"]["training_seeds"]
        roots = self.protocol["seed_roles"]["validation_scenario_roots"]
        config_metrics: dict[str, dict[str, Any]] = {}
        for configuration in CONFIGURATIONS:
            seed_aggregates = []
            for seed in seeds:
                values = [
                    results[(configuration, seed, root)]["metrics"]
                    for root in roots
                ]
                seed_aggregates.append(
                    {
                        "training_seed": seed,
                        "mean_validation_sharpe": float(
                            np.mean([value["sharpe_ratio"] for value in values])
                        ),
                        "mean_maximum_drawdown": float(
                            np.mean([value["maximum_drawdown"] for value in values])
                        ),
                        "mean_maximum_drawdown_magnitude": float(
                            np.mean([abs(value["maximum_drawdown"]) for value in values])
                        ),
                        "mean_turnover": float(
                            np.mean([value["portfolio_turnover"] for value in values])
                        ),
                    }
                )
            config_metrics[configuration] = {
                "seed_aggregates": seed_aggregates,
                "mean_validation_sharpe": float(
                    np.mean(
                        [value["mean_validation_sharpe"] for value in seed_aggregates]
                    )
                ),
                "mean_maximum_drawdown": float(
                    np.mean(
                        [value["mean_maximum_drawdown"] for value in seed_aggregates]
                    )
                ),
                "mean_maximum_drawdown_magnitude": float(
                    np.mean(
                        [
                            value["mean_maximum_drawdown_magnitude"]
                            for value in seed_aggregates
                        ]
                    )
                ),
                "mean_turnover": float(
                    np.mean([value["mean_turnover"] for value in seed_aggregates])
                ),
            }
        selected, tie_outcomes = self._select_configuration(config_metrics)
        references = [
            {
                "key": self._result_key(self._run_from_entry(entry), root),
                **dict(manifest["validation_results"][
                    self._result_key(self._run_from_entry(entry), root)
                ]),
            }
            for entry in manifest["runs"]
            for root in roots
        ]
        selection: dict[str, Any] = {
            "format": "phase72-selection-v1",
            "protocol_id": self.protocol["protocol_id"],
            "protocol_version": self.protocol["protocol_version"],
            "protocol_sha256": self.protocol_sha256,
            "selection_timestamp_utc": _utc_now(),
            "selection_tolerance": PHASE72_SELECTION_TOLERANCE,
            "selection_rule": {
                "primary": "highest mean validation Sharpe ratio",
                "aggregation": (
                    "mean over 20 roots per seed, then mean over all five seeds"
                ),
                "tie_breakers": [
                    "lowest mean maximum drawdown magnitude",
                    "lowest mean portfolio turnover",
                    "lexicographically smallest configuration id",
                ],
                "drawdown_note": (
                    "The evaluation metric is non-positive; minimize its absolute "
                    "magnitude to prefer a smaller drawdown."
                ),
            },
            "configuration_metrics": config_metrics,
            "selected_configuration": selected,
            "tie_breaker_outcomes": tie_outcomes,
            "training_seeds_retained": list(seeds),
            "validation_result_references": references,
            "reserved_test_roots_accessed": [],
            "software_versions": _software_versions(),
        }
        selection["selection_manifest_sha256"] = sha256_bytes(
            canonical_json_bytes(selection)
        )
        return selection

    def _select_configuration(
        self,
        metrics: Mapping[str, Mapping[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        remaining = list(CONFIGURATIONS)
        outcomes: dict[str, Any] = {}
        primary = max(float(metrics[name]["mean_validation_sharpe"]) for name in remaining)
        remaining = [
            name for name in remaining
            if abs(float(metrics[name]["mean_validation_sharpe"]) - primary)
            <= PHASE72_SELECTION_TOLERANCE
        ]
        outcomes["primary_metric"] = {
            "best_value": primary,
            "candidates_within_tolerance": list(remaining),
        }
        drawdown = min(
            float(metrics[name]["mean_maximum_drawdown_magnitude"])
            for name in remaining
        )
        remaining = [
            name for name in remaining
            if abs(
                float(metrics[name]["mean_maximum_drawdown_magnitude"]) - drawdown
            ) <= PHASE72_SELECTION_TOLERANCE
        ]
        outcomes["drawdown_tie_breaker"] = {
            "best_magnitude": drawdown,
            "candidates_within_tolerance": list(remaining),
        }
        turnover = min(float(metrics[name]["mean_turnover"]) for name in remaining)
        remaining = [
            name for name in remaining
            if abs(float(metrics[name]["mean_turnover"]) - turnover)
            <= PHASE72_SELECTION_TOLERANCE
        ]
        outcomes["turnover_tie_breaker"] = {
            "best_value": turnover,
            "candidates_within_tolerance": list(remaining),
        }
        selected = min(remaining)
        outcomes["configuration_id_tie_breaker"] = {
            "remaining_candidates": list(remaining),
            "selected": selected,
        }
        return selected, outcomes

    def _verified_selection(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        reference = manifest.get("selection")
        if not isinstance(reference, Mapping) or reference.get("path") != SELECTION_MANIFEST_NAME:
            raise ValueError("selection manifest reference is missing")
        path = self.output_root / SELECTION_MANIFEST_NAME
        if reference.get("sha256") != sha256_file(path):
            raise ValueError("selection manifest checksum mismatch")
        selection = _json_object(path, "selection manifest")
        digest = selection.get("selection_manifest_sha256")
        unsigned = {
            key: value
            for key, value in selection.items()
            if key != "selection_manifest_sha256"
        }
        if digest != sha256_bytes(canonical_json_bytes(unsigned)):
            raise ValueError("selection manifest integrity check failed")
        if (
            selection.get("protocol_sha256") != self.protocol_sha256
            or selection.get("protocol_id") != self.protocol["protocol_id"]
            or selection.get("protocol_version") != self.protocol["protocol_version"]
            or selection.get("selected_configuration") != reference.get("configuration")
        ):
            raise ValueError("selection manifest does not match the planned protocol")
        expected = self._build_selection(manifest)
        expected.pop("selection_timestamp_utc")
        expected.pop("selection_manifest_sha256")
        actual = dict(selection)
        actual.pop("selection_timestamp_utc", None)
        actual.pop("selection_manifest_sha256", None)
        if actual != expected:
            raise ValueError("selection manifest does not match validated results")
        return selection

    @staticmethod
    def _selection_matches(
        existing: Mapping[str, Any],
        expected: Mapping[str, Any],
    ) -> bool:
        if existing.get("selection_manifest_sha256") is None:
            return False
        unsigned = {
            key: value
            for key, value in existing.items()
            if key != "selection_manifest_sha256"
        }
        if existing["selection_manifest_sha256"] != sha256_bytes(
            canonical_json_bytes(unsigned)
        ):
            return False
        comparable_existing = dict(existing)
        comparable_expected = dict(expected)
        for value in (comparable_existing, comparable_expected):
            value.pop("selection_timestamp_utc", None)
            value.pop("selection_manifest_sha256", None)
        return comparable_existing == comparable_expected

    @staticmethod
    def _final_test_reports_match(
        existing: Mapping[str, Any],
        expected: Mapping[str, Any],
    ) -> bool:
        comparable_existing = dict(existing)
        comparable_expected = dict(expected)
        comparable_existing.pop("created_at_utc", None)
        comparable_expected.pop("created_at_utc", None)
        return comparable_existing == comparable_expected

    def _build_final_test_report(
        self,
        manifest: Mapping[str, Any],
        selection: Mapping[str, Any],
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        references = manifest["test_results"]
        selected = selection["selected_configuration"]
        for seed in self.protocol["seed_roles"]["training_seeds"]:
            run = next(
                self._run_from_entry(entry)
                for entry in manifest["runs"]
                if entry["configuration"] == selected
                and entry["training_seed"] == seed
            )
            for root in self.protocol["seed_roles"]["reserved_test_scenario_roots"]:
                key = self._result_key(run, root)
                reference = references.get(key)
                path = self._test_result_path(run, root)
                if (
                    not isinstance(reference, Mapping)
                    or reference.get("sha256") != sha256_file(path)
                ):
                    raise ValueError("final-test result integrity check failed")
                results.append(
                    self._load_evaluation_result(
                        path,
                        stage="test",
                        configuration=selected,
                        seed=seed,
                        root=root,
                    )
                )
        if len(results) != 100:
            raise ValueError("final held-out evaluation must contain exactly 100 results")
        seed_aggregates = []
        for seed in self.protocol["seed_roles"]["training_seeds"]:
            values = [
                result["metrics"]
                for result in results
                if result["training_seed"] == seed
            ]
            seed_aggregates.append(
                {
                    "training_seed": seed,
                    "scenario_count": len(values),
                    "mean_sharpe_ratio": float(
                        np.mean([value["sharpe_ratio"] for value in values])
                    ),
                    "mean_maximum_drawdown": float(
                        np.mean([value["maximum_drawdown"] for value in values])
                    ),
                    "mean_turnover": float(
                        np.mean([value["portfolio_turnover"] for value in values])
                    ),
                }
            )
        all_metrics = [result["metrics"] for result in results]
        return {
            "format": "phase72-final-test-v1",
            "evaluation_stage": "one_time_final_held_out_test",
            "protocol_id": self.protocol["protocol_id"],
            "protocol_version": self.protocol["protocol_version"],
            "protocol_sha256": self.protocol_sha256,
            "selected_configuration": selected,
            "selection_manifest_sha256": manifest["selection"]["sha256"],
            "scenario_roots": list(
                self.protocol["seed_roles"]["reserved_test_scenario_roots"]
            ),
            "scenario_count": len(results),
            "seed_count": len(seed_aggregates),
            "results": results,
            "seed_aggregates": seed_aggregates,
            "overall_aggregate": {
                "scenario_count": len(all_metrics),
                "mean_sharpe_ratio": float(
                    np.mean([value["sharpe_ratio"] for value in all_metrics])
                ),
                "mean_maximum_drawdown": float(
                    np.mean([value["maximum_drawdown"] for value in all_metrics])
                ),
                "mean_turnover": float(
                    np.mean([value["portfolio_turnover"] for value in all_metrics])
                ),
            },
            "test_results_used_for_selection": False,
            "software_versions": _software_versions(),
            "created_at_utc": _utc_now(),
        }

    def _verify_completed_test_results(
        self,
        manifest: Mapping[str, Any],
        selection: Mapping[str, Any],
    ) -> None:
        """Verify the exact frozen test matrix without regenerating any results."""
        selected = selection.get("selected_configuration")
        references = manifest.get("test_results")
        if selected not in CONFIGURATIONS or not isinstance(references, Mapping):
            raise ValueError("completed test result references are malformed")
        entries_by_seed = {
            entry.get("training_seed"): entry
            for entry in manifest["runs"]
            if entry.get("configuration") == selected
        }
        seeds = self.protocol["seed_roles"]["training_seeds"]
        roots = self.protocol["seed_roles"]["reserved_test_scenario_roots"]
        if set(entries_by_seed) != set(seeds):
            raise ValueError("completed test does not retain exactly the selected five seeds")
        expected_keys = {
            f"{selected}:{seed}:{root}"
            for seed in seeds
            for root in roots
        }
        if len(references) != len(expected_keys) or set(references) != expected_keys:
            raise ValueError("completed test result matrix is incomplete or unexpected")

        for seed in seeds:
            run = self._run_from_entry(entries_by_seed[seed])
            for root in roots:
                key = self._result_key(run, root)
                path = self._test_result_path(run, root)
                reference = references[key]
                if (
                    not isinstance(reference, Mapping)
                    or reference.get("path")
                    != str(path.relative_to(self.output_root))
                    or reference.get("sha256") != sha256_file(path)
                ):
                    raise ValueError(
                        f"completed test result reference is invalid: {key}"
                    )
                self._load_evaluation_result(
                    path,
                    stage="test",
                    configuration=selected,
                    seed=seed,
                    root=root,
                )


def build_argument_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plan or execute staged Phase 7.2 synthetic PPO experiment workflows."
    )
    parser.add_argument("stage", choices=("plan", "train", "validate", "test", "retry"))
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--configuration")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reason")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else repository_root / ".artifacts" / "phase7_2"
    )
    runner = Phase72ExperimentRunner(
        output_root=output_root,
        repository_root=repository_root,
        protocol_path=args.protocol,
    )
    if args.stage == "plan":
        runner.plan()
    elif args.stage == "train":
        runner.train()
    elif args.stage == "validate":
        runner.validate_and_select()
    elif args.stage == "test":
        runner.final_test()
    else:
        if args.configuration is None or args.seed is None or args.reason is None:
            raise SystemExit(
                "retry requires --configuration, --seed, and --reason"
            )
        runner.retry_failed_run(args.configuration, args.seed, args.reason)


if __name__ == "__main__":
    main()
