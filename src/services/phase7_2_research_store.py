"""Read-only access to frozen Phase 7.2 synthetic PPO research artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


V2_MANIFEST_FORMAT = "phase72-run-manifest-v2"
SELECTION_FORMAT = "phase72-selection-v1"
FINAL_REPORT_FORMAT = "phase72-final-test-v1"
EVALUATION_FORMAT = "phase72-evaluation-v1"
PROTOCOL_ID = "phase7_2_synthetic_ppo_comparison"
PROTOCOL_VERSION = 1
ALLOWED_SEEDS = frozenset({101, 202, 303, 404, 505})
ALLOWED_A_SEEDS = frozenset(ALLOWED_SEEDS)
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class Phase72ResearchStoreError(RuntimeError):
    """Base exception for a failed frozen-artifact read."""


class MissingArtifactError(Phase72ResearchStoreError):
    """A required artifact or referenced file is missing."""


class MalformedArtifactError(Phase72ResearchStoreError):
    """An artifact is malformed or has an unsupported format."""


class UnsupportedArtifactFormatError(Phase72ResearchStoreError):
    """A manifest or artifact format is not supported."""


class ChecksumMismatchError(Phase72ResearchStoreError):
    """A stored SHA-256 reference does not match a file."""


class InvalidArtifactReferenceError(Phase72ResearchStoreError):
    """A manifest reference is invalid or inconsistent."""


class UnsafeArtifactPathError(Phase72ResearchStoreError):
    """A filesystem path is outside the approved artifact root."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return _sha256_bytes(_canonical_json_bytes(payload))


def _selection_digest(selection: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in selection.items()
        if key != "selection_manifest_sha256"
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise UnsafeArtifactPathError("artifact path cannot be inspected") from error
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _path_has_reparse_component(path: Path) -> bool:
    current = path.absolute()
    while True:
        if _is_reparse_point(current):
            return True
        if current.parent == current:
            return False
        current = current.parent


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise MalformedArtifactError("artifact contains a non-finite numeric value")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MalformedArtifactError(f"{name} must be numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise MalformedArtifactError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise MalformedArtifactError(f"{name} must be finite")
    return result


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MalformedArtifactError(f"{name} must be an integer")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MalformedArtifactError(f"{name} must be a JSON object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise MalformedArtifactError(f"{name} must be a JSON array")
    return value


class Phase72ResearchStore:
    """Verify and read frozen artifacts under a trusted-local, single-writer assumption.

    Filesystem checks are not race-free against a hostile concurrent replacement
    between validation and later reads.
    """

    def __init__(
        self,
        artifact_root: str | Path | None = None,
        *,
        project_root: str | Path | None = None,
    ) -> None:
        if project_root is None:
            project_root_path = Path(__file__).absolute().parents[2]
        else:
            project_root_path = Path(project_root).expanduser()
        if _path_has_reparse_component(project_root_path):
            raise UnsafeArtifactPathError("project root contains a reparse point")
        try:
            self.project_root = project_root_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise MissingArtifactError("project root is missing") from error
        except (OSError, RuntimeError) as error:
            raise UnsafeArtifactPathError("project root cannot be resolved") from error

        if artifact_root is None:
            supplied_root = self.project_root / ".artifacts" / "phase7_2"
        else:
            supplied_root = Path(artifact_root).expanduser()
            if not supplied_root.is_absolute():
                supplied_root = project_root_path / supplied_root

        if _path_has_reparse_component(supplied_root):
            raise UnsafeArtifactPathError("artifact root contains a reparse point")
        try:
            root = supplied_root.resolve(strict=False)
            root_metadata = root.stat()
        except FileNotFoundError as error:
            raise MissingArtifactError("artifact root is missing") from error
        except (OSError, RuntimeError) as error:
            raise UnsafeArtifactPathError("artifact root cannot be resolved or inspected") from error
        if _path_has_reparse_component(root):
            raise UnsafeArtifactPathError("resolved artifact root contains a reparse point")
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise UnsafeArtifactPathError("artifact root is unsafe")
        self.artifact_root = root

        self._manifest: dict[str, Any] | None = None
        self._selection: dict[str, Any] | None = None
        self._final_report: dict[str, Any] | None = None
        self._verification_complete = False

    def _safe_relative_file(self, relative: str, name: str) -> Path:
        if not isinstance(relative, str) or not relative:
            raise InvalidArtifactReferenceError(f"{name} path is missing")
        normalized = relative.replace("\\", "/")
        if (
            normalized.startswith("/")
            or normalized.startswith("//")
            or _WINDOWS_DRIVE_RE.match(normalized)
            or "://" in normalized
        ):
            raise UnsafeArtifactPathError(f"{name} path is absolute")
        parts = normalized.split("/")
        if any(
            part in ("", ".", "..")
            or ":" in part
            or "\x00" in part
            for part in parts
        ):
            raise UnsafeArtifactPathError(f"{name} path is invalid")
        path = self.artifact_root.joinpath(*parts)
        if _path_has_reparse_component(path):
            raise UnsafeArtifactPathError(f"{name} path contains a reparse point")
        resolved_root = self.artifact_root.resolve(strict=False)
        resolved_path = path.resolve(strict=False)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as error:
            raise UnsafeArtifactPathError(f"{name} path escapes the artifact root") from error
        if not path.is_file():
            raise MissingArtifactError(f"{name} file is missing")
        if _is_reparse_point(path):
            raise UnsafeArtifactPathError(f"{name} path is redirected")
        return path

    def _read_json_file(self, path: Path, name: str) -> dict[str, Any]:
        if _path_has_reparse_component(path):
            raise UnsafeArtifactPathError(f"{name} path contains a reparse point")
        if not path.is_file():
            raise MissingArtifactError(f"{name} is missing")
        if _is_reparse_point(path):
            raise UnsafeArtifactPathError(f"{name} path is redirected")
        try:
            parsed = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-JSON numeric constant {value}")
                ),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise MalformedArtifactError(f"{name} is unreadable or malformed") from error
        if not isinstance(parsed, dict):
            raise MalformedArtifactError(f"{name} must be a JSON object")
        _reject_nonfinite(parsed)
        return parsed

    def _read_primary(self, name: str) -> dict[str, Any]:
        path = self.artifact_root / name
        if _path_has_reparse_component(path):
            raise UnsafeArtifactPathError(f"{name} path contains a reparse point")
        return self._read_json_file(path, name)

    def _read_referenced_result(
        self,
        reference_value: Any,
        *,
        stage: str,
        configuration: str,
        seed: int,
        scenario_root: int,
    ) -> dict[str, Any]:
        reference = _mapping(reference_value, f"{stage} result reference")
        relative = reference.get("path")
        declared_sha256 = reference.get("sha256")
        if not isinstance(relative, str):
            raise InvalidArtifactReferenceError(f"{stage} result path is missing")
        if (
            not isinstance(declared_sha256, str)
            or len(declared_sha256) != 64
            or any(char not in "0123456789abcdef" for char in declared_sha256)
        ):
            raise InvalidArtifactReferenceError(f"{stage} result checksum is invalid")
        if configuration == "A" and seed == 101:
            active_prefix = "retries/attempt_0002/config_A/seed_101/"
            if not relative.replace("\\", "/").startswith(active_prefix):
                raise InvalidArtifactReferenceError(
                    "A/101 result does not use attempt 2"
                )

        path = self._safe_relative_file(relative, f"{stage} result")
        try:
            actual_sha256 = _sha256_file(path)
        except OSError as error:
            raise MissingArtifactError(f"{stage} result cannot be read") from error
        if actual_sha256 != declared_sha256:
            raise ChecksumMismatchError(f"{stage} result checksum mismatch")

        sidecar_relative = path.with_name(path.name + ".sha256").relative_to(
            self.artifact_root
        ).as_posix()
        sidecar = self._safe_relative_file(
            sidecar_relative,
            f"{stage} result sidecar",
        )
        try:
            sidecar_sha256 = sidecar.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError) as error:
            raise MalformedArtifactError(
                f"{stage} result sidecar is unreadable"
            ) from error
        if (
            len(sidecar_sha256) != 64
            or any(char not in "0123456789abcdef" for char in sidecar_sha256)
            or sidecar_sha256 != declared_sha256
            or sidecar_sha256 != actual_sha256
        ):
            raise ChecksumMismatchError(f"{stage} result sidecar mismatch")

        record = self._read_json_file(path, f"{stage} result")
        if (
            record.get("format") != EVALUATION_FORMAT
            or record.get("stage") != stage
            or record.get("configuration") != configuration
            or _int(record.get("training_seed"), f"{stage} training_seed") != seed
            or _int(record.get("scenario_root"), f"{stage} scenario_root")
            != scenario_root
            or record.get("scenario_id") != f"{stage}-{scenario_root}"
            or record.get("protocol_id") != PROTOCOL_ID
            or _int(record.get("protocol_version"), f"{stage} protocol_version")
            != PROTOCOL_VERSION
            or record.get("protocol_sha256") != self._manifest["protocol_sha256"]
        ):
            raise InvalidArtifactReferenceError(
                f"{stage} evaluation result metadata mismatch"
            )
        return record

    def verify(self) -> dict[str, Any]:
        """Explicitly verify the complete frozen artifact set once."""
        manifest = self._read_primary("run_manifest.json")
        if manifest.get("format") != V2_MANIFEST_FORMAT:
            raise UnsupportedArtifactFormatError("unsupported Phase 7.2 manifest format")
        digest = manifest.get("manifest_sha256")
        if not isinstance(digest, str) or digest != _manifest_digest(manifest):
            raise ChecksumMismatchError("manifest embedded digest mismatch")
        if (
            manifest.get("protocol_id") != PROTOCOL_ID
            or _int(manifest.get("protocol_version"), "manifest protocol_version")
            != PROTOCOL_VERSION
        ):
            raise InvalidArtifactReferenceError("manifest protocol identity mismatch")
        protocol_sha256 = manifest.get("protocol_sha256")
        if not isinstance(protocol_sha256, str):
            raise MalformedArtifactError("manifest protocol SHA-256 is missing")

        protocol_path = self.project_root / "configs" / "phase7_2_experiment_protocol.json"
        if _path_has_reparse_component(protocol_path):
            raise UnsafeArtifactPathError("protocol path contains a reparse point")
        if not protocol_path.is_file():
            raise MissingArtifactError("canonical Phase 7.2 protocol is missing")
        actual_protocol_sha256 = _sha256_file(protocol_path)
        if actual_protocol_sha256 != protocol_sha256:
            raise ChecksumMismatchError("protocol SHA-256 mismatch")
        try:
            protocol = json.loads(
                protocol_path.read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-JSON numeric constant {value}")
                ),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise MalformedArtifactError(
                "canonical Phase 7.2 protocol is unreadable"
            ) from error
        if not isinstance(protocol, dict):
            raise MalformedArtifactError("canonical protocol must be a JSON object")
        _reject_nonfinite(protocol)
        if (
            protocol.get("protocol_id") != PROTOCOL_ID
            or _int(protocol.get("protocol_version"), "protocol_version")
            != PROTOCOL_VERSION
        ):
            raise InvalidArtifactReferenceError(
                "canonical protocol identity differs from frozen protocol"
            )

        if manifest.get("status") != "test_complete":
            raise InvalidArtifactReferenceError("manifest is not in test_complete state")
        runs = _list(manifest.get("runs"), "manifest runs")
        if len(runs) != 15:
            raise InvalidArtifactReferenceError("manifest must contain exactly 15 runs")

        validation_roots = _list(manifest.get("validation_roots"), "validation_roots")
        test_roots = _list(manifest.get("reserved_test_roots"), "reserved_test_roots")
        if len(validation_roots) != 20 or len(test_roots) != 20:
            raise InvalidArtifactReferenceError("manifest root counts are invalid")
        try:
            seed_roles = _mapping(protocol.get("seed_roles"), "protocol seed_roles")
            protocol_training_seeds = _list(
                seed_roles.get("training_seeds"),
                "protocol training seeds",
            )
            protocol_validation_roots = _list(
                seed_roles.get("validation_scenario_roots"),
                "protocol validation roots",
            )
            protocol_test_roots = _list(
                seed_roles.get("reserved_test_scenario_roots"),
                "protocol reserved test roots",
            )
        except (KeyError, TypeError) as error:
            raise MalformedArtifactError(
                "canonical protocol seed roles are malformed"
            ) from error
        if (
            protocol_training_seeds != [101, 202, 303, 404, 505]
            or protocol_validation_roots != validation_roots
            or protocol_test_roots != test_roots
        ):
            raise InvalidArtifactReferenceError(
                "manifest seed roles differ from the canonical protocol"
            )
        if not all(isinstance(root, int) and not isinstance(root, bool) for root in validation_roots):
            raise MalformedArtifactError("validation roots must be integers")
        if not all(isinstance(root, int) and not isinstance(root, bool) for root in test_roots):
            raise MalformedArtifactError("reserved test roots must be integers")
        if set(validation_roots).intersection(test_roots):
            raise InvalidArtifactReferenceError("validation and test roots overlap")

        expected_pairs = [
            (configuration, seed)
            for configuration in ("A", "B", "C")
            for seed in (101, 202, 303, 404, 505)
        ]
        actual_pairs: list[tuple[Any, Any]] = []
        for entry_value in runs:
            entry = _mapping(entry_value, "manifest run")
            configuration = entry.get("configuration")
            seed = entry.get("training_seed")
            actual_pairs.append((configuration, seed))
            if entry.get("status") != "trained":
                raise InvalidArtifactReferenceError("manifest run is not trained")
            training = _mapping(entry.get("training"), "manifest training record")
            if (
                training.get("configuration") != configuration
                or training.get("training_seed") != seed
                or training.get("protocol_id") != PROTOCOL_ID
                or _int(
                    training.get("protocol_version"),
                    "training protocol_version",
                )
                != PROTOCOL_VERSION
                or training.get("protocol_sha256") != protocol_sha256
            ):
                raise InvalidArtifactReferenceError("training record is inconsistent")
        if actual_pairs != expected_pairs:
            raise InvalidArtifactReferenceError("manifest run matrix is incomplete or reordered")

        self._verify_a101_attempt(runs)

        selection = self._read_primary("selection.json")
        if selection.get("format") != SELECTION_FORMAT:
            raise UnsupportedArtifactFormatError("unsupported selection format")
        selection_digest = selection.get("selection_manifest_sha256")
        if not isinstance(selection_digest, str) or selection_digest != _selection_digest(selection):
            raise ChecksumMismatchError("selection embedded digest mismatch")
        if (
            selection.get("protocol_id") != PROTOCOL_ID
            or _int(selection.get("protocol_version"), "selection protocol_version")
            != PROTOCOL_VERSION
            or selection.get("protocol_sha256") != protocol_sha256
            or selection.get("selected_configuration") != "A"
            or selection.get("training_seeds_retained") != [101, 202, 303, 404, 505]
        ):
            raise InvalidArtifactReferenceError("selection is inconsistent with the manifest")

        selection_reference = _mapping(manifest.get("selection"), "manifest selection reference")
        if (
            selection_reference.get("path") != "selection.json"
            or not isinstance(selection_reference.get("sha256"), str)
        ):
            raise InvalidArtifactReferenceError("manifest selection reference is malformed")
        selection_path = self.artifact_root / "selection.json"
        if _sha256_file(selection_path) != selection_reference["sha256"]:
            raise ChecksumMismatchError("selection manifest checksum mismatch")

        report = self._read_primary("final_test_report.json")
        if report.get("format") != FINAL_REPORT_FORMAT:
            raise UnsupportedArtifactFormatError("unsupported final-test report format")
        if (
            report.get("protocol_id") != PROTOCOL_ID
            or _int(report.get("protocol_version"), "report protocol_version")
            != PROTOCOL_VERSION
            or report.get("protocol_sha256") != protocol_sha256
            or report.get("selected_configuration") != "A"
            or report.get("selection_manifest_sha256") != selection_reference["sha256"]
            or report.get("test_results_used_for_selection") is not False
        ):
            raise InvalidArtifactReferenceError("final-test report is inconsistent")

        report_path = self.artifact_root / "final_test_report.json"
        report_sha256 = manifest.get("final_test_report_sha256")
        if not isinstance(report_sha256, str) or _sha256_file(report_path) != report_sha256:
            raise ChecksumMismatchError("final-test report checksum mismatch")

        validation_refs = _mapping(manifest.get("validation_results"), "validation result references")
        test_refs = _mapping(manifest.get("test_results"), "test result references")
        if len(validation_refs) != 300:
            raise InvalidArtifactReferenceError("manifest must reference exactly 300 validation results")
        if len(test_refs) != 100:
            raise InvalidArtifactReferenceError("manifest must reference exactly 100 final-test results")

        self._manifest = manifest
        self._selection = selection
        self._final_report = report
        self._verify_selected_a_artifacts(runs)
        self._verify_result_references(validation_refs, "validation", runs)
        self._verify_result_references(test_refs, "test", runs)
        self._verify_selection_aggregates(validation_refs)

        if _int(report.get("scenario_count"), "final report scenario_count") != 100:
            raise InvalidArtifactReferenceError("final report must contain exactly 100 results")
        if _int(report.get("seed_count"), "final report seed_count") != 5:
            raise InvalidArtifactReferenceError("final report must contain exactly 5 seeds")
        report_results = _list(report.get("results"), "final report results")
        if len(report_results) != 100:
            raise InvalidArtifactReferenceError("final report result count is incomplete")
        self._verify_report_aggregates(report, test_refs)

        selection_configuration_metrics = _mapping(
            selection.get("configuration_metrics"),
            "selection configuration_metrics",
        )
        if set(selection_configuration_metrics) != {"A", "B", "C"}:
            raise InvalidArtifactReferenceError("selection configuration metrics are incomplete")

        self._manifest = manifest
        self._selection = selection
        self._final_report = report
        self._verification_complete = True
        return self.status()

    def _verify_a101_attempt(self, runs: list[Any]) -> None:
        entry = next(
            (
                item
                for item in runs
                if isinstance(item, Mapping)
                and item.get("configuration") == "A"
                and item.get("training_seed") == 101
            ),
            None,
        )
        if not isinstance(entry, Mapping):
            raise InvalidArtifactReferenceError("A/101 run entry is missing")
        if entry.get("active_attempt_id") != "attempt_0002":
            raise InvalidArtifactReferenceError("A/101 active attempt is not attempt_0002")
        history = _list(entry.get("attempt_history"), "A/101 attempt_history")
        if len(history) != 2:
            raise InvalidArtifactReferenceError("A/101 attempt history is incomplete")
        attempt_1 = _mapping(history[0], "A/101 attempt 1")
        attempt_2 = _mapping(history[1], "A/101 attempt 2")
        if (
            attempt_1.get("attempt_id") != "attempt_0001"
            or attempt_1.get("status") != "failed"
            or attempt_1.get("output_directory")
            != ".artifacts\\phase7_2\\config_A\\seed_101"
        ):
            raise InvalidArtifactReferenceError("A/101 attempt 1 history changed")
        if (
            attempt_2.get("attempt_id") != "attempt_0002"
            or attempt_2.get("status") != "trained"
            or attempt_2.get("output_directory")
            != ".artifacts\\phase7_2\\retries\\attempt_0002\\config_A\\seed_101"
        ):
            raise InvalidArtifactReferenceError("A/101 attempt 2 history is inconsistent")

        original = self.artifact_root / "config_A" / "seed_101"
        if _path_has_reparse_component(original):
            raise UnsafeArtifactPathError("A/101 original directory is redirected")
        if not original.is_dir():
            raise MissingArtifactError("A/101 original directory is missing")
        try:
            if any(original.iterdir()):
                raise InvalidArtifactReferenceError(
                    "A/101 original failed directory is no longer empty"
                )
        except OSError as error:
            raise MissingArtifactError("A/101 original directory is unreadable") from error

        active = self.artifact_root / "retries" / "attempt_0002" / "config_A" / "seed_101"
        if _path_has_reparse_component(active) or not active.is_dir():
            raise UnsafeArtifactPathError("A/101 active retry directory is unsafe")

    def _resolve_run_directory(self, entry: Mapping[str, Any]) -> Path:
        configuration = entry.get("configuration")
        seed = entry.get("training_seed")
        if configuration == "A" and seed == 101:
            directory = self.artifact_root / "retries" / "attempt_0002" / "config_A" / "seed_101"
        else:
            directory = self.artifact_root / f"config_{configuration}" / f"seed_{seed}"
        if _path_has_reparse_component(directory) or not directory.is_dir():
            raise UnsafeArtifactPathError("active run directory is missing or redirected")
        resolved_root = self.artifact_root.resolve(strict=False)
        resolved_directory = directory.resolve(strict=False)
        try:
            resolved_directory.relative_to(resolved_root)
        except ValueError as error:
            raise UnsafeArtifactPathError("active run directory escapes the artifact root") from error
        return directory

    def _verify_selected_a_artifacts(self, runs: list[Any]) -> None:
        for entry_value in runs:
            entry = _mapping(entry_value, "manifest run")
            if entry.get("configuration") != "A":
                continue
            seed = _int(entry.get("training_seed"), "A training seed")
            if seed not in ALLOWED_A_SEEDS:
                raise InvalidArtifactReferenceError("A run seed is not allow-listed")
            training = _mapping(entry.get("training"), "A training record")
            model_relative = training.get("model_path")
            if not isinstance(model_relative, str):
                raise MalformedArtifactError("A training model path is missing")
            expected_model_relative = (
                "retries/attempt_0002/config_A/seed_101/model.zip"
                if seed == 101
                else f"config_A/seed_{seed}/model.zip"
            )
            if model_relative.replace("\\", "/") != expected_model_relative:
                raise InvalidArtifactReferenceError(
                    "A training model path does not use the active attempt"
                )
            model_path = self._safe_relative_file(model_relative, "A model")
            model_manifest = model_path.parent / "model.manifest.json"
            experiment = model_path.parent / "experiment.json"
            training_result = model_path.parent / "training_result.json"
            for name, path in (
                ("model manifest", model_manifest),
                ("experiment metadata", experiment),
                ("training result", training_result),
            ):
                if _path_has_reparse_component(path) or not path.is_file():
                    raise MissingArtifactError(f"{name} is missing")
            if _sha256_file(model_path) != training.get("model_sha256"):
                raise ChecksumMismatchError("A model checksum mismatch")
            if _sha256_file(model_manifest) != training.get("model_manifest_sha256"):
                raise ChecksumMismatchError("A model manifest checksum mismatch")
            if _sha256_file(experiment) != training.get("experiment_sha256"):
                raise ChecksumMismatchError("A experiment metadata checksum mismatch")
            model_manifest_record = self._read_json_file(
                model_manifest,
                "A model manifest",
            )
            if (
                model_manifest_record.get("source_type") != "synthetic"
                or model_manifest_record.get("synthetic") is not True
            ):
                raise InvalidArtifactReferenceError("A model is not marked synthetic-only")
            training_record = self._read_json_file(
                training_result,
                "A training result",
            )
            if training_record != training:
                raise InvalidArtifactReferenceError(
                    "A training manifest record differs from training_result.json"
                )

    def _verify_result_references(
        self,
        references: Mapping[str, Any],
        stage: str,
        runs: list[Any],
    ) -> None:
        entries_by_key = {
            (entry["configuration"], entry["training_seed"]): entry
            for entry in runs
            if isinstance(entry, Mapping)
        }
        seen: set[str] = set()
        for key, reference_value in references.items():
            if not isinstance(key, str):
                raise InvalidArtifactReferenceError("result key is not a string")
            reference = _mapping(reference_value, "result reference")
            relative = reference.get("path")
            declared_sha256 = reference.get("sha256")
            if not isinstance(relative, str) or not isinstance(declared_sha256, str):
                raise InvalidArtifactReferenceError("result reference fields are invalid")
            try:
                configuration, seed_text, root_text = key.split(":")
                seed = int(seed_text)
                root = int(root_text)
            except (ValueError, AttributeError) as error:
                raise InvalidArtifactReferenceError("result key is malformed") from error
            if (
                configuration not in ("A", "B", "C")
                or seed_text != str(seed)
                or root_text != str(root)
            ):
                raise InvalidArtifactReferenceError(
                    "result key is not in canonical <configuration>:<seed>:<root> form"
                )
            entry = entries_by_key.get((configuration, seed))
            if entry is None:
                raise InvalidArtifactReferenceError("result key does not identify a run")
            if seed not in ALLOWED_SEEDS:
                raise InvalidArtifactReferenceError("result seed is not allow-listed")
            if stage == "validation":
                allowed_roots = set(
                    _list(self._manifest["validation_roots"], "validation_roots")
                    if self._manifest is not None
                    else []
                )
            else:
                allowed_roots = set(
                    _list(self._manifest["reserved_test_roots"], "reserved_test_roots")
                    if self._manifest is not None
                    else []
                )
            if root not in allowed_roots:
                raise InvalidArtifactReferenceError("result root is outside its stage")
            if stage == "test" and configuration != "A":
                raise InvalidArtifactReferenceError("non-A configuration has final-test results")
            if configuration == "A" and seed == 101:
                expected_root = (
                    "retries/attempt_0002/config_A/seed_101/"
                )
                normalized_relative = relative.replace("\\", "/")
                if not normalized_relative.startswith(expected_root):
                    raise InvalidArtifactReferenceError(
                        "A/101 result does not use attempt 2"
                    )

            path = self._safe_relative_file(relative, f"{stage} result")
            if _sha256_file(path) != declared_sha256:
                raise ChecksumMismatchError("manifest result checksum mismatch")
            sidecar = path.with_name(path.name + ".sha256")
            try:
                sidecar_relative = sidecar.relative_to(self.artifact_root).as_posix()
            except ValueError as error:
                raise UnsafeArtifactPathError("result sidecar escapes the artifact root") from error
            sidecar = self._safe_relative_file(sidecar_relative, "result sidecar")
            if not sidecar.is_file():
                raise MissingArtifactError("result checksum sidecar is missing")
            try:
                sidecar_digest = sidecar.read_text(encoding="ascii").strip()
            except (OSError, UnicodeDecodeError) as error:
                raise MalformedArtifactError("result checksum sidecar is unreadable") from error
            if (
                len(sidecar_digest) != 64
                or any(char not in "0123456789abcdef" for char in sidecar_digest)
                or sidecar_digest != declared_sha256
                or sidecar_digest != _sha256_file(path)
            ):
                raise ChecksumMismatchError("result checksum sidecar mismatch")

            record = self._read_json_file(path, f"{stage} result")
            if (
                record.get("format") != EVALUATION_FORMAT
                or record.get("stage") != stage
                or record.get("configuration") != configuration
                or record.get("training_seed") != seed
                or record.get("scenario_root") != root
                or record.get("scenario_id") != f"{stage}-{root}"
                or record.get("protocol_id") != PROTOCOL_ID
                or _int(
                    record.get("protocol_version"),
                    f"{stage} protocol_version",
                )
                != PROTOCOL_VERSION
                or record.get("protocol_sha256") != self._manifest["protocol_sha256"]
            ):
                raise InvalidArtifactReferenceError("evaluation result metadata mismatch")
            metrics = _mapping(record.get("metrics"), "evaluation metrics")
            for metric_name in (
                "initial_portfolio_value",
                "final_portfolio_value",
                "cumulative_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "sortino_ratio",
                "maximum_drawdown",
                "episode_reward",
                "gross_turnover_sum",
                "one_way_turnover_sum",
                "portfolio_turnover",
                "transaction_fees_sum",
                "slippage_cost_sum",
                "transaction_cost_sum",
                "total_transaction_costs",
            ):
                _finite_number(metrics.get(metric_name), f"{stage} metric {metric_name}")
            if (
                metrics.get("source_type") != "synthetic"
                or metrics.get("synthetic") is not True
                or metrics.get("scenario_id") != f"{stage}-{root}"
                or metrics.get("seed") != root
            ):
                raise InvalidArtifactReferenceError("evaluation metrics are not synthetic/consistent")
            steps = metrics.get("steps")
            if isinstance(steps, bool) or not isinstance(steps, int) or steps != 33:
                raise MalformedArtifactError("evaluation steps must be the integer 33")
            if not isinstance(metrics.get("terminated"), bool):
                raise MalformedArtifactError("terminated must be a boolean")
            if not isinstance(metrics.get("truncated"), bool):
                raise MalformedArtifactError("truncated must be a boolean")
            if metrics.get("terminated") is not True:
                raise InvalidArtifactReferenceError("evaluation result is not terminated")
            ending_weights = _mapping(metrics.get("ending_weights"), "ending_weights")
            if not ending_weights:
                raise MalformedArtifactError("ending_weights must not be empty")
            for weight_name, weight_value in ending_weights.items():
                if not isinstance(weight_name, str) or not weight_name:
                    raise MalformedArtifactError("ending weight name is invalid")
                _finite_number(weight_value, f"ending weight {weight_name}")
            evaluation_output = _mapping(record.get("evaluation_output"), "evaluation_output")
            if evaluation_output.get("evaluation_type") != "synthetic-only":
                raise InvalidArtifactReferenceError("evaluation output is not synthetic-only")
            if evaluation_output.get("ppo") != metrics:
                raise InvalidArtifactReferenceError("evaluation PPO output differs from metrics")
            benchmarks = _mapping(
                evaluation_output.get("benchmarks"),
                "evaluation benchmarks",
            )
            benchmark_metric_fields = {
                "initial_portfolio_value",
                "final_portfolio_value",
                "cumulative_return",
                "annualized_return",
                "annualized_volatility",
                "maximum_drawdown",
                "portfolio_turnover",
                "sharpe_ratio",
                "sortino_ratio",
                "total_transaction_costs",
            }
            if set(benchmarks) != {
                "Equal-weight",
                "Monthly rebalancing",
                "SPY buy-and-hold",
                "Threshold rebalancing",
            }:
                raise InvalidArtifactReferenceError("benchmark metric set is invalid")
            for benchmark_name, benchmark_value in benchmarks.items():
                benchmark = _mapping(
                    benchmark_value,
                    f"benchmark {benchmark_name}",
                )
                if set(benchmark) != benchmark_metric_fields:
                    raise MalformedArtifactError(
                        f"benchmark {benchmark_name} fields are incomplete"
                    )
                for metric_name, value in benchmark.items():
                    _finite_number(value, f"{benchmark_name} {metric_name}")
            self._verify_allocation_diagnostics(metrics)
            seen.add(key)
        if stage == "validation" and len(seen) != 300:
            raise InvalidArtifactReferenceError("validation result matrix is incomplete")
        if stage == "test" and len(seen) != 100:
            raise InvalidArtifactReferenceError("final-test result matrix is incomplete")

    def _verify_allocation_diagnostics(self, metrics: Mapping[str, Any]) -> None:
        diagnostics = _mapping(
            metrics.get("allocation_diagnostics"),
            "allocation diagnostics",
        )
        if set(diagnostics) != {"summary", "rebalance_history"}:
            raise MalformedArtifactError("allocation diagnostics fields are invalid")
        summary = _mapping(diagnostics.get("summary"), "allocation summary")
        summary_fields = {
            "asset_allocation_distribution",
            "average_cash_allocation",
            "average_gross_turnover",
            "average_one_way_turnover",
            "average_risky_asset_allocation",
            "decision_count",
            "maximum_cash_allocation",
            "minimum_cash_allocation",
            "total_slippage_costs",
            "total_transaction_costs",
            "total_transaction_fees",
        }
        if set(summary) != summary_fields:
            raise MalformedArtifactError("allocation summary fields are incomplete")
        if _int(summary.get("decision_count"), "allocation decision_count") != 33:
            raise InvalidArtifactReferenceError("allocation decision count is invalid")
        for field in (
            "average_cash_allocation",
            "average_gross_turnover",
            "average_one_way_turnover",
            "average_risky_asset_allocation",
            "maximum_cash_allocation",
            "minimum_cash_allocation",
            "total_slippage_costs",
            "total_transaction_costs",
            "total_transaction_fees",
        ):
            _finite_number(summary.get(field), f"allocation summary {field}")
        asset_distribution = _mapping(
            summary["asset_allocation_distribution"],
            "asset_allocation_distribution",
        )
        if set(asset_distribution) != {
            "CASH", "GLD", "IEF", "QQQ", "SPY", "XLU", "XLV",
        }:
            raise MalformedArtifactError("asset allocation distribution is invalid")
        for asset_name, asset_value in asset_distribution.items():
            distribution = _mapping(
                asset_value,
                f"asset distribution {asset_name}",
            )
            if set(distribution) != {"average", "maximum", "minimum"}:
                raise MalformedArtifactError(
                    f"asset distribution {asset_name} fields are incomplete"
                )
            for field in ("average", "maximum", "minimum"):
                _finite_number(distribution[field], f"{asset_name} {field}")
        expected_asset_keys = set(asset_distribution)
        ending_weights = _mapping(
            metrics.get("ending_weights"),
            "ending_weights",
        )
        if set(ending_weights) != expected_asset_keys:
            raise MalformedArtifactError("ending_weights asset keys are invalid")
        for asset_name, value in ending_weights.items():
            _finite_number(value, f"ending weight {asset_name}")
        history = _list(
            diagnostics.get("rebalance_history"),
            "allocation rebalance_history",
        )
        if len(history) != 33:
            raise InvalidArtifactReferenceError("rebalance history must contain 33 decisions")
        required_history_fields = {
            "decision_number",
            "raw_action",
            "requested_target_weights",
            "applied_portfolio_weights",
            "cash_allocation",
            "gross_turnover",
            "one_way_turnover",
            "portfolio_nav",
            "slippage_costs",
            "transaction_fees",
            "total_transaction_costs",
        }
        for index, item_value in enumerate(history, start=1):
            item = _mapping(item_value, "rebalance history item")
            if set(item) != required_history_fields:
                raise MalformedArtifactError("rebalance history item fields are incomplete")
            if _int(item.get("decision_number"), "decision_number") != index:
                raise InvalidArtifactReferenceError("rebalance decision sequence is invalid")
            for field in (
                "cash_allocation",
                "gross_turnover",
                "one_way_turnover",
                "portfolio_nav",
                "slippage_costs",
                "transaction_fees",
                "total_transaction_costs",
            ):
                _finite_number(item.get(field), f"rebalance history {field}")
            for field in (
                "requested_target_weights",
                "applied_portfolio_weights",
            ):
                weights = _mapping(item[field], field)
                if set(weights) != expected_asset_keys:
                    raise MalformedArtifactError(
                        f"{field} asset keys are invalid"
                    )
                for asset_name, value in weights.items():
                    _finite_number(value, f"{field} {asset_name}")
            raw_action = _list(item["raw_action"], "raw_action")
            if len(raw_action) != 7:
                raise MalformedArtifactError("raw_action must contain exactly 7 values")
            for value in raw_action:
                _finite_number(value, "raw_action")

    def _verify_selection_aggregates(
        self,
        references: Mapping[str, Any],
    ) -> None:
        if self._selection is None:
            raise MalformedArtifactError("selection has not been loaded")
        if self._manifest is None:
            raise MalformedArtifactError("manifest has not been loaded")
        configurations = ("A", "B", "C")
        seeds = (101, 202, 303, 404, 505)
        roots = tuple(
            _list(self._manifest["validation_roots"], "validation_roots")
        )
        if len(roots) != 20 or len(set(roots)) != 20:
            raise InvalidArtifactReferenceError(
                "validation roots are not exactly 20 distinct roots"
            )
        try:
            stored_metrics = _mapping(
                self._selection["configuration_metrics"],
                "selection configuration metrics",
            )
            selection_rule = _mapping(
                self._selection["selection_rule"],
                "selection_rule",
            )
            tie_outcomes = _mapping(
                self._selection["tie_breaker_outcomes"],
                "tie_breaker_outcomes",
            )
        except (KeyError, TypeError) as error:
            raise MalformedArtifactError(
                "selection nested structure is malformed"
            ) from error

        expected_rule = {
            "primary": "highest mean validation Sharpe ratio",
            "aggregation": "mean over 20 roots per seed, then mean over all five seeds",
            "tie_breakers": [
                "lowest mean maximum drawdown magnitude",
                "lowest mean portfolio turnover",
                "lexicographically smallest configuration id",
            ],
            "drawdown_note": (
                "The evaluation metric is non-positive; minimize its absolute "
                "magnitude to prefer a smaller drawdown."
            ),
        }
        if selection_rule != expected_rule:
            raise InvalidArtifactReferenceError("selection rule is not the frozen rule")

        recomputed_configs: dict[str, dict[str, Any]] = {}
        for configuration in configurations:
            seed_aggregates: list[dict[str, Any]] = []
            for seed in seeds:
                values: list[Mapping[str, Any]] = []
                for root in roots:
                    key = f"{configuration}:{seed}:{root}"
                    try:
                        reference = _mapping(
                            references[key],
                            "validation result reference",
                        )
                        path = self._safe_relative_file(
                            reference["path"],
                            "validation result",
                        )
                        record = self._read_json_file(path, "validation result")
                        values.append(
                            _mapping(record["metrics"], "validation metrics")
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        raise MalformedArtifactError(
                            "validation result reference or metric structure is malformed"
                        ) from error
                sharpe = sum(
                    _finite_number(value.get("sharpe_ratio"), "validation Sharpe")
                    for value in values
                ) / len(values)
                drawdown = sum(
                    _finite_number(
                        value.get("maximum_drawdown"),
                        "validation drawdown",
                    )
                    for value in values
                ) / len(values)
                drawdown_magnitude = sum(
                    abs(
                        _finite_number(
                            value.get("maximum_drawdown"),
                            "validation drawdown",
                        )
                    )
                    for value in values
                ) / len(values)
                turnover = sum(
                    _finite_number(
                        value.get("portfolio_turnover"),
                        "validation turnover",
                    )
                    for value in values
                ) / len(values)
                seed_aggregates.append(
                    {
                        "training_seed": seed,
                        "mean_validation_sharpe": sharpe,
                        "mean_maximum_drawdown": drawdown,
                        "mean_maximum_drawdown_magnitude": drawdown_magnitude,
                        "mean_turnover": turnover,
                    }
                )
            recomputed_configs[configuration] = {
                "mean_validation_sharpe": sum(
                    seed["mean_validation_sharpe"] for seed in seed_aggregates
                ) / len(seed_aggregates),
                "mean_maximum_drawdown": sum(
                    seed["mean_maximum_drawdown"] for seed in seed_aggregates
                ) / len(seed_aggregates),
                "mean_maximum_drawdown_magnitude": sum(
                    seed["mean_maximum_drawdown_magnitude"]
                    for seed in seed_aggregates
                ) / len(seed_aggregates),
                "mean_turnover": sum(
                    seed["mean_turnover"] for seed in seed_aggregates
                ) / len(seed_aggregates),
                "seed_aggregates": seed_aggregates,
            }
            try:
                stored_config = _mapping(
                    stored_metrics[configuration],
                    "stored configuration metrics",
                )
                stored_seeds = _list(
                    stored_config["seed_aggregates"],
                    "stored seed aggregates",
                )
            except (KeyError, TypeError) as error:
                raise MalformedArtifactError(
                    "stored configuration metrics are malformed"
                ) from error
            if len(stored_seeds) != 5:
                raise ChecksumMismatchError("selection seed aggregates are incomplete")
            by_seed: dict[int, Mapping[str, Any]] = {}
            for stored_value in stored_seeds:
                try:
                    stored_seed = _mapping(
                        stored_value,
                        "stored seed aggregate",
                    )
                    seed_id = _int(
                        stored_seed["training_seed"],
                        "stored aggregate training_seed",
                    )
                except (KeyError, TypeError) as error:
                    raise MalformedArtifactError(
                        "stored seed aggregate identity is malformed"
                    ) from error
                if seed_id not in seeds or seed_id in by_seed:
                    raise InvalidArtifactReferenceError(
                        "stored seed aggregates contain duplicates or unknown seeds"
                    )
                by_seed[seed_id] = stored_seed
            if set(by_seed) != set(seeds):
                raise InvalidArtifactReferenceError(
                    "stored seed aggregates do not contain exactly five retained seeds"
                )
            for expected in seed_aggregates:
                seed_id = expected["training_seed"]
                stored_seed = by_seed[seed_id]
                for name in (
                    "mean_validation_sharpe",
                    "mean_maximum_drawdown",
                    "mean_maximum_drawdown_magnitude",
                    "mean_turnover",
                ):
                    try:
                        stored_value = _finite_number(
                            stored_seed[name],
                            f"stored seed aggregate {name}",
                        )
                    except (KeyError, TypeError) as error:
                        raise MalformedArtifactError(
                            f"stored seed aggregate {name} is malformed"
                        ) from error
                    if not math.isclose(
                        stored_value,
                        expected[name],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        raise ChecksumMismatchError(
                            "selection seed aggregate differs from validation results"
                        )
            for name, expected_name in (
                ("mean_validation_sharpe", "mean_validation_sharpe"),
                ("mean_maximum_drawdown", "mean_maximum_drawdown"),
                ("mean_maximum_drawdown_magnitude", "mean_maximum_drawdown_magnitude"),
                ("mean_turnover", "mean_turnover"),
            ):
                expected = sum(
                    seed[expected_name]
                    for seed in seed_aggregates
                ) / len(seed_aggregates)
                try:
                    stored_value = _finite_number(
                        stored_config[name],
                        f"stored configuration aggregate {name}",
                    )
                except (KeyError, TypeError) as error:
                    raise MalformedArtifactError(
                        f"stored configuration aggregate {name} is malformed"
                    ) from error
                if not math.isclose(
                    stored_value,
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ChecksumMismatchError(
                        "selection configuration aggregate differs from validation results"
                    )

        tolerance = _finite_number(
            self._selection.get("selection_tolerance"),
            "selection tolerance",
        )
        if tolerance != 1e-12:
            raise InvalidArtifactReferenceError("selection tolerance changed")

        remaining = list(configurations)
        primary = max(
            recomputed_configs[name]["mean_validation_sharpe"]
            for name in remaining
        )
        remaining = [
            name
            for name in remaining
            if abs(
                recomputed_configs[name]["mean_validation_sharpe"] - primary
            ) <= tolerance
        ]
        primary_candidates = list(remaining)
        drawdown = min(
            recomputed_configs[name]["mean_maximum_drawdown_magnitude"]
            for name in remaining
        )
        remaining = [
            name
            for name in remaining
            if abs(
                recomputed_configs[name]["mean_maximum_drawdown_magnitude"]
                - drawdown
            ) <= tolerance
        ]
        drawdown_candidates = list(remaining)
        turnover = min(
            recomputed_configs[name]["mean_turnover"]
            for name in remaining
        )
        remaining = [
            name
            for name in remaining
            if abs(recomputed_configs[name]["mean_turnover"] - turnover)
            <= tolerance
        ]
        turnover_candidates = list(remaining)
        recomputed_selection = min(remaining)
        if recomputed_selection != "A":
            raise InvalidArtifactReferenceError(
                "recomputed selection does not select Configuration A"
            )
        try:
            primary_outcome = _mapping(
                tie_outcomes["primary_metric"],
                "primary tie-break outcome",
            )
            drawdown_outcome = _mapping(
                tie_outcomes["drawdown_tie_breaker"],
                "drawdown tie-break outcome",
            )
            turnover_outcome = _mapping(
                tie_outcomes["turnover_tie_breaker"],
                "turnover tie-break outcome",
            )
            config_outcome = _mapping(
                tie_outcomes["configuration_id_tie_breaker"],
                "configuration tie-break outcome",
            )
        except (KeyError, TypeError) as error:
            raise MalformedArtifactError(
                "selection tie-break metadata is malformed"
            ) from error
        if not math.isclose(
            _finite_number(
                primary_outcome.get("best_value"),
                "primary tie-break value",
            ),
            primary,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise InvalidArtifactReferenceError(
                "primary tie-break metadata differs from recomputed result"
            )
        if primary_outcome.get("candidates_within_tolerance") != primary_candidates:
            raise InvalidArtifactReferenceError(
                "primary tie-break candidates differ from recomputed result"
            )
        if not math.isclose(
            _finite_number(
                drawdown_outcome.get("best_magnitude"),
                "drawdown tie-break value",
            ),
            drawdown,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise InvalidArtifactReferenceError(
                "drawdown tie-break metadata differs from recomputed result"
            )
        if (
            drawdown_outcome.get("candidates_within_tolerance")
            != drawdown_candidates
        ):
            raise InvalidArtifactReferenceError(
                "drawdown tie-break candidates differ from recomputed result"
            )
        if not math.isclose(
            _finite_number(
                turnover_outcome.get("best_value"),
                "turnover tie-break value",
            ),
            turnover,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise InvalidArtifactReferenceError(
                "turnover tie-break metadata differs from recomputed result"
            )
        if (
            turnover_outcome.get("candidates_within_tolerance")
            != turnover_candidates
        ):
            raise InvalidArtifactReferenceError(
                "turnover tie-break candidates differ from recomputed result"
            )
        if (
            config_outcome.get("selected") != recomputed_selection
            or config_outcome.get("remaining_candidates") != [recomputed_selection]
        ):
            raise InvalidArtifactReferenceError(
                "configuration tie-break metadata differs from recomputed result"
            )

    def _verify_report_aggregates(
        self,
        report: Mapping[str, Any],
        references: Mapping[str, Any],
    ) -> None:
        roots = tuple(
            _list(self._manifest["reserved_test_roots"], "reserved_test_roots")
        )
        if tuple(report.get("scenario_roots", [])) != roots:
            raise InvalidArtifactReferenceError(
                "final report scenario roots differ from the manifest"
            )
        report_results = _list(report.get("results"), "final report results")
        if len(report_results) != 100:
            raise InvalidArtifactReferenceError(
                "final report must contain exactly 100 results"
            )
        seen_report_keys: set[str] = set()
        for report_result_value in report_results:
            try:
                report_result = _mapping(
                    report_result_value,
                    "final report result",
                )
                configuration = report_result["configuration"]
                seed = _int(
                    report_result["training_seed"],
                    "report training_seed",
                )
                root = _int(
                    report_result["scenario_root"],
                    "report scenario_root",
                )
            except (KeyError, TypeError) as error:
                raise MalformedArtifactError(
                    "final report result identity is malformed"
                ) from error
            if configuration != "A" or seed not in ALLOWED_SEEDS:
                raise InvalidArtifactReferenceError(
                    "final report contains an unexpected configuration or seed"
                )
            if root not in roots:
                raise InvalidArtifactReferenceError(
                    "final report result uses an unexpected reserved root"
                )
            key = f"{configuration}:{seed}:{root}"
            if key in seen_report_keys:
                raise InvalidArtifactReferenceError(
                    "final report contains duplicate records"
                )
            seen_report_keys.add(key)
            try:
                reference = _mapping(
                    references[key],
                    "test result reference",
                )
                path = self._safe_relative_file(
                    reference["path"],
                    "test result",
                )
                stored_record = self._read_json_file(
                    path,
                    "stored test result",
                )
            except (KeyError, TypeError) as error:
                raise MalformedArtifactError(
                    "final report cannot resolve its corresponding test record"
                ) from error
            if report_result != stored_record:
                raise ChecksumMismatchError(
                    "final report result differs from its stored test record"
                )
        expected_report_keys = {
            f"A:{seed}:{root}"
            for seed in ALLOWED_SEEDS
            for root in roots
        }
        if seen_report_keys != expected_report_keys:
            raise InvalidArtifactReferenceError(
                "final report 5-seed x 20-root matrix is incomplete or unexpected"
            )

        overall = _mapping(report.get("overall_aggregate"), "overall aggregate")
        if set(overall) != {
            "scenario_count",
            "mean_sharpe_ratio",
            "mean_maximum_drawdown",
            "mean_turnover",
        }:
            raise MalformedArtifactError("overall aggregate fields are invalid")
        if _int(overall.get("scenario_count"), "overall scenario_count") != 100:
            raise InvalidArtifactReferenceError("overall aggregate count is invalid")
        for name in (
            "mean_sharpe_ratio",
            "mean_maximum_drawdown",
            "mean_turnover",
        ):
            _finite_number(overall.get(name), f"overall aggregate {name}")
        seeds = _list(report.get("seed_aggregates"), "seed aggregates")
        if len(seeds) != 5:
            raise InvalidArtifactReferenceError("report must contain five seed aggregates")
        seen: set[int] = set()
        for seed_value in seeds:
            seed = _mapping(seed_value, "seed aggregate")
            if set(seed) != {
                "training_seed",
                "scenario_count",
                "mean_sharpe_ratio",
                "mean_maximum_drawdown",
                "mean_turnover",
            }:
                raise MalformedArtifactError("seed aggregate fields are invalid")
            seed_id = _int(seed.get("training_seed"), "aggregate training_seed")
            if seed_id not in ALLOWED_SEEDS or seed_id in seen:
                raise InvalidArtifactReferenceError("seed aggregate is invalid")
            seen.add(seed_id)
            if _int(seed.get("scenario_count"), "seed scenario_count") != 20:
                raise InvalidArtifactReferenceError("seed aggregate count is invalid")
            for name in (
                "mean_sharpe_ratio",
                "mean_maximum_drawdown",
                "mean_turnover",
            ):
                _finite_number(seed.get(name), f"seed aggregate {name}")
        if seen != set(ALLOWED_SEEDS):
            raise InvalidArtifactReferenceError("seed aggregate set is incomplete")

        roots = tuple(
            _list(self._manifest["reserved_test_roots"], "reserved_test_roots")
        )
        all_metrics: list[Mapping[str, Any]] = []
        seed_metrics: dict[int, list[Mapping[str, Any]]] = {
            seed: [] for seed in ALLOWED_SEEDS
        }
        for seed in ALLOWED_SEEDS:
            for root in roots:
                key = f"A:{seed}:{root}"
                reference = _mapping(references[key], "test result reference")
                path = self._safe_relative_file(reference["path"], "test result")
                record = self._read_json_file(path, "test result")
                metrics = _mapping(record["metrics"], "test metrics")
                all_metrics.append(metrics)
                seed_metrics[seed].append(metrics)

        for seed_value in seeds:
            seed = _mapping(seed_value, "seed aggregate")
            seed_id = int(seed["training_seed"])
            values = seed_metrics[seed_id]
            expected = {
                "mean_sharpe_ratio": sum(
                    float(_finite_number(value["sharpe_ratio"], "final Sharpe"))
                    for value in values
                ) / len(values),
                "mean_maximum_drawdown": sum(
                    float(_finite_number(value["maximum_drawdown"], "final drawdown"))
                    for value in values
                ) / len(values),
                "mean_turnover": sum(
                    float(
                        _finite_number(
                            value["portfolio_turnover"],
                            "final turnover",
                        )
                    )
                    for value in values
                ) / len(values),
            }
            for name, value in expected.items():
                if not math.isclose(
                    float(seed[name]),
                    value,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ChecksumMismatchError(
                        "final-test seed aggregate differs from stored results"
                    )

        expected_overall = {
            "mean_sharpe_ratio": sum(
                float(_finite_number(value["sharpe_ratio"], "overall final Sharpe"))
                for value in all_metrics
            ) / len(all_metrics),
            "mean_maximum_drawdown": sum(
                float(_finite_number(value["maximum_drawdown"], "overall final drawdown"))
                for value in all_metrics
            ) / len(all_metrics),
            "mean_turnover": sum(
                float(
                    _finite_number(
                        value["portfolio_turnover"],
                        "overall final turnover",
                    )
                )
                for value in all_metrics
            ) / len(all_metrics),
        }
        for name, value in expected_overall.items():
            if not math.isclose(
                float(overall[name]),
                value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ChecksumMismatchError(
                    "final-test overall aggregate differs from stored results"
                )

        report_results = _list(report["results"], "final report results")
        if len(report_results) != 100:
            raise InvalidArtifactReferenceError("final report result count is incomplete")
        seen_report_keys: set[str] = set()
        for result_value in report_results:
            result = _mapping(result_value, "final report result")
            configuration = result.get("configuration")
            seed = result.get("training_seed")
            root = result.get("scenario_root")
            key = f"{configuration}:{seed}:{root}"
            if configuration != "A" or seed not in ALLOWED_SEEDS:
                raise InvalidArtifactReferenceError("final report contains invalid configuration")
            if root not in roots:
                raise InvalidArtifactReferenceError("final report contains invalid root")
            seen_report_keys.add(key)
        expected_keys = {
            f"A:{seed}:{root}"
            for seed in ALLOWED_SEEDS
            for root in roots
        }
        if seen_report_keys != expected_keys:
            raise InvalidArtifactReferenceError("final report result matrix is incomplete")

    def status(self) -> dict[str, Any]:
        if not self._verification_complete:
            raise MalformedArtifactError("verify() must complete before status is available")
        manifest = _mapping(self._manifest, "manifest")
        selection = _mapping(self._selection, "selection")
        report = _mapping(self._final_report, "final report")
        return {
            "format": manifest["format"],
            "manifest_status": manifest["status"],
            "protocol_id": manifest["protocol_id"],
            "protocol_version": manifest["protocol_version"],
            "protocol_sha256": manifest["protocol_sha256"],
            "synthetic_only": True,
            "run_count": len(_list(manifest["runs"], "runs")),
            "trained_run_count": len(
                [entry for entry in manifest["runs"] if entry["status"] == "trained"]
            ),
            "validation_result_count": len(
                _mapping(manifest["validation_results"], "validation_results")
            ),
            "final_test_result_count": len(
                _mapping(manifest["test_results"], "test_results")
            ),
            "selected_configuration": selection["selected_configuration"],
            "final_test_scenario_count": report["scenario_count"],
            "integrity_verified": True,
        }

    def read_status(self) -> dict[str, Any]:
        if not self._verification_complete:
            self.verify()
        return deepcopy(self.status())

    def read_selection(self) -> dict[str, Any]:
        if not self._verification_complete:
            self.verify()
        selection = _mapping(self._selection, "selection")
        return {
            "format": selection["format"],
            "protocol_id": selection["protocol_id"],
            "protocol_version": selection["protocol_version"],
            "protocol_sha256": selection["protocol_sha256"],
            "selected_configuration": selection["selected_configuration"],
            "selection_sha256": selection["selection_manifest_sha256"],
            "selection_rule": deepcopy(selection["selection_rule"]),
            "selection_tolerance": selection["selection_tolerance"],
            "training_seeds_retained": list(selection["training_seeds_retained"]),
            "configuration_metrics": deepcopy(selection["configuration_metrics"]),
            "tie_breaker_outcomes": deepcopy(selection["tie_breaker_outcomes"]),
        }

    def read_validation_aggregates(self) -> dict[str, Any]:
        if not self._verification_complete:
            self.verify()
        selection = _mapping(self._selection, "selection")
        configuration_metrics = _mapping(
            selection["configuration_metrics"],
            "configuration_metrics",
        )
        return {
            "synthetic_only": True,
            "configuration_metrics": deepcopy(configuration_metrics),
        }

    def read_final_test_overall(self) -> dict[str, Any]:
        if not self._verification_complete:
            self.verify()
        report = _mapping(self._final_report, "final report")
        return {
            "synthetic_only": True,
            "selected_configuration": report["selected_configuration"],
            "scenario_count": report["scenario_count"],
            "seed_count": report["seed_count"],
            "overall_aggregate": deepcopy(report["overall_aggregate"]),
        }

    def read_final_test_seed_aggregates(self) -> list[dict[str, Any]]:
        if not self._verification_complete:
            self.verify()
        report = _mapping(self._final_report, "final report")
        return deepcopy(_list(report["seed_aggregates"], "seed aggregates"))

    def read_evaluation_metrics(
        self,
        *,
        stage: str,
        seed: int,
        scenario_root: int,
    ) -> dict[str, Any]:
        if not self._verification_complete:
            self.verify()
        if stage not in ("validation", "test"):
            raise InvalidArtifactReferenceError("unsupported evaluation stage")
        seed = _int(seed, "training seed")
        scenario_root = _int(scenario_root, "scenario root")
        if seed not in ALLOWED_SEEDS:
            raise InvalidArtifactReferenceError("seed is not allow-listed")
        manifest = _mapping(self._manifest, "manifest")
        roots_name = (
            "validation_roots"
            if stage == "validation"
            else "reserved_test_roots"
        )
        refs_name = (
            "validation_results"
            if stage == "validation"
            else "test_results"
        )
        roots = set(_list(manifest[roots_name], roots_name))
        if scenario_root not in roots:
            raise InvalidArtifactReferenceError("scenario root is outside the stage")
        key = f"A:{seed}:{scenario_root}"
        references = _mapping(manifest[refs_name], refs_name)
        reference = references.get(key)
        if not isinstance(reference, Mapping):
            raise InvalidArtifactReferenceError("requested evaluation result is not stored")
        record = self._read_referenced_result(
            reference,
            stage=stage,
            configuration="A",
            seed=seed,
            scenario_root=scenario_root,
        )
        metrics = _mapping(record["metrics"], "evaluation metrics")
        return deepcopy(metrics)

    def read_allocation_diagnostics(
        self,
        *,
        stage: str,
        seed: int,
        scenario_root: int,
    ) -> dict[str, Any]:
        metrics = self.read_evaluation_metrics(
            stage=stage,
            seed=seed,
            scenario_root=scenario_root,
        )
        diagnostics = _mapping(
            metrics["allocation_diagnostics"],
            "allocation diagnostics",
        )
        return deepcopy(diagnostics)
