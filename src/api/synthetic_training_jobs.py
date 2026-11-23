"""In-process launcher for explicit synthetic PPO dashboard experiments.

Only the existing synthetic-only training script is launched.  Historical and
India-market workflows are intentionally outside this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

_RUN_ID = re.compile(r"^synthetic-ppo-\d{8}T\d{6}Z$")
_active: dict[str, Any] | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runs_root() -> Path:
    return _project_root() / ".artifacts" / "dashboard_runs"


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid synthetic training run identifier")
    return run_id


def run_directory(run_id: str) -> Path:
    return runs_root() / validate_run_id(run_id)


def start_training() -> dict[str, object]:
    """Launch one new, isolated synthetic PPO experiment in the background."""
    global _active
    if _active is not None and _active["process"].poll() is None:
        raise RuntimeError("a synthetic PPO training job is already running")

    run_id = "synthetic-ppo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = run_directory(run_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "training.log"
    command = [
        sys.executable,
        "-m",
        "scripts.train_synthetic_ppo_dashboard",
        "--output-dir",
        str(output_dir),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=_project_root(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    _active = {"run_id": run_id, "process": process, "output_dir": output_dir, "log_path": log_path}
    return status(run_id)


def status(run_id: str) -> dict[str, object]:
    """Return safe status and the tail of the selected experiment's log."""
    global _active
    directory = run_directory(run_id)
    log_path = directory / "training.log"
    process = _active["process"] if _active and _active["run_id"] == run_id else None
    return_code = process.poll() if process is not None else None
    if process is not None and return_code is None:
        state = "running"
    elif process is not None and return_code == 0:
        state = "completed"
    elif process is not None:
        state = "failed"
    elif (directory / "model.zip").is_file():
        state = "completed"
    else:
        state = "unknown"
    log_tail = ""
    if log_path.is_file():
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
    return {"run_id": run_id, "state": state, "return_code": return_code, "log_tail": log_tail}
