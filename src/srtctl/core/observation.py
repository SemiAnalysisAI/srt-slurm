# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative single-generation Slurm observation for accepted intents."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from srtctl.core.prepared import known_receipt, validate_receipt

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def _run(command: list[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False, timeout=max(0.01, timeout))
    except (OSError, subprocess.TimeoutExpired):
        return None


def observe_job(job_id: str, *, command_timeout: float = 10, expected_comment: str | None = None) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("Numeric job ID required")
    deadline = time.monotonic() + command_timeout
    controller = _run(["scontrol", "show", "job", "--oneliner", job_id], command_timeout)
    # A live controller record always wins over stale accounting. Requeue is
    # rejected rather than combining records from different execution epochs.
    if controller is not None and controller.returncode == 0:
        fields = dict(re.findall(r"(\w+)=([^\s]+)", controller.stdout))
        if fields.get("JobId") != job_id or "JobState" not in fields:
            return {"state": "unknown", "job_id": job_id, "observation": "unavailable"}
        if expected_comment is not None and fields.get("Comment") != expected_comment:
            return {
                "state": "unknown",
                "job_id": job_id,
                "observation": "controller",
                "identity_mismatch": True,
                "diagnostic": "Controller allocation does not match intent comment",
            }
        state = fields["JobState"].split("+")[0]
        restarts = fields.get("Restarts", fields.get("RestartCnt", "0"))
        if restarts != "0":
            return {
                "state": "failed",
                "job_id": job_id,
                "slurm_state": state,
                "exit_code": None,
                "observation": "controller",
                "diagnostic": "Execution generation changed (requeue)",
                "terminal": state in TERMINAL,
            }
        exit_code = fields.get("ExitCode")
        verdict = (
            "completed" if state == "COMPLETED" and exit_code == "0:0" else "failed" if state in TERMINAL else "active"
        )
        return {
            "state": verdict,
            "job_id": job_id,
            "slurm_state": state,
            "exit_code": exit_code,
            "observation": "controller",
            "terminal": state in TERMINAL,
        }
    # Only explicit controller absence permits terminal accounting fallback.
    # A timeout/permission/network error cannot establish absence.
    if controller is None or "invalid job id" not in (controller.stderr + controller.stdout).lower():
        return {"state": "unknown", "job_id": job_id, "observation": "unavailable"}
    accounting = _run(
        [
            "sacct",
            "-X",
            "--noheader",
            "--parsable2",
            "--jobs",
            job_id,
            "--format=JobIDRaw,State,ExitCode,Restarts,Comment",
        ],
        deadline - time.monotonic(),
    )
    if accounting is None or accounting.returncode != 0:
        return {"state": "unknown", "job_id": job_id, "observation": "unavailable"}
    records = [line.strip().split("|") for line in accounting.stdout.splitlines() if line.strip()]
    records = [row for row in records if len(row) >= 4 and row[0] == job_id]
    if len(records) != 1 or records[0][3] != "0":
        return {"state": "unknown", "job_id": job_id, "observation": "accounting"}
    if expected_comment is not None and (len(records[0]) < 5 or records[0][4] != expected_comment):
        return {"state": "unknown", "job_id": job_id, "observation": "accounting", "identity_mismatch": True}
    _, state, exit_code, *_ = records[0]
    state = state.split()[0].split("+")[0]
    verdict = (
        "completed" if state == "COMPLETED" and exit_code == "0:0" else "failed" if state in TERMINAL else "unknown"
    )
    return {
        "state": verdict,
        "job_id": job_id,
        "slurm_state": state,
        "exit_code": exit_code,
        "observation": "accounting",
        "terminal": state in TERMINAL,
    }


def wait_receipt(
    receipt_path: Path, *, timeout: float, poll: float = 5, until_terminal: bool = False
) -> dict[str, Any]:
    if timeout <= 0 or poll <= 0:
        raise ValueError("Observation timeout and poll interval must be positive")
    receipt = validate_receipt(receipt_path)
    deadline = time.monotonic() + timeout
    while True:
        observation = observe_job(
            receipt["job_id"],
            command_timeout=min(10, max(0.01, deadline - time.monotonic())),
            expected_comment=receipt.get("scheduler_comment"),
        )
        observation["receipt_path"] = str(receipt_path)
        if observation["state"] in {"completed", "failed"} and (not until_terminal or observation.get("terminal")):
            if observation["state"] == "completed":
                completion_path = Path(receipt["output_dir"]) / "completion.json"
                try:
                    completion = json.loads(completion_path.read_text())
                except (OSError, ValueError):
                    completion = None
                if not completion or completion.get("manifest_sha256") != receipt["manifest_sha256"]:
                    observation.update(state="failed", diagnostic="Missing or foreign runtime completion evidence")
                elif any(
                    completion.get(key) is not True
                    for key in ("execution_success", "cleanup_complete", "restoration_success")
                ):
                    observation.update(state="failed", diagnostic="Runtime execution/cleanup/restoration incomplete")
            return observation
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            observation.update(state="unknown", diagnostic="Observation deadline exceeded; allocation remains owned")
            return observation
        time.sleep(min(poll, remaining))


def wait_known_receipt(receipt_path: Path, *, timeout: float, poll: float = 5) -> dict[str, Any]:
    """Observe closure of all known owned allocations without qualifying results."""
    if timeout <= 0 or poll <= 0:
        raise ValueError("Observation timeout and poll interval must be positive")
    receipt = known_receipt(receipt_path)
    deadline = time.monotonic() + timeout
    while True:
        outcomes = [
            observe_job(
                job_id,
                command_timeout=min(10, max(0.01, deadline - time.monotonic())),
                expected_comment=receipt["scheduler_comment"],
            )
            for job_id in receipt["accepted_ids"]
        ]
        closed = all(outcome.get("terminal") is True and not outcome.get("identity_mismatch") for outcome in outcomes)
        result = {
            "state": "closed" if closed else "unknown",
            "terminal": closed,
            "receipt_path": str(receipt_path),
            "jobs": outcomes,
        }
        if closed or time.monotonic() >= deadline:
            return result
        time.sleep(min(poll, max(0, deadline - time.monotonic())))
