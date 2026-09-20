# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-scoped step discovery through real subprocesses with controlled Slurm replies."""

import json
import os
import sys
from pathlib import Path
from subprocess import Popen
from unittest.mock import MagicMock

import pytest

from srtctl.core.processes import ManagedProcess, ProcessRegistry, find_step_id, list_step_ids


def step_record(step_id: str = "18324.5", name: str = "agg_0_hpc-gpu-1-0", state: str = "RUNNING") -> str:
    return (
        f"StepId={step_id} UserId=1000 StartTime=2026-09-20T02:19:13 TimeLimit=UNLIMITED "
        f"State={state} Partition=compute NodeList=hpc-gpu-1-0 "
        f"Nodes=1 CPUs=224 Tasks=1 Name={name} Network=(null) "
        "TRES=cpu=224,mem=0,node=1 ResvPorts=(null) SrunHost:Pid=hpc-gpu-1-0:1234\n"
    )


@pytest.fixture
def scheduler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    responses = tmp_path / "responses.json"
    calls = tmp_path / "calls.jsonl"
    responses.write_text(
        json.dumps(
            {
                # The observed Step Manager failure: squeue sees only controller steps.
                "squeue": {"stdout": "18324.batch batch\n18324.extern extern\n"},
                "scontrol": {
                    "stdout": step_record("18324.batch", "batch")
                    + step_record("18324.extern", "extern")
                    + step_record()
                },
                "scancel": {},
            }
        )
    )
    script = (
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        "from pathlib import Path\n"
        f"with Path({str(calls)!r}).open('a') as stream:\n"
        "    stream.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')\n"
        f"response = json.loads(Path({str(responses)!r}).read_text())[Path(sys.argv[0]).name]\n"
        "time.sleep(response.get('delay', 0))\n"
        "sys.stdout.write(response.get('stdout', ''))\n"
        "sys.stderr.write(response.get('stderr', ''))\n"
        "sys.exit(response.get('returncode', 0))\n"
    )
    for name in ("scontrol", "squeue", "scancel"):
        executable = tmp_path / name
        executable.write_text(script)
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("SLURM_JOB_ID", "18324")
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    return responses, calls


def reply(scheduler: tuple[Path, Path], **values: str | float) -> None:
    path, _ = scheduler
    responses = json.loads(path.read_text())
    responses["scontrol"] = values
    path.write_text(json.dumps(responses))


def read_calls(scheduler: tuple[Path, Path]) -> list[list[str]]:
    return [json.loads(line) for line in scheduler[1].read_text().splitlines()]


def test_step_manager_worker_is_found_when_squeue_has_only_batch_and_extern(scheduler: tuple[Path, Path]) -> None:
    assert list_step_ids("18324", timeout=2) == {
        "batch": "18324.batch",
        "extern": "18324.extern",
        "agg_0_hpc-gpu-1-0": "18324.5",
    }
    assert read_calls(scheduler) == [["scontrol", "--oneliner", "show", "steps", "18324"]]


def test_named_lookup_preserves_the_entire_step_name(scheduler: tuple[Path, Path]) -> None:
    reply(scheduler, stdout=step_record(name="worker with spaces"))
    assert find_step_id("worker with spaces") == "18324.5"
    assert find_step_id("worker") is None


def test_empty_successful_response_has_no_running_steps(scheduler: tuple[Path, Path]) -> None:
    reply(scheduler, stdout="\n")
    assert list_step_ids("18324") == {}


@pytest.mark.parametrize("state", ["COMPLETED", "FAILED", "COMPLETING", "SUSPENDED", "PENDING"])
def test_non_running_steps_cannot_satisfy_worker_lookup(scheduler: tuple[Path, Path], state: str) -> None:
    reply(scheduler, stdout=step_record(state=state))
    assert list_step_ids("18324") == {}
    assert find_step_id("agg_0_hpc-gpu-1-0", "18324") is None


@pytest.mark.parametrize(
    "record",
    [
        step_record("18325.5"),
        step_record("18325.batch", "batch"),
        step_record("18324.5+1"),
        step_record("18324.other"),
        step_record().replace("State=RUNNING ", ""),
        step_record().replace("Name=agg_0_hpc-gpu-1-0 ", ""),
        step_record().replace("StepId=18324.5 ", ""),
        step_record(name=""),
        step_record(state="RUNNING garbage"),
        "unexpected diagnostic\n",
        "prefix " + step_record(),
        step_record().replace("State=RUNNING", "State=COMPLETED State=RUNNING"),
        step_record().replace("StepId=18324.5", "StepId=18325.5 StepId=18324.5"),
        step_record().replace("Name=agg_0_hpc-gpu-1-0", "Name=foreign Name=agg_0_hpc-gpu-1-0"),
    ],
)
def test_foreign_or_malformed_record_invalidates_the_whole_response(scheduler: tuple[Path, Path], record: str) -> None:
    reply(scheduler, stdout=step_record("18324.4", "other-worker") + record)
    assert list_step_ids("18324") is None


@pytest.mark.parametrize(
    "second",
    [step_record("18324.6"), step_record(name="other-worker"), step_record()],
)
def test_duplicate_name_or_step_identifier_is_not_used_for_signalling(
    scheduler: tuple[Path, Path], second: str
) -> None:
    reply(scheduler, stdout=step_record() + second)
    assert list_step_ids("18324") is None


@pytest.mark.parametrize("job_id", ["18324,18325", "18324.5", "0", "-1", "18324 --all", "all"])
def test_invalid_job_id_never_queries_slurm(scheduler: tuple[Path, Path], job_id: str) -> None:
    assert list_step_ids(job_id) is None
    assert not scheduler[1].exists()


def test_unavailable_slurm_is_unknown(scheduler: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(scheduler[0].parent / "missing"))
    assert list_step_ids("18324") is None


def test_slurm_command_failure_is_unknown_even_with_valid_partial_output(scheduler: tuple[Path, Path]) -> None:
    reply(scheduler, stdout=step_record(), stderr="permission denied", returncode=1)
    assert list_step_ids("18324") is None


def test_slurm_timeout_is_unknown(scheduler: tuple[Path, Path]) -> None:
    reply(scheduler, stdout=step_record(), delay=1)
    assert list_step_ids("18324", timeout=0.05) is None


@pytest.mark.parametrize("foreign", [False, True])
def test_registry_cleanup_signals_only_the_discovered_owned_worker_step(
    scheduler: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, foreign: bool
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "99999")
    if foreign:
        reply(scheduler, stdout=step_record("18325.5"))
    popen = MagicMock(spec=Popen)
    popen.pid = 1234
    alive = [True]
    popen.poll.side_effect = lambda: None if alive[0] else 0

    def wait(timeout: float | None = None) -> int:
        alive[0] = False
        return 0

    popen.wait.side_effect = wait
    registry = ProcessRegistry(job_id="18324")
    registry.add_process(ManagedProcess(name="worker", popen=popen, step_name="agg_0_hpc-gpu-1-0"))

    assert registry.cleanup(timeout=2)
    calls = read_calls(scheduler)
    assert calls[0] == ["scontrol", "--oneliner", "show", "steps", "18324"]
    if foreign:
        assert len(calls) == 1
        popen.terminate.assert_called_once()
    else:
        assert calls[1:] == [["scancel", "--signal=TERM", "--full", "18324.5"]]
        popen.terminate.assert_not_called()
    popen.kill.assert_not_called()
