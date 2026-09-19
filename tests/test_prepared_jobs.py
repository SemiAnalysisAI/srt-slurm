# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior at the recipe/allocator/process boundary, with external stubs."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from srtctl.backends.vllm import _config_to_cli_args
from srtctl.core import prepared as native
from srtctl.core.config import cluster_config_scope, load_config
from srtctl.core.observation import observe_job, wait_receipt
from srtctl.core.processes import ManagedProcess, ProcessRegistry, start_process_monitor
from srtctl.core.slurm import start_srun_process


def executable(path, body):
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)
    return path


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("""schema: 2
name: literal-client
model: {path: 'hf:test/model', container: 'test/image@sha256:abc', precision: bf16}
resources: {gpu_type: h100, gpus_per_node: 8}
engine: vllm
frontend: {type: vllm, enable_multiple_frontends: false}
roles:
  agg:
    nodes: 1
    workers: 1
    gpus: 8
    args: {tensor-parallel-size: 8}
benchmark:
  type: custom
  argv: [python3, '-m', example.client]
  cwd: /ix
  concurrencies: [1]
""")
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "cluster": "test",
                "output_dir": str(tmp_path / "outputs"),
                "default_account": "test",
                "default_partition": "test",
                "srtctl_root": str(Path(__file__).resolve().parents[1]),
            }
        )
    )
    monkeypatch.setattr(native, "runtime_identity", lambda _: {"controlled_runtime": "v1"})
    return recipe, profile, tmp_path / "prepared"


def prepare(inputs, nodes=1):
    return native.prepare_job(*inputs, expected_nodes=nodes, runtime_python=sys.executable)


def test_prepare_freezes_profile_and_rejects_mutated_snapshot(inputs):
    job = prepare(inputs)
    inputs[1].write_text("default_account: changed\n")
    with cluster_config_scope({"default_account": "foreign", "default_mounts": {"/foreign": "/bad"}}):
        config = load_config(job.prepared_dir / "config.yaml", frozen=True)
    assert config.slurm.account == "test"
    assert job.manifest["resources"] == {
        "nodes": 1,
        "gpus_per_node": 8,
        "serving_gpus": 8,
        "workers": 1,
        "cardinality": 1,
    }
    snapshot = job.prepared_dir / "profile.yaml"
    snapshot.chmod(0o644)
    snapshot.write_text("default_account: foreign\n")
    with pytest.raises(ValueError, match="Prepared input changed"):
        native.load_prepared(job.prepared_dir)


@pytest.mark.parametrize(
    "change", ["duplicate", "missing_profile", "sweep", "nested_sweep", "dedicated", "override_nodes"]
)
def test_invalid_inputs_never_reach_scheduler(inputs, tmp_path, monkeypatch, change):
    called = tmp_path / "called"
    executable(tmp_path / "sbatch", f"from pathlib import Path\nPath({str(called)!r}).touch()\nprint('101')\n")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    recipe, profile, _ = inputs
    raw = yaml.safe_load(recipe.read_text())
    if change == "duplicate":
        recipe.write_text(recipe.read_text() + "name: second\n")
    elif change == "missing_profile":
        profile.unlink()
    else:
        if change == "sweep":
            raw["sweep"] = {"concurrency": [1, 2]}
        elif change == "nested_sweep":
            raw["benchmark"]["concurrencies"] = [1, 2]
        elif change == "dedicated":
            raw["benchmark"]["client_dedicated_node"] = True
        else:
            raw["sbatch_directives"] = {"nodes": "2"}
        recipe.write_text(yaml.safe_dump(raw))
    with pytest.raises((ValueError, OSError)):
        prepare(inputs)
    assert not called.exists()
    assert not inputs[2].exists()


def test_prepared_exists_at_scheduler_acceptance_and_environment_is_clean(inputs, tmp_path, monkeypatch):
    job = prepare(inputs)
    audit = tmp_path / "audit.json"
    executable(
        tmp_path / "sbatch",
        f"""import json, os, pathlib, sys
p=pathlib.Path(sys.argv[-1]).parent
assert (p/'config.yaml').is_file() and (p/'profile.yaml').is_file() and (p/'manifest.json').is_file()
pathlib.Path({str(audit)!r}).write_text(json.dumps({{'argv':sys.argv,'env':dict(os.environ)}}))
print('12345')
""",
    )
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SBATCH_NODES", "99")
    monkeypatch.setenv("SLURM_JOB_ID", "foreign")
    monkeypatch.setenv("EVAL_ONLY", "true")
    receipt = native.submit_prepared(job.prepared_dir, "point-a", "test", tmp_path / "journal")
    assert receipt["state"] == "accepted"
    assert receipt["job_id"] == "12345"
    record = json.loads(audit.read_text())
    assert not {"SBATCH_NODES", "SLURM_JOB_ID", "EVAL_ONLY"}.intersection(record["env"])
    assert "--no-requeue" in record["argv"]
    audit.unlink()
    assert native.submit_prepared(job.prepared_dir, "point-a", "test", tmp_path / "journal") == receipt
    assert not audit.exists()


def test_unknown_never_authorizes_resubmission(inputs, tmp_path, monkeypatch):
    job = prepare(inputs)
    record = tmp_path / "allocations"
    executable(
        tmp_path / "sbatch",
        f"from pathlib import Path\np=Path({str(record)!r})\np.write_text(p.read_text()+'allocation\\n' if p.exists() else 'allocation\\n')\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    first = native.submit_prepared(job.prepared_dir, "lost-ack", "test", tmp_path / "journal")
    second = native.submit_prepared(job.prepared_dir, "lost-ack", "test", tmp_path / "journal")
    assert first["state"] == second["state"] == "unknown"
    assert record.read_text() == "allocation\n"


def test_lost_receipt_recovers_independent_stdout_journal(inputs, tmp_path, monkeypatch):
    job = prepare(inputs)
    executable(tmp_path / "sbatch", "print('12345')\n")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    original = native.durable_write

    def fail_receipt(path, text, **kwargs):
        if path.name == "accepted.json":
            raise OSError("injected bookkeeping failure after acceptance")
        return original(path, text, **kwargs)

    monkeypatch.setattr(native, "durable_write", fail_receipt)
    uncertain = native.submit_prepared(job.prepared_dir, "point", "test", tmp_path / "journal")
    assert uncertain["state"] == "unknown"
    monkeypatch.setattr(native, "durable_write", original)
    receipt_path = next((tmp_path / "journal").rglob("receipt.json"))
    assert json.loads(receipt_path.read_text())["state"] == "unknown"
    # Cleanup ownership must survive execution inputs disappearing afterwards.
    (job.prepared_dir / "manifest.json").unlink()
    assert native.reconcile_receipt(receipt_path)["accepted_ids"] == ["12345"]


def test_concurrent_intent_claim_invokes_one_allocator(inputs, tmp_path, monkeypatch):
    job = prepare(inputs)
    count = tmp_path / "allocations"
    executable(
        tmp_path / "sbatch",
        f"import time\nfrom pathlib import Path\nwith Path({str(count)!r}).open('a') as f: f.write('allocation\\n')\ntime.sleep(.1)\nprint('111')\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    results = []

    def submit():
        try:
            results.append(native.submit_prepared(job.prepared_dir, "same", "test", tmp_path / "journal"))
        except ValueError as exc:
            results.append(str(exc))

    threads = [threading.Thread(target=submit) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert count.read_text() == "allocation\n"
    assert any(isinstance(value, dict) and value["state"] == "accepted" for value in results)


def test_literal_client_reaches_recording_child_with_cwd_env_and_endpoint(tmp_path, monkeypatch):
    record = tmp_path / "child.json"
    child = tmp_path / "child.py"
    child.write_text(
        "import json, os, pathlib, sys\npathlib.Path(sys.argv[1]).write_text(json.dumps({'argv':sys.argv[2:], 'cwd':os.getcwd(), 'set':os.environ.get('SET_VALUE'), 'unset':os.environ.get('REMOVE_ME'), 'endpoint':os.environ['SRT_ENDPOINT']}))\n"
    )
    executable(tmp_path / "srun", "import os, sys\ni=sys.argv.index('bash')\nos.execvp('bash',sys.argv[i:])\n")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("REMOVE_ME", "inherited")
    literal = ['{"mode":"$(touch injected)","v":true}', "one two", "`id`", "${HOME}", "a;b", ""]
    with cluster_config_scope({}):
        process = start_srun_process(
            [sys.executable, str(child), str(record), *literal],
            env_to_set={"SET_VALUE": "literal $HOME ; `id`", "SRT_ENDPOINT": "http://10.0.0.2:8000"},
            env_to_unset=["REMOVE_ME"],
            working_directory=str(tmp_path),
        )
        assert process.wait(timeout=10) == 0
    assert json.loads(record.read_text()) == {
        "argv": literal,
        "cwd": str(tmp_path),
        "set": "literal $HOME ; `id`",
        "unset": None,
        "endpoint": "http://10.0.0.2:8000",
    }
    assert not (tmp_path / "injected").exists()


def test_json_engine_flag_remains_valid_json():
    args = _config_to_cli_args({"speculative-config": {"method": "mtp", "enabled": True, "optional": None}})
    assert args[0] == "--speculative-config"
    assert json.loads(args[1]) == {"method": "mtp", "enabled": True, "optional": None}


def test_server_exit_zero_is_failure_before_or_during_client():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=5)
    registry = ProcessRegistry("123")
    registry.add_process(ManagedProcess("required-server", child))
    stopped = threading.Event()
    monitor = start_process_monitor(stopped, registry, poll_interval=0.01)
    monitor.join(timeout=1)
    assert stopped.is_set() and registry.check_failures()
    assert registry.cleanup(timeout=0.1)


def test_group_cleanup_has_one_deadline_and_is_idempotent(tmp_path):
    children = []
    registry = ProcessRegistry("123")
    for i in range(3):
        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(30)",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert child.stdout.readline() == "ready\n"
        children.append(child)
        registry.add_process(ManagedProcess(str(i), child, terminate_timeout=10))
    start = time.monotonic()
    try:
        assert registry.cleanup(timeout=0.4)
        assert registry.cleanup(timeout=0.4)
        assert time.monotonic() - start < 1.5
        assert all(child.poll() is not None for child in children)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize(
    "controller,expected", [("RUNNING", "active"), ("COMPLETING", "active"), ("COMPLETED", "completed")]
)
def test_controller_precedes_stale_accounting(tmp_path, monkeypatch, controller, expected):
    called = tmp_path / "sacct-called"
    executable(tmp_path / "scontrol", f"print('JobId=123 JobState={controller} ExitCode=0:0 Restarts=0')\n")
    executable(
        tmp_path / "sacct", f"from pathlib import Path\nPath({str(called)!r}).touch()\nprint('123|COMPLETED|0:0|0')\n"
    )
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    assert observe_job("123")["state"] == expected
    assert not called.exists()


def test_accounting_requires_controller_absence_and_same_generation(tmp_path, monkeypatch):
    executable(
        tmp_path / "scontrol",
        "import sys\nprint('slurm_load_jobs error: Invalid job id specified',file=sys.stderr)\nsys.exit(1)\n",
    )
    executable(tmp_path / "sacct", "print('123|COMPLETED|0:0|0')\n")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    assert observe_job("123")["state"] == "completed"
    executable(
        tmp_path / "scontrol", "import sys\nprint('Controller connection refused',file=sys.stderr)\nsys.exit(1)\n"
    )
    assert observe_job("123")["state"] == "unknown"
    executable(tmp_path / "scontrol", "print('JobId=123 JobState=COMPLETED ExitCode=0:0 Restarts=1')\n")
    assert observe_job("123")["state"] == "failed"


def test_scheduler_success_requires_runtime_cleanup_evidence(tmp_path, monkeypatch):
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "state": "accepted",
                "job_id": "123",
                "accepted_ids": ["123"],
                "manifest_sha256": "expected",
                "output_dir": str(tmp_path),
            }
        )
    )
    executable(tmp_path / "scontrol", "print('JobId=123 JobState=COMPLETED ExitCode=0:0 Restarts=0')\n")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    assert wait_receipt(receipt_path, timeout=0.5)["state"] == "failed"
    (tmp_path / "completion.json").write_text(
        json.dumps(
            {
                "manifest_sha256": "expected",
                "execution_success": True,
                "cleanup_complete": True,
                "restoration_success": True,
            }
        )
    )
    assert wait_receipt(receipt_path, timeout=0.5)["state"] == "completed"


def test_parent_killed_after_acceptance_is_recoverable_in_fresh_process(inputs, tmp_path, monkeypatch):
    job = prepare(inputs)
    allocations = tmp_path / "allocations"
    executable(
        tmp_path / "sbatch",
        f"""import os, signal, sys
from pathlib import Path
Path({str(allocations)!r}).write_text('12345\\n')
print('12345',flush=True)
os.kill(os.getppid(),signal.SIGKILL)
""",
    )
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PYTHONPATH", str(Path(native.__file__).resolve().parents[2]))
    script = tmp_path / "submitter.py"
    script.write_text(f"""from pathlib import Path
from srtctl.core import prepared as p
p.runtime_identity=lambda _: {{'controlled_runtime':'v1'}}
p.submit_prepared(Path({str(job.prepared_dir)!r}), 'crash', 'test', Path({str(tmp_path / "journal")!r}))
""")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == -signal.SIGKILL
    receipt = next((tmp_path / "journal").rglob("receipt.json"))
    assert json.loads(receipt.read_text())["state"] == "claimed"
    recovered = subprocess.run(
        [sys.executable, "-m", "srtctl.cli.submit", "reconcile", "--receipt", str(receipt), "--json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert json.loads(recovered.stdout)["accepted_ids"] == ["12345"]
    assert allocations.read_text() == "12345\n"


def test_repeated_term_during_cleanup_does_not_reenter_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path(native.__file__).resolve().parents[2]))
    ready, cleanup, result = (tmp_path / name for name in ("ready", "cleaning", "result"))
    script = tmp_path / "controller.py"
    script.write_text(f"""import subprocess,sys,threading,time
from pathlib import Path
from srtctl.core.processes import ProcessRegistry,ManagedProcess,setup_signal_handlers
r=ProcessRegistry('123'); e=threading.Event(); setup_signal_handlers(e,r)
p=subprocess.Popen([sys.executable,'-u','-c',"import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(30)"],stdout=subprocess.PIPE,text=True)
assert p.stdout.readline()=='ready\\n'
r.add_process(ManagedProcess('server',p,terminate_timeout=10))
Path({str(ready)!r}).touch()
try:
    while True: time.sleep(.01)
except InterruptedError:
    pass
finally:
    r.finalizing=True
    Path({str(cleanup)!r}).touch()
    ok=r.cleanup(timeout=1.3)
    Path({str(result)!r}).write_text(str(ok)+':'+str(p.poll()))
""")
    child = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        child.send_signal(signal.SIGTERM)
        while not cleanup.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cleanup.exists()
        child.send_signal(signal.SIGTERM)
        child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == 0, stdout + stderr
        assert result.read_text() == "True:-9"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_cancel_known_checks_each_intent_and_never_cancels_foreign_job(tmp_path, monkeypatch):
    from srtctl.core.observation import wait_known_receipt

    token = "srtctl:" + "a" * 64
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"schema": 1, "state": "unknown", "accepted_ids": ["111", "222", "333"], "scheduler_comment": token})
    )
    audit = tmp_path / "cancelled"
    executable(
        tmp_path / "scontrol",
        f"""import sys
job=sys.argv[-1]
comment={token!r} if job != '333' else 'foreign'
print(f'JobId={{job}} JobState=RUNNING ExitCode=0:0 Restarts=0 Comment={{comment}}')
""",
    )
    executable(
        tmp_path / "scancel",
        f"import sys\nfrom pathlib import Path\nwith Path({str(audit)!r}).open('a') as f: f.write(sys.argv[-1]+'\\n')\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    result = native.cancel_known_receipt(receipt)
    assert result["state"] == "unknown"
    assert audit.read_text().splitlines() == ["111", "222"]
    assert result["jobs"][2]["identity_mismatch"]
    executable(
        tmp_path / "scontrol",
        f"import sys\nprint(f'JobId={{sys.argv[-1]}} JobState=CANCELLED ExitCode=0:15 Restarts=0 Comment={token}')\n",
    )
    closed = wait_known_receipt(receipt, timeout=1, poll=0.01)
    assert closed["state"] == "closed" and closed["terminal"]


def test_required_missing_setup_does_not_launch_engine(tmp_path):
    from types import SimpleNamespace

    from srtctl.cli.mixins.worker_stage import WorkerStageMixin

    stage = WorkerStageMixin()
    stage.config = SimpleNamespace(
        setup_script="missing-phase1-" + tmp_path.name, frontend=SimpleNamespace(type="vllm")
    )
    engine = tmp_path / "engine-started"
    preamble = stage._build_worker_preamble()
    result = subprocess.run(
        ["bash", "-c", preamble + " && " + f"touch {engine}"], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert not engine.exists()


def test_intent_path_is_pure_and_matches_submitted_journal(inputs, tmp_path, monkeypatch):
    from srtctl.cli.prepared import main

    journal = tmp_path / "journal"
    assert main(["intent-path", "--intent", "point", "--cluster", "test", "--journal-dir", str(journal), "--json"]) == 0
    assert not journal.exists()
    expected = native.intent_receipt_path("point", "test", journal)
    job = prepare(inputs)
    executable(tmp_path / "sbatch", "print('12345')\n")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    result = native.submit_prepared(job.prepared_dir, "point", "test", journal)
    assert result["receipt_path"] == str(expected)
    assert expected.exists()


def test_hf_snapshot_relative_blob_is_readable_through_native_model_argument(inputs, tmp_path, monkeypatch):
    from srtctl.core.runtime import RuntimeContext

    cache = tmp_path / "hf-cache"
    repo = cache / "models--org--model"
    snapshot = repo / "snapshots" / "immutable-sha"
    snapshot.mkdir(parents=True)
    (repo / "blobs").mkdir()
    (repo / "blobs" / "shard-hash").write_bytes(b"real-shard-fixture")
    (snapshot / "model.safetensors").symlink_to("../../blobs/shard-hash")
    raw = yaml.safe_load(inputs[0].read_text())
    raw["model"]["path"] = str(snapshot)
    inputs[0].write_text(yaml.safe_dump(raw))
    profile = yaml.safe_load(inputs[1].read_text())
    profile["default_mounts"] = {str(cache): str(cache)}
    monkeypatch.setenv("SRTCTL_OUTPUT_DIR", str(tmp_path / "runtime-output"))
    monkeypatch.setattr("srtctl.core.runtime.get_slurm_nodelist", lambda: ["node1"])
    monkeypatch.setattr("srtctl.core.runtime.get_slurm_het_nodelists", lambda: None)
    monkeypatch.setattr("srtctl.core.runtime.get_hostname_ip", lambda *args: "127.0.0.1")
    monkeypatch.setattr("srtctl.core.slurm.get_hostname_ip", lambda *args: "127.0.0.1")
    with cluster_config_scope(profile):
        config = load_config(inputs[0])
        runtime = RuntimeContext.from_config(config, "123")
    endpoints = config.backend.allocate_endpoints(
        num_prefill=0,
        num_decode=0,
        num_agg=1,
        gpus_per_prefill=0,
        gpus_per_decode=0,
        gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=["node1"],
    )
    processes = config.backend.endpoints_to_processes(endpoints, frontend_type="vllm")
    command = config.backend.build_worker_command(processes[0], processes, runtime, frontend_type="vllm")
    model_arg = command[command.index("serve") + 1]
    assert model_arg == str(snapshot.resolve())
    # Resolve the actual argument through the declared container mount, then
    # follow the unchanged relative shard symlink in that mounted cache tree.
    mounts = [
        (host, container)
        for host, container in runtime.container_mounts.items()
        if Path(model_arg).is_relative_to(container)
    ]
    host, container = max(mounts, key=lambda mount: len(mount[1].parts))
    host_model = host / Path(model_arg).relative_to(container)
    assert (host_model / "model.safetensors").is_symlink()
    assert (host_model / "model.safetensors").read_bytes() == b"real-shard-fixture"
