# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in benchmark log mirroring, using real child processes without Slurm."""

from __future__ import annotations

import io
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from srtctl.benchmarks import get_runner
from srtctl.cli.mixins import benchmark_stage
from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
from srtctl.core.config import load_config
from srtctl.core.log_stream import LogOutputStreamer
from srtctl.core.runtime import Nodes


def _stage(tmp_path, monkeypatch, benchmark_type="custom", stream_output=None):
    benchmark = {"type": benchmark_type}
    if benchmark_type == "custom":
        benchmark["command"] = "echo custom-benchmark"
    else:
        benchmark.update(isl=128, osl=128, concurrencies=[1])
    if stream_output is not None:
        benchmark["stream_output"] = stream_output
    recipe = {
        "schema": 2,
        "name": "stream-test",
        "model": {"path": "hf:fake/model", "container": "fake:latest", "precision": "bf16"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "engine": "sglang",
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 1}},
        "frontend": {"type": "sglang-router", "enable_multiple_frontends": False},
        "benchmark": benchmark,
    }
    config_file = tmp_path / "recipe.yaml"
    config_file.write_text(yaml.safe_dump(recipe))

    class Harness(BenchmarkStageMixin):
        @property
        def backend_processes(self):
            return []

        @property
        def endpoints(self):
            return []

    stage = Harness()
    stage.config = load_config(config_file)
    stage.runtime = SimpleNamespace(
        log_dir=tmp_path,
        container_image="fake:latest",
        container_mounts={},
        srun_options={},
        nodes=Nodes(head="node-a", bench="node-a", infra="node-a", worker=("node-a",)),
        frontend_port=8000,
        network_interface=None,
        environment={},
    )
    monkeypatch.setattr(benchmark_stage, "get_hostname_ip", lambda *args: "127.0.0.1")
    return stage, get_runner(benchmark_type)


@pytest.fixture
def local_srun(monkeypatch):
    """Replace remote execution only; retain real polling, signals and reaping."""
    children = []
    calls = []

    def install(script, *, already_exited=False):
        def launch(**kwargs):
            calls.append(kwargs)
            with Path(kwargs["output"]).open("wb") as output:
                child = subprocess.Popen(
                    [sys.executable, "-u", "-c", script],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE,
                )
            children.append(child)
            if already_exited:
                child.wait(timeout=5)
            else:
                deadline = time.monotonic() + 5
                while Path(kwargs["output"]).stat().st_size == 0:
                    assert child.poll() is None, "child exited before writing its ready marker"
                    assert time.monotonic() < deadline, "child did not become ready"
                    threading.Event().wait(0.01)
            return child

        monkeypatch.setattr(benchmark_stage, "start_srun_process", launch)
        return children, calls

    yield install
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        if child.stdin is not None:
            child.stdin.close()


@pytest.mark.parametrize("benchmark_type", ["custom", "sglang-bench"])
def test_enabled_streams_before_child_exits(tmp_path, monkeypatch, local_srun, benchmark_type):
    stage, runner = _stage(tmp_path, monkeypatch, benchmark_type, stream_output=True)
    children, calls = local_srun(
        "import select, sys\nprint('live', flush=True)\n"
        "if not select.select([sys.stdin], [], [], 5)[0]: sys.exit(99)\n"
        "sys.stdin.readline()\n"
        "print('stderr tail', file=sys.stderr, end='', flush=True)\n"
    )

    class LiveOutput(io.StringIO):
        saw_live_child = False

        def write(self, text):
            if "live" in text:
                assert children[0].poll() is None
                self.saw_live_child = True
                children[0].stdin.write(b"finish\n")
                children[0].stdin.flush()
            return super().write(text)

    output = LiveOutput()
    monkeypatch.setattr(sys, "stdout", output)
    log = tmp_path / "benchmark.out"
    assert stage._run_benchmark_script(runner, log, threading.Event()) == 0
    assert output.saw_live_child
    assert output.getvalue() == log.read_text() == "live\nstderr tail"
    assert calls[0]["output"] == str(log)
    assert stage.benchmark_child_reaped is True


@pytest.mark.parametrize("benchmark_type,stream_output", [("custom", None), ("sglang-bench", False)])
def test_default_and_explicit_off_preserve_file_without_stdout(
    tmp_path, monkeypatch, local_srun, benchmark_type, stream_output
):
    stage, runner = _stage(tmp_path, monkeypatch, benchmark_type, stream_output)
    assert stage.config.benchmark.stream_output is False
    local_srun("print('artifact only')", already_exited=True)
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    log = tmp_path / "benchmark.out"
    assert stage._run_benchmark_script(runner, log, threading.Event()) == 0
    assert log.read_text() == "artifact only\n"
    assert output.getvalue() == ""


def test_fast_failure_drains_without_changing_exit_code(tmp_path, monkeypatch, local_srun):
    stage, runner = _stage(tmp_path, monkeypatch, stream_output=True)
    local_srun("import sys\nprint('failed', end='')\nsys.exit(7)", already_exited=True)
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    assert stage._run_benchmark_script(runner, tmp_path / "benchmark.out", threading.Event()) == 7
    assert output.getvalue() == "failed"
    assert stage.benchmark_child_reaped is True


@pytest.mark.parametrize("signal_unwind", [False, True])
def test_cancellation_drains_termination_output_and_reaps(tmp_path, monkeypatch, local_srun, signal_unwind):
    stage, runner = _stage(tmp_path, monkeypatch, stream_output=True)
    children, _ = local_srun(
        "import signal, sys, time\n"
        "def finish(*args):\n    print('terminated', end='', flush=True)\n    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, finish)\nprint('ready', flush=True)\n"
        "while True: time.sleep(1)\n"
    )
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    stop = threading.Event()
    if signal_unwind:

        def interrupt(_seconds):
            raise SystemExit(128 + signal.SIGTERM)

        monkeypatch.setattr(benchmark_stage, "time", SimpleNamespace(sleep=interrupt))
        with pytest.raises(SystemExit):
            stage._run_benchmark_script(runner, tmp_path / "benchmark.out", stop)
    else:
        stop.set()
        assert stage._run_benchmark_script(runner, tmp_path / "benchmark.out", stop) == 1
    assert output.getvalue() == "ready\nterminated"
    assert children[0].returncode == 0
    assert stage.benchmark_child_reaped is True
    assert stage.benchmark_child_allows_window_mutation is True


def test_missing_log_partial_lines_unicode_and_truncation(tmp_path):
    log = tmp_path / "benchmark.out"
    output = io.StringIO()
    stream = LogOutputStreamer(log, output)
    stream.poll()  # Slurm may not have created the log yet.
    log.write_bytes(b"progress\r\xe2")
    stream.poll()
    assert output.getvalue() == "progress\r"
    with log.open("ab") as writer:
        writer.write(b"\x82\xac done")
    stream.poll()
    stream.poll()  # Already streamed bytes must never be repeated.
    assert output.getvalue() == "progress\r€ done"
    log.write_bytes(b"new")  # A rerun may truncate the same artifact path.
    stream.poll(final=True)
    assert output.getvalue() == "progress\r€ donenew"


def test_final_drain_spans_chunks_and_replaces_incomplete_utf8(tmp_path):
    log = tmp_path / "benchmark.out"
    log.write_bytes(b"a" * 65535 + "€".encode() + b"\xe2")
    output = io.StringIO()
    stream = LogOutputStreamer(log, output)
    stream.poll(final=True)
    stream.poll(final=True)
    assert output.getvalue() == "a" * 65535 + "€�"


@pytest.mark.parametrize("fail_on", ["write", "flush"])
def test_broken_stdout_does_not_fail_benchmark(tmp_path, monkeypatch, local_srun, fail_on):
    stage, runner = _stage(tmp_path, monkeypatch, stream_output=True)
    local_srun("print('still successful')", already_exited=True)

    class BrokenOutput(io.StringIO):
        def write(self, text):
            if fail_on == "write":
                raise BrokenPipeError("consumer disconnected")
            return super().write(text)

        def flush(self):
            if fail_on == "flush":
                raise BrokenPipeError("consumer disconnected")
            return super().flush()

    monkeypatch.setattr(sys, "stdout", BrokenOutput())
    log = tmp_path / "benchmark.out"
    assert stage._run_benchmark_script(runner, log, threading.Event()) == 0
    assert log.read_text() == "still successful\n"
