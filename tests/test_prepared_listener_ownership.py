# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise listener ownership against controlled proc records and real Linux sockets."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.processes import ManagedProcess, ProcessRegistry
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.runtime_scripts import owned_worker


def process(proc, pid, parent, start, inodes=()):
    directory = proc / str(pid)
    directory.mkdir(exist_ok=True)
    (directory / "stat").write_text(f"{pid} (worker (test)) S {parent}" + " 0" * 17 + f" {start}\n")
    (directory / "fd").mkdir(exist_ok=True)
    for index, inode in enumerate(inodes):
        (directory / "fd" / str(index)).symlink_to(f"socket:[{inode}]")


def listeners(proc, inodes):
    (proc / "net").mkdir(exist_ok=True)
    (proc / "net/tcp").write_text(
        "header\n" + "".join(f"0: 00000000:1F40 00000000:0000 0A 0 0 0 0 0 {inode}\n" for inode in inodes)
    )


@pytest.fixture
def proc_tree(tmp_path):
    proc = tmp_path / "proc"
    (proc / "self/ns").mkdir(parents=True)
    for name in ("pid", "net"):
        (proc / f"self/ns/{name}").touch()
    identity = tmp_path / "worker.json"
    identity.write_text(
        json.dumps(
            {
                "pid": 100,
                "start_ticks": 1000,
                "pid_namespace": owned_worker.namespace_identity(proc / "self/ns/pid"),
                "net_namespace": owned_worker.namespace_identity(proc / "self/ns/net"),
            }
        )
    )
    process(proc, 100, 1, 1000)
    listeners(proc, [])
    return proc, identity


def test_descendant_listener_is_owned_and_foreign_listener_is_rejected(proc_tree):
    proc, identity = proc_tree
    process(proc, 101, 100, 1001)
    process(proc, 102, 101, 1002, [123])
    listeners(proc, [123])
    assert owned_worker.listener_owned(identity, 8000, proc=proc)
    process(proc, 200, 1, 2000, [456])
    listeners(proc, [123, 456])
    with pytest.raises(RuntimeError, match="outside the prepared worker"):
        owned_worker.listener_owned(identity, 8000, proc=proc)


def test_descendant_pid_reuse_during_fd_scan_is_rejected(proc_tree, monkeypatch):
    proc, identity = proc_tree
    process(proc, 101, 100, 1001, [123])
    listeners(proc, [456])
    readlink = os.readlink

    def replace_process(fd):
        if str(fd).endswith("/101/fd/0"):
            (proc / "101/stat").write_text("101 (foreign) S 200" + " 0" * 17 + " 2000\n")
            return "socket:[456]"
        return readlink(fd)

    monkeypatch.setattr(os, "readlink", replace_process)
    with pytest.raises(RuntimeError, match="descendant changed during"):
        owned_worker.listener_owned(identity, 8000, proc=proc)


@pytest.mark.parametrize("change", ["pid-reuse", "pid-namespace", "net-namespace", "missing-worker"])
def test_unverifiable_worker_fails_closed(proc_tree, change):
    proc, identity = proc_tree
    if change == "pid-reuse":
        (proc / "100/stat").write_text("100 (new process) S 1" + " 0" * 17 + " 2000\n")
    elif change == "missing-worker":
        (proc / "100/stat").unlink()
    else:
        data = json.loads(identity.read_text())
        data[change.replace("-", "_")] = [0, 0]
        identity.write_text(json.dumps(data))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        owned_worker.listener_owned(identity, 8000, proc=proc)


def test_no_listener_is_pending_and_listener_without_identity_is_foreign(proc_tree):
    proc, identity = proc_tree
    assert not owned_worker.listener_owned(identity, 8000, proc=proc)
    identity.unlink()
    assert not owned_worker.listener_owned(identity, 8000, proc=proc)
    listeners(proc, [456])
    with pytest.raises(RuntimeError, match="before the prepared worker starts"):
        owned_worker.listener_owned(identity, 8000, proc=proc)


def test_free_port_probe_does_not_steal_a_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        with pytest.raises(OSError):
            owned_worker.require_free_port(port)
    owned_worker.require_free_port(port)


def orchestrator(tmp_path, port):
    config = SrtConfig.Schema().load(
        resolve_config_with_defaults(
            {
                "schema": 2,
                "name": "owned-listener",
                "model": {"path": str(tmp_path), "container": "controlled/image", "precision": "fp4"},
                "resources": {"gpu_type": "h100", "gpus_per_node": 8},
                "engine": "vllm",
                "frontend": {"type": "vllm", "enable_multiple_frontends": False},
                "roles": {
                    "agg": {
                        "nodes": 1,
                        "workers": 1,
                        "gpus": 8,
                        "args": {"tensor-parallel-size": 8, "served-model-name": "SAME-MODEL"},
                    }
                },
                "health_check": {"max_attempts": 3, "interval_seconds": 1},
                "observability": {"tachometer": {"enabled": False}},
                "benchmark": {
                    "type": "custom",
                    "argv": [
                        sys.executable,
                        "-I",
                        "-c",
                        (
                            "import os,urllib.request; "
                            "print(urllib.request.urlopen(os.environ['SRT_ENDPOINT']+'/client-request').status)"
                        ),
                    ],
                },
            },
            {"default_account": "test", "default_partition": "test"},
        )
    )
    runtime = RuntimeContext(
        job_id="1",
        run_name="owner-test",
        nodes=Nodes("127.0.0.1", "127.0.0.1", "127.0.0.1", ("127.0.0.1",)),
        head_node_ip="127.0.0.1",
        infra_node_ip="127.0.0.1",
        log_dir=tmp_path,
        model_path=tmp_path,
        container_image=tmp_path / "image",
        gpus_per_node=8,
        network_interface=None,
        frontend_port=port,
        prepared_direct_worker=tmp_path / "worker.json",
    )
    return SweepOrchestrator(config, runtime)


@pytest.mark.skipif(sys.platform != "linux", reason="Real listener ownership requires Linux procfs")
@pytest.mark.parametrize("foreign_after_probe", [False, True])
def test_same_model_foreign_listener_never_receives_client_traffic(tmp_path, monkeypatch, foreign_after_probe):
    requests = []

    class Foreign(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"SAME-MODEL"}]}')

    server = HTTPServer(("127.0.0.1", 0), Foreign, bind_and_activate=False)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    server.server_address = ("127.0.0.1", port)
    if foreign_after_probe:
        owned_worker.require_free_port(port)
    server.server_bind()
    server.server_activate()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    worker = subprocess.Popen(
        [
            sys.executable,
            owned_worker.__file__,
            "--identity",
            str(tmp_path / "worker.json"),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        start_new_session=True,
    )
    registry = ProcessRegistry("1")
    registry.add_process(ManagedProcess("agg", worker))
    launched = []
    monkeypatch.setattr(
        "srtctl.cli.mixins.benchmark_stage.start_srun_process", lambda **kwargs: launched.append(kwargs)
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "worker.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "worker.json").exists()
        with pytest.raises(RuntimeError, match="occupied before|outside the prepared worker"):
            orchestrator(tmp_path, port).run_benchmark(registry, threading.Event())
        assert launched == []
        assert requests == []
    finally:
        os.killpg(worker.pid, signal.SIGTERM)
        worker.wait(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.skipif(sys.platform != "linux", reason="Real listener ownership requires Linux procfs")
@pytest.mark.parametrize("takeover", [False, True, "client-exit"])
def test_worker_descendant_listener_and_midclient_takeover(tmp_path, monkeypatch, takeover):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    server_source = """import json
from http.server import BaseHTTPRequestHandler,HTTPServer
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  self.send_response(200); self.end_headers()
  self.wfile.write(json.dumps({'data':[{'id':'SAME-MODEL'}]}).encode())
HTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
""".replace("PORT", str(port))
    stop_file = tmp_path / "stop-http-child"
    stopped_file = tmp_path / "http-child-stopped"
    foreign_ready = tmp_path / "foreign-ready"
    parent = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-I','-c'," + repr(server_source) + "]); "
        "\nwhile not Path(" + repr(str(stop_file)) + ").exists(): time.sleep(.01)\n"
        "child.terminate(); child.wait(); Path(" + repr(str(stopped_file)) + ").touch(); time.sleep(30)"
    )
    worker = subprocess.Popen(
        [
            sys.executable,
            owned_worker.__file__,
            "--identity",
            str(tmp_path / "worker.json"),
            "--",
            sys.executable,
            "-I",
            "-c",
            parent,
        ],
        start_new_session=True,
    )
    registry = ProcessRegistry("1")
    registry.add_process(ManagedProcess("agg", worker))
    subject = orchestrator(tmp_path, port)
    replacements = []
    replace_stop = threading.Event()

    class Foreign(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"SAME-MODEL"}]}')

    def replace_listener():
        while not replace_stop.is_set() and not stopped_file.exists():
            replace_stop.wait(0.01)
        if replace_stop.is_set():
            return
        replacement = HTTPServer(("127.0.0.1", port), Foreign)
        replacements.append(replacement)
        foreign_ready.touch()
        replacement.serve_forever()

    replace_thread = threading.Thread(target=replace_listener, daemon=True)
    if takeover:
        client = (
            "import os,time,urllib.request; from pathlib import Path; "
            "print(urllib.request.urlopen(os.environ['SRT_ENDPOINT']+'/client-request').status,flush=True); "
            "Path(" + repr(str(stop_file)) + ").touch(); "
            "\nwhile not Path("
            + repr(str(foreign_ready))
            + ").exists(): time.sleep(.01)\n"
            + ("time.sleep(30)" if takeover is True else "pass")
        )
        subject.config = replace(
            subject.config, benchmark=replace(subject.config.benchmark, argv=[sys.executable, "-I", "-c", client])
        )
        replace_thread.start()

    def run_local_client(*, command, env_to_set, output, **_kwargs):
        with open(output, "w") as stream:
            return subprocess.Popen(command, env={**os.environ, **env_to_set}, stdout=stream, stderr=subprocess.STDOUT)

    monkeypatch.setattr("srtctl.cli.mixins.benchmark_stage.start_srun_process", run_local_client)
    monkeypatch.setattr("srtctl.cli.mixins.benchmark_stage.FRONTEND_PUBLIC_PORT", port)
    try:
        assert subject.run_benchmark(registry, threading.Event()) == (1 if takeover else 0)
        assert (tmp_path / "benchmark.out").read_text().strip() == "200"
        assert json.loads((tmp_path / "worker.json").read_text())["pid"] == worker.pid
        assert worker.poll() is None
        assert subject.benchmark_child_reaped is True
        if takeover:
            assert stopped_file.exists() and foreign_ready.exists()
    finally:
        replace_stop.set()
        for replacement in replacements:
            replacement.shutdown()
            replacement.server_close()
        if takeover:
            replace_thread.join(timeout=2)
        os.killpg(worker.pid, signal.SIGTERM)
        worker.wait(timeout=5)
