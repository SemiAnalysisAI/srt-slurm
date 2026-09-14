# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the TileRT adapter using real normalization, launch planning, and conversion."""

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from marshmallow import ValidationError

from srtctl.backends import TileRTProtocol
from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
from srtctl.cli.mixins.worker_stage import WorkerStageMixin
from srtctl.cli.submit import show_config_details
from srtctl.core.config import resolve_container_aliases
from srtctl.core.processes import ProcessRegistry
from srtctl.core.roles import expand_roles
from srtctl.core.schema import SrtConfig, TachometerConfig
from srtctl.core.telemetry import generate_tachometer_config
from srtctl.core.topology import Process
from srtctl.frontends import get_frontend
from srtctl.frontends.static_router import RouterWorker


def recipe():
    return {
        "schema": 2,
        "name": "tile-pd",
        "model": {"path": "hf:org/model", "container": "decode", "precision": "fp8"},
        "resources": {"gpu_type": "b200", "gpus_per_node": 8},
        "engine": {"type": "tilert", "prefill_container": "prefill", "weights_dir": "/weights/model"},
        "roles": {
            "prefill": {"nodes": 1, "workers": 1, "gpus": 8, "args": {"enforce-eager": True}},
            "decode": {"nodes": 1, "workers": 1, "gpus": 8},
        },
        "frontend": {"type": "tilert-router", "enable_multiple_frontends": False},
        "benchmark": {"type": "manual"},
    }


def load(data):
    resolve_container_aliases(data, {"prefill": "prefill.sqsh", "decode": "decode.sqsh"})
    return SrtConfig.Schema().load(expand_roles(data))


def workers(backend):
    endpoints = backend.allocate_endpoints(1, 1, 0, 8, 8, 0, 8, ["p0", "d0"])
    return backend.endpoints_to_processes(endpoints, base_sys_port=7200)


@pytest.mark.parametrize("mtp", [True, False])
def test_roles_render_prefill_transfer_and_decode_commands(mtp):
    data = recipe()
    data["engine"].update(with_mtp=mtp, max_seq_len=8192, served_model_name="served")
    backend = load(data).backend
    prefill, decode = workers(backend)
    runtime = SimpleNamespace(worker_model_arg="org/model", model_path="org/model")
    prefill_cmd = backend.build_worker_command(prefill, [prefill], runtime)
    decode_cmd = backend.build_worker_command(decode, [decode], runtime)
    assert prefill_cmd[:3] == ["vllm", "serve", "org/model"]
    assert prefill_cmd[prefill_cmd.index("--served-model-name") + 1] == "served"
    assert prefill_cmd[prefill_cmd.index("--tensor-parallel-size") + 1] == "8"
    assert "--enforce-eager" in prefill_cmd
    assert json.loads(prefill_cmd[prefill_cmd.index("--kv-transfer-config") + 1]) == {
        "kv_connector": "TileRTConnector",
        "kv_connector_module_path": "tilert.pd_vllm.prefill_connector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "tilert_ctrl_port": 7200,
            "tilert_model": "glm5",
            "tilert_max_seq_len": 8192,
            "tilert_transport": "nixl",
        },
    }
    assert decode_cmd == [
        "python",
        "-m",
        "tilert.pd_vllm.decode_server",
        "--engine",
        "tilert",
        "--model",
        "glm5",
        "--model-weights-dir",
        "/weights/model",
        "--max-seq-len",
        "8192",
        "--kv-cache-dtype",
        "fp8",
        "--transport",
        "nixl",
        "--ctrl-port",
        "7201",
        "--http-port",
        "6100",
    ] + (["--with-mtp"] if mtp else [])
    assert ("--speculative-config" in prefill_cmd) == mtp


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("frontend", "type", "dynamo", "requires frontend.type"),
        ("frontend", "enable_multiple_frontends", True, "multiple_frontends"),
        ("frontend", "args", {"decode": "wrong:1:2"}, "managed"),
        ("engine", "prefill_container", None, "prefill image"),
        ("engine", "weights_dir", "/", "dedicated"),
        ("engine", "max_seq_len", 0, "positive"),
    ],
)
def test_invalid_settings_rejected_before_submission(section, key, value, message):
    data = recipe()
    data[section][key] = value
    with pytest.raises(ValidationError, match=message):
        load(data)


@pytest.mark.parametrize("mutation", ["two_prefills", "cross_node", "decode_gpu_count", "managed_arg"])
def test_invalid_role_topology_and_managed_overrides(mutation):
    data = recipe()
    if mutation == "two_prefills":
        data["roles"]["prefill"].update(nodes=2, workers=2)
    elif mutation == "cross_node":
        data["roles"]["decode"].update(nodes=2, gpus=16)
    elif mutation == "decode_gpu_count":
        data["roles"]["decode"]["gpus"] = 4
    else:
        data["roles"]["prefill"]["args"]["--tensor_parallel_size"] = 2
    with pytest.raises(ValidationError, match="exactly one|fit on one|managed|exactly 8"):
        load(data)


def test_router_routes_multiple_decodes_and_uses_prepared_tokenizer():
    frontend = get_frontend("tilert-router")
    config = load(recipe())
    command = frontend.build_router_command(
        [
            RouterWorker("prefill", "http://10.0.0.1:6100"),
            RouterWorker("decode", "http://10.0.0.2:6100", 7201),
            RouterWorker("decode", "http://10.0.0.3:6110", 7202),
        ],
        "0.0.0.0",
        8000,
    )
    assert command == [
        "python",
        "-m",
        "tilert.pd_vllm.pd_router",
        "--vllm-url",
        "http://10.0.0.1:6100",
        "--decode",
        "10.0.0.2:7201:6100",
        "--decode",
        "10.0.0.3:7202:6110",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    assert frontend.get_managed_frontend_args(config, config.backend, []) == ["--model-path", "/weights/model"]
    assert frontend.parse_health({"status": "ok", "decode_free": 0}, 1, 2).ready
    assert not frontend.parse_health({"status": "starting"}, 1, 2).ready
    with pytest.raises(ValueError, match="exactly one"):
        frontend.build_router_command([RouterWorker("decode", "http://d:6100", 7201)], "0.0.0.0", 8000)


def test_router_waits_for_workers_and_aborts_before_launch():
    config = load(recipe())
    frontend = get_frontend("tilert-router")
    stopped = threading.Event()
    stopped.set()
    with (
        patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=lambda node, _: node),
        pytest.raises(RuntimeError, match="did not become ready"),
    ):
        frontend.start_frontends(
            None, SimpleNamespace(network_interface="eth0"), config, config.backend, workers(config.backend), stopped
        )


class Harness(WorkerStageMixin, BenchmarkStageMixin):
    @property
    def backend_processes(self):
        return workers(self.backend)


def test_runtime_selects_role_images_and_only_real_metrics(capsys):
    config = load(recipe())
    runner = Harness()
    runner.config = config
    runner.runtime = SimpleNamespace(
        container_image="decode.sqsh", network_interface="eth0", job_id="1", run_name="tile", log_dir=Path("/logs")
    )
    assert runner._worker_container("prefill") == "prefill.sqsh"
    assert runner._worker_container("decode") == "decode.sqsh"
    with patch("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", side_effect=lambda node, _: node):
        assert runner._get_aiperf_server_metrics_env(logical_workers_only=True) == {
            "AIPERF_SERVER_METRICS_URLS": "http://p0:6100/metrics"
        }
    with patch("srtctl.core.telemetry.get_hostname_ip", side_effect=lambda node, _: node):
        toml = generate_tachometer_config(
            processes=runner.backend_processes,
            frontend_topology=SimpleNamespace(frontend_nodes=["p0"], frontend_port=8000),
            runtime=runner.runtime,
            tachometer=TachometerConfig(default_exporters=False),
            frontend_type="tilert-router",
            worker_metrics_port=config.backend.get_metrics_port,
            frontend_metrics_enabled=get_frontend("tilert-router").has_metrics,
        )
    assert re.findall(r'url = "([^"\n]+)"', toml) == ["http://p0:6100/metrics"]
    show_config_details(config)
    output = capsys.readouterr().out
    assert "prefill.sqsh" in output and "decode.sqsh" in output


def test_preparation_uses_decode_gpu_step_and_propagates_failure(tmp_path, monkeypatch):
    runner = Harness()
    runner.config = load(recipe())
    runner.runtime = SimpleNamespace(
        container_image="decode.sqsh",
        environment={},
        srun_options={},
        container_mounts={},
        model_path="org/model",
        worker_model_arg="org/model",
        is_hf_model=True,
        log_dir=tmp_path,
    )
    launched = []

    def srun(**kwargs):
        launched.append(kwargs)
        return subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])

    monkeypatch.setattr("srtctl.cli.mixins.worker_stage.start_srun_process", srun)
    registry = ProcessRegistry("1")
    with pytest.raises(RuntimeError, match="exit code 7"):
        runner.prepare_backend(registry)
    step = launched[0]
    assert (step["nodelist"], step["container_image"], step["srun_options"], step["env_to_set"]) == (
        ["d0"],
        "decode.sqsh",
        {"gpus-per-task": "1"},
        {"TILERT_ROLE": "decode"},
    )
    assert registry.get_process("backend_preparation").exit_code == 7


@pytest.mark.parametrize("broken", [False, True])
def test_weight_conversion_executes_and_reuses_only_complete_cache(tmp_path, broken):
    """Stub only TileRT's external converter; execute the generated preparation verbatim."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "chat_template.jinja").write_text("{{ messages }}")
    (source / "tokenizer.json").write_text('{"version": "1.0"}')
    module = tmp_path / "packages" / "tilert" / "models" / "preprocess"
    module.mkdir(parents=True)
    (module / "weight_converter.py").write_text("""
import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--model_type'); p.add_argument('--model_dir'); p.add_argument('--save_dir')
a = p.parse_args()
target = Path(a.save_dir); target.mkdir(parents=True)
with open(os.environ['CALLS'], 'a') as f: f.write(a.model_type + '\\n')
(target / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'shard.safetensors'}}))
if os.environ['BROKEN'] == '0': (target / 'shard.safetensors').write_bytes(b'converted')
""")
    target = tmp_path / "cache"
    backend = TileRTProtocol(weights_dir=str(target))
    runtime = SimpleNamespace(is_hf_model=False, worker_model_arg=str(source))
    preparation = backend.get_preparation(runtime, [Process("d0", frozenset(range(8)), 7201, 6100, "decode", 0)])
    calls = tmp_path / "calls"
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "packages"), CALLS=str(calls), BROKEN=str(int(broken)))
    result = subprocess.run(preparation.command, env=env, capture_output=True, text=True, check=False)
    if broken:
        assert result.returncode != 0
        assert "incomplete shards" in result.stderr
        assert not target.exists()
        assert list(tmp_path.glob(".cache.tmp.*")) == []
    else:
        assert result.returncode == 0, result.stderr
        assert (target / "shard.safetensors").read_bytes() == b"converted"
        assert (target / "chat_template.jinja").read_text() == "{{ messages }}"
        assert (target / "tokenizer.json").read_text() == '{"version": "1.0"}'
        subprocess.run(preparation.command, env=env, check=True, capture_output=True)
        assert calls.read_text() == "glm-5\n"
        (target / "shard.safetensors").unlink()
        subprocess.run(preparation.command, env=env, check=True, capture_output=True)
        assert calls.read_text() == "glm-5\nglm-5\n"


def test_native_frontend_launches_after_health_checks(tmp_path, monkeypatch):
    config = load(recipe())
    frontend = get_frontend("tilert-router")
    runtime = SimpleNamespace(
        network_interface="eth0",
        log_dir=tmp_path,
        container_image="decode.sqsh",
        container_mounts={},
        environment={},
        nodes=SimpleNamespace(het_group_for=lambda node: None),
    )
    events = []

    def request(url, **kwargs):
        events.append(url)
        return SimpleNamespace(status_code=200)

    def launch(**kwargs):
        events.append(kwargs)
        return subprocess.Popen([sys.executable, "-c", "pass"])

    monkeypatch.setattr("srtctl.core.health.requests.get", request)
    monkeypatch.setattr("srtctl.frontends.static_router.get_hostname_ip", lambda node, _: node)
    monkeypatch.setattr("srtctl.frontends.static_router.start_srun_process", launch)
    processes = frontend.start_frontends(
        SimpleNamespace(frontend_nodes=["p0"], frontend_port=8000),
        runtime,
        config,
        config.backend,
        workers(config.backend),
    )
    processes[0].popen.wait(timeout=5)
    assert events[:2] == ["http://p0:6100/health", "http://d0:6100/health"]
    assert events[2]["command"] == [
        "python",
        "-m",
        "tilert.pd_vllm.pd_router",
        "--vllm-url",
        "http://p0:6100",
        "--decode",
        "d0:7201:6100",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--model-path",
        "/weights/model",
    ]
    assert events[2]["container_image"] == "decode.sqsh"
    assert processes[0].step_name == "tilert_router_0"
