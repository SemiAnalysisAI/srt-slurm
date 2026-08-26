# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-signal contracts for native ATOM and AToMesh orchestration."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from srtctl.backends import AtomProtocol, AtomServerConfig
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Process
from srtctl.frontends import AtomeshFrontend, get_frontend
from srtctl.frontends.static_router import RouterWorker


def _config() -> dict:
    return {
        "name": "atom-atomesh",
        "model": {
            "path": "hf:Qwen/Qwen3-0.6B",
            "container": "rocm/atom:latest",
            "precision": "bf16",
        },
        "resources": {
            "gpu_type": "mi300x",
            "gpus_per_node": 8,
            "prefill_nodes": 1,
            "decode_nodes": 1,
            "prefill_workers": 1,
            "decode_workers": 1,
        },
        "backend": {
            "type": "atom",
            "atom_config": {"decode": {"gpu-memory-utilization": 0.9}},
        },
        "frontend": {"type": "atomesh", "enable_multiple_frontends": False},
    }


def test_atom_atomesh_schema_roundtrip() -> None:
    config = SrtConfig.Schema().load(_config())

    assert isinstance(config.backend, AtomProtocol)
    assert isinstance(get_frontend("atomesh"), AtomeshFrontend)
    assert SrtConfig.Schema().load(SrtConfig.Schema().dump(config)) == config


def test_atom_builds_native_aggregate_command() -> None:
    backend = AtomProtocol(atom_config=AtomServerConfig(aggregated={"trust-remote-code": True}))
    process = Process("node0", frozenset(range(8)), 7500, 6100, "agg", 0, nixl_port=5400)
    runtime = SimpleNamespace(worker_model_arg="/model", network_interface="hsn0")

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.20"):
        command = backend.build_worker_command(process, [process], runtime)

    assert command[:5] == ["env", "ATOM_HOST_IP=10.0.0.20", "python3", "-m", "atom.entrypoints.openai_server"]
    assert command[command.index("--server-port") + 1] == "6100"
    assert command[command.index("-tp") + 1] == "8"
    assert "--kv-transfer-config" not in command
    assert command[-1] == "--trust-remote-code"


def test_atom_builds_official_mooncake_pd_contract() -> None:
    backend = AtomProtocol()
    process = Process("node0", frozenset(range(4)), 7500, 6100, "prefill", 0, nixl_port=6301)
    runtime = SimpleNamespace(worker_model_arg="/model", network_interface=None)

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.20"):
        command = backend.build_worker_command(process, [process], runtime)

    payload = json.loads(command[command.index("--kv-transfer-config") + 1])
    assert payload == {
        "kv_role": "kv_producer",
        "kv_connector": "mooncake",
        "handshake_port": 6301,
    }


def test_atomesh_builds_native_static_pd_command() -> None:
    frontend = AtomeshFrontend()
    workers = [
        RouterWorker("prefill", "http://10.0.0.20:6100"),
        RouterWorker("decode", "http://10.0.0.21:6101"),
    ]

    command = frontend.build_router_command(workers, "0.0.0.0", 8000)
    command.extend(frontend.get_managed_frontend_args(SimpleNamespace(frontend=SimpleNamespace(args={})), None, []))

    assert command == [
        "atomesh",
        "launch",
        "--pd-disaggregation",
        "--prefill",
        "http://10.0.0.20:6100",
        "--decode",
        "http://10.0.0.21:6101",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--backend",
        "atom",
    ]


def test_atomesh_suppresses_bootstrap_port_for_atom_workers() -> None:
    frontend = AtomeshFrontend()
    process = Process("node0", frozenset({0}), 7500, 6100, "prefill", 0, nixl_port=6301)

    assert frontend.worker_bootstrap_port(AtomProtocol(), process) is None
