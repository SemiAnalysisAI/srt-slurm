# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-signal contracts for heterogeneous TileRT P/D orchestration."""

import json
from types import SimpleNamespace

from srtctl.backends import TileRTProtocol
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Process
from srtctl.frontends import TileRTRouterFrontend, get_frontend
from srtctl.frontends.static_router import RouterWorker


def _config() -> dict:
    return {
        "name": "tilert-pd",
        "model": {
            "path": "hf:zai-org/GLM-5.1-FP8",
            "container": "ghcr.io/tile-ai/tilert:0.1.5",
            "precision": "fp8",
        },
        "resources": {
            "gpu_type": "b200",
            "gpus_per_node": 8,
            "prefill_nodes": 1,
            "decode_nodes": 1,
            "prefill_workers": 1,
            "decode_workers": 1,
        },
        "backend": {
            "type": "tilert",
            "prefill_container": "vllm/vllm-openai:v0.26.0",
            "model_profile": "glm5",
            "weights_dir": "/tilert_weights/glm5.1-fp8",
        },
        "frontend": {"type": "tilert-router", "enable_multiple_frontends": False},
    }


def test_tilert_schema_preserves_heterogeneous_runtime_contract() -> None:
    config = SrtConfig.Schema().load(_config())

    assert isinstance(config.backend, TileRTProtocol)
    assert config.backend.get_container_image_for_mode("prefill", "decode.sqsh") == "vllm/vllm-openai:v0.26.0"
    assert config.backend.get_container_image_for_mode("decode", "decode.sqsh") == "decode.sqsh"
    assert isinstance(get_frontend("tilert-router"), TileRTRouterFrontend)
    assert SrtConfig.Schema().load(SrtConfig.Schema().dump(config)) == config


def test_tilert_builds_managed_prefill_and_decode_commands() -> None:
    backend = TileRTProtocol(weights_dir="/tilert_weights/glm5.1-fp8")
    runtime = SimpleNamespace(worker_model_arg="zai-org/GLM-5.1-FP8", model_path="zai-org/GLM-5.1-FP8")
    prefill = Process("p0", frozenset(range(8)), 7500, 6100, "prefill", 0)
    decode = Process("d0", frozenset(range(8)), 7501, 6100, "decode", 0)

    prefill_command = backend.build_worker_command(prefill, [prefill], runtime)
    decode_command = backend.build_worker_command(decode, [decode], runtime)

    assert prefill_command[:3] == ["vllm", "serve", "zai-org/GLM-5.1-FP8"]
    transfer = json.loads(prefill_command[prefill_command.index("--kv-transfer-config") + 1])
    assert transfer["kv_connector"] == "TileRTConnector"
    assert transfer["kv_connector_extra_config"]["tilert_transport"] == "nixl"
    assert decode_command[:4] == ["python", "-m", "tilert.pd_vllm.decode_server", "--engine"]
    assert decode_command[decode_command.index("--ctrl-port") + 1] == "7501"
    assert decode_command[decode_command.index("--http-port") + 1] == "6100"
    assert "--with-mtp" in decode_command
    assert backend.get_metrics_port(prefill) == 6100
    assert backend.get_metrics_port(decode) is None


def test_tilert_prepares_weights_once_on_decode_runtime() -> None:
    backend = TileRTProtocol(weights_dir="/shared/tilert/glm5.1-fp8")
    runtime = SimpleNamespace(
        is_hf_model=True,
        model_path="zai-org/GLM-5.1-FP8",
        worker_model_arg="zai-org/GLM-5.1-FP8",
    )
    prefill = Process("p0", frozenset(range(8)), 7500, 6100, "prefill", 0)
    decode = Process("d0", frozenset(range(8)), 7501, 6100, "decode", 0)

    preparation = backend.get_preparation(runtime, [prefill, decode])
    script = preparation.command[-1]

    assert preparation.node == "d0"
    assert preparation.mode == "decode"
    assert preparation.gpus_per_task == 1
    assert "fcntl.LOCK_EX" in script
    assert "snapshot_download('zai-org/GLM-5.1-FP8')" in script
    assert "local_files_only=True" not in script
    assert "tilert.models.preprocess.weight_converter" in script
    assert "\"--model_type\",\n            'glm-5'" in script
    assert "os.replace(tmp, target)" in script


def test_tilert_router_uses_exact_prefill_and_decode_endpoints() -> None:
    frontend = TileRTRouterFrontend()
    workers = [
        RouterWorker("prefill", "http://10.0.0.20:6100"),
        RouterWorker("decode", "http://10.0.0.21:6100", bootstrap_port=7501),
    ]

    command = frontend.build_router_command(workers, "0.0.0.0", 8000)
    managed = frontend.get_managed_frontend_args(
        SimpleNamespace(model=SimpleNamespace(path="hf:zai-org/GLM-5.1-FP8"), frontend=SimpleNamespace(args={})),
        None,
        [],
    )

    assert command + managed == [
        "python",
        "-m",
        "tilert.pd_vllm.pd_router",
        "--vllm-url",
        "http://10.0.0.20:6100",
        "--decode",
        "10.0.0.21:7501:6100",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--model-path",
        "zai-org/GLM-5.1-FP8",
    ]
