# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TileRT backend and TileRT P/D router contracts."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from marshmallow import ValidationError

from srtctl.backends import TileRTProtocol, TileRTWeightConverter
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Process
from srtctl.frontends import TileRTRouterFrontend

CLUSTER = {"containers": {"tilert-decode": "/images/decode.sqsh", "tilert-prefill": "/images/prefill.sqsh"}}


def _recipe(decode_workers: int = 1) -> dict:
    return {
        "schema": 2,
        "name": "tilert-pd",
        "model": {"path": "/models/GLM-5", "container": "tilert-decode", "precision": "fp8"},
        "resources": {"gpu_type": "mi355x", "gpus_per_node": 8},
        "engine": {
            "type": "tilert",
            "prefill_container": "tilert-prefill",
            "served_model_name": "glm5",
            "model_profile": "glm5_2",
            "weights_dir": "/weights/glm5-tp8",
            "max_seq_len": 1048576,
            "kv_cache_dtype": "bf16",
            "prefill_kv_cache_dtype": "bfloat16",
            "transport": "mooncake",
        },
        "roles": {
            "prefill": {
                "nodes": 1,
                "workers": 1,
                "gpus": 8,
                "env": {"VLLM_ROCM_USE_AITER": "1"},
                "args": {"block-size": 64, "enforce-eager": True},
            },
            "decode": {
                "nodes": decode_workers,
                "workers": decode_workers,
                "gpus": 8,
                "env": {"GLM5_AR_N": "2"},
                "args": {"num-mtp": 3},
            },
        },
        "frontend": {"type": "tilert-router", "enable_multiple_frontends": False, "args": {"parser": "none"}},
        "benchmark": {"type": "manual"},
    }


def _load(data: dict) -> SrtConfig:
    return SrtConfig.Schema().load(resolve_config_with_defaults(data, CLUSTER))


def _orchestrator(config: SrtConfig, tmp_path: Path, workers: tuple[str, ...]) -> SweepOrchestrator:
    runtime = RuntimeContext(
        job_id="7",
        run_name=config.name,
        nodes=Nodes(head=workers[0], bench=workers[0], infra=workers[0], worker=workers),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=tmp_path,
        model_path=Path("/models/GLM-5"),
        container_image=Path(config.model.container),
        container_mounts={tmp_path: Path("/logs")},
        gpus_per_node=8,
        network_interface=None,
    )
    return SweepOrchestrator(config, runtime)


def _flag(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_v2_recipe_launches_each_role_in_its_own_image(tmp_path: Path) -> None:
    """Load a roles recipe and start both workers through the real worker stage."""
    config = _load(_recipe())
    orchestrator = _orchestrator(config, tmp_path, ("p0", "d0"))
    workers = orchestrator.backend_processes

    with (
        patch("srtctl.cli.mixins.worker_stage.get_hostname_ip", return_value="10.0.0.2"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as srun,
    ):
        for worker in workers:
            orchestrator.start_worker(worker, [worker])

    prefill, decode = (call.kwargs for call in srun.call_args_list)
    assert (prefill["nodelist"], decode["nodelist"]) == (["p0"], ["d0"])
    assert prefill["container_image"] == "/images/prefill.sqsh"
    assert decode["container_image"] == "/images/decode.sqsh"
    assert prefill["env_to_set"]["TILERT_ROLE"] == "prefill"
    assert prefill["env_to_set"]["VLLM_ROCM_USE_AITER"] == "1"
    assert decode["env_to_set"]["TILERT_ROLE"] == "decode"
    assert decode["env_to_set"]["GLM5_AR_N"] == "2"

    prefill_cmd = prefill["command"]
    assert prefill_cmd[:3] == ["vllm", "serve", "/model"]
    assert _flag(prefill_cmd, "--served-model-name") == "glm5"
    assert _flag(prefill_cmd, "--tensor-parallel-size") == "8"
    assert _flag(prefill_cmd, "--max-model-len") == "1048576"
    assert _flag(prefill_cmd, "--kv-cache-dtype") == "bfloat16"
    assert _flag(prefill_cmd, "--block-size") == "64"
    assert "--enforce-eager" in prefill_cmd
    assert json.loads(_flag(prefill_cmd, "--kv-transfer-config")) == {
        "kv_connector": "TileRTConnector",
        "kv_connector_module_path": "tilert.pd_vllm.prefill_connector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "tilert_model": "glm5_2",
            "tilert_max_seq_len": 1048576,
            "tilert_transport": "mooncake",
        },
    }
    assert json.loads(_flag(prefill_cmd, "--speculative-config")) == {"method": "mtp", "num_speculative_tokens": 1}

    decode_worker = workers[1]
    assert decode["command"] == [
        "python",
        "-m",
        "tilert.pd_vllm.decode_server",
        "--engine",
        "tilert",
        "--model",
        "glm5_2",
        "--model-weights-dir",
        "/weights/glm5-tp8",
        "--max-seq-len",
        "1048576",
        "--kv-cache-dtype",
        "bf16",
        "--transport",
        "mooncake",
        "--ctrl-port",
        str(decode_worker.nixl_port),
        "--http-port",
        str(decode_worker.http_port),
        "--with-mtp",
        "--num-mtp",
        "3",
    ]


def test_router_advertises_every_decode_ctrl_port_under_one_decode_flag(tmp_path: Path) -> None:
    """pd_router's --decode is nargs='+': repeating the flag would drop all but the last worker."""
    config = _load(_recipe(decode_workers=2))
    orchestrator = _orchestrator(config, tmp_path, ("p0", "d0", "d1"))
    processes = orchestrator.backend_processes
    frontend = TileRTRouterFrontend()
    hosts = {"p0": "10.0.0.10", "d0": "10.0.0.11", "d1": "10.0.0.12"}

    with patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=lambda node, _: hosts[node]):
        workers = frontend.collect_workers(config.backend, processes, None)
        health = frontend.get_backend_health_urls(config.backend, processes, None)
    command = frontend.build_router_command(workers, "0.0.0.0", 8000, config.backend)
    command.extend(frontend.get_managed_frontend_args(config, config.backend, processes))
    command.extend(frontend.get_frontend_args_list(config.frontend.args))

    prefill, decode0, decode1 = processes
    decode_cmd = config.backend.build_worker_command(decode1, [decode1], SimpleNamespace(worker_model_arg="/model"))
    assert _flag(decode_cmd, "--ctrl-port") == str(decode1.nixl_port)
    assert command == [
        "python",
        "-m",
        "tilert.pd_vllm.pd_router",
        "--vllm-url",
        f"http://10.0.0.10:{prefill.http_port}",
        "--decode",
        f"10.0.0.11:{decode0.nixl_port}:{decode0.http_port}",
        f"10.0.0.12:{decode1.nixl_port}:{decode1.http_port}",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--parser",
        "none",
    ]
    assert health == [
        f"http://10.0.0.10:{prefill.http_port}/health",
        f"http://10.0.0.11:{decode0.http_port}/health",
        f"http://10.0.0.12:{decode1.http_port}/health",
    ]


def _mutate(data: dict, path: str, value) -> dict:
    *parents, leaf = path.split(".")
    node = data
    for key in parents:
        node = node[key]
    if value is None:
        node.pop(leaf)
    else:
        node[leaf] = value
    return data


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("frontend.enable_multiple_frontends", True, "enable_multiple_frontends: false"),
        ("frontend.args", {"decode": "h:1:2"}, "srtctl-managed TileRT router"),
        ("engine.prefill_container", None, "prefill_container"),
        ("engine.weights_dir", "weights", "absolute"),
        ("engine.max_seq_len", 0, "positive"),
        ("roles.prefill.workers", 2, "exactly one prefill worker"),
        ("roles.prefill.gpus", 16, "fit on one node"),
        ("roles.prefill.args", {"kv_cache_dtype": "fp8"}, "prefill config cannot override"),
        ("roles.decode.args", {"--ctrl-port": 1}, "decode config cannot override"),
        ("model.path", "hf:zai-org/GLM-5", "local model.path"),
        ("engine.weight_converter.args", {"save_dir": "/elsewhere"}, "weight_converter config cannot override"),
    ],
)
def test_invalid_recipes_fail_at_load(path: str, value, message: str) -> None:
    data = _recipe()
    data["engine"]["weight_converter"] = {"module": "tilert.models.preprocess.weight_converter"}
    with pytest.raises(ValidationError, match=message):
        _load(_mutate(data, path, value))


def test_tilert_engine_requires_the_tilert_router() -> None:
    process = Process("n0", frozenset(range(8)), 7500, 6100, "decode", 0, nixl_port=5400)
    with pytest.raises(ValueError, match="requires frontend.type: tilert-router"):
        TileRTProtocol(weights_dir="/w").build_worker_command(process, [process], SimpleNamespace(), "dynamo")


def test_router_scrapes_only_prefill_metrics() -> None:
    frontend = TileRTRouterFrontend()
    prefill = Process("n0", frozenset(range(8)), 7500, 6100, "prefill", 0)
    decode = Process("n1", frozenset(range(8)), 7501, 6101, "decode", 0, nixl_port=5400)
    assert frontend.worker_metrics_port(prefill, SimpleNamespace()) == 6100
    assert frontend.worker_metrics_port(decode, SimpleNamespace()) is None


def test_router_health_requires_ok_status() -> None:
    frontend = TileRTRouterFrontend()
    assert frontend.parse_health({"status": "ok", "decode_free": 1}, 1, 2).ready
    assert not frontend.parse_health({"status": "starting"}, 1, 2).ready


@pytest.mark.skipif(shutil.which("flock") is None, reason="needs util-linux flock")
def test_decode_converts_once_stages_tokenizer_and_starts_server(tmp_path: Path) -> None:
    """Run the generated decode command with fake converter and server modules."""
    source = tmp_path / "model"
    source.mkdir()
    (source / "model-00001.safetensors").write_text("w")
    (source / "model.safetensors.index.json").write_text("{}")
    (source / "tokenizer.json").write_text("tok")
    (source / "chat_template.jinja").write_text("tmpl")
    weights = tmp_path / "weights"
    log = tmp_path / "calls.log"

    packages = tmp_path / "py"
    (packages / "fakeconv").mkdir(parents=True)
    (packages / "fakeconv" / "__init__.py").write_text("")
    (packages / "fakeconv" / "convert.py").write_text(
        "import os, sys\n"
        "argv = sys.argv[1:]\n"
        f"open({str(log)!r}, 'a').write('convert ' + ' '.join(argv) + '\\n')\n"
        "save = argv[argv.index('--save_dir') + 1]\n"
        "if os.environ.get('FAIL_CONVERT') != '1':\n"
        "    open(os.path.join(save, 'tilert_meta.json'), 'w').write('{}')\n"
    )
    server = packages / "tilert" / "pd_vllm"
    server.mkdir(parents=True)
    (packages / "tilert" / "__init__.py").write_text("")
    (server / "__init__.py").write_text("")
    (server / "decode_server.py").write_text(
        f"import sys\nopen({str(log)!r}, 'a').write('serve ' + ' '.join(sys.argv[1:]) + '\\n')\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)

    backend = TileRTProtocol(
        weights_dir=str(weights),
        weight_converter=TileRTWeightConverter(
            module="fakeconv.convert", args={"num_mtp": 3, "device": "cuda:7"}, ready_file="tilert_meta.json"
        ),
    )
    process = Process("n0", frozenset(range(8)), 7500, 6100, "decode", 0, nixl_port=5400)
    command = backend.build_worker_command(process, [process], SimpleNamespace(worker_model_arg=str(source)))
    assert command[:2] == ["bash", "-c"]
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "PYTHONPATH": str(packages)}

    for _ in range(2):
        subprocess.run(command, env=env, check=True)

    calls = log.read_text().splitlines()
    assert calls[0] == f"convert --model_dir {source} --save_dir {weights} --num_mtp 3 --device cuda:7"
    assert [call.split()[0] for call in calls] == ["convert", "serve", "serve"]
    assert "--ctrl-port 5400 --http-port 6100" in calls[1]
    assert sorted(path.name for path in weights.iterdir() if not path.name.startswith(".")) == [
        "chat_template.jinja",
        "tilert_meta.json",
        "tokenizer.json",
    ]

    empty = tmp_path / "empty-weights"
    failing = TileRTProtocol(
        weights_dir=str(empty),
        weight_converter=TileRTWeightConverter(module="fakeconv.convert", ready_file="tilert_meta.json"),
    )
    command = failing.build_worker_command(process, [process], SimpleNamespace(worker_model_arg=str(source)))
    result = subprocess.run(command, env={**env, "FAIL_CONVERT": "1"}, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "did not produce" in result.stderr
    assert log.read_text().splitlines()[-1].startswith("convert ")
