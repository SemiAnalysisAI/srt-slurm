# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline Docker exports preserve native argv and Mooncake wiring."""

import copy
import json
import os
import shlex
import socket
import subprocess
import threading
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import yaml

from srtctl.cli import submit
from srtctl.core.docker_render import load_docker_config, render_docker
from srtctl.core.slurm import get_hostname_ip


def recipe(engine="vllm", mooncake=False):
    data = {
        "schema": 2,
        "name": "docker-test",
        "model": {
            "path": "hf:deepseek-ai/DeepSeek-V4-Flash",
            "container": "vllm/vllm-openai:nightly",
            "precision": "fp8",
        },
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "engine": engine,
        "roles": {
            "agg": {
                "nodes": 1,
                "workers": 1,
                "gpus": 4,
                "args": {"tensor-parallel-size": 4, "trust-remote-code": True},
                "env": {"VLLM_USE_RUST_FRONTEND": "1", "PYTHONHASHSEED": "0"},
            }
        },
        "environment": {"SHARED": "global"},
        "benchmark": {"type": "manual"},
    }
    if mooncake:
        data["services"] = [
            {
                "name": "mooncake-master",
                "type": "mooncake-master",
                "args": ["--rpc_thread_num=4"],
                "options": {
                    "store_config": {
                        "mode": "embedded",
                        "metadata_server": "P2PHANDSHAKE",
                        "global_segment_size": "100GB",
                        "local_buffer_size": "4GB",
                        "protocol": "rdma",
                        "device_name": "",
                        "enable_offload": False,
                    }
                }
                if engine == "vllm"
                else {},
            }
        ]
        if engine == "vllm":
            data["roles"]["agg"]["args"]["kv-transfer-config"] = json.dumps(
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both",
                }
            )
        else:
            data["roles"]["agg"]["args"].update(
                {
                    "enable-hierarchical-cache": True,
                    "hicache-storage-backend": "mooncake",
                }
            )
    return data


@pytest.fixture(autouse=True)
def no_cluster():
    with (
        patch("srtctl.core.docker_render.load_cluster_config", return_value=None),
        patch("srtctl.core.docker_render.get_srtslurm_setting", side_effect=lambda key, default=None: default),
    ):
        yield


def render(data=None, **kwargs):
    return render_docker(load_docker_config(data or recipe()), **kwargs)


def worker_argv(text):
    return shlex.split(text.split("# Server\n")[1].replace("\\\n", ""))


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_native_command_and_no_slurm(engine):
    with (
        patch.dict(os.environ, {"SLURM_JOB_ID": "42"}),
        patch("srtctl.core.slurm.get_node_ip", side_effect=AssertionError("must not call srun")),
    ):
        text = render(recipe(engine))
    argv = worker_argv(text)
    assert "--network=host" in argv
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in argv
    assert argv[argv.index("--entrypoint") + 1] == ("vllm" if engine == "vllm" else "python3")
    assert "serve" in argv if engine == "vllm" else "sglang.launch_server" in argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "4"
    assert argv[argv.index("--port") + 1] == "8000"
    assert "deepseek-ai/DeepSeek-V4-Flash" in argv
    assert "dynamo.vllm" not in argv


def test_literal_ip_never_launches_slurm():
    with patch("srtctl.core.slurm.get_node_ip") as lookup, patch.dict(os.environ, {"SLURM_JOB_ID": "42"}):
        assert get_hostname_ip("192.0.2.1") == "192.0.2.1"
        lookup.assert_not_called()


@pytest.mark.parametrize("mapped", [False, True])
def test_mooncake_config_and_master(mapped):
    data = recipe(mooncake=True)
    if mapped:
        data["services"][0]["options"]["device_names_by_gpu"] = [f"mlx5_{i}" for i in range(8)]
    before = copy.deepcopy(data)
    text = render(data, host_ip="192.0.2.5")
    payload = json.loads(text.split("<<'SRT_MOONCAKE_JSON'\n")[1].split("\nSRT_MOONCAKE_JSON")[0])
    assert payload["master_server_address"] == "192.0.2.5:8700"
    assert payload["global_segment_size"] == "100GB"
    assert payload["enable_offload"] is False
    assert payload["device_name"] == ("mlx5_0,mlx5_1,mlx5_2,mlx5_3" if mapped else "")
    filename = "mooncake_store_config_gpu0-1-2-3.json" if mapped else "mooncake_store_config.json"
    argv = worker_argv(text)
    assert f"MOONCAKE_CONFIG_PATH=/logs/{filename}" in argv
    assert "MOONCAKE_MASTER=192.0.2.5:8700" in argv
    assert "MOONCAKE_LOCAL_HOSTNAME=192.0.2.5" in argv
    assert json.loads(argv[argv.index("--kv-transfer-config") + 1])["kv_connector"] == "MooncakeStoreConnector"
    assert "--entrypoint mooncake_master" in text
    assert "--rpc_thread_num=4" in text
    assert "readiness timed out" in text
    assert data == before
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)


def test_sglang_mooncake_and_required_local_address():
    text = render(recipe("sglang", mooncake=True))
    assert "MOONCAKE_CONFIG_PATH" not in text
    assert "--hicache-storage-backend mooncake" in text
    assert "export MOONCAKE_LOCAL_HOSTNAME=" in text
    assert "\n  -e MOONCAKE_LOCAL_HOSTNAME \\" in text
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)


def test_mooncake_requires_the_store_connector_and_accepts_multi_connector():
    data = recipe(mooncake=True)
    del data["roles"]["agg"]["args"]["kv-transfer-config"]
    with pytest.raises(ValueError, match="MooncakeStoreConnector"):
        render(data)
    connector = {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"connectors": [{"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"}]},
    }
    data["roles"]["agg"]["args"]["kv-transfer-config"] = json.dumps(connector)
    argv = worker_argv(render(data))
    assert json.loads(argv[argv.index("--kv-transfer-config") + 1]) == connector


def test_mooncake_separate_image_environment_and_preamble():
    data = recipe(mooncake=True)
    data["services"][0].update(
        container="/containers/master.sqsh",
        env={"MASTER_HOST": "{node_ip}", "MASTER_TOKEN": "private-token"},
        preamble="echo 'master setup'\nexport CUSTOM=1",
    )
    text = render(data, mooncake_image="mooncake:custom", host_ip="192.0.2.5")
    assert "--entrypoint bash" in text
    assert "mooncake:custom" in text
    assert "MASTER_HOST=192.0.2.5" in text
    assert "private-token" not in text
    assert "export MASTER_TOKEN=" in text
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)


def test_quoting_secrets_and_environment_precedence(tmp_path):
    data = recipe()
    value = """spaces 'quotes' "double" $HOME $(touch /tmp/should-not-exist); {}"""
    data["roles"]["agg"]["env"]["SHARED"] = "role"
    data["environment"].update({"WEIRD": value, "HF_TOKEN": "never-export-this-token"})
    data["roles"]["agg"]["args"]["served-model-name"] = "model with 'quotes'"
    data["model"]["path"] = str(tmp_path / "model with spaces")
    data["extra_mount"] = [f"{tmp_path}/extra files:/extra"]
    text = render(data)
    argv = worker_argv(text)
    assert f"WEIRD={value}" in argv
    assert "SHARED=global" in argv
    assert "never-export-this-token" not in text
    assert "HF_TOKEN" in argv
    assert f"{tmp_path}/model with spaces:/model:ro" in argv
    assert f"{tmp_path}/extra files:/extra" in argv
    assert "/model" in argv
    assert "model with 'quotes'" in argv
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)


def test_image_identity_and_enroot_conversion():
    data = recipe()
    data["model"]["container"] = "/images/vllm.sqsh"
    with pytest.raises(ValueError, match="registry image"):
        render(data)
    data["identity"] = {"container": {"image": "docker://nvcr.io#nvidia/vllm:latest"}}
    assert "nvcr.io/nvidia/vllm:latest" in worker_argv(render(data))
    assert "custom:v1" in worker_argv(render(data, image="custom:v1"))


def test_amd_devices():
    data = recipe()
    data["resources"]["gpu_type"] = "mi355x"
    argv = worker_argv(render(data))
    assert "/dev/kfd" in argv and "/dev/dri" in argv
    assert "--gpus" not in argv
    assert "ROCR_VISIBLE_DEVICES=0,1,2,3" in argv


@pytest.mark.parametrize("nodes,workers,gpus", [(2, 1, 8), (1, 2, 4)])
def test_reject_unsupported_topology(nodes, workers, gpus):
    data = recipe()
    data["roles"]["agg"].update(nodes=nodes, workers=workers, gpus=gpus)
    with pytest.raises(ValueError, match="exactly one aggregate"):
        render(data)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("setup_script", "install.sh", "setup_script"),
        ("host_setup", {"commands": ["echo setup"]}, "host_setup"),
        ("services", [{"name": "store", "command": ["sleep", "infinity"]}], "service 'store'"),
    ],
)
def test_reject_dependencies_instead_of_silently_omitting(field, value, message):
    data = recipe()
    data[field] = value
    with pytest.raises(ValueError, match=message):
        render(data)


def test_variants_sweeps_and_v1():
    data = recipe()
    variants = {"schema": 2, "base": data, "override_a": {"name": "a"}, "override_b": {"name": "b"}}
    assert load_docker_config(variants, "override_a").name == "a"
    with pytest.raises(ValueError, match="exactly one variant"):
        load_docker_config(variants)
    with pytest.raises(ValueError, match="sweeps"):
        load_docker_config({**data, "sweep": {"parameters": {}}})
    v1 = {
        "name": "v1",
        "model": data["model"],
        "resources": {"gpus_per_node": 8, "agg_nodes": 1, "agg_workers": 1, "gpus_per_agg": 4},
        "backend": {"type": "vllm", "vllm_config": {"aggregated": {"tensor-parallel-size": 4}}},
        "benchmark": {"type": "manual"},
    }
    assert "--tensor-parallel-size 4" in render(v1)


def test_cli_writes_text_and_applies_overrides_without_submitting(tmp_path, capsys):
    source = tmp_path / "recipe.yaml"
    source.write_text(yaml.safe_dump(recipe(mooncake=True)))
    destination = tmp_path / "rendered" / "server.txt"
    argv = [
        "srtctl",
        "render-docker",
        "-f",
        str(source),
        "--to",
        str(destination),
        "--set",
        "roles.agg.args.tensor-parallel-size=2",
        "--set",
        "roles.agg.gpus=2",
    ]
    with (
        patch.object(submit.sys, "argv", argv),
        patch.object(submit, "submit_single", side_effect=AssertionError("must not submit")),
        patch.object(submit, "validate_setup", side_effect=AssertionError("must not preflight")),
    ):
        submit.main()
    assert capsys.readouterr().out.strip() == str(destination)
    assert "--tensor-parallel-size 2" in destination.read_text()
    assert "tensor-parallel-size: 4" in source.read_text()


def test_export_executes_with_fake_docker_and_real_mooncake_readiness(tmp_path):
    """Run the entire .txt with fake Docker and real sockets, without a GPU."""
    script = tmp_path / "docker"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['DOCKER_CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if sys.argv[1] == 'run' and '-d' in sys.argv: print('fake-master')\n"
        "elif sys.argv[1] == 'inspect': print('true')\n"
    )
    script.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    out = tmp_path / "output with spaces"
    stop = threading.Event()

    def drain(sock):
        sock.settimeout(0.1)
        while not stop.is_set():
            try:
                client, _ = sock.accept()
                client.close()
            except TimeoutError:
                pass

    with ExitStack() as stack:
        threads = []
        for port in (8700, 8701, 8702):
            sock = stack.enter_context(socket.socket())
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen()
            thread = threading.Thread(target=drain, args=(sock,), daemon=True)
            thread.start()
            threads.append(thread)
        try:
            subprocess.run(
                ["bash"],
                input=render(recipe(mooncake=True)),
                text=True,
                check=True,
                timeout=10,
                env={
                    **os.environ,
                    "PATH": f"{tmp_path}:{os.environ['PATH']}",
                    "DOCKER_CALLS": str(calls),
                    "SRT_DOCKER_DIR": str(out),
                    "MOONCAKE_LOCAL_HOSTNAME": "192.0.2.5",
                },
            )
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=1)
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    master, worker = [call for call in recorded if call[0] == "run"]
    assert "-d" in master and "-d" not in worker
    assert worker[worker.index("--entrypoint") + 1] == "vllm"
    assert f"{out}:/logs" in worker
    assert recorded[-1] == ["stop", "fake-master"]
    payload = json.loads((out / "mooncake_store_config.json").read_text())
    assert payload["master_server_address"] == "127.0.0.1:8700"
