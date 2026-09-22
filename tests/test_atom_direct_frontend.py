# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct ATOM commands, topology validation, and readiness."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from marshmallow import ValidationError

from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Process
from srtctl.frontends import get_frontend


def recipe():
    return {
        "schema": 2,
        "engine": "atom",
        "name": "direct-test",
        "model": {"path": "hf:test/model", "container": "test:tag", "precision": "bf16"},
        "resources": {"gpu_type": "mi300x", "gpus_per_node": 8},
        "frontend": {"type": "atom", "enable_multiple_frontends": False},
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 4, "args": {"kv_cache_dtype": "fp8"}}},
    }


def load(data):
    return SrtConfig.Schema().load(resolve_config_with_defaults(data, None))


def test_direct_worker_owns_public_port_without_router():
    config = load(recipe())
    frontend = get_frontend(config.frontend.type)
    runtime = SimpleNamespace(worker_model_arg="test/model", network_interface=None, frontend_port=9017)
    process = Process("node0", frozenset(range(4)), 7500, 6100, "agg", 0)
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.4"):
        command = config.backend.build_worker_command(process, [process], runtime, frontend_type=config.frontend.type)
    assert command == [
        "env",
        "ATOM_HOST_IP=10.0.0.4",
        "python3",
        "-m",
        "atom.entrypoints.openai_server",
        "--model",
        "test/model",
        "--host",
        "0.0.0.0",
        "--server-port",
        "9017",
        "-tp",
        "4",
        "--kv_cache_dtype",
        "fp8",
    ]
    topology = SimpleNamespace(uses_nginx=False, frontend_nodes=["node0"], public_port=9017)
    assert frontend.start_frontends(topology, runtime, config, config.backend, [process]) == []
    assert frontend.worker_endpoint_port(process, config, runtime) == 9017
    assert frontend.direct_endpoint_nodes([process]) == ["node0"]


@pytest.mark.parametrize(
    "change,error",
    [
        ({"roles": {"agg": {"nodes": 1, "workers": 2, "gpus": 1}}}, "exactly one aggregate worker"),
        ({"roles": {"prefill": {"nodes": 1, "workers": 1}, "decode": {"nodes": 1, "workers": 1}}}, "prefill/decode"),
        ({"frontend": {"type": "atom", "enable_multiple_frontends": True}}, "enable_multiple_frontends: false"),
        ({"engine": "sglang"}, "requires backend.type: atom"),
    ],
)
def test_direct_layout_rejects_jobs_requiring_a_router(change, error):
    with pytest.raises(ValidationError, match=error):
        load({**recipe(), **change})


@pytest.mark.parametrize("models,ready", [([], False), ([{"id": "test/model"}], True)])
def test_direct_readiness_requires_loaded_model(models, ready):
    health = requests.Response()
    health.status_code = 200
    listing = requests.Response()
    listing.status_code = 200
    import json

    listing._content = json.dumps({"data": models}).encode()
    with patch("srtctl.core.health.requests.get", side_effect=[health, listing]):
        result = get_frontend("atom").probe_ready("worker", 9017, 0, 1, load(recipe()))
    assert result.ready is ready
