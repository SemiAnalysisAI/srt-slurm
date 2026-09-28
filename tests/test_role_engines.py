# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic dispatch tests; a test frontend makes no claims about KV interoperability."""

import copy

import pytest
from marshmallow import ValidationError

from srtctl.backends import SGLangProtocol, VLLMProtocol
from srtctl.core.roles import expand_roles, roles_from_legacy
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import NodePortAllocator
from srtctl.frontends.base import _FRONTENDS
from srtctl.frontends.sglang import SGLangRouterFrontend
from srtctl.ports import HTTP_PORTS


@pytest.fixture(autouse=True)
def test_frontend(monkeypatch):
    class RoleTestFrontend(SGLangRouterFrontend):
        type = "role-test-router"
        required_backend = None

    monkeypatch.setitem(_FRONTENDS, "role-test-router", RoleTestFrontend)


def recipe():
    return {
        "schema": 2,
        "name": "independent-engines",
        "model": {"path": "/models/test", "container": "default-image", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8, "het_jobs": False},
        "roles": {
            "prefill": {
                "engine": "vllm",
                "nodes": 1,
                "workers": 1,
                "gpus": 2,
                "args": {"tensor-parallel-size": 2},
                "env": {"PREFILL_ONLY": "1"},
                "container": "prefill-image",
            },
            "decode": {"engine": "sglang", "nodes": 1, "workers": 1, "gpus": 2, "args": {"tp-size": 2}},
        },
        "frontend": {"type": "role-test-router", "enable_multiple_frontends": False},
    }


def load(data):
    return SrtConfig.Schema().load(expand_roles(copy.deepcopy(data)))


def test_independent_engine_arguments_environments_and_images():
    config = load(recipe())
    assert isinstance(config.backend_for_role("prefill"), VLLMProtocol)
    assert isinstance(config.backend_for_role("decode"), SGLangProtocol)
    assert config.backend_for_role("prefill").get_config_for_mode("prefill") == {"tensor-parallel-size": 2}
    assert config.backend_for_role("decode").get_config_for_mode("decode") == {"tp-size": 2}
    assert config.backend_for_role("prefill").prefill_environment == {"PREFILL_ONLY": "1"}
    assert config.backend_for_role("decode").prefill_environment == {}
    assert config.worker_container_for_role("prefill") == "prefill-image"
    assert config.worker_container_for_role("decode") == "default-image"


@pytest.mark.parametrize("shared_engine", ["sglang", "vllm"])
def test_shared_and_role_engines_are_rejected_at_preflight(shared_engine):
    from srtctl.core.validation import preflight_config_variants

    data = recipe()
    data["engine"] = shared_engine
    with pytest.raises(ValueError, match="cannot be combined with a top-level engine"):
        preflight_config_variants(data, cluster_config=None)


def test_explicit_role_engines_do_not_inherit_sibling_options():
    data = recipe()
    data["roles"]["prefill"]["engine"] = "sglang"
    data["roles"]["decode"]["engine"] = {"type": "sglang", "gpu_type": "h100"}
    config = load(data)
    assert config.backend_for_role("decode").gpu_type == "h100"
    assert config.backend_for_role("prefill").gpu_type == SGLangProtocol().gpu_type
    assert config.backend_for_role("prefill").get_config_for_mode("prefill") == {"tensor-parallel-size": 2}
    assert config.backend_for_role("decode").get_config_for_mode("decode") == {"tp-size": 2}


@pytest.mark.parametrize("missing_role", ["prefill", "decode"])
def test_no_default_requires_an_engine_on_every_role(missing_role):
    data = recipe()
    data["roles"]["decode"]["engine"] = "sglang"
    data["roles"][missing_role].pop("engine")
    with pytest.raises(ValueError, match=rf"roles\.{missing_role}\.engine must name a type"):
        load(data)


def test_role_mapping_and_containers_round_trip():
    expanded = expand_roles(recipe())
    migrated = roles_from_legacy(expanded)
    assert "role_backends" not in migrated
    assert "role_containers" not in migrated
    assert "engine" not in migrated
    assert expand_roles(migrated) == expanded
    config = load(recipe())
    dumped = SrtConfig.Schema().dump(config)
    reloaded = SrtConfig.Schema().load(dumped)
    assert reloaded.role_backends == config.role_backends
    assert reloaded.role_containers == config.role_containers


def test_shared_placement_does_not_allocate_both_roles_on_node_zero():
    config = load(recipe())
    endpoints = config.allocate_worker_endpoints(["node0", "node1"])
    assert [(ep.mode, ep.nodes) for ep in endpoints] == [("prefill", ("node0",)), ("decode", ("node1",))]


def test_colocation_shares_allocator_and_allocates_distinct_ports():
    data = recipe()
    data["roles"]["decode"]["nodes"] = "colocate"
    config = load(data)
    endpoints = config.allocate_worker_endpoints(["node0"])
    assert not endpoints[0].gpu_indices.intersection(endpoints[1].gpu_indices)
    allocator = NodePortAllocator()
    reserved_port = allocator.next(HTTP_PORTS, "node0")
    processes = config.worker_processes(endpoints, allocator)
    assert len(processes) == 2
    assert {p.node for p in processes} == {"node0"}
    assert reserved_port not in {p.http_port for p in processes}
    assert len({p.http_port for p in processes}) == len(processes)
    assert len({p.sys_port for p in processes}) == len(processes)


def test_homogeneous_colocation_still_separates_worker_gpus():
    data = recipe()
    data["roles"]["prefill"]["engine"] = "sglang"
    data["roles"]["decode"]["nodes"] = "colocate"
    config = load(data)
    assert config.worker_container_for_role("prefill") == "prefill-image"
    endpoints = config.allocate_worker_endpoints(["node0"])
    processes = config.worker_processes(endpoints)
    assert [(p.endpoint_mode, p.node, p.gpu_indices) for p in processes] == [
        ("prefill", "node0", frozenset({0, 1})),
        ("decode", "node0", frozenset({2, 3})),
    ]


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("frontend", {"type": "dynamo"}, "Dynamo"),
        ("dynamo", {"sidecar": True}, "sidecars"),
        ("profiling", {"type": "nsys"}, "profiling"),
        ("observability", {"enabled": True}, "profiling"),
    ],
)
def test_unsupported_job_wide_features_fail_early(field, value, message):
    data = recipe()
    data[field] = value
    with pytest.raises(ValidationError, match=message):
        load(data)


def test_heterogeneous_slurm_job_is_rejected():
    data = recipe()
    data["resources"]["het_jobs"] = True
    with pytest.raises(ValidationError, match="het_jobs"):
        load(data)


def test_required_frontend_engine_checks_overridden_role():
    data = recipe()
    data["frontend"]["type"] = "sglang-router"
    with pytest.raises(ValidationError, match="prefill=vllm"):
        load(data)


def test_role_local_implicit_service_is_not_silently_ignored():
    data = recipe()
    data["roles"]["prefill"]["engine"] = {"type": "vllm", "mooncake_kv_store": {}}
    with pytest.raises(ValidationError, match="Mooncake"):
        load(data)


def test_role_local_discovery_connector_is_not_silently_ignored():
    data = recipe()
    data["roles"]["prefill"]["engine"] = {"type": "vllm", "connector": "moriio"}
    with pytest.raises(ValidationError, match="discovery connectors"):
        load(data)


def test_served_model_name_comes_from_the_selected_decode_engine():
    data = recipe()
    data["roles"]["decode"]["engine"] = "vllm"
    data["roles"]["decode"]["args"] = {"served-model-name": "decode-model"}
    assert load(data).served_model_name == "decode-model"


def test_multinode_trt_packing_is_explicitly_rejected():
    data = recipe()
    data["roles"]["prefill"].update(engine="trtllm", gpus=16, nodes=2)
    with pytest.raises(ValidationError, match="multi-node TRT-LLM"):
        load(data)
