# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline, native-server Docker export for a single aggregate worker."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shlex
from pathlib import Path

from srtctl.backends.sglang import SGLangProtocol
from srtctl.backends.vllm import VLLMProtocol
from srtctl.core.config import (
    expand_engine_config_defaults,
    generate_override_configs,
    get_srtslurm_setting,
    load_cluster_config,
    resolve_config_with_defaults,
)
from srtctl.core.launch_plan import _is_secret_name
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Endpoint
from srtctl.ports import MOONCAKE_HTTP_METADATA_PORT, MOONCAKE_MASTER_PORT, MOONCAKE_METRICS_PORT
from srtctl.services.implicit import effective_services
from srtctl.services.mooncake_master import mooncake_master_command
from srtctl.services.registry import ServiceLaunchContext


def load_docker_config(raw: dict, selector: str | None = None) -> SrtConfig:
    """Resolve exactly one recipe, including v1/v2, aliases and CLI overrides."""
    if "base" in raw:
        variants = generate_override_configs(raw, selector=selector)
        if len(variants) != 1:
            raise ValueError("render-docker needs exactly one variant; use -f recipe.yaml:base or :override_name")
        raw = variants[0][1]
    elif selector:
        raise ValueError("render-docker selectors require an override recipe")
    if "sweep" in raw:
        raise ValueError("render-docker does not support sweeps; select one single-node aggregate recipe")
    resolved = resolve_config_with_defaults(raw, load_cluster_config())
    expand_engine_config_defaults(resolved)
    return SrtConfig.Schema().load(resolved)


def _image(value: str) -> str:
    image = value.removeprefix("docker://").replace("#", "/", 1)
    if (
        not image
        or image.startswith(("/", ".", "~", "-"))
        or image.endswith((".sqsh", ".squashfs", ".sif"))
        or re.search(r"\s|\$", image)
    ):
        raise ValueError(
            f"Docker needs a registry image, not {value!r}; set identity.container.image "
            "or pass --image (and --mooncake-image for a separate master image)"
        )
    return image


def _command_lines(command: list[str]) -> list[str]:
    """Put each option and its values on one copyable shell line."""
    groups: list[list[str]] = []
    for arg in command:
        if not groups or arg.startswith("--"):
            groups.append([])
        groups[-1].append(arg)
    return [shlex.join(group) for group in groups]


def _docker(
    image: str,
    command: list[str],
    env: dict[str, str],
    mounts: list[str],
    *,
    gpu_type: str,
    preamble: str = "",
    detached: bool = False,
    passthrough: set[str] | None = None,
) -> str:
    lines = ["docker run --rm" + (" -d" if detached else "")]
    if gpu_type.lower().startswith(("mi", "amd")):
        lines.append("--device /dev/kfd --device /dev/dri --group-add video")
    else:
        lines.append("--gpus all")
    # Host networking also carries Mooncake's dynamically allocated RDMA/TCP ports.
    lines.extend(["--privileged --ipc=host --network=host", "--ulimit memlock=-1:-1", *mounts])
    for name, value in sorted(env.items()):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid environment variable name: {name!r}")
        inherited = _is_secret_name(name) or name in (passthrough or set())
        lines.append("-e " + shlex.quote(name if inherited else f"{name}={value}"))
    # Images such as vllm/vllm-openai have their own ENTRYPOINT. Override it so
    # the exported native command is not appended to an unrelated entrypoint.
    if preamble:
        lines.append("--entrypoint bash")
        lines.append(shlex.quote(image))
        lines.append("-c " + shlex.quote("set -e\n" + preamble + "\nexec " + shlex.join(command)))
    else:
        lines.append("--entrypoint " + shlex.quote(command[0]))
        lines.append(shlex.quote(image))
        lines.extend(_command_lines(command[1:]))
    return " \\\n  ".join(lines)


def render_docker(
    config: SrtConfig,
    *,
    image: str | None = None,
    mooncake_image: str | None = None,
    host_ip: str = "127.0.0.1",
) -> str:
    """Return Bash-compatible text without running Docker, SLURM or setup scripts.

    This deliberately exports the native server, not the job's router, client,
    profiling or telemetry. Refuse dependencies we cannot faithfully reproduce.
    """
    ipaddress.IPv4Address(host_ip)
    resources = config.resources
    gpu_type = resources.gpu_type or "nvidia"
    backend = config.backend
    if not isinstance(backend, SGLangProtocol | VLLMProtocol):
        raise TypeError("render-docker supports SGLang and vLLM engines")
    if (
        resources.num_prefill
        or resources.num_decode
        or resources.num_agg != 1
        or resources.agg_nodes != 1
        or resources.gpus_per_agg > resources.gpus_per_node
        or config.services_node_count
    ):
        raise ValueError("render-docker requires exactly one aggregate worker on one node, without service pools")
    if getattr(backend, "failover", None) or config.dynamo.sidecar:
        raise ValueError("render-docker does not support failover or Dynamo sidecars")
    if config.dynamo.source or config.host_setup.commands or config.host_setup.teardown:
        raise ValueError("render-docker cannot reproduce dynamo.source or host_setup; bake setup into the Docker image")
    if config.setup_script:
        raise ValueError("render-docker cannot reproduce setup_script; bake its changes into the Docker image")
    if get_srtslurm_setting("default_bash_preamble"):
        raise ValueError("render-docker cannot reproduce default_bash_preamble; bake its changes into the Docker image")
    if config.model.stage_dir:
        raise ValueError("render-docker does not stage models; use the final local model.path without stage_dir")

    worker_image = _image(image or config.identity.container.image or config.model.container)
    model = os.path.expandvars(config.model.path)
    is_hf = model.startswith("hf:")
    model_path = Path(model[3:]) if is_hf else Path(model).expanduser().absolute()
    node = "127.0.0.1"  # command builders resolve a literal locally, never a SLURM hostname
    runtime = RuntimeContext(
        job_id="docker",
        run_name=config.name,
        nodes=Nodes(head=node, bench=node, infra=node, worker=(node,)),
        head_node_ip=host_ip,
        infra_node_ip=host_ip,
        log_dir=Path("/logs"),
        model_path=model_path,
        container_image=Path(worker_image),
        gpus_per_node=resources.gpus_per_node,
        network_interface=None,
        is_hf_model=is_hf,
    )
    endpoint = Endpoint(
        mode="agg",
        index=0,
        nodes=(node,),
        gpu_indices=frozenset(range(resources.gpus_per_agg)),
        gpus_per_node=resources.gpus_per_node,
    )
    process = backend.endpoints_to_processes([endpoint], frontend_type=backend.type)[0]
    command = backend.build_worker_command(process, [process], runtime, frontend_type=backend.type)
    # Global environment wins, matching WorkerStageMixin.start_worker.
    env = {**backend.get_environment_for_mode("agg"), **config.environment}
    env = {k: v.replace("{node}", host_ip).replace("{node_id}", "0") for k, v in env.items()}
    if resources.gpus_per_agg < resources.gpus_per_node:
        env["CUDA_VISIBLE_DEVICES"] = process.cuda_visible_devices
        if gpu_type.lower().startswith(("mi", "amd")):
            env["ROCR_VISIBLE_DEVICES"] = process.cuda_visible_devices

    mounts = ['-v "${SRT_DOCKER_DIR}:/logs"', '-v "${HOME}/.cache/huggingface:/root/.cache/huggingface"']
    if not is_hf:
        mounts.append("-v " + shlex.quote(f"{model_path}:/model:ro"))
    for host, container in (get_srtslurm_setting("default_mounts", {}) or {}).items():
        mounts.append("-v " + shlex.quote(f"{Path(os.path.expandvars(host)).expanduser().absolute()}:{container}"))
    for mount in config.extra_mount or ():
        host, container = mount.split(":", 1)
        mounts.append("-v " + shlex.quote(f"{Path(os.path.expandvars(host)).expanduser().absolute()}:{container}"))
    for host, container in config.container_mounts.items():
        if any("{" in template.template for template in (host, container)):
            raise ValueError("render-docker requires literal container_mounts paths, without runtime placeholders")
        mounts.append(
            "-v "
            + shlex.quote(
                f"{host.get_path(runtime, ensure_exists=False)}:"
                f"{container.get_path(runtime, make_absolute=False, ensure_exists=False)}"
            )
        )

    services = effective_services(config)
    masters = [entry.service for entry in services if entry.service.type == "mooncake-master"]
    for entry in services:
        if entry.service.type != "mooncake-master" and not entry.implicit:
            raise ValueError(f"render-docker cannot reproduce service {entry.service.name!r}")
    mooncake = backend.mooncake_kv_store
    if mooncake is not None and len(masters) != 1:
        raise ValueError("render-docker needs one enabled mooncake-master service")
    if mooncake is not None and isinstance(backend, VLLMProtocol):
        flags = {key.replace("_", "-"): value for key, value in backend.get_config_for_mode("agg").items()}
        connector = flags.get("kv-transfer-config", "")
        if isinstance(connector, str):
            try:
                connector = json.loads(connector)
            except json.JSONDecodeError:
                connector = {}

        def uses_store(value: object) -> bool:
            if isinstance(value, dict):
                return value.get("kv_connector") == "MooncakeStoreConnector" or any(
                    uses_store(child) for child in value.values()
                )
            if isinstance(value, list):
                return any(uses_store(child) for child in value)
            return False

        if not uses_store(connector):
            raise ValueError(
                "render-docker with Mooncake requires roles.agg.args.kv-transfer-config "
                "to include MooncakeStoreConnector (directly or inside MultiConnector)"
            )
    if mooncake_image and mooncake is None:
        raise ValueError("--mooncake-image requires a Mooncake recipe")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Native single-node aggregate server. Run on a Linux Docker GPU host.",
        "# Router, benchmark, SLURM, profiling and telemetry are intentionally omitted.",
        "# Uses host networking: the API listens on port 8000; no -p mapping is needed.",
        'export SRT_DOCKER_DIR="${SRT_DOCKER_DIR:-${PWD}/srtctl-docker}"',
        'mkdir -p "${SRT_DOCKER_DIR}" "${HOME}/.cache/huggingface"',
    ]
    master_env: dict[str, str] = {}
    master_text = ""
    passthrough: set[str] = set()
    if mooncake is not None:
        master = masters[0]
        if master.external or master.command or master.source or master.build_command or master.srun_options:
            raise ValueError(
                "render-docker requires a local managed Mooncake master without custom command/source/srun"
            )
        context = ServiceLaunchContext(runtime, node, host_ip, 0, 0, master.effective_placement, config=config)

        def render_placeholders(value: str) -> str:
            for key, replacement in context.template_vars().items():
                value = value.replace("{" + key + "}", replacement)
            return value

        master_env = {key: render_placeholders(value) for key, value in master.env.items()}
        master_image = _image(mooncake_image or master.container or worker_image)
        env.update(backend.get_mooncake_worker_env(host_ip, env.get("MOONCAKE_LOCAL_HOSTNAME", host_ip)))
        # Do not advertise loopback as the transfer engine's NIC address. The
        # local master can stay on loopback; only the worker needs the NIC IP.
        if env["MOONCAKE_LOCAL_HOSTNAME"] == "127.0.0.1":
            passthrough.add("MOONCAKE_LOCAL_HOSTNAME")
            lines.append(
                'export MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:'
                '?Set the host transfer NIC IPv4 address, or render with --host-ip}"'
            )
        if isinstance(backend, VLLMProtocol):
            local = backend.build_mooncake_process_config(process, host_ip, resources.gpus_per_node)
            filename, payload = local or ("mooncake_store_config.json", backend.build_mooncake_store_config(host_ip))
            env["MOONCAKE_CONFIG_PATH"] = f"/logs/{filename}"
            lines.extend(
                [
                    "",
                    "# Mooncake config. Segment sizes and HCA selection come from the recipe.",
                    f"cat > \"${{SRT_DOCKER_DIR}}/{filename}\" <<'SRT_MOONCAKE_JSON'",
                    json.dumps(payload, indent=2),
                    "SRT_MOONCAKE_JSON",
                ]
            )
        master_text = _docker(
            master_image,
            [render_placeholders(arg) for arg in mooncake_master_command(master.args)],
            master_env,
            mounts,
            gpu_type=gpu_type,
            preamble=render_placeholders(master.preamble or ""),
            detached=True,
        )
        lines.extend(
            [
                "",
                "# Mooncake master. Both containers share the host network.",
                "# For RDMA, render with --host-ip <this host's RDMA NIC IPv4 address>.",
                "# The image must already contain compatible Mooncake binaries and Python dependencies.",
            ]
        )
    secrets = sorted(name for name in set(env) | set(master_env) if _is_secret_name(name))
    for name in secrets:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid environment variable name: {name!r}")
        lines.append(f'export {name}="${{{name}:?Set {name} before running this file}}"')
    if master_text:
        lines.extend(
            [
                'SRT_MOONCAKE_CID="$(' + master_text + ')"',
                "trap 'docker stop \"${SRT_MOONCAKE_CID}\" >/dev/null 2>&1 || true' EXIT",
                "# Wait for all managed master ports, and fail if the master exits.",
                "SRT_MOONCAKE_READY=0",
                "for ((attempt=0; attempt<120; attempt++)); do",
                '  if [[ "$(docker inspect -f \'{{.State.Running}}\' "${SRT_MOONCAKE_CID}")" != true ]]; then',
                '    docker logs "${SRT_MOONCAKE_CID}" >&2; exit 1',
                "  fi",
                (
                    f"  if (echo > /dev/tcp/{host_ip}/{MOONCAKE_MASTER_PORT}) 2>/dev/null && "
                    f"(echo > /dev/tcp/{host_ip}/{MOONCAKE_HTTP_METADATA_PORT}) 2>/dev/null && "
                    f"(echo > /dev/tcp/{host_ip}/{MOONCAKE_METRICS_PORT}) 2>/dev/null; then"
                ),
                "    SRT_MOONCAKE_READY=1; break",
                "  fi",
                "  sleep 1",
                "done",
                '[[ "${SRT_MOONCAKE_READY}" == 1 ]] || { echo "Mooncake master readiness timed out" >&2; exit 1; }',
            ]
        )
    lines.extend(
        [
            "",
            "# Server",
            _docker(worker_image, command, env, mounts, gpu_type=gpu_type, passthrough=passthrough),
            "",
        ]
    )
    return "\n".join(lines)
