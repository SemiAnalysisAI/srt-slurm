# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental Infera discovery frontend for ATOM workers."""

from __future__ import annotations

import logging
import shlex
import threading
from typing import TYPE_CHECKING, Any

from srtctl.core.health import WorkerHealthResult
from srtctl.core.slurm import start_srun_process
from srtctl.frontends.static_router import build_setup_script_preamble
from srtctl.ports import ETCD_CLIENT_PORT

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


def check_infera_workers(
    response_json: dict[str, Any],
    expected_prefill: int,
    expected_decode: int,
) -> WorkerHealthResult:
    workers = response_json.get("workers")
    if not isinstance(workers, list):
        return WorkerHealthResult(ready=False, message=f"Key 'workers' not found in response: {response_json}")

    active = [worker for worker in workers if worker.get("status") == "active"]
    prefills = sum(worker.get("disagg_mode") == "prefill" for worker in active)
    decodes = sum(worker.get("disagg_mode") in {"decode", "mixed"} for worker in active)
    ready = prefills >= expected_prefill and decodes >= expected_decode
    return WorkerHealthResult(
        ready=ready,
        message=(
            f"Infera {'is ready' if ready else 'is not ready'}: "
            f"{prefills}/{expected_prefill} prefill and {decodes}/{expected_decode} decode workers."
        ),
        prefill_ready=prefills,
        prefill_expected=expected_prefill,
        decode_ready=decodes,
        decode_expected=expected_decode,
    )


class InferaFrontend:
    """Dynamic-discovery Infera router for its ATOM worker adapter."""

    @property
    def type(self) -> str:
        return "infera"

    @property
    def health_endpoint(self) -> str:
        return "/v1/workers"

    def parse_health(
        self,
        response_json: dict[str, Any],
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        return check_infera_workers(response_json, expected_prefill, expected_decode)

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        del backend, backend_processes, network_interface
        return []

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        result: list[str] = []
        for key, value in (args or {}).items():
            flag = f"--{key.replace('_', '-')}"
            if value is True:
                result.append(flag)
            elif value is False or value is None:
                continue
            elif isinstance(value, list):
                for item in value:
                    result.extend([flag, str(item)])
            else:
                result.extend([flag, str(value)])
        return result

    def start_frontends(
        self,
        topology: Any,
        runtime: RuntimeContext,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
        stop_event: threading.Event | None = None,
    ) -> list[ManagedProcess]:
        del backend_processes, stop_event
        from srtctl.core.processes import ManagedProcess

        if backend.type != "atom":
            raise ValueError(f"frontend.type: infera requires backend.type: atom (got {backend.type!r})")

        model_arg = str(runtime.model_path) if runtime.is_hf_model else "/model"
        processes: list[ManagedProcess] = []
        for index, node in enumerate(topology.frontend_nodes):
            log_file = runtime.log_dir / f"{node}_infera_{index}.out"
            command = [
                "python3",
                "-m",
                "infera.server",
                "--host",
                "0.0.0.0",
                "--port",
                str(topology.frontend_port),
                "--router-tokenizer-path",
                model_arg,
                "--discovery-backend",
                "etcd",
                "--etcd-endpoint",
                f"{runtime.infra_node_ip}:{ETCD_CLIENT_PORT}",
                "--request-transport",
                "http",
                "--kv-event-transport",
                "zmq",
            ]
            command.extend(self.get_frontend_args_list(config.frontend.args))
            environment = dict(runtime.environment)
            environment.update(config.frontend.env or {})
            logger.info("Starting Infera frontend %d on %s: %s", index, node, shlex.join(command))
            process = start_srun_process(
                command=command,
                nodelist=[node],
                output=str(log_file),
                container_image=getattr(config.frontend, "container_image", None) or str(runtime.container_image),
                container_mounts=runtime.container_mounts,
                env_to_set=environment or None,
                bash_preamble=build_setup_script_preamble(getattr(config, "setup_script", None)),
                het_group=runtime.nodes.het_group_for(node),
                srun_options=runtime.srun_options,
            )
            processes.append(ManagedProcess(f"infera_{index}", process, log_file, node, critical=True))
        return processes
