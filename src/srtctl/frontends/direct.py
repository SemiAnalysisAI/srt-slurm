# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared lifecycle for a single direct OpenAI server."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from srtctl.core.health import WorkerHealthResult, probe_direct_server
from srtctl.frontends.base import agg_leader_nodes, logical_health_expectations

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process
    from srtctl.services.implicit import EffectiveService

logger = logging.getLogger(__name__)


class DirectServerFrontend:
    """One aggregate worker owns the public OpenAI endpoint without a router."""

    required_backend: ClassVar[str]
    server_name: ClassVar[str]
    router_hint: ClassVar[str]
    worker_launch: ClassVar[Literal["dynamo", "direct"]] = "direct"
    expands_node_local_dp: ClassVar[bool] = False

    @property
    def type(self) -> str:
        return self.required_backend

    def worker_api_port(self, mode: str) -> Literal["public", "allocated"]:
        """The one server is the endpoint, so it binds the public port."""
        return "public"

    metrics_path: ClassVar[str] = "/metrics"

    def worker_metrics_port(self, process: Process, runtime: RuntimeContext) -> int | None:
        """The aggregate leader binds the public port; its followers serve nothing."""
        if process.endpoint_mode == "agg" and process.is_leader:
            return runtime.frontend_port
        return None

    def worker_endpoint_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        return runtime.frontend_port if process.is_leader else None

    def profiling_control_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        """The leader's server on the public port carries the control routes; followers have none."""
        return runtime.frontend_port if process.is_leader else None

    def profiling_control_is_leader_only(self, config: Any) -> bool:
        return False

    def direct_endpoint_nodes(self, processes: list[Process]) -> list[str]:
        return agg_leader_nodes(processes)

    def worker_ready_port(self, process: Process) -> int:
        return process.sys_port

    def probe_ready(
        self, host: str, port: int, expected_prefill: int, expected_decode: int, config: Any
    ) -> WorkerHealthResult:
        """The worker's own /health, then /v1/models must list the model."""
        return probe_direct_server(host, port)

    def health_expectations(self, config: Any, processes: list[Process] | None) -> tuple[int, int, str]:
        return logical_health_expectations(config)

    def validate(self, config: Any) -> None:
        """Reject layouts that need a router before allocating a job."""
        prefix = f"frontend.type: {self.type}"
        if config.frontend.enable_multiple_frontends:
            raise ValueError(
                f"{prefix} binds {self.server_name} directly; set frontend.enable_multiple_frontends: false"
            )
        if config.resources.is_disaggregated:
            raise ValueError(
                f"{prefix} supports one aggregate worker only, not a prefill/decode layout. {self.router_hint}"
            )
        if config.resources.num_agg != 1:
            raise ValueError(
                f"{prefix} supports exactly one aggregate worker, got {config.resources.num_agg}. {self.router_hint}"
            )
        if config.dynamo.sidecar:
            raise ValueError(f"{prefix} does not support dynamo.sidecar; use frontend.type: dynamo")

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        return []

    def implied_services(self, config: Any) -> list[EffectiveService]:
        return []

    def frontend_metrics_port(self, frontend_args: dict[str, Any] | None) -> int | None:
        return None

    def start_frontends(
        self,
        topology: Any,
        runtime: RuntimeContext,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
        stop_event: threading.Event | None = None,
    ) -> list[ManagedProcess]:
        if config.backend.type != self.required_backend:
            raise ValueError(
                f"frontend.type: {self.type} requires engine {self.required_backend} (got {config.backend.type!r})"
            )
        self.validate(config)
        if topology.uses_nginx or len(topology.frontend_nodes) != 1:
            raise ValueError(
                f"frontend.type: {self.type} binds {self.server_name} directly to the public port; "
                "set frontend.enable_multiple_frontends: false"
            )
        logger.info(
            "frontend.type=%s: no separate frontend process; %s owns port %d",
            self.type,
            self.server_name,
            topology.public_port,
        )
        return []
