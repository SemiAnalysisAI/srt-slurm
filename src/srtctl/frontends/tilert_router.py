# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TileRT's native vLLM-prefill/TileRT-decode router frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit

from srtctl.core.health import WorkerHealthResult
from srtctl.frontends.static_router import RouterWorker, StaticRouterFrontend

if TYPE_CHECKING:
    from srtctl.core.topology import Process


class TileRTRouterFrontend(StaticRouterFrontend):
    """Launch ``tilert.pd_vllm.pd_router`` from an explicit worker topology."""

    type: ClassVar[str] = "tilert-router"
    backend_type: ClassVar[str] = "tilert"
    executable: ClassVar[tuple[str, ...]] = ("python", "-m", "tilert.pd_vllm.pd_router")
    pd_flag: ClassVar[str] = ""
    process_name: ClassVar[str] = "tilert_router"

    @property
    def health_endpoint(self) -> str:
        return "/health"

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        ready = response_json.get("status") == "ok"
        return WorkerHealthResult(
            ready=ready,
            message="TileRT P/D router healthy" if ready else "TileRT P/D router not ready",
            prefill_ready=expected_prefill if ready else 0,
            prefill_expected=expected_prefill,
            decode_ready=expected_decode if ready else 0,
            decode_expected=expected_decode,
        )

    def worker_bootstrap_port(self, backend: Any, process: Process) -> int | None:
        del backend
        return process.sys_port if process.endpoint_mode == "decode" else None

    def get_pre_start_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        return [
            f"{worker.url.rstrip('/')}/health"
            for worker in self.collect_workers(backend, backend_processes, network_interface)
        ]

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        return self.get_pre_start_backend_health_urls(backend, backend_processes, network_interface)

    def build_router_command(
        self,
        workers: list[RouterWorker],
        host: str,
        port: int,
        *,
        dynamic_discovery: bool = False,
    ) -> list[str]:
        if dynamic_discovery:
            raise ValueError("TileRT router requires a static worker topology")
        prefills = [worker for worker in workers if worker.mode == "prefill"]
        decodes = [worker for worker in workers if worker.mode == "decode"]
        aggregate = [worker for worker in workers if worker.mode == "agg"]
        if aggregate or len(prefills) != 1 or not decodes:
            raise ValueError("TileRT router requires exactly one prefill worker and at least one decode worker")

        command = [*self.executable, "--vllm-url", prefills[0].url]
        for worker in decodes:
            parsed = urlsplit(worker.url)
            if parsed.hostname is None or parsed.port is None or worker.bootstrap_port is None:
                raise ValueError(f"Incomplete TileRT decode endpoint: {worker}")
            command.extend(["--decode", f"{parsed.hostname}:{worker.bootstrap_port}:{parsed.port}"])
        command.extend(["--host", host, "--port", str(port)])
        return command

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
    ) -> list[str]:
        del backend, backend_processes
        normalized = {str(key).lstrip("-").replace("_", "-") for key in (config.frontend.args or {})}
        reserved = {"vllm-url", "decode", "host", "port", "model-path"}
        overlap = normalized.intersection(reserved)
        if overlap:
            raise ValueError(f"frontend.args cannot override srtctl-managed TileRT argument(s): {sorted(overlap)}")
        model_path = str(config.model.path).removeprefix("hf:")
        return ["--model-path", model_path]
