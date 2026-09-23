# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TileRT decode with vLLM prefill for native P/D disaggregation."""

from __future__ import annotations

import builtins
import json
import shlex
from collections.abc import Sequence
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from marshmallow import Schema
from marshmallow_dataclass import dataclass

from srtctl.ports import DYN_SYSTEM_PORT_BASE

if TYPE_CHECKING:
    from srtctl.backends.base import SrunConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process

WorkerMode = Literal["prefill", "decode", "agg"]

PREFILL_MANAGED_ARGS = frozenset(
    {
        "host",
        "port",
        "served-model-name",
        "tensor-parallel-size",
        "max-model-len",
        "return-tokens-as-token-ids",
        "kv-cache-dtype",
        "kv-transfer-config",
        "speculative-config",
    }
)

DECODE_MANAGED_ARGS = frozenset(
    {
        "engine",
        "model",
        "model-weights-dir",
        "max-seq-len",
        "kv-cache-dtype",
        "transport",
        "ctrl-port",
        "http-port",
        "with-mtp",
    }
)

CONVERTER_MANAGED_ARGS = frozenset({"model-dir", "save-dir"})


@dataclass(frozen=True)
class TileRTServerConfig:
    """Role-specific CLI arguments not owned by srtctl."""

    prefill: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class TileRTWeightConverter:
    """TileRT weight conversion run by each decode worker before its server starts.

    Attributes:
        module: Python module of the converter, run with ``python -m``
            (for example ``tilert.models.preprocess.weight_converter``).
        args: Converter arguments; keys keep their spelling (``model_type``,
            ``num_mtp``). srtctl supplies ``--model_dir`` and ``--save_dir``.
        ready_file: Path relative to ``weights_dir`` that exists only in a
            complete conversion. Its presence skips conversion.
    """

    module: str
    args: dict[str, Any] = field(default_factory=dict)
    ready_file: str = "model.safetensors.index.json"

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class TileRTProtocol:
    """Run vLLM prefill workers and TileRT decode workers behind the TileRT P/D router.

    ``model.container`` is the decode image and ``prefill_container`` the vLLM
    prefill image. Ports, model paths, prefill TP, KV transfer, and MTP flags are
    derived by srtctl; ``roles.<role>.args`` carries every other engine flag.
    """

    type: Literal["tilert"] = "tilert"
    # OpenAI model name; defaults to the final component of model.path.
    served_model_name: str | None = None
    # Image or cluster container alias for prefill workers.
    prefill_container: str | None = None
    prefill_environment: dict[str, str] = field(default_factory=dict)
    decode_environment: dict[str, str] = field(default_factory=dict)
    aggregated_environment: dict[str, str] = field(default_factory=dict)
    tilert_config: TileRTServerConfig | None = None
    # TileRT P/D model profile (decode_server --model, connector tilert_model).
    model_profile: str = "glm5"
    # Converted TileRT weights, at this container path in every decode worker.
    weights_dir: str | None = None
    # Maximum sequence length shared by prefill and decode.
    max_seq_len: int = 202752
    # TileRT decode KV cache dtype.
    kv_cache_dtype: str = "fp8"
    # vLLM prefill KV cache dtype; must match the layout decode expects.
    prefill_kv_cache_dtype: str = "fp8_ds_mla"
    # KV transfer data plane shared by prefill and decode.
    transport: Literal["nixl", "mooncake"] = "nixl"
    # Enable MTP on both sides of the P/D transfer.
    with_mtp: bool = True
    # Speculative tokens advertised by vLLM prefill when with_mtp is set.
    speculative_tokens: int = 1
    # Optional conversion into weights_dir when it has no complete conversion.
    weight_converter: TileRTWeightConverter | None = None

    Schema: ClassVar[builtins.type[Schema]] = Schema

    def validate_recipe(self, config: Any) -> None:
        """Reject unsupported topology and managed-argument overrides before submission."""
        resources = config.resources
        if resources.num_agg or resources.num_prefill != 1 or resources.num_decode < 1:
            raise ValueError("TileRT requires exactly one prefill worker and at least one decode worker")
        if max(resources.gpus_per_prefill, resources.gpus_per_decode) > resources.gpus_per_node:
            raise ValueError("Each TileRT prefill and decode worker must fit on one node")
        if not self.prefill_container:
            raise ValueError("engine.prefill_container must select the vLLM prefill image")
        if not self.weights_dir or not Path(self.weights_dir).is_absolute() or self.weights_dir == "/":
            raise ValueError("engine.weights_dir must be an absolute, dedicated converted-weight directory")
        if self.max_seq_len < 1 or self.speculative_tokens < 1:
            raise ValueError("TileRT max_seq_len and speculative_tokens must be positive")
        if config.profiling.is_torch:
            raise ValueError("TileRT does not support torch profiling; use nsys")
        if self.weight_converter is not None:
            if str(config.model.path).startswith("hf:"):
                raise ValueError("engine.weight_converter needs a local model.path directory to convert")
            _reject_managed_args(self.weight_converter.args, CONVERTER_MANAGED_ARGS, "weight_converter")
        _reject_managed_args(self.get_config_for_mode("prefill"), PREFILL_MANAGED_ARGS, "prefill")
        _reject_managed_args(self.get_config_for_mode("decode"), DECODE_MANAGED_ARGS, "decode")

    def get_srun_config(self) -> SrunConfig:
        from srtctl.backends.base import SrunConfig

        return SrunConfig(mpi=None, oversubscribe=False, launch_per_endpoint=False)

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        if self.tilert_config is None:
            return {}
        values = {"prefill": self.tilert_config.prefill, "decode": self.tilert_config.decode}
        return dict(values.get(mode) or {})

    def get_environment_for_mode(self, mode: WorkerMode) -> dict[str, str]:
        values = {
            "prefill": self.prefill_environment,
            "decode": self.decode_environment,
            "agg": self.aggregated_environment,
        }
        return dict(values.get(mode) or {})

    def get_process_environment(self, process: Process) -> dict[str, str]:
        return {"TILERT_ROLE": process.endpoint_mode}

    def get_served_model_name(self, default: str) -> str:
        return self.served_model_name or default

    @property
    def mooncake_kv_store(self) -> None:
        return None

    @property
    def failover(self) -> None:
        return None

    def get_mooncake_worker_env(self, infra_node_ip: str, local_hostname: str) -> dict[str, str]:
        return {}

    def get_failover_environment(self, process: Process, job_id: str) -> dict[str, str]:
        return {}

    def should_set_visible_devices(self) -> bool:
        return True

    def allocate_endpoints(
        self,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
        available_nodes: Sequence[str],
        spread_workers: bool = False,
    ) -> list[Endpoint]:
        if num_agg:
            raise ValueError("TileRT supports prefill/decode topology only")
        from srtctl.core.topology import allocate_endpoints

        return allocate_endpoints(
            num_prefill=num_prefill,
            num_decode=num_decode,
            num_agg=num_agg,
            gpus_per_prefill=gpus_per_prefill,
            gpus_per_decode=gpus_per_decode,
            gpus_per_agg=gpus_per_agg,
            gpus_per_node=gpus_per_node,
            available_nodes=available_nodes,
            spread_workers=spread_workers,
        )

    def endpoints_to_processes(
        self,
        endpoints: list[Endpoint],
        base_sys_port: int = DYN_SYSTEM_PORT_BASE,
        port_allocator: NodePortAllocator | None = None,
        frontend_type: str = "tilert-router",
        dynamo_sidecar: bool = False,
    ) -> list[Process]:
        if dynamo_sidecar:
            raise ValueError("TileRT does not support Dynamo sidecars")
        _require_router(frontend_type)
        from srtctl.core.topology import endpoints_to_processes

        return endpoints_to_processes(endpoints, base_sys_port=base_sys_port, port_allocator=port_allocator)

    def decode_ctrl_port(self, process: Process) -> int:
        """The decode server's KV control listener, which prefill connects to per request."""
        if process.nixl_port is None:
            raise ValueError("TileRT decode worker is missing its allocated control port")
        return process.nixl_port

    def build_worker_command(
        self,
        process: Process,
        endpoint_processes: list[Process],
        runtime: RuntimeContext,
        frontend_type: str = "tilert-router",
        nsys_prefix: list[str] | None = None,
        dump_config_path: Path | None = None,
        profiling: ProfilingConfig | None = None,
    ) -> list[str]:
        _require_router(frontend_type)
        if len({item.node for item in endpoint_processes}) != 1:
            raise ValueError("TileRT requires each prefill and decode worker to fit on one node")
        if process.endpoint_mode == "decode":
            return self._decode_command(process, runtime, nsys_prefix)
        if process.endpoint_mode == "prefill":
            return self._prefill_command(process, runtime, nsys_prefix)
        raise ValueError("TileRT supports prefill/decode workers only")

    def _decode_command(self, process: Process, runtime: RuntimeContext, nsys_prefix: list[str] | None) -> list[str]:
        if not self.weights_dir:
            raise ValueError("engine.weights_dir is required for TileRT decode workers")
        config = self.get_config_for_mode("decode")
        _reject_managed_args(config, DECODE_MANAGED_ARGS, "decode")
        server = [
            *(nsys_prefix or []),
            "python",
            "-m",
            "tilert.pd_vllm.decode_server",
            "--engine",
            "tilert",
            "--model",
            self.model_profile,
            "--model-weights-dir",
            self.weights_dir,
            "--max-seq-len",
            str(self.max_seq_len),
            "--kv-cache-dtype",
            self.kv_cache_dtype,
            "--transport",
            self.transport,
            "--ctrl-port",
            str(self.decode_ctrl_port(process)),
            "--http-port",
            str(process.http_port),
        ]
        if self.with_mtp:
            server.append("--with-mtp")
        server.extend(_config_to_cli_args(config))
        if self.weight_converter is None:
            return server
        script = f"{_conversion_script(self.weight_converter, runtime.worker_model_arg, self.weights_dir)}exec {shlex.join(server)}\n"
        return ["bash", "-c", script]

    def _prefill_command(self, process: Process, runtime: RuntimeContext, nsys_prefix: list[str] | None) -> list[str]:
        config = self.get_config_for_mode("prefill")
        _reject_managed_args(config, PREFILL_MANAGED_ARGS, "prefill")
        # The router supplies tilert_host and tilert_ctrl_port per request in kv_transfer_params.
        transfer = {
            "kv_connector": "TileRTConnector",
            "kv_connector_module_path": "tilert.pd_vllm.prefill_connector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "tilert_model": self.model_profile,
                "tilert_max_seq_len": self.max_seq_len,
                "tilert_transport": self.transport,
            },
        }
        command = [*(nsys_prefix or []), "vllm", "serve", runtime.worker_model_arg]
        command.extend(
            [
                "--served-model-name",
                self.get_served_model_name(Path(str(runtime.model_path)).name),
                "--host",
                "0.0.0.0",
                "--port",
                str(process.http_port),
                "--tensor-parallel-size",
                str(len(process.gpu_indices)),
                "--max-model-len",
                str(self.max_seq_len),
                "--return-tokens-as-token-ids",
                "--kv-cache-dtype",
                self.prefill_kv_cache_dtype,
                "--kv-transfer-config",
                json.dumps(transfer, separators=(",", ":")),
            ]
        )
        if self.with_mtp:
            speculative = {"method": "mtp", "num_speculative_tokens": self.speculative_tokens}
            command.extend(["--speculative-config", json.dumps(speculative, separators=(",", ":"))])
        command.extend(_config_to_cli_args(config))
        return command


def _require_router(frontend_type: str) -> None:
    if frontend_type != "tilert-router":
        raise ValueError(f"engine.type: tilert requires frontend.type: tilert-router (got {frontend_type!r})")


def _config_to_cli_args(config: dict[str, Any], *, normalize: bool = True) -> list[str]:
    args: list[str] = []
    for key, value in config.items():
        name = key.lstrip("-")
        flag = f"--{name.replace('_', '-') if normalize else name}"
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            for item in value:
                args.extend([flag, str(item)])
        else:
            args.extend([flag, str(value)])
    return args


def _reject_managed_args(config: dict[str, Any], reserved: frozenset[str], role: str) -> None:
    overlap = {key.lstrip("-").replace("_", "-") for key in config}.intersection(reserved)
    if overlap:
        raise ValueError(f"TileRT {role} config cannot override srtctl-managed argument(s): {sorted(overlap)}")


def _conversion_script(converter: TileRTWeightConverter, source: str, weights_dir: str) -> str:
    """Convert under a shared lock unless complete, then stage tokenizer files next to the weights."""
    target = shlex.quote(weights_dir)
    ready = shlex.quote(f"{weights_dir.rstrip('/')}/{converter.ready_file}")
    convert = shlex.join(
        [
            "python",
            "-m",
            converter.module,
            "--model_dir",
            source,
            "--save_dir",
            weights_dir,
            *_config_to_cli_args(converter.args, normalize=False),
        ]
    )
    return (
        "set -euo pipefail\n"
        f"mkdir -p {target}\n"
        f"exec 9>{target}/.srtctl-convert.lock\n"
        "flock 9\n"
        f"if [ ! -e {ready} ]; then\n"
        f"  echo 'Converting TileRT weights into '{target}\n"
        f"  {convert}\n"
        f"  [ -e {ready} ] || {{ echo 'TileRT conversion did not produce '{ready} >&2; exit 1; }}\n"
        "fi\n"
        f"for f in {shlex.quote(source)}/*; do\n"
        '  b="$(basename "$f")"\n'
        '  case "$b" in *.safetensors|model.safetensors.index.json) continue ;; esac\n'
        f'  if [ -f "$f" ] && [ ! -e {target}/"$b" ]; then cp -p "$f" {target}/"$b"; fi\n'
        "done\n"
        "exec 9>&-\n"
    )


def worker_container_image(backend: object, mode: str, default: str) -> str:
    """Worker image for one role: TileRT prefill may use its own image; everything else uses ``default``."""
    if isinstance(backend, TileRTProtocol) and mode == "prefill" and backend.prefill_container:
        return backend.prefill_container
    return default
