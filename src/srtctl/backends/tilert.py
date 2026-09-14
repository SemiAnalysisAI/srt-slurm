# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TileRT decode with vLLM prefill for native P/D disaggregation."""

from __future__ import annotations

import builtins
import json
from collections.abc import Sequence
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from marshmallow import Schema, ValidationError
from marshmallow_dataclass import dataclass

from srtctl.ports import DYN_SYSTEM_PORT_BASE

if TYPE_CHECKING:
    from srtctl.backends.base import BackendPreparation, SrunConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process

WorkerMode = Literal["prefill", "decode", "agg"]

_PREFILL_MANAGED_ARGS = {
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

_DECODE_MANAGED_ARGS = {
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


@dataclass(frozen=True)
class TileRTServerConfig:
    """Role-specific CLI arguments not owned by the topology adapter."""

    prefill: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class TileRTProtocol:
    """Run vLLM prefill workers and TileRT decode workers without Dynamo.

    TileRT's public P/D implementation intentionally uses different runtime
    images for the two roles. ``model.container`` remains the decode image;
    ``prefill_container`` selects the vLLM image used only for prefill sruns.
    """

    type: Literal["tilert"] = "tilert"
    # OpenAI model name; defaults to the final component of model.path.
    served_model_name: str | None = None
    # vLLM image or cluster container alias for prefill workers only.
    prefill_container: str | None = None
    prefill_environment: dict[str, str] = field(default_factory=dict)
    decode_environment: dict[str, str] = field(default_factory=dict)
    tilert_config: TileRTServerConfig | None = None
    # TileRT P/D model profile (for example glm5).
    model_profile: str = "glm5"
    # Model type passed to the TileRT weight converter.
    weight_model_type: str = "glm-5"
    # Dedicated shared converted-weight directory, mounted at this path in all decode containers.
    weights_dir: str = "/tilert_weights/converted"
    # Maximum sequence length shared by prefill and decode.
    max_seq_len: int = 202752
    # TileRT decode KV cache dtype.
    kv_cache_dtype: str = "fp8"
    # vLLM KV cache dtype; must be wire-compatible with the decode dtype.
    prefill_kv_cache_dtype: str = "fp8_ds_mla"
    # KV transfer transport (the adapter supports NIXL).
    transport: Literal["nixl"] = "nixl"
    # Enable MTP on both sides of the P/D transfer.
    with_mtp: bool = True
    # Number of speculative tokens advertised by vLLM prefill.
    speculative_tokens: int = 1

    Schema: ClassVar[builtins.type[Schema]] = Schema

    def validate_recipe(self, config: Any) -> None:
        """Reject unsupported topology and adapter overrides before allocating Slurm nodes."""
        resources = config.resources
        if config.frontend.type != "tilert-router":
            raise ValidationError("engine.type: tilert requires frontend.type: tilert-router")
        if config.frontend.enable_multiple_frontends:
            raise ValidationError("TileRT requires frontend.enable_multiple_frontends: false")
        if resources.num_agg or resources.num_prefill != 1 or resources.num_decode < 1:
            raise ValidationError("TileRT requires exactly one prefill worker and at least one decode worker")
        if max(resources.gpus_per_prefill, resources.gpus_per_decode) > resources.gpus_per_node:
            raise ValidationError("Each TileRT prefill and decode worker must fit on one node")
        if self.model_profile in {"glm5", "glm_5", "glm-5"} and resources.gpus_per_decode != 8:
            raise ValidationError("TileRT GLM-5 decode kernels require exactly 8 GPUs per worker")
        if not self.prefill_container:
            raise ValidationError("engine.prefill_container must select the vLLM prefill image")
        if not Path(self.weights_dir).is_absolute() or self.weights_dir == "/":
            raise ValidationError("engine.weights_dir must be an absolute, dedicated converted-weight directory")
        if self.max_seq_len < 1 or self.speculative_tokens < 1:
            raise ValidationError("TileRT max_seq_len and speculative_tokens must be positive")
        if config.profiling.is_torch:
            raise ValidationError("TileRT does not support torch profiling; use nsys")
        # Command builders own these flags even when supplied through roles.*.args.
        for mode, reserved in (("prefill", _PREFILL_MANAGED_ARGS), ("decode", _DECODE_MANAGED_ARGS)):
            try:
                _reject_managed_args(self.get_config_for_mode(mode), reserved)
            except ValueError as error:
                raise ValidationError(str(error)) from error

    def get_srun_config(self) -> SrunConfig:
        from srtctl.backends.base import SrunConfig

        return SrunConfig(mpi=None, oversubscribe=False, launch_per_endpoint=False)

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        if self.tilert_config is None:
            return {}
        values = {"prefill": self.tilert_config.prefill, "decode": self.tilert_config.decode}
        return dict(values.get(mode) or {})

    def get_environment_for_mode(self, mode: WorkerMode) -> dict[str, str]:
        values = {"prefill": self.prefill_environment, "decode": self.decode_environment}
        return dict(values.get(mode) or {})

    def get_process_environment(self, process: Process) -> dict[str, str]:
        return {"TILERT_ROLE": process.endpoint_mode}

    def get_container_image_for_mode(self, mode: WorkerMode, default: str) -> str:
        if mode == "prefill" and self.prefill_container:
            return self.prefill_container
        return default

    def get_served_model_name(self, default: str) -> str:
        return self.served_model_name or default

    def get_metrics_port(self, process: Process) -> int | None:
        """Return only real Prometheus endpoints.

        The vLLM prefill API exports ``/metrics``. TileRT's decode API exports
        health and token-transfer routes but no Prometheus handler.
        """
        return process.http_port if process.endpoint_mode == "prefill" else None

    def get_preparation(self, runtime: RuntimeContext, processes: list[Process]) -> BackendPreparation:
        """Materialize converted TileRT weights once under a shared lock."""
        from srtctl.backends.base import BackendPreparation

        decode_leaders = [process for process in processes if process.endpoint_mode == "decode" and process.is_leader]
        if not decode_leaders:
            raise ValueError("TileRT model preparation requires a decode worker")
        source = str(runtime.model_path) if runtime.is_hf_model else runtime.worker_model_arg
        if not runtime.is_hf_model and Path(source) == Path(self.weights_dir):
            raise ValueError("TileRT weights_dir must differ from the source model directory")
        script = _weight_preparation_script(
            source=source,
            is_hf_model=runtime.is_hf_model,
            weights_dir=self.weights_dir,
            weight_model_type=self.weight_model_type,
        )
        return BackendPreparation(
            command=["bash", "-ceu", script],
            node=decode_leaders[0].node,
            mode="decode",
            log_name="tilert_weight_conversion.out",
            gpus_per_task=1,
        )

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
        if frontend_type != "tilert-router":
            raise ValueError(f"backend.type: tilert requires frontend.type: tilert-router (got {frontend_type!r})")
        from srtctl.core.topology import endpoints_to_processes

        return endpoints_to_processes(endpoints, base_sys_port=base_sys_port, port_allocator=port_allocator)

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
        del dump_config_path, profiling
        if frontend_type != "tilert-router":
            raise ValueError(f"backend.type: tilert requires frontend.type: tilert-router (got {frontend_type!r})")
        if len({item.node for item in endpoint_processes}) != 1:
            raise ValueError("TileRT currently requires every prefill and decode endpoint to fit on one node")
        if process.endpoint_mode not in {"prefill", "decode"}:
            raise ValueError("TileRT supports prefill/decode workers only")

        config = self.get_config_for_mode(process.endpoint_mode)
        if process.endpoint_mode == "decode":
            _reject_managed_args(config, _DECODE_MANAGED_ARGS)
            command = [*(nsys_prefix or []), "python", "-m", "tilert.pd_vllm.decode_server"]
            command.extend(
                [
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
                    str(process.sys_port),
                    "--http-port",
                    str(process.http_port),
                ]
            )
            if self.with_mtp:
                command.append("--with-mtp")
            command.extend(_config_to_cli_args(config))
            return command

        _reject_managed_args(config, _PREFILL_MANAGED_ARGS)
        transfer = {
            "kv_connector": "TileRTConnector",
            "kv_connector_module_path": "tilert.pd_vllm.prefill_connector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "tilert_ctrl_port": process.sys_port,
                "tilert_model": self.model_profile,
                "tilert_max_seq_len": self.max_seq_len,
                "tilert_transport": self.transport,
            },
        }
        speculative = {"method": "mtp", "num_speculative_tokens": self.speculative_tokens}
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
            command.extend(["--speculative-config", json.dumps(speculative, separators=(",", ":"))])
        command.extend(_config_to_cli_args(config))
        return command


def _config_to_cli_args(config: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in config.items():
        flag = key if key.startswith("-") else f"--{key.replace('_', '-')}"
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


def _reject_managed_args(config: dict[str, Any], reserved: set[str]) -> None:
    overlap = {key.lstrip("-").replace("_", "-") for key in config}.intersection(reserved)
    if overlap:
        raise ValueError(f"TileRT config cannot override srtctl-managed argument(s): {sorted(overlap)}")


def _weight_preparation_script(*, source: str, is_hf_model: bool, weights_dir: str, weight_model_type: str) -> str:
    """Build a lock-safe, atomic TileRT conversion command."""
    # Backend preparation is the final completeness boundary before conversion.
    # The best-effort orchestrator pre-download may time out, and
    # ``local_files_only=True`` can still resolve a partially populated snapshot.
    # Let snapshot_download resume missing files here under its normal HF cache
    # locks; a complete cache remains a metadata-only fast path.
    resolve_source = (
        f"from huggingface_hub import snapshot_download; source = snapshot_download({source!r})"
        if is_hf_model
        else f"source = {source!r}"
    )
    python = f"""
import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path

{resolve_source}
source_path = Path(source)
target = Path({weights_dir!r})
target.parent.mkdir(parents=True, exist_ok=True)
lock_path = target.parent / f".{{target.name}}.convert.lock"
with lock_path.open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    def complete(path):
        index = path / "model.safetensors.index.json"
        if not index.is_file() or not (path / "chat_template.jinja").is_file():
            return False
        if not ((path / "tokenizer.json").is_file() or (path / "tokenizer_config.json").is_file()):
            return False
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
            return bool(shards) and all((path / shard).is_file() for shard in shards)
        except (ValueError, KeyError, TypeError):
            return False

    if complete(target):
        print(f"TileRT weight cache hit: {{target}}", flush=True)
        raise SystemExit(0)
    tmp = target.parent / f".{{target.name}}.tmp.{{os.environ.get('SLURM_JOB_ID', os.getpid())}}"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        subprocess.run(
            [
                "python",
                "-m",
                "tilert.models.preprocess.weight_converter",
                "--model_type",
                {weight_model_type!r},
                "--model_dir",
                str(source_path),
                "--save_dir",
                str(tmp),
            ],
            check=True,
        )
        for item in source_path.iterdir():
            if not item.is_file() or item.suffix == ".safetensors" or item.name == "model.safetensors.index.json":
                continue
            shutil.copy2(item, tmp / item.name)
        converted_index = tmp / "model.safetensors.index.json"
        if not converted_index.is_file():
            raise RuntimeError(f"TileRT conversion produced no index: {{converted_index}}")
        if not (tmp / "chat_template.jinja").is_file():
            raise RuntimeError(f"TileRT conversion produced no chat template: {{tmp}}")
        if not ((tmp / "tokenizer.json").is_file() or (tmp / "tokenizer_config.json").is_file()):
            raise RuntimeError(f"TileRT conversion produced no tokenizer metadata: {{tmp}}")
        if not complete(tmp):
            raise RuntimeError(f"TileRT conversion produced incomplete shards: {{tmp}}")
        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp, target)
        print(f"TileRT weight cache ready: {{target}}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
"""
    return f"python - <<'PY'\n{python}PY\n"
