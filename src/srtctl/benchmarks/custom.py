# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom benchmark runner."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from srtctl.benchmarks.base import BenchmarkRunner, register_benchmark
from srtctl.core.runtime import RuntimeContext
from srtctl.core.schema import SrtConfig


@register_benchmark("custom")
class CustomBenchmarkRunner(BenchmarkRunner):
    """Run an arbitrary benchmark command inside a container.

    IMPORTANT — no templating on ``benchmark.command``.

    The string in ``benchmark.command`` is passed to ``bash -lc`` verbatim.
    srtctl does NOT substitute placeholders like ``{nginx_url}``,
    ``{slurm_job_id}``, ``{log_dir}``, ``{target}``, etc. Any literal
    ``{…}`` in the command will reach the shell unchanged and almost
    certainly produce a confusing error (e.g. ``bash: {nginx_url}: not
    found``).

    Practical consequences:

    * The benchmark runs inside the job's container with pyxis/enroot's
      default networking, so services on the head node are reachable at
      ``localhost``. Hit ``http://localhost:<port>`` directly.
    * If ``frontend.enable_multiple_frontends`` is ``False`` there is no
      nginx proxy; point the benchmark at the master router port (or a
      worker) directly — again via ``localhost``.
    * If you need to parameterize the command, render it yourself when
      you generate the recipe and paste the final string into
      ``benchmark.command``.
    * Runtime-discovered frontend and logical worker endpoints are injected
      through ``SRT_*`` environment variables. Custom AIPerf commands also
      receive ``AIPERF_SERVER_METRICS_URLS``. Multi-node follower ranks are
      intentionally excluded; see ``docs/config-reference.md`` for the full
      contract.
    """

    # BenchmarkConfig fields this runner reads (beyond the shared ones); see benchmark_config_fields().
    config_fields: ClassVar[frozenset[str]] = frozenset(
        {"command", "argv", "cwd", "env_unset", "container_image", "env"}
    )

    @property
    def name(self) -> str:
        return "Custom"

    @property
    def script_path(self) -> str:
        return "<custom command>"

    def validate_config(self, config: SrtConfig) -> list[str]:
        b = config.benchmark
        errors = []
        if bool(b.command) == bool(b.argv):
            errors.append("Exactly one of benchmark.command or benchmark.argv is required for benchmark.type=custom")
        if b.argv is not None and (not b.argv or any(not isinstance(a, str) or "\0" in a for a in b.argv)):
            errors.append("benchmark.argv must be a nonempty list of NUL-free strings")
        if b.argv and not b.argv[0]:
            errors.append("benchmark.argv executable must not be empty")
        if b.cwd is not None and (not b.cwd.startswith("/") or "\0" in b.cwd):
            errors.append("benchmark.cwd must be an absolute container path")
        for key in [*b.env, *b.env_unset]:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                errors.append(f"Invalid client environment variable: {key!r}")
        if set(b.env) & set(b.env_unset):
            errors.append("benchmark.env and benchmark.env_unset must be disjoint")
        if any(key.startswith("SRT_") for key in [*b.env, *b.env_unset]):
            errors.append("Runtime SRT_* client context cannot be overridden or removed")
        return errors

    def build_command(self, config: SrtConfig, runtime: RuntimeContext) -> list[str]:
        del runtime
        if config.benchmark.argv is not None:
            return list(config.benchmark.argv)
        assert config.benchmark.command is not None
        return ["bash", "-lc", config.benchmark.command]

    def get_container_image(self, config: SrtConfig, runtime: RuntimeContext) -> str | Path:
        return config.benchmark.container_image or runtime.container_image

    def get_environment(self, config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
        del runtime
        return dict(config.benchmark.env)
