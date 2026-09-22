# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct ATOM OpenAI server for a single aggregate worker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from srtctl.frontends.base import register_frontend
from srtctl.frontends.direct import DirectServerFrontend

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process


@register_frontend("atom")
class AtomFrontend(DirectServerFrontend):
    required_backend = "atom"
    server_name = "atom.entrypoints.openai_server"
    router_hint = "Use frontend.type: atomesh for replicas or prefill/decode."

    def worker_metrics_port(self, process: Process, runtime: RuntimeContext) -> None:
        return None

    def profiling_control_port(self, process: Process, config: Any, runtime: RuntimeContext) -> None:
        return None
