# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct SGLang server; use sglang-router for replicas or prefill/decode."""

from srtctl.frontends.base import register_frontend
from srtctl.frontends.direct import DirectServerFrontend


@register_frontend("sglang")
class SGLangFrontend(DirectServerFrontend):
    required_backend = "sglang"
    server_name = "sglang.launch_server"
    router_hint = "Use frontend.type: sglang-router (the SGLang Model Gateway, renamed in 2.0) or dynamo."
