# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose AMD SMI GPU socket power to the existing head-node HTTP collector.

Standalone stdlib script, mounted at /srtctl-runtime in the exporter container.
GPU identity is fixed for the exporter lifetime; each request reads fresh watts.
AMD SMI owns sensor sampling. The orchestrator owns timestamps.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger(__name__)


def _value(raw: object, unit: str) -> str:
    """ROCm 7.2 JSON carries explicit units; never substitute a limit or zero."""
    if not isinstance(raw, dict) or raw.get("unit") != unit:
        return "NaN"
    value = raw.get("value")
    if type(value) not in (int, float):
        return "NaN"
    return str(value)


def render_metrics(identities: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    """Translate native JSON without changing watts, identities or duplicates.

    Invalid metric values and absent labels remain visible to the shared
    parser, which assigns the established power-contract reason codes.
    """
    # Preserve ambiguous joins as duplicates for the shared per-GPU validation.
    by_index: dict[str, list[dict[str, Any]]] = {}
    for device in identities:
        index = str(device.get("gpu", ""))
        by_index.setdefault(index, []).append(device)

    lines = [
        "# HELP amd_smi_socket_power_watts AMD SMI power.socket_power in watts.",
        "# TYPE amd_smi_socket_power_watts gauge",
        "# TYPE amd_smi_gfx_activity_percent gauge",
    ]
    for device in metrics["gpu_data"]:
        index = str(device.get("gpu", ""))
        power = device.get("power")
        raw_power = power.get("socket_power") if isinstance(power, dict) else None
        usage = device.get("usage")
        for identity in by_index.get(index, [{}]):
            uuid = identity.get("uuid", "")
            if uuid == "N/A":
                uuid = ""
            labels = f"gpu={json.dumps(index)},uuid={json.dumps(uuid)}"
            partition = identity.get("partition_id")
            if partition not in (None, 0, "0", "N/A"):
                labels += f",partition_id={json.dumps(str(partition))}"
            lines.append(f"amd_smi_socket_power_watts{{{labels}}} {_value(raw_power, 'W')}")
            if isinstance(usage, dict) and "gfx_activity" in usage:
                lines.append(f"amd_smi_gfx_activity_percent{{{labels}}} {_value(usage['gfx_activity'], '%')}")
    return "\n".join(lines) + "\n"


def _run_amd_smi(binary: str, command_timeout: float, *args: str) -> Any:
    completed = subprocess.run(
        [binary, *args, "--json"], capture_output=True, text=True, check=True, timeout=command_timeout
    )
    return json.loads(completed.stdout)


def make_server(host: str, port: int, *, binary: str = "amd-smi", command_timeout: float = 1.0) -> HTTPServer:
    """One collector per node; GPU identity stays fixed within the allocation."""
    identities = _run_amd_smi(binary, command_timeout, "list")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/metrics":
                self.send_error(404)
                return
            try:
                metrics = _run_amd_smi(binary, command_timeout, "metric", "--power", "--usage")
                payload = render_metrics(identities, metrics).encode()
            except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, AttributeError) as exc:
                logger.warning("AMD SMI scrape failed: %s", exc)
                self.send_error(503, "AMD SMI scrape failed")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            logger.debug(format, *args)

    return HTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9401)
    parser.add_argument("--amd-smi", default="amd-smi")
    parser.add_argument("--command-timeout", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if not math.isfinite(args.command_timeout) or args.command_timeout <= 0:
        parser.error("command-timeout must be finite and positive")
    logging.basicConfig(level=logging.INFO)
    with make_server(args.host, args.port, binary=args.amd_smi, command_timeout=args.command_timeout) as server:
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
