# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose AMD SMI GPU socket power to the existing head-node HTTP collector.

Standalone stdlib script, mounted at /srtctl-runtime in the exporter container.
Each request obtains fresh identity and metric snapshots; failures never serve
cached watts. AMD SMI owns sensor sampling. The orchestrator owns timestamps.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger(__name__)


def _devices(payload: object) -> list[dict[str, Any]]:
    # `list --json` returns an array; ROCm 7.2 `metric --json` wraps
    # device records in gpu_data, separately from any CPU/core records.
    if isinstance(payload, dict) and "gpu_data" in payload:
        payload = payload["gpu_data"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("expected AMD SMI JSON device objects")
    return payload


def _index(value: object) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, int) and value >= 0:
        return str(value)
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return str(int(value))
    return ""


def _value(raw: object, unit: str) -> str:
    """ROCm 7.2 JSON carries explicit units; never substitute a limit or zero."""
    if not isinstance(raw, dict) or raw.get("unit") != unit:
        return "NaN"
    value = raw.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "NaN"
    return str(value) if math.isfinite(value) else "NaN"


def render_metrics(identities: object, metrics: object) -> str:
    """Translate native JSON without changing watts, identities or duplicates.

    Invalid metric values and absent labels remain visible to the shared
    parser, which assigns the established power-contract reason codes.
    """
    by_index: dict[str, str] = {}
    seen_uuids: set[str] = set()
    for device in _devices(identities):
        index = _index(device.get("gpu"))
        uuid = device.get("uuid")
        uuid = uuid.strip() if isinstance(uuid, str) else ""
        if uuid.lower() in {"n/a", "none", "null"}:
            uuid = ""
        if device.get("partition_id") not in (None, 0, "0", "N/A"):
            raise ValueError("partitioned GPU identity is unsupported")
        if index in by_index or (uuid and uuid in seen_uuids):
            raise ValueError("ambiguous GPU identity")
        if index:
            by_index[index] = uuid
        if uuid:
            seen_uuids.add(uuid)

    lines = [
        "# HELP amd_smi_socket_power_watts AMD SMI power.socket_power in watts.",
        "# TYPE amd_smi_socket_power_watts gauge",
        "# TYPE amd_smi_gfx_activity_percent gauge",
    ]
    for device in _devices(metrics):
        index = _index(device.get("gpu"))
        # JSON string escaping is also valid for the label characters emitted
        # here (UUIDs and decimal indices); emit Unicode without JSON-only escapes.
        uuid = by_index.get(index, "")
        labels = f"gpu={json.dumps(index, ensure_ascii=False)},uuid={json.dumps(uuid, ensure_ascii=False)}"
        power = device.get("power")
        raw_power = power.get("socket_power") if isinstance(power, dict) else None
        lines.append(f"amd_smi_socket_power_watts{{{labels}}} {_value(raw_power, 'W')}")
        usage = device.get("usage")
        if isinstance(usage, dict) and "gfx_activity" in usage:
            lines.append(f"amd_smi_gfx_activity_percent{{{labels}}} {_value(usage['gfx_activity'], '%')}")
    return "\n".join(lines) + "\n"


def collect_metrics(binary: str, command_timeout: float) -> str:
    """Bound both CLI invocations; HTTP failures cannot return stale samples."""

    def run(*args: str) -> object:
        completed = subprocess.run(
            [binary, *args, "--json"], capture_output=True, text=True, check=True, timeout=command_timeout
        )
        return json.loads(completed.stdout)

    identities = run("list")
    metrics = run("metric", "--power", "--usage")
    return render_metrics(identities, metrics)


def make_server(host: str, port: int, *, binary: str = "amd-smi", command_timeout: float = 1.0) -> HTTPServer:
    """A serial server bounds subprocess concurrency even after client timeout."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/metrics":
                self.send_error(404)
                return
            try:
                payload = collect_metrics(binary, command_timeout).encode()
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                logger.warning("AMD SMI scrape failed: %s", exc)
                self.send_error(503, "AMD SMI scrape failed")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            # A timed-out collector has already recorded the failure.
            with suppress(BrokenPipeError, ConnectionResetError):
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
