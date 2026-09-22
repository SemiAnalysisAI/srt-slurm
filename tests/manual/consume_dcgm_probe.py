# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real SRT collection and retained-artifact validation, never publish.

This is a telemetry fault-injection fixture, not a model benchmark result.
"""

import dataclasses
import importlib.util
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import requests

from srtctl.core.power.manifest import ExpectedWindow
from srtctl.core.power.session import PowerEndpoint, PowerSessionSettings, PowerTelemetrySession
from srtctl.core.power.topology import DeviceAssignment, ExpectedDevice
from srtctl.core.power.validate_artifacts import validate_power_artifacts

directory = Path(sys.argv[1]).resolve()
natural = len(sys.argv) > 2 and sys.argv[2] == "natural"
deadline = time.monotonic() + 60
while not (directory / "endpoint.txt").exists():
    if time.monotonic() > deadline:
        raise RuntimeError("native exporter probe did not become ready")
    time.sleep(0.1)
url = (directory / "endpoint.txt").read_text().strip()
hostname = socket.gethostname()
source = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "measurement_window", source / "src/srtctl/benchmarks/scripts/sa-bench/measurement_window.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

settings = PowerSessionSettings(
    power_dir=directory / "power",
    log_dir=directory,
    job_id=os.environ["SLURM_JOB_ID"],
    run_name="profiling-isolation-telemetry-fixture",
    sample_interval_seconds=1,
    startup_timeout_seconds=15,
    request_timeout_seconds=2,
    collector_join_timeout_seconds=10,
    required=True,
    exporter_port=int(url.rsplit(":", 1)[1]),
    exporter_image=os.environ["SRT_PROBE_IMAGE"],
    exporter_command=os.environ["SRT_PROBE_COMMAND"],
    producer_git_commit=os.environ["SRT_PROBE_COMMIT"],
)
devices = [ExpectedDevice(hostname, i, (DeviceAssignment("agg", 0, 0, None),)) for i in range(4)]
session = PowerTelemetrySession(
    settings=settings,
    expected_devices=devices,
    expected_windows=[ExpectedWindow("sa-bench", 1)],
    nodes=[hostname],
    endpoints=[PowerEndpoint(hostname, url + "/metrics")],
)
session.initialize()
secondary_stop = threading.Event()
secondary = None


def secondary_scraper():
    # Match the second one-second DCGM client in the retained tachometer config.
    with (directory / "secondary-scrapes.jsonl").open("w") as log:
        while not secondary_stop.is_set():
            started = time.monotonic()
            try:
                r = requests.get(url + "/metrics", timeout=2)
                item = {"status": r.status_code, "seconds": time.monotonic() - started}
                (directory / "metrics-last.txt").write_text(r.text)
            except requests.RequestException as exc:
                item = {"error": type(exc).__name__, "seconds": time.monotonic() - started}
            log.write(json.dumps(item) + "\n")
            secondary_stop.wait(max(0, 1 - (time.monotonic() - started)))


try:
    if not session.start_and_wait_for_readiness():
        raise RuntimeError("SRT power readiness failed")
    time.sleep(3)
    if natural:
        (directory / "metrics-first.txt").write_text(requests.get(url + "/metrics", timeout=2).text)
        secondary = threading.Thread(target=secondary_scraper)
        secondary.start()
    window = module.MeasurementWindow.create(
        save_result=True,
        result_dir=str(directory),
        result_filename="telemetry-fixture.json",
        concurrency=1,
        window_dir=str(session.windows_dir),
        log_root=str(directory),
    )
    start, start_mono = time.time(), time.monotonic()
    window.mark_running(start)
    if natural:
        time.sleep(60)
    else:
        time.sleep(8)
        response = requests.post(url + "/inject", timeout=2)
        response.raise_for_status()
        time.sleep(20)
    end, duration = time.time(), time.monotonic() - start_mono
    timing = {"benchmark_start_time_unix": start, "benchmark_end_time_unix": end, "duration": duration}
    (directory / "telemetry-fixture.json").write_text(
        json.dumps(
            {
                **timing,
                "diagnostic_only": True,
                "not_a_model_benchmark": True,
            },
            indent=2,
        )
        + "\n"
    )
    window.mark_completed(start_unix=start, end_unix=end, duration=duration)
    time.sleep(3)
finally:
    secondary_stop.set()
    if secondary is not None:
        secondary.join(timeout=5)
    outcome = session.stop_and_finalize()
    (directory / "srt-outcome.json").write_text(json.dumps(dataclasses.asdict(outcome), indent=2, default=str) + "\n")
    (directory / "stop").touch()

report = validate_power_artifacts(power_dir=session.power_dir, result_root=directory)
(directory / "artifact-validation.json").write_text(json.dumps(dataclasses.asdict(report), indent=2) + "\n")
print(report.render())
if not report.ok:
    sys.exit(1)
