# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check a retained five-second fault probe; never launch work or publish data."""

import csv
import itertools
import json
import sys
from pathlib import Path


def check(directory: Path) -> dict:
    trace = [json.loads(line) for line in (directory / "dcgm-reads.jsonl").read_text().splitlines()]
    events = {row["event"]: row for row in trace if row["event"] != "power_read"}
    start = events["repair_entered"]["at"]
    end = events["repair_returned"]["at"]
    assert events["repair_returned"]["success"], "profiling repair did not succeed"
    assert end - start >= 5_000_000, "the five-second fault was not exercised"
    assert events["fault_requested"]["at"] <= start
    with (directory / "power/samples.csv").open() as source:
        samples = list(csv.DictReader(source))
    assert samples, "no SRT samples"
    gpus = {int(row["gpu_index"]) for row in samples}
    assert len(gpus) == 4, "this fixture expects four GPUs"
    result = {}
    for gpu in sorted(gpus):
        rows = [row for row in samples if int(row["gpu_index"]) == gpu]
        assert len({row["gpu_uuid"] for row in rows}) == 1, "GPU identity changed"
        times = sorted(float(row["timestamp_unix"]) for row in rows)
        gap = max(b - a for a, b in itertools.pairwise(times))
        assert gap < 3, f"GPU {gpu}: power sample gap {gap:.4f}s exceeds the 3s gate"
        during = [
            row for row in trace if row["event"] == "power_read" and row["gpu"] == gpu and start <= row["at"] <= end
        ]
        assert len({row["dcgm_timestamp"] for row in during}) >= 4, f"GPU {gpu}: power did not advance during repair"
        assert all(row["status"] == 0 for row in during), "DCGM power read failed"
        ages = [(row["at"] - row["dcgm_timestamp"]) / 1e6 for row in during]
        assert all(0 <= age < 0.5 for age in ages), f"GPU {gpu}: stale source power"
        assert any(row["sm_active"] != "" and float(row["timestamp_unix"]) * 1e6 > end for row in rows), (
            f"GPU {gpu}: profiling never recovered"
        )
        result[str(gpu)] = {
            "max_gap_seconds": gap,
            "fresh_reads_during_repair": len(during),
            "max_source_age_seconds": max(ages),
        }
    timings = [json.loads(line) for line in (directory / "power/scrape-timings.jsonl").read_text().splitlines()]
    first_sample = min(float(row["timestamp_unix"]) for row in samples)
    errors = [
        endpoint["error_type"]
        for row in timings
        if row["cycle_started_at_unix"] >= first_sample
        for endpoint in row["endpoints"]
        if endpoint["error_type"]
    ]
    assert not errors, f"post-readiness request errors: {errors}"
    assert json.loads((directory / "artifact-validation.json").read_text())["ok"], "SRT artifact validation failed"
    return result


if __name__ == "__main__":
    print(json.dumps(check(Path(sys.argv[1])), indent=2))
