# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exporter metric names mapped onto the fixed GPU power artifact columns."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from srtctl.core.power.contract import (
    GPU_UTIL_METRIC,
    POWER_METRIC,
    POWER_SCOPE,
    SM_ACTIVE_METRIC,
    UTILIZATION_METRICS,
    UtilizationMetric,
)


@dataclass(frozen=True)
class PowerMetricProfile:
    """Map exporter metrics to the artifact's fixed columns, units, and bounds.

    Utilization sources are keyed by artifact column; omit a column when the
    exporter cannot supply that measurement. Partition labels mark unsupported
    logical devices without dropping other GPUs from the same scrape.
    """

    name: str = "dcgm"
    power_metric: str = POWER_METRIC
    gpu_index_label: str = "gpu"
    gpu_uuid_label: str = "UUID"
    power_scope: str = POWER_SCOPE
    utilization_sources: tuple[tuple[str, str], ...] = (
        ("gpu_util_pct", GPU_UTIL_METRIC),
        ("sm_active", SM_ACTIVE_METRIC),
    )
    partition_label: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.power_scope.strip():
            raise ValueError("power profile name and scope must be non-empty")
        labels = [self.gpu_index_label, self.gpu_uuid_label]
        if self.partition_label is not None:
            labels.append(self.partition_label)
        if any(not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", label) for label in labels):
            raise ValueError("power profile labels must be Prometheus label names")
        if len(labels) != len(set(labels)):
            raise ValueError("power profile labels must differ")
        columns = [column for column, _ in self.utilization_sources]
        if len(columns) != len(set(columns)) or set(columns) - {metric.column for metric in UTILIZATION_METRICS}:
            raise ValueError("power profile utilization columns must be distinct artifact columns")
        metrics = [self.power_metric, *(source for _, source in self.utilization_sources)]
        if any(not re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*", metric) for metric in metrics):
            raise ValueError("power profile metrics must be Prometheus metric names")
        if len(metrics) != len(set(metrics)):
            raise ValueError("power profile metric names must be distinct")

    @property
    def utilization_metrics(self) -> tuple[UtilizationMetric, ...]:
        """Keep the artifact's column order and semantics, regardless of mapping order."""
        sources = dict(self.utilization_sources)
        return tuple(
            replace(spec, metric=sources[spec.column]) for spec in UTILIZATION_METRICS if spec.column in sources
        )

    def to_dict(self) -> dict[str, Any]:
        """Self-contained JSON mapping for retained artifact provenance."""
        payload = asdict(self)
        payload["utilization_sources"] = [list(source) for source in self.utilization_sources]
        return payload

    @classmethod
    def from_manifest(cls, payload: object) -> PowerMetricProfile:
        """Validate untrusted JSON types before applying the mapping's semantics."""
        if not isinstance(payload, dict) or set(payload) != {field.name for field in fields(cls)}:
            raise ValueError("expected a complete metric profile object")
        for key in ("name", "power_metric", "gpu_index_label", "gpu_uuid_label", "power_scope"):
            if not isinstance(payload[key], str):
                raise TypeError(f"power profile {key} must be a string")
        if payload["partition_label"] is not None and not isinstance(payload["partition_label"], str):
            raise TypeError("power profile partition_label must be a string or null")
        sources = payload["utilization_sources"]
        if not isinstance(sources, list) or any(
            not isinstance(source, list) or len(source) != 2 or not all(isinstance(value, str) for value in source)
            for source in sources
        ):
            raise ValueError("power profile utilization_sources must contain column and metric string pairs")
        profile_fields: dict[str, Any] = {**payload, "utilization_sources": tuple(tuple(source) for source in sources)}
        return cls(**profile_fields)


DEFAULT_POWER_PROFILE = PowerMetricProfile()

# The adapter reads AMD SMI's GPU socket sensor. This is deliberately not
# described as the same measurement boundary as DCGM's device-board metric.
AMD_SMI_POWER_PROFILE = PowerMetricProfile(
    name="amd-smi-socket",
    power_metric="amd_smi_socket_power_watts",
    gpu_index_label="gpu",
    gpu_uuid_label="uuid",
    power_scope="gpu_socket_as_reported_by_amd_smi",
    utilization_sources=(("gpu_util_pct", "amd_smi_gfx_activity_percent"),),
    partition_label="partition_id",
)
