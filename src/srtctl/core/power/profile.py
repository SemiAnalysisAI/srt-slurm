# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exporter metric names mapped onto the fixed GPU power artifact columns."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
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
    """A Prometheus exporter's identity and watts mapping, independent of vendor.

    Optional utilization sources may be disabled with null. Their output
    columns, units, and bounds remain owned by the artifact contract.
    """

    name: str = "dcgm"
    power_metric: str = POWER_METRIC
    gpu_index_label: str = "gpu"
    gpu_uuid_label: str = "UUID"
    power_scope: str = POWER_SCOPE
    gpu_util_metric: str | None = GPU_UTIL_METRIC
    sm_active_metric: str | None = SM_ACTIVE_METRIC

    def __post_init__(self) -> None:
        for name in ("name", "power_scope"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"power profile {name} must be a non-empty string")
        for name in ("power_metric", "gpu_util_metric", "sm_active_metric"):
            value = getattr(self, name)
            if value is None and name != "power_metric":
                continue
            if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*", value):
                raise ValueError(f"power profile {name} must be a Prometheus metric name")
        for name in ("gpu_index_label", "gpu_uuid_label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", value):
                raise ValueError(f"power profile {name} must be a Prometheus label name")
        if self.gpu_index_label == self.gpu_uuid_label:
            raise ValueError("power profile index and UUID labels must differ")
        metrics = [metric for metric in (self.power_metric, self.gpu_util_metric, self.sm_active_metric) if metric]
        if len(metrics) != len(set(metrics)):
            raise ValueError("power profile metric names must be distinct")

    @property
    def utilization_metrics(self) -> tuple[UtilizationMetric, ...]:
        """Keep the artifact columns and their semantics fixed across exporters."""
        sources = (self.gpu_util_metric, self.sm_active_metric)
        return tuple(
            replace(spec, metric=source)
            for spec, source in zip(UTILIZATION_METRICS, sources, strict=True)
            if source is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Self-contained mapping for retained artifact provenance."""
        return asdict(self)


DEFAULT_POWER_PROFILE = PowerMetricProfile()

# The adapter reads AMD SMI's GPU socket sensor. This is deliberately not
# described as the same measurement boundary as DCGM's device-board metric.
AMD_SMI_POWER_PROFILE = PowerMetricProfile(
    name="amd-smi-socket",
    power_metric="amd_smi_socket_power_watts",
    gpu_index_label="gpu",
    gpu_uuid_label="uuid",
    power_scope="gpu_socket_as_reported_by_amd_smi",
    gpu_util_metric="amd_smi_gfx_activity_percent",
    sm_active_metric=None,
)
