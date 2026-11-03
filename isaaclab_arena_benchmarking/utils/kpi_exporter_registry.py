# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Name-to-class mapping of the KPI exporters, filled by the ``register_kpi_exporter`` decorator."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import InfluxdbKpiExporter

EXPORTERS_BY_NAME: dict[str, type[InfluxdbKpiExporter]] = {}
"""KPI exporter classes by the name used to select them, once their modules are imported."""


def register_kpi_exporter(name: str) -> Callable[[type[InfluxdbKpiExporter]], type[InfluxdbKpiExporter]]:
    """Return a class decorator that registers a KPI exporter under ``name``.

    Args:
        name: Name that selects the exporter; no other exporter may use it.
    """

    def decorator(exporter_type: type[InfluxdbKpiExporter]) -> type[InfluxdbKpiExporter]:
        assert name not in EXPORTERS_BY_NAME, (
            f"KPI exporter name {name!r} is already registered by {EXPORTERS_BY_NAME[name].__name__}, cannot register"
            f" {exporter_type.__name__}"
        )
        EXPORTERS_BY_NAME[name] = exporter_type
        return exporter_type

    return decorator
