# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import DatasetExportMode
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.utils.configclass import make_configclass


def metrics_to_recorder_manager_cfg(metrics: list[MetricBase] | None) -> RecorderManagerBaseCfg | None:
    """Converts a list of metrics to a recorder manager configuration.

    Args:
        metrics(list[MetricBase] | None): The list of metrics to convert to a
            recorder manager configuration.

    Returns:
        The recorder manager configuration. None if no metrics are provided.
    """
    if metrics is None:
        return None
    # Build the configclass field list from each metric's RecorderTermCfg. Multiple
    # metrics may legitimately read the same underlying HDF5 stream (e.g. an episode-wide
    # reduction plus a windowed reduction of the same per-step signal share a recorder
    # term). Register each recorder term only once, keyed by the recorder cfg's ``name``.
    configclass_fields: list[tuple[str, type, object]] = []
    seen_recorder_names: set[str] = set()
    for metric in metrics:
        recorder_cfg = metric.get_recorder_term_cfg()
        if recorder_cfg is None:
            continue
        if recorder_cfg.name in seen_recorder_names:
            continue
        seen_recorder_names.add(recorder_cfg.name)
        configclass_fields.append((metric.name, type(recorder_cfg), recorder_cfg))
    # Make a configclass for the recorder manager configuration.
    recorder_cfg_cls = make_configclass("RecorderManagerCfg", configclass_fields, bases=(RecorderManagerBaseCfg,))
    recorder_cfg = recorder_cfg_cls()
    # Export all episodes to file.
    recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    return recorder_cfg
