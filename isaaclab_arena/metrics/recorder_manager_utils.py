# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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
    # For each metric, grad it's RecorderTermCfg and add it to the output configclass fields list.
    configclass_fields: list[tuple[str, type, object]] = []
    for metric in metrics:
        configclass_fields.append((metric.name, type(metric.get_recorder_term_cfg()), metric.get_recorder_term_cfg()))
    # Make a configclass for the recorder manager configuration.
    recorder_cfg_cls = make_configclass("RecorderManagerCfg", configclass_fields, bases=(RecorderManagerBaseCfg,))
    recorder_cfg = recorder_cfg_cls()
    # Export all episodes to file.
    recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    return recorder_cfg
