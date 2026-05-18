# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from abc import ABC, abstractmethod

from isaaclab.managers.recorder_manager import RecorderTermCfg


class MetricBase(ABC):

    name: str
    recorder_term_name: str

    @property
    def extra_recorder_term_dependencies(self) -> list[str]:
        """Names of additional recorder terms (beyond this metric's own ``recorder_term_name``)
        that must be loaded from HDF5 and passed to ``compute_metric_from_recording``.

        Metrics that declare extra dependencies must also accept a ``context`` keyword
        argument on ``compute_metric_from_recording`` — the dispatcher passes the loaded
        per-episode arrays keyed by recorder-term name as ``context``. Metrics that don't
        declare any extra dependencies are called the legacy way (single positional arg),
        so existing subclasses need no changes.
        """
        return []

    @abstractmethod
    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
        raise NotImplementedError("Function not implemented yet.")
