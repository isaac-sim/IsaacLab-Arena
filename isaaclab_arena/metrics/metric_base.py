# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from abc import ABC, abstractmethod

from isaaclab.managers.recorder_manager import RecorderTermCfg

from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg


class MetricBase(ABC):

    name: str
    recorder_term_name: str

    @abstractmethod
    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_metric_term_cfg(self) -> MetricTermCfg:
        """Return the metric term configuration.

        The returned config carries a pointer to the module-level offline compute
        function (``func``) together with the parameters needed to call it. Used by
        the runtime ``MetricsManager`` to compute the metric value once a rollout has
        finished.
        """
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
        raise NotImplementedError("Function not implemented yet.")
