# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for metrics."""

from abc import ABC, abstractmethod

from isaaclab.managers.recorder_manager import RecorderTermCfg

from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg


class MetricBase(ABC):
    """A metric that pairs a rollout-time recorder with an offline function that computes the metric value."""

    name: str
    recorder_term_name: str

    @abstractmethod
    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        """Return the configuration of the recorder term that records the data this metric needs."""
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_metric_term_cfg(self) -> MetricTermCfg:
        """Return the metric term configuration.

        The returned config carries a pointer to the module-level offline compute
        function (``compute_metric_func``) together with the parameters needed to call
        it.
        """
        raise NotImplementedError("Function not implemented yet.")
