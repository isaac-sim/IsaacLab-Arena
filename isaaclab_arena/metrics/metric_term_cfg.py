# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

from isaaclab.utils import configclass


@configclass
class MetricTermCfg:
    """Configuration for a metric term.

    Metrics are computed offline from data captured by an associated recorder term during
    rollout. The ``func`` callable receives the recorded data for the recorder term named
    by :attr:`recorder_term_name` (a list of per-episode arrays) plus any parameters in
    :attr:`params` and returns a scalar metric value.
    """

    func: Callable[..., Any] = MISSING
    """The function used to compute the metric value from the recorded data.

    Signature: ``func(recorded_metric_data: list[np.ndarray], **params) -> Any``.

    The return is typically a ``float`` (e.g. success rate), but a metric may also
    return a list (e.g. per-subtask success rates).
    """

    params: dict[str, Any] = dict()
    """The keyword arguments forwarded to :attr:`func`. Defaults to an empty dict."""

    recorder_term_name: str = MISSING
    """The name of the recorder term whose recorded data is fed to :attr:`func`."""
