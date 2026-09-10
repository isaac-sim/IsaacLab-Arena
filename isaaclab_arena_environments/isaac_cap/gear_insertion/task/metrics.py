# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode-level completion metrics for gear insertion."""

from __future__ import annotations

import logging
import numpy as np
import torch

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg

logger = logging.getLogger(__name__)


def _terminal_diagnostics(success_term, env_ids) -> list[dict[str, object]]:
    names = [cfg.name for cfg in success_term.gear_asset_cfgs]
    per_gear = success_term.success_per_gear[env_ids].tolist()
    diagnostics = {name: values[env_ids].tolist() for name, values in success_term.diagnostics_per_gear.items()}
    episodes = []
    for env_index, completion in enumerate(per_gear):
        episode = {}
        for gear_index, gear_name in enumerate(names):
            episode[gear_name] = {
                "success": bool(completion[gear_index]),
                **{name: values[env_index][gear_index] for name, values in diagnostics.items()},
            }
        episodes.append(episode)
    return episodes


class GearInsertionFractionRecorder(RecorderTerm):
    """Record the terminal fraction of gears satisfying the success criteria."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.first_reset = True

    def record_pre_reset(self, env_ids):
        if self.first_reset:
            assert len(env_ids) == self._env.num_envs
            self.first_reset = False
            return None, None

        success_term = self._env.termination_manager.get_term_cfg("success").func
        if not hasattr(success_term, "success_per_gear"):
            raise TypeError("gear insertion success term does not expose per-gear completion")
        per_gear = success_term.success_per_gear[env_ids]
        logger.warning(
            "terminal per-gear diagnostics: %s",
            _terminal_diagnostics(success_term, env_ids),
        )
        fractions = per_gear.to(torch.float32).mean(dim=-1)
        return self.name, fractions


@configclass
class GearInsertionFractionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = GearInsertionFractionRecorder
    name: str = "gear_insertion_fraction"


def compute_gear_insertion_fraction(recorded_metric_data: list[np.ndarray]) -> float:
    """Average terminal per-episode gear completion fractions."""

    if not recorded_metric_data:
        return 0.0
    values = np.concatenate(recorded_metric_data)
    if values.ndim != 1:
        raise ValueError("gear insertion fraction samples must be one-dimensional")
    return float(np.mean(values))


class GearInsertionFractionMetric(MetricBase):
    """Report the fraction of individually seated gears at episode termination."""

    name = "gear_insertion_fraction"
    recorder_term_name = "gear_insertion_fraction"

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return GearInsertionFractionRecorderCfg(name=self.recorder_term_name)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=compute_gear_insertion_fraction,
            params={},
            recorder_term_name=self.recorder_term_name,
        )
