# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode-level completion metrics for gear insertion."""

from __future__ import annotations

import logging
import numpy as np
import torch
from dataclasses import MISSING

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg

logger = logging.getLogger(__name__)

_GEAR_GATE_NAMES = ("xy", "z", "upright", "support", "velocity")


def _terminal_diagnostics(success_predicate, env_ids, gear_names: tuple[str, ...]) -> list[dict[str, dict[str, bool]]]:
    """Build per-episode completion and gate diagnostics for each gear."""

    assert set(success_predicate.per_gear_results) == set(
        gear_names
    ), "Gear diagnostics must match the requested gears."
    episodes = [{} for _ in env_ids]
    for gear_name in gear_names:
        gear_results = success_predicate.per_gear_results[gear_name]
        gate_results = success_predicate.per_gear_gate_results[gear_name]
        assert gear_results.ndim == 1, "Gear completion results must have shape (num_envs,)."
        assert set(gate_results) == set(_GEAR_GATE_NAMES), "Gear diagnostics must report all five placement gates."
        completion = gear_results[env_ids].tolist()
        diagnostics = {}
        for gate_name, results in gate_results.items():
            assert results.shape == gear_results.shape, "Gear gate results must have shape (num_envs,)."
            diagnostics[gate_name] = results[env_ids].tolist()
        for env_index, episode in enumerate(episodes):
            episode[gear_name] = {"success": completion[env_index]}
            for gate_name, values in diagnostics.items():
                episode[gear_name][gate_name] = values[env_index]
    return episodes


class GearInsertionFractionRecorder(RecorderTerm):
    """Record the terminal fraction of gears satisfying the success criteria."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.gear_names = tuple(cfg.gear_names)
        self.first_reset = True
        self._success_objective_name = "gear_insertion"
        # CompositeTaskBase suffixes recorder names but prefixes objective names.
        # TODO(cvolk): Replace this CAP naming workaround with an explicit diagnostics reference.
        _, subtask_marker, subtask_index = self.name.rpartition("_subtask_")
        if subtask_marker:
            assert subtask_index.isdecimal(), f"Invalid composite recorder name: {self.name!r}."
            self._success_objective_name = f"subtask_{subtask_index}/gear_insertion"

    def record_pre_reset(self, env_ids):
        if self.first_reset:
            assert len(env_ids) == self._env.num_envs
            self.first_reset = False
            return None, None

        progress_tracker = self._env.progress_tracker
        assert progress_tracker is not None, "Gear insertion diagnostics require task success tracking."
        success_predicate = progress_tracker.get_predicate(self._success_objective_name)
        per_gear_results = [success_predicate.per_gear_results[gear_name][env_ids] for gear_name in self.gear_names]
        per_gear = torch.stack(per_gear_results, dim=-1)
        assert per_gear.ndim == 2 and per_gear.shape[1] == len(self.gear_names), (
            f"Gear completion results have shape {tuple(per_gear.shape)}; expected "
            f"(num_episodes, {len(self.gear_names)})."
        )
        logger.warning(
            "terminal per-gear diagnostics: %s",
            _terminal_diagnostics(success_predicate, env_ids, self.gear_names),
        )
        fractions = per_gear.to(torch.float32).mean(dim=-1)
        return self.name, fractions


@configclass
class GearInsertionFractionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = GearInsertionFractionRecorder
    name: str = "gear_insertion_fraction"
    gear_names: tuple[str, ...] = MISSING


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

    def __init__(self, gear_names: tuple[str, ...]):
        self.gear_names = gear_names

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return GearInsertionFractionRecorderCfg(name=self.recorder_term_name, gear_names=self.gear_names)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=compute_gear_insertion_fraction,
            params={},
            recorder_term_name=self.recorder_term_name,
        )
