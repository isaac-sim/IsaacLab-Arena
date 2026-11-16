# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Suite metrics: collision rate, time to success and searched-again rate, plus per-episode JSONL fields.

Success rate reuses Arena's ``SuccessRateMetric``.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import MISSING
from typing import Any

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena_environments.spatial_memory.runtime import NOT_SET, get_runtime


class RuntimeFieldRecorder(RecorderTerm):
    """Records one per-episode value from the spatial-memory runtime just before an environment resets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.field = cfg.field

    def record_pre_reset(self, env_ids):
        runtime = get_runtime(self._env)
        if self.field == "time_to_success_s":
            success = self._env.termination_manager.get_term("success")[env_ids]
            steps = self._env.episode_length_buf[env_ids].float()
            value = torch.where(success, steps * self._env.step_dt, torch.full_like(steps, float("nan")))
        else:
            value = getattr(runtime, self.field)[env_ids].float()
        return self.name, value


@configclass
class RuntimeFieldRecorderCfg(RecorderTermCfg):
    """Configuration of ``RuntimeFieldRecorder``."""

    class_type: type[RecorderTerm] = RuntimeFieldRecorder
    name: str = MISSING
    """Recorder term name."""
    field: str = MISSING
    """Runtime attribute (or ``time_to_success_s``) to record."""


def compute_episode_rate(recorded_metric_data: list[np.ndarray]) -> float:
    """Return the fraction of episodes whose recorded flag is set."""
    if len(recorded_metric_data) == 0:
        return 0.0
    flags = np.concatenate([np.asarray(data).reshape(-1) for data in recorded_metric_data])
    return float(np.mean(flags > 0.5))


def compute_mean_time_to_success(recorded_metric_data: list[np.ndarray]) -> float:
    """Return the mean time to success in seconds over successful episodes (NaN when none succeeded)."""
    if len(recorded_metric_data) == 0:
        return float("nan")
    times = np.concatenate([np.asarray(data).reshape(-1) for data in recorded_metric_data])
    times = times[np.isfinite(times)]
    return float(np.mean(times)) if times.size else float("nan")


class _RuntimeFieldMetric(MetricBase):
    field: str
    compute: Any

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return RuntimeFieldRecorderCfg(name=self.recorder_term_name, field=self.field)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=type(self).compute, params={}, recorder_term_name=self.recorder_term_name
        )


class CollisionRateMetric(_RuntimeFieldMetric):
    """Fraction of episodes with at least one collision with geometry the task does not need."""

    name = "collision_rate"
    recorder_term_name = "spatial_memory_collided"
    field = "collided"
    compute = staticmethod(compute_episode_rate)


class SearchedAgainRateMetric(_RuntimeFieldMetric):
    """Fraction of episodes in which the robot re-observed the memorised scene beyond the allowed glimpse."""

    name = "searched_again_rate"
    recorder_term_name = "spatial_memory_searched_again"
    field = "searched_again"
    compute = staticmethod(compute_episode_rate)


class TimeToSuccessMetric(_RuntimeFieldMetric):
    """Mean episode time, in seconds, of successful episodes."""

    name = "time_to_success_s"
    recorder_term_name = "spatial_memory_time_to_success"
    field = "time_to_success_s"
    compute = staticmethod(compute_mean_time_to_success)


def record_spatial_memory_episode(env, env_id: int) -> dict[str, Any]:
    """Per-episode JSONL fields: layout parameters, collision details, visibility summary and phase timings."""
    runtime = get_runtime(env)
    step_dt = env.step_dt

    def seconds(step: torch.Tensor) -> float | None:
        value = int(step[env_id])
        return None if value == NOT_SET else value * step_dt

    params = {}
    for key, value in runtime.params.items():
        params[key] = value[env_id].tolist()
    hidden_violations = []
    for target_index, name in enumerate(runtime.target_names):
        if runtime.hidden_target_mask[target_index] and int(runtime.visible_steps_in_action[env_id, target_index]) > 0:
            hidden_violations.append(name)
    seen_before_action = [
        name
        for target_index, name in enumerate(runtime.target_names)
        if bool(runtime.visible_before_action[env_id, target_index])
    ]
    success_terms = {name: bool(value[env_id]) for name, value in runtime.logic.success_terms(env, runtime).items()}
    collided = bool(runtime.collided[env_id])
    return {
        "spatial_memory": {
            "task": runtime.logic.name,
            "difficulty": runtime.logic.difficulty,
            "layout": params,
            "collided": collided,
            "collision_steps": int(runtime.collision_steps[env_id]),
            "first_collision": (
                {
                    "time_s": seconds(runtime.first_collision_step),
                    "body": runtime.first_collision_body[env_id],
                    "partner": runtime.first_collision_partner[env_id],
                    "position_e": runtime.first_collision_position_e[env_id].tolist(),
                }
                if collided
                else None
            ),
            "searched_again": bool(runtime.searched_again[env_id]),
            "reobservation_s": int(runtime.reobservation_steps[env_id]) * step_dt,
            "instruction_time_s": seconds(runtime.reveal_step),
            "action_phase_start_s": seconds(runtime.action_phase_start_step),
            "action_phase_s": int(runtime.action_phase_steps[env_id]) * step_dt,
            "hidden_targets_seen_in_action_phase": hidden_violations,
            "targets_seen_before_action_phase": seen_before_action,
            "visible_in_action_phase_s": {
                name: int(runtime.visible_steps_in_action[env_id, index]) * step_dt
                for index, name in enumerate(runtime.target_names)
            },
            "static_max_displacement_m": float(runtime.static_max_displacement_m[env_id]),
            "tool_path_length_m": float(runtime.path_length_m[env_id]),
            "success_terms_at_end": success_terms,
        }
    }


def get_suite_metrics() -> list[MetricBase]:
    """Return the metrics every spatial-memory task reports."""
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric

    return [SuccessRateMetric(), CollisionRateMetric(), TimeToSuccessMetric(), SearchedAgainRateMetric()]
