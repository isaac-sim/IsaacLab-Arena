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
from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.metrics.success_rate import SuccessRateMetric, compute_success_rate
from isaaclab_arena_environments.spatial_memory.episode_state import NOT_SET, get_episode_state


class EpisodeFieldRecorder(RecorderTerm):
    """Records one per-episode value from the spatial-memory episode state just before an environment resets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.field = cfg.field

    def record_pre_reset(self, env_ids):
        episode = get_episode_state(self._env)
        if self.field == "time_to_success_s":
            success = self._env.termination_manager.get_term("success")[env_ids]
            steps = self._env.episode_length_buf[env_ids].float()
            value = torch.where(success, steps * self._env.step_dt, torch.full_like(steps, float("nan")))
        else:
            value = getattr(episode, self.field)[env_ids].float()
        return self.name, value


@configclass
class EpisodeFieldRecorderCfg(RecorderTermCfg):
    """Configuration of ``EpisodeFieldRecorder``."""

    class_type: type[RecorderTerm] = EpisodeFieldRecorder

    name: str = MISSING
    """Recorder term name."""

    field: str = MISSING
    """Episode attribute (or ``time_to_success_s``) to record."""


def compute_mean_time_to_success(recorded_metric_data: list[np.ndarray]) -> float:
    """Return the mean time to success in seconds over successful episodes (NaN when none succeeded)."""
    if len(recorded_metric_data) == 0:
        return float("nan")
    times = np.concatenate([np.asarray(data).reshape(-1) for data in recorded_metric_data])
    times = times[np.isfinite(times)]
    return float(np.mean(times)) if times.size else float("nan")


class _EpisodeFieldMetric(MetricBase):
    """Metric over one per-episode field of the episode state, recorded just before each reset."""

    field: str
    """Episode state attribute (or ``time_to_success_s``) to record."""

    compute: Callable[[list[np.ndarray]], float]
    """Function that reduces the recorded values of all episodes to the metric."""

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return EpisodeFieldRecorderCfg(name=self.recorder_term_name, field=self.field)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=type(self).compute, params={}, recorder_term_name=self.recorder_term_name
        )


class CollisionRateMetric(_EpisodeFieldMetric):
    """Fraction of episodes with at least one collision with geometry the task does not need."""

    name = "collision_rate"
    recorder_term_name = "spatial_memory_collided"
    field = "collided"
    compute = staticmethod(compute_success_rate)


class SearchedAgainRateMetric(_EpisodeFieldMetric):
    """Fraction of episodes in which the robot re-observed the memorized scene beyond the allowed glimpse."""

    name = "searched_again_rate"
    recorder_term_name = "spatial_memory_searched_again"
    field = "searched_again"
    compute = staticmethod(compute_success_rate)


class TimeToSuccessMetric(_EpisodeFieldMetric):
    """Mean episode time, in seconds, of successful episodes."""

    name = "time_to_success_s"
    recorder_term_name = "spatial_memory_time_to_success"
    field = "time_to_success_s"
    compute = staticmethod(compute_mean_time_to_success)


def record_spatial_memory_episode(env, env_id: int) -> dict[str, Any]:
    """Return the episode's JSONL record: layout parameters, collision details, visibility summary and phase timings."""
    episode = get_episode_state(env)
    step_dt = env.step_dt

    def seconds(step: torch.Tensor) -> float | None:
        value = int(step[env_id])
        return None if value == NOT_SET else value * step_dt

    layout_params = {key: value[env_id].tolist() for key, value in episode.layout_params.items()}
    hidden_violations = []
    for target_index, name in enumerate(episode.target_names):
        if (
            episode.hidden_target_mask[target_index]
            and int(episode.visible_steps_in_hidden_phase[env_id, target_index]) > 0
        ):
            hidden_violations.append(name)
    seen_before_hidden_phase = []
    for target_index, name in enumerate(episode.target_names):
        if bool(episode.visible_before_hidden_phase[env_id, target_index]):
            seen_before_hidden_phase.append(name)
    success_terms = {name: bool(value[env_id]) for name, value in episode.logic.success_terms(env, episode).items()}
    collided = bool(episode.collided[env_id])
    return {
        "spatial_memory": {
            "task": episode.logic.name,
            "difficulty": episode.logic.difficulty,
            "layout": layout_params,
            "collided": collided,
            "collision_steps": int(episode.collision_steps[env_id]),
            "first_collision": (
                {
                    "time_s": seconds(episode.first_collision_step),
                    "body": episode.first_collision_body[env_id],
                    "partner": episode.first_collision_partner[env_id],
                    "body_position_e": episode.first_collision_body_position_e[env_id].tolist(),
                }
                if collided
                else None
            ),
            "searched_again": bool(episode.searched_again[env_id]),
            "reobservation_s": int(episode.reobservation_steps[env_id]) * step_dt,
            "instruction_time_s": seconds(episode.instruction_reveal_step),
            "hidden_phase_start_s": seconds(episode.hidden_phase_start_step),
            "hidden_phase_s": int(episode.hidden_phase_steps[env_id]) * step_dt,
            "hidden_targets_seen_in_hidden_phase": hidden_violations,
            "targets_seen_before_hidden_phase": seen_before_hidden_phase,
            "visible_in_hidden_phase_s": {
                name: int(episode.visible_steps_in_hidden_phase[env_id, index]) * step_dt
                for index, name in enumerate(episode.target_names)
            },
            "static_max_displacement_m": float(episode.static_max_displacement_m[env_id]),
            "tracked_path_length_m": float(episode.tracked_path_length_m[env_id]),
            "success_terms_at_end": success_terms,
        }
    }


def get_suite_metrics() -> list[MetricBase]:
    """Return the metrics every spatial-memory task reports."""
    return [SuccessRateMetric(), CollisionRateMetric(), TimeToSuccessMetric(), SearchedAgainRateMetric()]
