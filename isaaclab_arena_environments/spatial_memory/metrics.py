# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record contact, visibility and task outcomes for each completed spatial-memory episode."""

from __future__ import annotations

import torch
from typing import Any

from isaaclab_arena_environments.spatial_memory.episode_state import NOT_SET, get_episode_state


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
        "task": episode.logic.name,
        "time_to_success_s": (
            float(env.episode_length_buf[env_id]) * step_dt
            if env.termination_manager.get_term("success")[env_id]
            else None
        ),
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
