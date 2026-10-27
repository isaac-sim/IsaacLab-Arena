# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode-end measurements for diagnosing pick-and-place success."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.tasks.predicates.spatial import object_on_destination_checks

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def record_placement_conditions(
    env: IsaacLabArenaManagerBasedRLEnv,
    env_id: int,
    object_cfg: SceneEntityCfg,
    destination_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg | None,
    force_threshold: float,
    velocity_threshold: float,
    support_cone_half_angle_rad: float = math.pi / 4,
) -> dict[str, Any]:
    """Record final placement checks before resetting a finishing environment.

    Args:
        env: Environment whose termination and progress managers have completed the current step.
        env_id: Finishing environment index.
        object_cfg: Object being placed.
        destination_cfg: Destination object.
        contact_sensor_cfg: Filtered contact sensor, or None for deformables.
        force_threshold: Minimum upward support force in newtons.
        velocity_threshold: Maximum linear speed in meters per second.
        support_cone_half_angle_rad: Maximum support-force angle from world +Z.

    Returns:
        JSON-serializable placement measurements and termination results.
    """
    checks = object_on_destination_checks(
        env,
        object_cfg,
        destination_cfg,
        contact_sensor_cfg,
        force_threshold,
        velocity_threshold,
        support_cone_half_angle_rad,
    )
    conditions = {name: value[env_id].detach().cpu().tolist() for name, value in checks.items()}
    conditions["velocity_threshold_m_s"] = velocity_threshold
    conditions["force_threshold_N"] = force_threshold
    conditions["object"] = object_cfg.name
    conditions["destination"] = destination_cfg.name
    bounds = env.arena_world.get_aabb_w(destination_cfg.name)
    conditions["destination_min_W"] = bounds.min_point[env_id].cpu().tolist()
    conditions["destination_max_W"] = bounds.max_point[env_id].cpu().tolist()
    if object_cfg.name in env.scene.deformable_objects:
        data = env.scene.deformable_objects[object_cfg.name].data
        conditions["nodal_positions_W"] = data.nodal_pos_w.torch[env_id].cpu().tolist()
        conditions["nodal_velocities_W"] = data.nodal_vel_w.torch[env_id].cpu().tolist()
    termination_terms = {
        name: bool(env.termination_manager.get_term(name)[env_id].item())
        for name in env.termination_manager.active_terms
    }
    summary = {name: value for name, value in conditions.items() if not name.startswith("nodal_")}
    print(
        "[placement_conditions] "
        + json.dumps({
            "env_id": env_id,
            "episode_in_env": env.get_episode_index(env_id),
            "episode_step": int(env.episode_length_buf[env_id]),
            "termination_terms": termination_terms,
            **summary,
        }),
        flush=True,
    )
    return {"placement_conditions": conditions, "termination_terms": termination_terms}
