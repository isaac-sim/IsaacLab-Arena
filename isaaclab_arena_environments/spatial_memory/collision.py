# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collision term: contacts between the robot (or a carried item) and scene geometry that the task does not need.

PhysX reports per-partner forces only for a sensor on a single body, so the suite uses:

* one unfiltered sensor over the arm links: any contact there is a collision;
* one filtered sensor per gripper body and per item, against every scene part. Contacts that the task needs (fingers
  on the grasped item, an item on its supports) are removed with an allowed-contact mask from the task logic.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena_environments.spatial_memory.parts import PartKind

if TYPE_CHECKING:
    from isaaclab_arena_environments.spatial_memory.runtime import SpatialMemoryRuntime
    from isaaclab_arena_environments.spatial_memory.task_logic import SpatialMemoryTaskLogic

ARM_SENSOR_NAME = "sm_arm_contacts"
"""Scene key of the unfiltered arm-link contact sensor."""

CONTACT_FORCE_THRESHOLD_N = 0.5
"""Contact force above which two bodies count as touching."""


def filtered_sensor_name(body_label: str) -> str:
    """Return the scene key of the filtered contact sensor on ``body_label``."""
    return f"sm_contacts_{body_label}"


def make_contact_sensor_cfgs(logic: SpatialMemoryTaskLogic) -> dict:
    """Return the contact sensor configs, keyed by scene key, that the collision term needs."""
    from isaaclab.sensors import ContactSensorCfg

    part_paths = ["{ENV_REGEX_NS}/" + spec.name for spec in logic.part_specs()]
    sensors = {}
    adapter = logic.embodiment
    if adapter.arm_link_regex is not None:
        sensors[ARM_SENSOR_NAME] = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/" + adapter.arm_link_regex)
    for body_label in adapter.gripper_bodies:
        sensors[filtered_sensor_name(body_label)] = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + body_label, filter_prim_paths_expr=part_paths
        )
    for spec in logic.part_specs():
        if spec.kind != PartKind.ITEM:
            continue
        sensors[filtered_sensor_name(spec.name)] = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/" + spec.name, filter_prim_paths_expr=part_paths
        )
    return sensors


def filtered_sensor_labels(logic: SpatialMemoryTaskLogic) -> list[str]:
    """Return the body labels that carry a filtered contact sensor, gripper bodies first, then items."""
    labels = list(logic.embodiment.gripper_bodies)
    labels += [spec.name for spec in logic.part_specs() if spec.kind == PartKind.ITEM]
    return labels


def detect_collisions(env, runtime: SpatialMemoryRuntime, active: torch.Tensor) -> torch.Tensor:
    """Update the runtime's collision bookkeeping from the contact sensors and return this step's collision flags."""
    logic = runtime.logic
    colliding = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    body_labels: list[list[str]] = [[] for _ in range(env.num_envs)]
    partner_labels: list[list[str]] = [[] for _ in range(env.num_envs)]
    positions_w = torch.zeros(env.num_envs, 3, device=env.device)

    if ARM_SENSOR_NAME in env.scene.keys():
        arm_sensor = env.scene[ARM_SENSOR_NAME]
        arm_forces = arm_sensor.data.net_forces_w.torch.norm(dim=-1)
        arm_touching = arm_forces > CONTACT_FORCE_THRESHOLD_N
        arm_any = arm_touching.any(dim=-1)
        if arm_any.any():
            strongest = arm_forces.argmax(dim=-1)
            robot = env.scene["robot"]
            for env_id in torch.nonzero(arm_any).flatten().tolist():
                body_name = arm_sensor.body_names[int(strongest[env_id])]
                body_labels[env_id].append(body_name)
                partner_labels[env_id].append("scene")
                positions_w[env_id] = robot.data.body_link_pose_w.torch[
                    env_id, robot.data.body_names.index(body_name), :3
                ]
        colliding |= arm_any

    labels = filtered_sensor_labels(logic)
    allowed = logic.allowed_contacts(env, runtime)  # (N, S, K)
    for sensor_index, label in enumerate(labels):
        sensor = env.scene[filtered_sensor_name(label)]
        forces = sensor.data.force_matrix_w.torch[:, 0].norm(dim=-1)  # (N, K)
        touching = (forces > CONTACT_FORCE_THRESHOLD_N) & ~allowed[:, sensor_index]
        any_touching = touching.any(dim=-1)
        if any_touching.any():
            strongest = torch.where(touching, forces, torch.zeros_like(forces)).argmax(dim=-1)
            for env_id in torch.nonzero(any_touching).flatten().tolist():
                body_labels[env_id].append(label)
                partner_labels[env_id].append(runtime.part_names[int(strongest[env_id])])
                if label in runtime.part_index:
                    positions_w[env_id] = env.arena_world.get_pose_w(label)[env_id, :3]
                else:
                    positions_w[env_id] = runtime.get_robot_body_poses_w(env, [label])[env_id, 0, :3]
        colliding |= any_touching

    colliding &= active & logic.collisions_counted(env, runtime)
    first = colliding & (runtime.first_collision_step == -1)
    for env_id in torch.nonzero(first).flatten().tolist():
        runtime.first_collision_step[env_id] = int(env.episode_length_buf[env_id])
        runtime.first_collision_body[env_id] = body_labels[env_id][0]
        runtime.first_collision_partner[env_id] = partner_labels[env_id][0]
        runtime.first_collision_position_e[env_id] = positions_w[env_id] - env.scene.env_origins[env_id]
    runtime.collision_steps += colliding.long()
    runtime.collided |= colliding
    return colliding
