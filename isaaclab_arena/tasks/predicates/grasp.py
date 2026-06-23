# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Contact / grasp predicates backed by Isaac Lab ``ContactSensor`` instances.

Grasp detection in Isaac Lab requires a pre-configured ``ContactSensor`` whose
filter prim is the gripper (or whatever body you want to detect contact with).
These predicates do not create sensors; they query sensors that the task author
already wired into the scene (the same sensors used by existing terminations
like ``object_on_destination``).

Naming convention: ``object_grabbed("foo")`` defaults to looking up the sensor
``"foo_contact_sensor"``. Pass ``contact_sensor_name="..."`` to override.
"""

from __future__ import annotations

import torch

import warp as wp
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor

from isaaclab_arena.tasks.predicates.decorators import atomic, composite
from isaaclab_arena.tasks.predicates.geometry import get_env, logical_and, logical_not, select
from isaaclab_arena.tasks.predicates.pose import object_stationary


def _contact_force_norm(env, sensor_name: str) -> torch.Tensor:
    e = get_env(env)
    sensor: ContactSensor = e.scene[sensor_name]
    force_matrix = wp.to_torch(sensor.data.force_matrix_w)
    return torch.linalg.norm(force_matrix, dim=-1).reshape(-1)


@atomic
def object_in_contact(
    env,
    contact_sensor_name: str,
    force_threshold: float = 0.1,
    env_id: int | None = None,
) -> torch.Tensor:
    """True when the contact-force magnitude on ``contact_sensor_name`` exceeds ``force_threshold``."""
    result = _contact_force_norm(env, contact_sensor_name) > force_threshold
    return select(result, env_id)


@composite
def object_grabbed(
    env,
    object_name: str,
    contact_sensor_name: str | None = None,
    force_threshold: float = 0.1,
    env_id: int | None = None,
) -> torch.Tensor:
    """True when ``object_name`` is in contact with its configured gripper sensor.

    Resolves ``contact_sensor_name`` from convention ``f"{object_name}_contact_sensor"``
    if not specified. The sensor must be filtered against the gripper body (or
    whichever body counts as "the gripper" for this task).
    """
    sensor_name = contact_sensor_name or f"{object_name}_contact_sensor"
    return object_in_contact(env, contact_sensor_name=sensor_name, force_threshold=force_threshold, env_id=env_id)


@composite
def object_settled_on(
    env,
    object_name: str,
    contact_sensor_name: str | None = None,
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
    env_id: int | None = None,
) -> torch.Tensor:
    """True when ``object_name`` is in stable contact with its desired surface."""
    sensor_name = contact_sensor_name or f"{object_name}_contact_sensor"
    in_contact = object_in_contact(env, contact_sensor_name=sensor_name, force_threshold=force_threshold, env_id=None)
    stationary = object_stationary(
        env,
        object_name=object_name,
        linear_threshold=velocity_threshold,
        check_angular=False,
        env_id=None,
    )
    return select(logical_and(in_contact, stationary), env_id)


@composite
def object_dropped(
    env,
    object_name: str,
    contact_sensor_name: str | None = None,
    force_threshold: float = 0.1,
    env_id: int | None = None,
) -> torch.Tensor:
    """True when ``object_name`` is NOT in contact with its gripper sensor.

    Note: this is the instantaneous negation. To track "was grasped then dropped"
    you can chain ``object_grabbed`` then ``object_dropped`` as conditions in a
    ``ProgressObjective`` — the state machine handles the sequencing.
    """
    grabbed = object_grabbed(
        env,
        object_name=object_name,
        contact_sensor_name=contact_sensor_name,
        force_threshold=force_threshold,
        env_id=None,
    )
    return select(logical_not(grabbed), env_id)
