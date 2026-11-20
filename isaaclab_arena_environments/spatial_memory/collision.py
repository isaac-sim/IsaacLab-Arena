# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Contact sensors of the collision check: the robot (or a carried item) touching scene geometry the task does not need.

PhysX reports per-partner forces only for a sensor on a single body, so the suite puts one filtered sensor on every
checked robot body, every hand body and every item, each filtered against every scene part. Robot self-contacts are
therefore never counted. Contacts the task needs (fingers on the grasped item, an item on its supports) are removed
with the task logic's allowed-contact mask.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena_environments.spatial_memory.parts import PartKind

if TYPE_CHECKING:
    from isaaclab_arena_environments.spatial_memory.task_logic import SpatialMemoryTaskLogic

CONTACT_FORCE_THRESHOLD_N = 0.5
"""Contact force above which two bodies count as touching."""


def filtered_sensor_name(body_label: str) -> str:
    """Return the scene key of the filtered contact sensor on ``body_label``."""
    return f"spatial_memory_contacts_{body_label}"


def make_contact_sensor_cfgs(logic: SpatialMemoryTaskLogic) -> dict:
    """Return the filtered contact sensor configs, keyed by scene key, of every robot body and item checked."""
    from isaaclab.sensors import ContactSensorCfg

    part_paths = ["{ENV_REGEX_NS}/" + spec.name for spec in logic.part_specs()]
    sensors = {}
    adapter = logic.embodiment
    for body_label in adapter.collision_bodies + adapter.hand_bodies:
        sensors[filtered_sensor_name(body_label)] = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + adapter.body_prim_paths.get(body_label, body_label),
            filter_prim_paths_expr=part_paths,
        )
    for spec in logic.part_specs():
        if spec.kind != PartKind.ITEM:
            continue
        sensors[filtered_sensor_name(spec.name)] = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/" + spec.name, filter_prim_paths_expr=part_paths
        )
    return sensors


def filtered_sensor_labels(logic: SpatialMemoryTaskLogic) -> list[str]:
    """Return the labels of the bodies that carry a filtered contact sensor: robot bodies, hand bodies, then items."""
    labels = list(logic.embodiment.collision_bodies) + list(logic.embodiment.hand_bodies)
    labels += [spec.name for spec in logic.part_specs() if spec.kind == PartKind.ITEM]
    return labels
