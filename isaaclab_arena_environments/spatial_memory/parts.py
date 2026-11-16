# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Primitive scene parts: kinematic boards for fixtures and dynamic items for manipulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object


class PartKind(str, Enum):
    """Whether a part is static fixture geometry or a movable item."""

    BOARD = "board"
    """Kinematic box; moved only by the layout reset (and by scripted scene changes)."""

    ITEM = "item"
    """Dynamic rigid body the robot can grasp, push or knock over."""


class PartShape(str, Enum):
    """Primitive spawned for a part."""

    BOX = "box"
    CYLINDER = "cylinder"


PARKING_POSE_Z_M = -0.5
"""Height below the floor at which unused boards are parked; the floor does not collide with kinematic boards."""

PALETTE: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.12, 0.10),
    "green": (0.10, 0.65, 0.20),
    "blue": (0.12, 0.30, 0.85),
    "yellow": (0.95, 0.80, 0.10),
    "purple": (0.55, 0.20, 0.70),
    "orange": (0.95, 0.50, 0.10),
    "cyan": (0.10, 0.75, 0.80),
    "pink": (0.95, 0.45, 0.65),
    "white": (0.90, 0.90, 0.88),
    "wood": (0.62, 0.45, 0.28),
    "grey": (0.55, 0.55, 0.58),
    "dark": (0.25, 0.25, 0.28),
}
"""Named colours. Item colours double as their identity in instructions."""

ITEM_COLOURS = ("red", "green", "blue", "yellow", "purple", "orange", "cyan", "pink")
"""Distinct colours assigned, in order, to items that are referred to by colour."""


@dataclass(frozen=True)
class PartSpec:
    """A primitive part of a spatial-memory scene."""

    name: str
    """Scene key and prim name of the part."""

    kind: PartKind
    """Board (kinematic fixture geometry) or item (dynamic body)."""

    size_xyz: tuple[float, float, float]
    """Full extents in metres. A cylinder uses ``(diameter, diameter, height)``."""

    colour: str = "grey"
    """Key into ``PALETTE``."""

    shape: PartShape = PartShape.BOX
    """Primitive to spawn."""

    mass_kg: float = 0.1
    """Mass of an item; ignored for boards."""

    friction: float = 1.0
    """Static and dynamic friction coefficient."""

    @property
    def half_extents(self) -> tuple[float, float, float]:
        """Half extents of the part's bounding box in its own frame."""
        return (0.5 * self.size_xyz[0], 0.5 * self.size_xyz[1], 0.5 * self.size_xyz[2])


def board(name: str, size_xyz: tuple[float, float, float], colour: str = "wood") -> PartSpec:
    """Return a kinematic box part."""
    return PartSpec(name=name, kind=PartKind.BOARD, size_xyz=size_xyz, colour=colour)


def box_item(name: str, size_xyz: tuple[float, float, float], colour: str, mass_kg: float = 0.1) -> PartSpec:
    """Return a dynamic box item."""
    return PartSpec(name=name, kind=PartKind.ITEM, size_xyz=size_xyz, colour=colour, mass_kg=mass_kg)


def cylinder_item(name: str, diameter_m: float, height_m: float, colour: str, mass_kg: float = 0.1) -> PartSpec:
    """Return a dynamic upright cylinder item."""
    return PartSpec(
        name=name,
        kind=PartKind.ITEM,
        size_xyz=(diameter_m, diameter_m, height_m),
        colour=colour,
        shape=PartShape.CYLINDER,
        mass_kg=mass_kg,
    )


def make_part_object(spec: PartSpec, position_xyz: tuple[float, float, float]) -> Object:
    """Create the Arena object that spawns ``spec`` at ``position_xyz`` (environment frame).

    The layout reset event owns the pose of every part, so the object's own pose-reset event is disabled.
    """
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    is_board = spec.kind == PartKind.BOARD
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=is_board,
            disable_gravity=is_board,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass_kg if not is_board else 10.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=spec.friction, dynamic_friction=spec.friction, restitution=0.0
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=PALETTE[spec.colour]),
        activate_contact_sensors=not is_board,
    )
    if spec.shape == PartShape.CYLINDER:
        spawner_cfg = sim_utils.CylinderCfg(radius=0.5 * spec.size_xyz[0], height=spec.size_xyz[2], axis="Z", **common)
    else:
        spawner_cfg = sim_utils.CuboidCfg(size=spec.size_xyz, **common)
    part = Object(
        name=spec.name,
        prim_path="{ENV_REGEX_NS}/" + spec.name,
        object_type=ObjectType.RIGID,
        spawner_cfg=spawner_cfg,
        initial_pose=Pose(position_xyz=tuple(float(v) for v in position_xyz)),
    )
    part.disable_reset_pose()
    return part
