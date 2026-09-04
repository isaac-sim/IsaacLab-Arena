# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena scene assets and geometry for YAM cable routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.cable import Cable
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.yam_cable_routing.physics import make_fixture_material

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.registries import AssetRegistry

EMBODIMENT_MIDPOINT = (-0.335, 0.0, 0.767)
TABLE_FRAME_POSITION = (-0.335, 0.0, 0.767)
TABLE_FRAME_ROTATION = (0.0, 0.0, 0.0, 1.0)
BOARD_POSITION = (0.0125, 0.0, 0.770175)

TABLE_TOP_Z = 0.767
TABLE_CENTER_X = EMBODIMENT_MIDPOINT[0] + 0.3475
BOARD_SIZE = (0.30, 0.40)
BOARD_THICKNESS = 0.00635
BOARD_TOP_Z = TABLE_TOP_Z + BOARD_THICKNESS
PEG_HEIGHT = 0.0235
PEG_CENTER_Z = BOARD_TOP_Z + 0.5 * PEG_HEIGHT
PEG_POSITIONS = (
    (0.0575, -0.055, 0.7851),
    (-0.0225, 0.085, 0.7851),
)

YAM_BASE_Z = EMBODIMENT_MIDPOINT[2]
YAM_VISUAL_BASE_WIDTH = 0.20
YAM_FRONT_X = EMBODIMENT_MIDPOINT[0]
YAM_LATERAL_OFFSET = 0.5 * (BOARD_SIZE[1] + YAM_VISUAL_BASE_WIDTH)
LEFT_YAM_POSITION = (YAM_FRONT_X, YAM_LATERAL_OFFSET, YAM_BASE_Z)
RIGHT_YAM_POSITION = (YAM_FRONT_X, -YAM_LATERAL_OFFSET, YAM_BASE_Z)

CABLE_LENGTH = 1.0
CABLE_SEGMENT_LENGTH = 0.01
CABLE_NUM_SEGMENTS = round(CABLE_LENGTH / CABLE_SEGMENT_LENGTH)
CABLE_THICKNESS = 0.006
CABLE_RADIUS = 0.5 * CABLE_THICKNESS
CABLE_CENTER_Z = BOARD_TOP_Z + CABLE_RADIUS + 0.002
CABLE_DENSITY = 1200.0
CABLE_TARGET_STRETCH_STIFFNESS = 2.0e5
CABLE_TARGET_BEND_STIFFNESS = 0.02
CABLE_CROSS_SECTION_AREA = math.pi * CABLE_RADIUS**2
CABLE_SECOND_MOMENT_OF_AREA = math.pi * CABLE_RADIUS**4 / 4.0
CABLE_STRETCH_MODULUS = CABLE_TARGET_STRETCH_STIFFNESS * CABLE_SEGMENT_LENGTH / CABLE_CROSS_SECTION_AREA
CABLE_BEND_MODULUS = CABLE_TARGET_BEND_STIFFNESS * CABLE_SEGMENT_LENGTH / CABLE_SECOND_MOMENT_OF_AREA


def make_neutral_rounded_cable_positions() -> list[tuple[float, float, float]]:
    """Return a smooth, exact-segment-length cable initialization curve."""
    corner_segments = 6
    corner_step = 0.5 * math.pi / corner_segments
    corner_radius = CABLE_SEGMENT_LENGTH / (2.0 * math.sin(0.5 * corner_step))
    horizontal_segments = 18
    vertical_segments = 30
    half_horizontal = 0.5 * horizontal_segments * CABLE_SEGMENT_LENGTH
    half_vertical = 0.5 * vertical_segments * CABLE_SEGMENT_LENGTH
    positions = [(-half_horizontal, -half_vertical - corner_radius, 0.0)]

    def append_straight(heading: float, count: int) -> None:
        for _ in range(count):
            x, y, z = positions[-1]
            positions.append((
                x + CABLE_SEGMENT_LENGTH * math.cos(heading),
                y + CABLE_SEGMENT_LENGTH * math.sin(heading),
                z,
            ))

    def append_corner(center_x: float, center_y: float, start_angle: float) -> None:
        for step in range(1, corner_segments + 1):
            angle = start_angle + step * corner_step
            positions.append((
                center_x + corner_radius * math.cos(angle),
                center_y + corner_radius * math.sin(angle),
                0.0,
            ))

    append_straight(0.0, horizontal_segments)
    append_corner(half_horizontal, -half_vertical, -0.5 * math.pi)
    append_straight(0.5 * math.pi, vertical_segments)
    append_corner(half_horizontal, half_vertical, 0.0)
    append_straight(math.pi, horizontal_segments)
    append_corner(-half_horizontal, half_vertical, 0.5 * math.pi)
    append_straight(-0.5 * math.pi, vertical_segments)
    append_corner(-half_horizontal, -half_vertical, math.pi)

    assert len(positions) > CABLE_NUM_SEGMENTS, "Rounded cable template is shorter than the requested cable."
    return positions[: CABLE_NUM_SEGMENTS + 1]


@dataclass(frozen=True)
class YamCableRoutingScene:
    """Built scene plus the assets consumed by the cable-routing task."""

    scene: Scene
    cable: Cable
    pegs: tuple[Object, Object]


def build_yam_cable_routing_scene(asset_registry: AssetRegistry) -> YamCableRoutingScene:
    """Build the fixed table, board, two pegs, cable, ground, and light."""
    table = asset_registry.get_asset_by_name("yam_cable_routing_table")(
        instance_name="table",
        prim_path="{ENV_REGEX_NS}/Table",
        initial_pose=Pose(position_xyz=TABLE_FRAME_POSITION, rotation_xyzw=TABLE_FRAME_ROTATION),
    )
    board = asset_registry.get_asset_by_name("yam_cable_routing_board")(
        instance_name="board",
        prim_path="{ENV_REGEX_NS}/Board",
        initial_pose=Pose(position_xyz=BOARD_POSITION),
    )
    peg_type = asset_registry.get_asset_by_name("yam_cable_routing_round_peg")
    peg_0 = peg_type(
        instance_name="peg_0",
        prim_path="{ENV_REGEX_NS}/Peg0",
        initial_pose=Pose(position_xyz=PEG_POSITIONS[0]),
    )
    peg_1 = peg_type(
        instance_name="peg_1",
        prim_path="{ENV_REGEX_NS}/Peg1",
        initial_pose=Pose(position_xyz=PEG_POSITIONS[1]),
    )
    # CableRoutingTask restores the complete scene in one ordered reset event.
    for fixture in (board, peg_0, peg_1):
        fixture.disable_reset_pose()

    cable = Cable(
        name="cable",
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=sim_utils.CableCfg(
            positions=make_neutral_rounded_cable_positions(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.07, 0.07, 0.08)),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=CABLE_THICKNESS,
                density=CABLE_DENSITY,
                stretch_stiffness=CABLE_STRETCH_MODULUS,
                bend_stiffness=CABLE_BEND_MODULUS,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
        initial_pose=Pose(position_xyz=(TABLE_CENTER_X, 0.0, CABLE_CENTER_Z)),
        tags=["deformable", "cable"],
    )
    ground = asset_registry.get_asset_by_name("ground_plane")(
        instance_name="ground",
        spawner_cfg=sim_utils.GroundPlaneCfg(
            color=(0.20, 0.20, 0.20),
            physics_material=make_fixture_material(),
        ),
    )
    sky_light = asset_registry.get_asset_by_name("light")(
        instance_name="sky_light",
        prim_path="/World/skyLight",
    )
    sky_light.set_intensity(1500.0)
    sky_light.set_color((0.75, 0.75, 0.75))

    scene = Scene(assets=[table, board, peg_0, peg_1, cable, ground, sky_light])
    return YamCableRoutingScene(scene=scene, cable=cable, pegs=(peg_0, peg_1))
