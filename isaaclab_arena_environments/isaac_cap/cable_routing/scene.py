# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Cap cable-routing layouts and Arena scene composition."""

from __future__ import annotations

import math
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg

from isaaclab_arena.assets.cable import Cable
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

from .physics import CONTACT_GAP, make_fixture_material

if TYPE_CHECKING:
    from isaaclab_arena.assets.registries import AssetRegistry, HDRImageRegistry

_PACKAGE_DIRECTORY = Path(__file__).resolve().parent
_MEDIUM_SCENE_SPEC = _PACKAGE_DIRECTORY / "cable_routing_medium.yaml"
_EASY_SCENE_SPEC = _PACKAGE_DIRECTORY / "cable_routing_easy.yaml"
_ASSET_ROOT = (
    "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/newton_envs/cap_envs/cable_routing/assets"
)
YAM_USD_PATH = f"{_ASSET_ROOT}/yam/i2rt_yam_cable_routing.usda"
YAM_INSTANCEABLE_USD_PATH = f"{_ASSET_ROOT}/yam/i2rt_yam_cable_routing_instanceable.usda"
_TABLE_USD_PATH = f"{_ASSET_ROOT}/industrial__yam_workcell_table/industrial__yam_workcell_table.usda"
_BOARD_USD_PATH = f"{_ASSET_ROOT}/industrial__cable_routing_board/board.usdc"
_ROUND_PEG_USD_PATH = f"{_ASSET_ROOT}/industrial__cable_routing_peg/round_peg.usdc"
_HDR_SHADOW_RECEIVER_USD_PATH = f"{_ASSET_ROOT}/industrial__hdr_shadow_receiver/industrial__hdr_shadow_receiver.usda"

_SCENE_LAYOUT = yaml.safe_load(_MEDIUM_SCENE_SPEC.read_text())
_EASY_SCENE_LAYOUT = yaml.safe_load(_EASY_SCENE_SPEC.read_text())
EMBODIMENT_MIDPOINT = tuple(float(value) for value in _SCENE_LAYOUT["embodiment_frame"]["midpoint_position_xyz"])
_FIXTURE_LAYOUT = _SCENE_LAYOUT["fixtures"]
TABLE_FRAME_POSITION = tuple(float(value) for value in _FIXTURE_LAYOUT["table"]["position_xyz"])
TABLE_FRAME_ROTATION = tuple(float(value) for value in _FIXTURE_LAYOUT["table"]["rotation_xyzw"])
BOARD_POSITION = tuple(float(value) for value in _FIXTURE_LAYOUT["board"]["position_xyz"])

TABLE_TOP_Z = float(_FIXTURE_LAYOUT["table"]["tabletop_z"])
TABLE_CENTER_X = EMBODIMENT_MIDPOINT[0] + 0.3475
BOARD_SIZE = (0.30, 0.40)
BOARD_THICKNESS = 0.00635
BOARD_TOP_Z = TABLE_TOP_Z + BOARD_THICKNESS
PEG_HEIGHT = 0.0235

_CABLE_LENGTH = 1.0
_CABLE_SEGMENT_LENGTH = 0.01
_CABLE_NUM_SEGMENTS = round(_CABLE_LENGTH / _CABLE_SEGMENT_LENGTH)
_CABLE_THICKNESS = 0.006
CABLE_RADIUS = 0.5 * _CABLE_THICKNESS
CABLE_CENTER_Z = BOARD_TOP_Z + CABLE_RADIUS + 0.002
_CABLE_DENSITY = 1200.0
_CABLE_TARGET_STRETCH_STIFFNESS = 2.0e5
_CABLE_TARGET_BEND_STIFFNESS = 0.02
_CABLE_CROSS_SECTION_AREA = math.pi * CABLE_RADIUS**2
_CABLE_SECOND_MOMENT_OF_AREA = math.pi * CABLE_RADIUS**4 / 4.0
_CABLE_STRETCH_MODULUS = _CABLE_TARGET_STRETCH_STIFFNESS * _CABLE_SEGMENT_LENGTH / _CABLE_CROSS_SECTION_AREA
_CABLE_BEND_MODULUS = _CABLE_TARGET_BEND_STIFFNESS * _CABLE_SEGMENT_LENGTH / _CABLE_SECOND_MOMENT_OF_AREA

_YAM_VISUAL_BASE_WIDTH = 0.20
_YAM_LATERAL_OFFSET = 0.5 * (BOARD_SIZE[1] + _YAM_VISUAL_BASE_WIDTH)
LEFT_YAM_POSITION = (EMBODIMENT_MIDPOINT[0], _YAM_LATERAL_OFFSET, EMBODIMENT_MIDPOINT[2])
RIGHT_YAM_POSITION = (EMBODIMENT_MIDPOINT[0], -_YAM_LATERAL_OFFSET, EMBODIMENT_MIDPOINT[2])


@dataclass(frozen=True)
class CableRoutingVariant:
    """One Cap cable layout and route objective."""

    name: str
    cable_local_positions: tuple[tuple[float, float, float], ...]
    peg_positions: tuple[tuple[float, float, float], ...]
    route_peg_indices: tuple[int, ...]
    route_directions: tuple[float, ...]
    task_description: str


def _make_neutral_rounded_cable_positions() -> tuple[tuple[float, float, float], ...]:
    """Return Cap's smooth, exact-segment-length medium cable curve."""
    corner_segments = 6
    corner_step = 0.5 * math.pi / corner_segments
    corner_radius = _CABLE_SEGMENT_LENGTH / (2.0 * math.sin(0.5 * corner_step))
    horizontal_segments = 18
    vertical_segments = 30
    half_horizontal = 0.5 * horizontal_segments * _CABLE_SEGMENT_LENGTH
    half_vertical = 0.5 * vertical_segments * _CABLE_SEGMENT_LENGTH
    positions = [(-half_horizontal, -half_vertical - corner_radius, 0.0)]

    def append_straight(heading: float, count: int) -> None:
        for _ in range(count):
            x, y, z = positions[-1]
            positions.append((
                x + _CABLE_SEGMENT_LENGTH * math.cos(heading),
                y + _CABLE_SEGMENT_LENGTH * math.sin(heading),
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

    assert len(positions) > _CABLE_NUM_SEGMENTS, "Rounded cable template is shorter than requested."
    return tuple(positions[: _CABLE_NUM_SEGMENTS + 1])


def _make_straight_cable_positions(
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    """Return exact-length local cable points between two env-frame endpoints."""
    distance = math.dist(start_xyz, end_xyz)
    segment_count = round(distance / _CABLE_SEGMENT_LENGTH)
    assert segment_count > 0 and math.isclose(
        distance,
        segment_count * _CABLE_SEGMENT_LENGTH,
        abs_tol=1.0e-9,
    ), f"Cable length {distance} must be a positive multiple of {_CABLE_SEGMENT_LENGTH} m."
    return tuple(
        (
            start_xyz[0] + (end_xyz[0] - start_xyz[0]) * index / segment_count - TABLE_CENTER_X,
            start_xyz[1] + (end_xyz[1] - start_xyz[1]) * index / segment_count,
            start_xyz[2] + (end_xyz[2] - start_xyz[2]) * index / segment_count - CABLE_CENTER_Z,
        )
        for index in range(segment_count + 1)
    )


def _medium_variant() -> CableRoutingVariant:
    peg_positions = tuple(
        tuple(float(value) for value in _FIXTURE_LAYOUT[name]["position_xyz"]) for name in ("peg_0", "peg_1")
    )
    return CableRoutingVariant(
        name="medium",
        cable_local_positions=_make_neutral_rounded_cable_positions(),
        peg_positions=peg_positions,
        route_peg_indices=(0, 1),
        route_directions=(-1.0, 1.0),
        task_description=(
            "Route the cable counterclockwise around the first peg, then clockwise around the second peg, "
            "using both YAM manipulators."
        ),
    )


def _easy_variant() -> CableRoutingVariant:
    fixtures = _EASY_SCENE_LAYOUT["fixtures"]
    assert tuple(_EASY_SCENE_LAYOUT["embodiment_frame"]["midpoint_position_xyz"]) == EMBODIMENT_MIDPOINT
    assert tuple(fixtures["table"]["position_xyz"]) == TABLE_FRAME_POSITION
    assert tuple(fixtures["table"]["rotation_xyzw"]) == TABLE_FRAME_ROTATION
    assert tuple(fixtures["board"]["position_xyz"]) == BOARD_POSITION
    assert float(fixtures["table"]["tabletop_z"]) == TABLE_TOP_Z
    peg_names = ("peg_0", "peg_1", "peg_2")
    peg_positions = tuple(tuple(float(value) for value in fixtures[name]["position_xyz"]) for name in peg_names)
    cable = _EASY_SCENE_LAYOUT["cable"]
    route = _EASY_SCENE_LAYOUT["route"]
    assert route["peg"] in peg_names, f"Unknown easy route peg: {route['peg']!r}."
    assert route["direction"] == "either", "The easy route must accept either winding direction."
    return CableRoutingVariant(
        name="easy",
        cable_local_positions=_make_straight_cable_positions(
            tuple(float(value) for value in cable["start_xyz"]),
            tuple(float(value) for value in cable["end_xyz"]),
        ),
        peg_positions=peg_positions,
        route_peg_indices=(peg_names.index(route["peg"]),),
        route_directions=(0.0,),
        task_description=(
            "Route the initially straight cable around the middle of three pegs using the YAM manipulators."
        ),
    )


MEDIUM_VARIANT = _medium_variant()
EASY_VARIANT = _easy_variant()


@dataclass(frozen=True)
class BuiltCableRoutingScene:
    """Composed Arena scene and task-facing cable fixtures."""

    scene: Scene
    cable: Cable
    pegs: tuple[Object, ...]


def _fixture_spawn_cfg() -> dict:
    return {
        "copy_from_source": False,
        "physics_material": make_fixture_material(),
        "rigid_props": sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
        "collision_props": NewtonCollisionPropertiesCfg(contact_margin=0.0, contact_gap=2.0 * CONTACT_GAP),
    }


def build_cable_routing_scene(
    asset_registry: AssetRegistry,
    hdr_registry: HDRImageRegistry,
    variant: CableRoutingVariant,
) -> BuiltCableRoutingScene:
    """Build the Cap table, board, pegs, and a native Isaac Lab cable."""
    table = Object(
        name="table",
        prim_path="{ENV_REGEX_NS}/Table",
        usd_path=_TABLE_USD_PATH,
        object_type=ObjectType.BASE,
        initial_pose=Pose(position_xyz=TABLE_FRAME_POSITION, rotation_xyzw=TABLE_FRAME_ROTATION),
        spawn_cfg_addon={"copy_from_source": False},
        tags=["background", "table"],
    )
    board = Object(
        name="board",
        prim_path="{ENV_REGEX_NS}/Board",
        usd_path=_BOARD_USD_PATH,
        object_type=ObjectType.RIGID,
        initial_pose=Pose(position_xyz=BOARD_POSITION),
        spawn_cfg_addon=_fixture_spawn_cfg(),
        tags=["fixture", "board"],
    )
    pegs = tuple(
        Object(
            name=f"peg_{index}",
            prim_path=f"{{ENV_REGEX_NS}}/Peg{index}",
            usd_path=_ROUND_PEG_USD_PATH,
            object_type=ObjectType.RIGID,
            initial_pose=Pose(position_xyz=position),
            spawn_cfg_addon=_fixture_spawn_cfg(),
            tags=["fixture", "peg"],
        )
        for index, position in enumerate(variant.peg_positions)
    )
    for fixture in (board, *pegs):
        fixture.disable_reset_pose()

    cable = Cable(
        name="cable",
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=sim_utils.CableCfg(
            positions=variant.cable_local_positions,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.07, 0.07, 0.08)),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=_CABLE_THICKNESS,
                density=_CABLE_DENSITY,
                stretch_stiffness=_CABLE_STRETCH_MODULUS,
                bend_stiffness=_CABLE_BEND_MODULUS,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
        initial_pose=Pose(position_xyz=(TABLE_CENTER_X, 0.0, CABLE_CENTER_Z)),
        tags=["deformable", "cable"],
    )
    ground = asset_registry.get_asset_by_name("ground_plane")(
        instance_name="ground",
        spawner_cfg=sim_utils.GroundPlaneCfg(
            visible=False,
            color=(0.20, 0.20, 0.20),
            physics_material=make_fixture_material(),
        ),
    )
    ground_visual = Object(
        name="ground_visual",
        prim_path="/World/GroundVisual",
        usd_path=_HDR_SHADOW_RECEIVER_USD_PATH,
        object_type=ObjectType.BASE,
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0005)),
        spawn_cfg_addon={"copy_from_source": False, "visible": False},
        asset_cfg_addon={"collision_group": -1},
        tags=["floor", "visual"],
    )
    sky_light = asset_registry.get_asset_by_name("light")(
        instance_name="sky_light",
        prim_path="/World/skyLight",
    )
    sky_light.set_intensity(1500.0)
    sky_light.set_color((0.75, 0.75, 0.75))
    sky_light.add_hdr(hdr_registry.get_hdr_by_name("empty_warehouse_robolab")())

    scene = Scene(assets=[table, board, *pegs, cable, ground, ground_visual, sky_light])
    return BuiltCableRoutingScene(scene=scene, cable=cable, pegs=pegs)
