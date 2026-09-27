# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Current Isaac CAP cable layouts expressed with Arena's native Cable asset."""

from __future__ import annotations

import math
import yaml
from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg, NewtonMaterialPropertiesCfg

from isaaclab_arena.assets.cable import Cable
from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

from .distribution import anchor_pose_from_cable, sample_cable_routing_layout

_PACKAGE_DIRECTORY = Path(__file__).resolve().parent
_MEDIUM_SCENE_SPEC = _PACKAGE_DIRECTORY / "cable_routing_medium.yaml"
_EASY_SCENE_SPEC = _PACKAGE_DIRECTORY / "cable_routing_easy.yaml"

_ASSET_ROOT = f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/latest/cable_routing/assets"

YAM_I2RT_USD_PATH = f"{_ASSET_ROOT}/yam_i2rt/yam_i2rt.usda"
TABLE_USD_PATH = f"{_ASSET_ROOT}/industrial__yam_workcell_table/industrial__yam_workcell_table.usda"
BOARD_USD_PATH = f"{_ASSET_ROOT}/industrial__cable_routing_board/board.usdc"
ROUND_PEG_USD_PATH = f"{_ASSET_ROOT}/industrial__cable_routing_peg/round_peg.usdc"
EASY_ROUND_PEG_USD_PATH = f"{_ASSET_ROOT}/industrial__cable_routing_peg/orange_round_peg.usda"
TERMINAL_ASSET_DIRECTORY = f"{_ASSET_ROOT}/industrial__cable_routing_terminals"
ANCHOR_USD_PATH = f"{TERMINAL_ASSET_DIRECTORY}/anchor.usda"
PORT_USD_PATH = f"{TERMINAL_ASSET_DIRECTORY}/port.usda"
HDR_SHADOW_RECEIVER_USD_PATH = f"{_ASSET_ROOT}/industrial__hdr_shadow_receiver/industrial__hdr_shadow_receiver.usda"
NATIVE_APPEARANCE_USD_PATH = f"{_ASSET_ROOT}/native_appearance/room.usda"

_MEDIUM_SCENE_LAYOUT = yaml.safe_load(_MEDIUM_SCENE_SPEC.read_text())
_EASY_SCENE_LAYOUT = yaml.safe_load(_EASY_SCENE_SPEC.read_text())
_EMBODIMENT_MIDPOINT = tuple(
    float(value) for value in _MEDIUM_SCENE_LAYOUT["embodiment_frame"]["midpoint_position_xyz"]
)
_FIXTURES = _MEDIUM_SCENE_LAYOUT["fixtures"]
TABLE_FRAME_POSITION = tuple(float(value) for value in _FIXTURES["table"]["position_xyz"])
TABLE_FRAME_ROTATION = tuple(float(value) for value in _FIXTURES["table"]["rotation_xyzw"])
TABLE_TOP_Z = float(_FIXTURES["table"]["tabletop_z"])
TABLE_CENTER_X = _EMBODIMENT_MIDPOINT[0] + 0.3475
BOARD_POSITION = tuple(float(value) for value in _FIXTURES["board"]["position_xyz"])
BOARD_THICKNESS = 0.00635
BOARD_TOP_Z = TABLE_TOP_Z + BOARD_THICKNESS
PEG_HEIGHT = 0.0235


@dataclass(frozen=True)
class CablePhysics:
    """Physical parameters for one procedural cable variant."""

    radius: float
    density: float
    stretch_stiffness: float
    stretch_damping: float
    bend_stiffness: float
    bend_damping: float
    cable_friction: float
    fixture_friction: float
    table_friction: float
    contact_stiffness: float
    contact_damping: float
    contact_gap: float
    color: tuple[float, float, float]


@dataclass
class TerminatedCableGoal:
    """World-axis bounds relative to each fixture's origin."""

    seat_regions: tuple[tuple[float, float, float, float], ...]
    seat_min_fractions: tuple[float, ...]
    port_region: tuple[float, float, float, float]
    port_min_fraction: float
    min_tcp_distance: float
    max_mean_speed: float


@dataclass(frozen=True)
class CableRoutingVariant:
    """One sampled current-CAP cable layout and route objective."""

    name: str
    cable_local_positions: tuple[tuple[float, float, float], ...]
    peg_positions: tuple[tuple[float, float, float], ...]
    route_peg_indices: tuple[int, ...]
    route_directions: tuple[float, ...]
    task_description: str
    physics: CablePhysics
    table_collision_center: tuple[float, float, float]
    table_collision_size: tuple[float, float, float]
    anchor_position: tuple[float, float, float]
    anchor_rotation_xyzw: tuple[float, float, float, float]
    port_position: tuple[float, float, float]
    port_rotation_xyzw: tuple[float, float, float, float]
    terminated_goal: TerminatedCableGoal
    episode_length_s: float
    pin_start: bool = True

    @property
    def cable_num_segments(self) -> int:
        """Return the number of native cable segments."""
        return len(self.cable_local_positions) - 1


@dataclass(frozen=True)
class BuiltCableRoutingScene:
    """Composed Arena scene and task-facing cable fixtures."""

    scene: Scene
    cable: Cable
    pegs: tuple[Object, ...]
    port: Object


def _surface_position(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    return TABLE_CENTER_X + xyz[0], xyz[1], TABLE_TOP_Z + xyz[2]


def _yaw_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def _make_sampled_cable_positions(cable: dict) -> tuple[tuple[float, float, float], ...]:
    """Reproduce one Berkeley sample's bowed cable at its declared length."""
    length = float(cable["length_m"])
    segments = int(cable["segments"])
    start = tuple(float(value) for value in cable["start_surface_xyz"])
    shape = cable["shape"]

    def sampled_value(name: str, range_name: str) -> float:
        low, high = (float(value) for value in shape[range_name])
        fraction = float(shape[name])
        return 0.5 * (low + high) + 0.5 * (high - low) * fraction

    bow = sampled_value("bow_fraction", "bow_range_m")
    wave = sampled_value("wave_fraction", "wave_range_m")
    shift = sampled_value("shift_fraction", "shift_range_m")
    yaw = math.radians(sampled_value("yaw_fraction", "yaw_range_deg"))
    points = []
    for index in range(segments + 1):
        t = index / segments
        lateral = bow * math.sin(math.pi * t) + wave * math.sin(2.0 * math.pi * t) + shift
        points.append([start[0] - lateral, start[1] + length * t, start[2]])

    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    for point in points:
        rel_x, rel_y = point[0] - center_x, point[1] - center_y
        point[0] = center_x + rel_x * cos_yaw - rel_y * sin_yaw
        point[1] = center_y + rel_x * sin_yaw + rel_y * cos_yaw
    return tuple(tuple(point) for point in points)


def _terminated_variant(name: str, scene_layout: dict, layout_seed: int | None = None) -> CableRoutingVariant:
    distribution = scene_layout["distribution"]
    if layout_seed is None:
        layout_seed = int(distribution["default_seed"])
    sampled = sample_cable_routing_layout(distribution, layout_seed)
    table = scene_layout["fixtures"]["table"]
    peg_positions = tuple(_surface_position((x, y, 0.5 * PEG_HEIGHT)) for x, y in sampled.peg_positions_xy)
    source_cable = scene_layout["cable"]
    cable = {
        **source_cable,
        "shape": {
            **source_cable["shape"],
            "bow_fraction": sampled.bow_fraction,
            "shift_fraction": sampled.shift_fraction,
            "yaw_fraction": sampled.yaw_fraction,
            "wave_fraction": sampled.wave_fraction,
        },
    }
    cable_positions = _make_sampled_cable_positions(cable)
    radius = float(cable["radius_m"])
    reference_pitch = 0.9144 / 28
    pitch = float(cable["length_m"]) / int(cable["segments"])
    bend_scale = (radius / 0.011) ** 4 * (reference_pitch / pitch)
    physics = CablePhysics(
        radius=radius,
        density=350.0,
        stretch_stiffness=2.0e5,
        stretch_damping=1.0,
        bend_stiffness=0.04 * bend_scale,
        bend_damping=0.06 * bend_scale,
        cable_friction=60.0,
        fixture_friction=2.0,
        table_friction=0.02,
        contact_stiffness=1.0e4,
        contact_damping=0.1,
        contact_gap=0.01,
        color=(0.88, 0.88, 0.87),
    )
    anchor_x, anchor_y, anchor_yaw = anchor_pose_from_cable(
        cable_positions, float(distribution["anchor_mouth_inset_m"])
    )
    port_x, port_y, port_yaw = sampled.port_pose_xyyaw
    goal = scene_layout["goal"]
    return CableRoutingVariant(
        name=name,
        cable_local_positions=cable_positions,
        peg_positions=peg_positions,
        route_peg_indices=tuple(range(len(peg_positions))),
        route_directions=tuple(float(value) for value in distribution["reference"]["route_sides"]),
        task_description=str(scene_layout["task_description"]),
        episode_length_s=float(scene_layout.get("max_episode_steps", 13000)) / 60.0,
        physics=physics,
        table_collision_center=_surface_position(
            tuple(float(value) for value in table["collision_center_surface_xyz"])
        ),
        table_collision_size=tuple(float(value) for value in table["collision_size_xyz"]),
        anchor_position=_surface_position((anchor_x, anchor_y, 0.0)),
        anchor_rotation_xyzw=_yaw_xyzw(anchor_yaw),
        port_position=_surface_position((port_x, port_y, 0.0)),
        port_rotation_xyzw=_yaw_xyzw(port_yaw),
        terminated_goal=TerminatedCableGoal(
            seat_regions=tuple(tuple(float(value) for value in region) for region in goal["seat_regions_xy"]),
            seat_min_fractions=tuple(float(value) for value in goal["seat_min_fractions"]),
            port_region=tuple(float(value) for value in goal["port_region_xy"]),
            port_min_fraction=float(goal["port_min_fraction"]),
            min_tcp_distance=float(goal["min_tcp_distance_m"]),
            max_mean_speed=float(goal["max_mean_speed_mps"]),
        ),
    )


def easy_variant(layout_seed: int | None = None) -> CableRoutingVariant:
    """Return one sampled current Easy layout."""
    return _terminated_variant("easy", _EASY_SCENE_LAYOUT, layout_seed)


def medium_variant(layout_seed: int | None = None) -> CableRoutingVariant:
    """Return one sampled current Medium layout."""
    return _terminated_variant("medium", _MEDIUM_SCENE_LAYOUT, layout_seed)


def _fixture_material(variant: CableRoutingVariant, friction: float) -> NewtonMaterialPropertiesCfg:
    return NewtonMaterialPropertiesCfg(
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
        contact_stiffness=variant.physics.contact_stiffness,
        contact_damping=variant.physics.contact_damping,
    )


def _fixture_spawn(variant: CableRoutingVariant, usd_path: str) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        copy_from_source=False,
        physics_material=_fixture_material(variant, variant.physics.fixture_friction),
        rigid_props=sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
        collision_props=NewtonCollisionPropertiesCfg(contact_margin=0.0, contact_gap=variant.physics.contact_gap),
    )


def build_cable_routing_scene(variant: CableRoutingVariant) -> BuiltCableRoutingScene:
    """Build the current CAP fixtures around an Arena-owned Cable."""
    table = Object(
        name="table",
        prim_path="{ENV_REGEX_NS}/Table",
        object_type=ObjectType.BASE,
        usd_path=str(TABLE_USD_PATH),
        initial_pose=Pose(position_xyz=TABLE_FRAME_POSITION, rotation_xyzw=TABLE_FRAME_ROTATION),
        spawn_cfg_addon={
            "copy_from_source": False,
            "visible": False,
            "make_uninstanceable": True,
            "physics_material": _fixture_material(variant, variant.physics.table_friction),
            "collision_props": NewtonCollisionPropertiesCfg(collision_enabled=False),
        },
        tags=["background", "table"],
    )
    table_collider = Object(
        name="table_collider",
        prim_path="{ENV_REGEX_NS}/TableCollider",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(
            size=variant.table_collision_size,
            visible=False,
            physics_material=_fixture_material(variant, variant.physics.table_friction),
            collision_props=NewtonCollisionPropertiesCfg(
                contact_margin=0.0,
                contact_gap=variant.physics.contact_gap,
            ),
        ),
        initial_pose=Pose(position_xyz=variant.table_collision_center),
        tags=["background", "table"],
    )
    pegs = tuple(
        Object(
            name=f"peg_{index}",
            prim_path=f"{{ENV_REGEX_NS}}/Peg{index}",
            object_type=ObjectType.RIGID,
            spawner_cfg=_fixture_spawn(variant, EASY_ROUND_PEG_USD_PATH),
            initial_pose=Pose(position_xyz=position),
            tags=["fixture", "peg"],
        )
        for index, position in enumerate(variant.peg_positions)
    )
    anchor = Object(
        name="anchor",
        prim_path="{ENV_REGEX_NS}/Anchor",
        object_type=ObjectType.RIGID,
        spawner_cfg=_fixture_spawn(variant, ANCHOR_USD_PATH),
        initial_pose=Pose(position_xyz=variant.anchor_position, rotation_xyzw=variant.anchor_rotation_xyzw),
        tags=["fixture", "terminal"],
    )
    port = Object(
        name="port",
        prim_path="{ENV_REGEX_NS}/Port",
        object_type=ObjectType.RIGID,
        spawner_cfg=_fixture_spawn(variant, PORT_USD_PATH),
        initial_pose=Pose(position_xyz=variant.port_position, rotation_xyzw=variant.port_rotation_xyzw),
        tags=["fixture", "terminal"],
    )

    nominal_pitch = math.dist(variant.cable_local_positions[0], variant.cable_local_positions[-1])
    nominal_pitch = max(nominal_pitch / variant.cable_num_segments, 1.0e-6)
    cross_section_area = math.pi * variant.physics.radius**2
    second_moment = math.pi * variant.physics.radius**4 / 4.0
    cable = Cable(
        name="cable",
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=sim_utils.CableCfg(
            positions=variant.cable_local_positions,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=variant.physics.color, roughness=0.9),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=2.0 * variant.physics.radius,
                density=variant.physics.density,
                stretch_stiffness=variant.physics.stretch_stiffness * nominal_pitch / cross_section_area,
                bend_stiffness=variant.physics.bend_stiffness * nominal_pitch / second_moment,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
        initial_pose=Pose(position_xyz=(TABLE_CENTER_X, 0.0, TABLE_TOP_Z)),
        tags=["deformable", "cable"],
    )
    native_appearance = Object(
        name="native_appearance",
        prim_path="/World/CableNativeAppearance",
        object_type=ObjectType.BASE,
        usd_path=str(NATIVE_APPEARANCE_USD_PATH),
        initial_pose=Pose(position_xyz=(-0.5875, 0.0, 0.017)),
        spawn_cfg_addon={"copy_from_source": False},
        asset_cfg_addon={"collision_group": -1},
        tags=["background", "visual"],
    )
    ground = Object(
        name="ground",
        prim_path="/World/GroundPlane",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.GroundPlaneCfg(
            visible=False,
            color=(0.20, 0.20, 0.20),
            physics_material=_fixture_material(variant, variant.physics.fixture_friction),
        ),
        asset_cfg_addon={"collision_group": -1},
    )
    ground_visual = Object(
        name="ground_visual",
        prim_path="/World/GroundVisual",
        object_type=ObjectType.BASE,
        usd_path=str(HDR_SHADOW_RECEIVER_USD_PATH),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0005)),
        spawn_cfg_addon={"copy_from_source": False, "visible": False},
        asset_cfg_addon={"collision_group": -1},
        tags=["background", "visual"],
    )
    scene = Scene(
        assets=[
            table,
            table_collider,
            *pegs,
            anchor,
            port,
            cable,
            native_appearance,
            ground,
            ground_visual,
        ]
    )
    return BuiltCableRoutingScene(scene=scene, cable=cable, pegs=pegs, port=port)


__all__ = [
    "BOARD_TOP_Z",
    "CablePhysics",
    "CableRoutingVariant",
    "TerminatedCableGoal",
    "YAM_I2RT_USD_PATH",
    "build_cable_routing_scene",
    "easy_variant",
    "medium_variant",
]
