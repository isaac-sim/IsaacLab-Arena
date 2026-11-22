# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assets of the tilted display shelf task: a static tilted shelf, look-alike friction samples, a grip mat and a stop pin.

The shelf is a procedural kinematic rigid body whose board is tilted inside the asset, so its root stays level. The
three samples share one size and color and differ only in their physics material, which is the task's hidden
variable. With PhysX's default average friction combine and the shelf's bare friction of 0.30, the slick sample slides
on the bare shelf and on the mat, the medium sample slides on the bare shelf but holds on the mat, and the grippy sample
holds on the bare shelf.

Shelf asset frame A: Z up, origin on the table under the board's low (front) edge, +X up the slope, away from a robot
standing on the -X side. Board frame B: origin on the board's top surface at the low edge, +X (u) up the slope, +Z (n)
along the surface normal, Y (v) shared with A. Bays are numbered from +Y (the robot's left) to -Y.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


@dataclass(frozen=True)
class TiltedShelfGeometry:
    """Dimensions (m), tilt (deg) and surface friction of the procedural tilted display shelf."""

    tilt_deg: float = 24.0
    """Angle of the board above horizontal."""

    bay_count: int = 3
    bay_width_m: float = 0.11
    """Clear width of one bay between its dividers."""

    divider_thickness_m: float = 0.008
    divider_height_m: float = 0.015
    board_length_m: float = 0.17
    """Length of the board along the slope."""

    board_thickness_m: float = 0.025
    front_edge_height_m: float = 0.06
    """Height of the board's top surface at its low edge, above the asset origin."""

    pocket_offset_m: float = 0.032
    """Distance along the slope from the low edge to a stop-pin pocket's lower wall."""

    pocket_length_m: float = 0.014
    """Pocket size along the slope."""

    pocket_width_m: float = 0.02
    """Pocket size across the slope. Narrower than a sample, so a sample sliding over an empty pocket stays supported
    on both sides of it instead of dropping an edge into it."""

    pocket_depth_m: float = 0.02
    static_friction: float = 0.30
    dynamic_friction: float = 0.27

    @property
    def tilt_rad(self) -> float:
        return math.radians(self.tilt_deg)

    @property
    def board_width_m(self) -> float:
        return self.bay_count * self.bay_width_m + (self.bay_count + 1) * self.divider_thickness_m

    @property
    def bay_pitch_m(self) -> float:
        return self.bay_width_m + self.divider_thickness_m

    def bay_center_v_m(self, bay_index: int) -> float:
        """Return the board-frame Y of a bay's center; bay 0 is at +Y."""
        assert 0 <= bay_index < self.bay_count, f"bay_index must be in [0, {self.bay_count})"
        return ((self.bay_count - 1) / 2 - bay_index) * self.bay_pitch_m

    @property
    def pocket_center_u_m(self) -> float:
        return self.pocket_offset_m + self.pocket_length_m / 2

    def board_point_to_asset(self, u: float, v: float, n: float) -> tuple[float, float, float]:
        """Return the asset-frame position of a board-frame point (u, v, n)."""
        sin_t, cos_t = math.sin(self.tilt_rad), math.cos(self.tilt_rad)
        return (u * cos_t - n * sin_t, v, self.front_edge_height_m + u * sin_t + n * cos_t)


DEFAULT_TILTED_SHELF_GEOMETRY = TiltedShelfGeometry()

_SHELF_WOOD = (0.58, 0.42, 0.26)
_SHELF_EDGE = (0.36, 0.25, 0.15)
_SUPPORT_GREY = (0.30, 0.31, 0.33)


def _cached_usd_path(prefix: str, key: str, writer) -> str:
    """Return a cached USD path for a content key, writing it atomically on first use."""
    cache_dir = Path(os.environ.get("ISAACLAB_ARENA_PROCEDURAL_CACHE", Path(tempfile.gettempdir()) / "isaaclab_arena"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = cache_dir / f"{prefix}_{digest}.usda"
    if not path.exists():
        # Write to a temporary file first so parallel processes never read a partial file.
        tmp_path = path.with_suffix(f".{os.getpid()}.usda")
        writer(tmp_path)
        os.replace(tmp_path, path)
    return str(path)


def write_tilted_shelf_usd(path: Path, geometry: TiltedShelfGeometry = DEFAULT_TILTED_SHELF_GEOMETRY) -> None:
    """Write the tilted display shelf as a kinematic rigid body of box colliders.

    Args:
        path: Output USD path; overwritten if it exists.
        geometry: Dimensions, tilt and surface friction of the shelf.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    g = geometry
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/TiltedShelf").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)

    material = UsdShade.Material.Define(stage, "/TiltedShelf/PhysicsMaterials/shelf_surface")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(g.static_friction)
    material_api.CreateDynamicFrictionAttr(g.dynamic_friction)
    material_api.CreateRestitutionAttr(0.0)

    def add_box(parent: str, name: str, center, size, color) -> None:
        cube = UsdGeom.Cube.Define(stage, f"{parent}/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    # Board, in the tilted board frame B. Boxes are given as (u_min, u_max, v_min, v_max, n_min, n_max).
    board_path = "/TiltedShelf/board"
    board = UsdGeom.Xform.Define(stage, board_path)
    board.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, g.front_edge_height_m))
    # A rotation of -tilt about +Y maps +X (u) up the slope.
    board.AddRotateYOp().Set(-g.tilt_deg)

    def add_board_box(name: str, u_range, v_range, n_range, color=_SHELF_WOOD) -> None:
        center = ((u_range[0] + u_range[1]) / 2, (v_range[0] + v_range[1]) / 2, (n_range[0] + n_range[1]) / 2)
        size = (u_range[1] - u_range[0], v_range[1] - v_range[0], n_range[1] - n_range[0])
        add_box(board_path, name, center, size, color)

    half_width = g.board_width_m / 2
    full_n = (-g.board_thickness_m, 0.0)
    pocket_u = (g.pocket_offset_m, g.pocket_offset_m + g.pocket_length_m)
    add_board_box("front_strip", (0.0, pocket_u[0]), (-half_width, half_width), full_n, _SHELF_EDGE)
    add_board_box("slab", (pocket_u[1], g.board_length_m), (-half_width, half_width), full_n)
    # Pocket row: solid segments between the stop-pin pockets, and a floor under each pocket.
    pocket_v_ranges = []
    for bay_index in range(g.bay_count):
        center_v = g.bay_center_v_m(bay_index)
        pocket_v_ranges.append((center_v - g.pocket_width_m / 2, center_v + g.pocket_width_m / 2))
    pocket_v_ranges.sort()
    solid_v_start = -half_width
    for segment_index, (pocket_v_min, pocket_v_max) in enumerate(pocket_v_ranges):
        add_board_box(f"pocket_row_{segment_index}", pocket_u, (solid_v_start, pocket_v_min), full_n, _SHELF_EDGE)
        add_board_box(
            f"pocket_floor_{segment_index}",
            pocket_u,
            (pocket_v_min, pocket_v_max),
            (-g.board_thickness_m, -g.pocket_depth_m),
            _SUPPORT_GREY,
        )
        solid_v_start = pocket_v_max
    add_board_box(f"pocket_row_{len(pocket_v_ranges)}", pocket_u, (solid_v_start, half_width), full_n, _SHELF_EDGE)
    # Dividers along the slope at every bay boundary.
    for divider_index in range(g.bay_count + 1):
        divider_v = half_width - g.divider_thickness_m / 2 - divider_index * g.bay_pitch_m
        add_board_box(
            f"divider_{divider_index}",
            (0.0, g.board_length_m),
            (divider_v - g.divider_thickness_m / 2, divider_v + g.divider_thickness_m / 2),
            (0.0, g.divider_height_m),
            _SHELF_EDGE,
        )

    # Two support rails from the table up into the board's underside, set back from the low edge so a sliding sample
    # falls clear of them.
    sin_t, cos_t = math.sin(g.tilt_rad), math.cos(g.tilt_rad)

    def board_bottom_z(x: float) -> float:
        return g.front_edge_height_m - g.board_thickness_m * cos_t + (x - g.board_thickness_m * sin_t) * sin_t / cos_t

    for rail_name, x_min, x_max in (("front_rail", 0.065, 0.09), ("back_rail", 0.12, 0.145)):
        rail_top = board_bottom_z(x_max) + 0.002
        add_box(
            "/TiltedShelf",
            rail_name,
            ((x_min + x_max) / 2, 0.0, rail_top / 2),
            (x_max - x_min, g.board_width_m, rail_top),
            _SUPPORT_GREY,
        )
    stage.GetRootLayer().Save()


def get_tilted_shelf_usd_path(geometry: TiltedShelfGeometry = DEFAULT_TILTED_SHELF_GEOMETRY) -> str:
    """Return a cached USD path for the tilted shelf, writing the file on first use."""
    return _cached_usd_path("tilted_shelf", repr(geometry), lambda path: write_tilted_shelf_usd(path, geometry))


@register_asset
class TiltedDisplayShelf(Object):
    """Static display shelf with a board tilted 24 degrees, three bays and a stop-pin pocket at each bay's low edge.

    The USD is generated procedurally on first use as a kinematic rigid body; see the module docstring for its frames.
    """

    registry_name = "tilted_display_shelf"
    tags = ["object", "fixture", "procedural", "tilted_shelf_supports"]
    geometry = DEFAULT_TILTED_SHELF_GEOMETRY

    def __init__(self, name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None):
        """Create the shelf.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Pose of the asset frame; its rotation must be a yaw only.
        """
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            usd_path=get_tilted_shelf_usd_path(self.geometry),
            initial_pose=initial_pose,
            tags=list(self.tags),
            spawn_cfg_addon={"rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)},
        )


SHELF_SAMPLE_SIZE_M = (0.045, 0.045, 0.03)
SHELF_SAMPLE_MASS_KG = 0.08
_SAMPLE_COLOR = (0.30, 0.42, 0.55)


class ShelfSample(Object):
    """Display sample: a small box whose look is shared by every sample and whose friction is set per class."""

    tags = ["object", "graspable", "procedural", "tilted_shelf_supports"]
    static_friction: float
    dynamic_friction: float

    def __init__(self, name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None):
        """Create a sample.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
        """
        spawner_cfg = sim_utils.CuboidCfg(
            size=SHELF_SAMPLE_SIZE_M,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, sleep_threshold=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=SHELF_SAMPLE_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=self.static_friction, dynamic_friction=self.dynamic_friction, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_SAMPLE_COLOR, metallic=0.1, roughness=0.5),
        )
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawner_cfg,
            initial_pose=initial_pose,
            tags=list(self.tags),
        )


@register_asset
class ShelfSampleSlick(ShelfSample):
    """Display sample with slick friction (0.05 static, 0.04 dynamic); needs the stop pin on the tilted shelf."""

    registry_name = "tilted_shelf_sample_slick"
    static_friction = 0.05
    dynamic_friction = 0.04


@register_asset
class ShelfSampleMedium(ShelfSample):
    """Display sample with medium friction (0.45 static, 0.38 dynamic); holds on the grip mat, not on bare shelf."""

    registry_name = "tilted_shelf_sample_medium"
    static_friction = 0.45
    dynamic_friction = 0.38


@register_asset
class ShelfSampleGrippy(ShelfSample):
    """Display sample with grippy friction (0.90 static, 0.77 dynamic); holds on the bare tilted shelf."""

    registry_name = "tilted_shelf_sample_grippy"
    static_friction = 0.90
    dynamic_friction = 0.77


GRIP_MAT_SIZE_M = (0.09, 0.08, 0.006)
"""Grip mat size: along the slope when placed, across it, and thickness."""


@dataclass(frozen=True)
class StopPinGeometry:
    """Dimensions (m) and mass (kg) of the T-shaped stop pin: a crossbar standing on the board and a peg under it.

    Pin frame P: origin at the center of the crossbar's bottom face, +X along the slope when inserted, +Y across it,
    +Z along the board normal. The peg hangs below the origin into a shelf pocket.
    """

    bar_thickness_m: float = 0.012
    bar_width_m: float = 0.07
    bar_height_m: float = 0.025
    """Height the crossbar stands above the board: the stop face for a sample."""

    peg_thickness_m: float = 0.01
    peg_width_m: float = 0.014
    peg_length_m: float = 0.018
    mass_kg: float = 0.04


DEFAULT_STOP_PIN_GEOMETRY = StopPinGeometry()
_PIN_BRASS = (0.80, 0.55, 0.10)


def write_stop_pin_usd(path: Path, geometry: StopPinGeometry = DEFAULT_STOP_PIN_GEOMETRY) -> None:
    """Write the T-shaped stop pin as one rigid body of two box colliders.

    Args:
        path: Output USD path; overwritten if it exists.
        geometry: Dimensions and mass of the pin.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    g = geometry
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/ShelfStopPin").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    # Only the mass is authored; PhysX derives the center of mass and inertia from the colliders.
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(g.mass_kg)
    material = UsdShade.Material.Define(stage, "/ShelfStopPin/PhysicsMaterials/pin_metal")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(0.5)
    material_api.CreateDynamicFrictionAttr(0.4)
    material_api.CreateRestitutionAttr(0.0)
    for name, center, size in (
        ("crossbar", (0.0, 0.0, g.bar_height_m / 2), (g.bar_thickness_m, g.bar_width_m, g.bar_height_m)),
        ("peg", (0.0, 0.0, -g.peg_length_m / 2), (g.peg_thickness_m, g.peg_width_m, g.peg_length_m)),
    ):
        cube = UsdGeom.Cube.Define(stage, f"/ShelfStopPin/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*_PIN_BRASS)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    stage.GetRootLayer().Save()


def get_stop_pin_usd_path(geometry: StopPinGeometry = DEFAULT_STOP_PIN_GEOMETRY) -> str:
    """Return a cached USD path for the stop pin, writing the file on first use."""
    return _cached_usd_path("stop_pin", repr(geometry), lambda path: write_stop_pin_usd(path, geometry))


@register_asset
class GripMat(Object):
    """Thin rubber grip mat (0.75 static, 0.70 dynamic friction) that holds on the bare tilted shelf by itself."""

    registry_name = "grip_mat"
    tags = ["object", "graspable", "procedural", "tilted_shelf_supports"]

    def __init__(self, name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None):
        """Create the grip mat.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
        """
        spawner_cfg = sim_utils.CuboidCfg(
            size=GRIP_MAT_SIZE_M,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, sleep_threshold=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.75, dynamic_friction=0.70, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.09, 0.09), roughness=0.9),
        )
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawner_cfg,
            initial_pose=initial_pose,
            tags=list(self.tags),
        )


@register_asset
class ShelfStopPin(Object):
    """T-shaped stop pin: its peg fits a tilted-shelf pocket and its 7 cm crossbar stands 25 mm proud of the board.

    The USD is generated procedurally on first use; see StopPinGeometry for its frame and dimensions.
    """

    registry_name = "stop_pin"
    tags = ["object", "graspable", "procedural", "tilted_shelf_supports"]
    geometry = DEFAULT_STOP_PIN_GEOMETRY

    def __init__(self, name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None):
        """Create the stop pin.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
        """
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            usd_path=get_stop_pin_usd_path(self.geometry),
            initial_pose=initial_pose,
            tags=list(self.tags),
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1, sleep_threshold=0.0
                ),
                "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            },
        )
