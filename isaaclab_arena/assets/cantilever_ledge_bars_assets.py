# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Look-alike bars with hidden center-of-mass offsets, a clamp-on counterweight, and a ledge over a channel.

All three assets are procedural USD written with pxr. Each bar is one rigid body whose center of mass is authored
explicitly (MassAPI centerOfMass), so bars that look the same balance differently. The ledge fixture is static
collision geometry: a raised ledge, a channel beside it and a yellow line on the channel floor marking how far a bar
must reach.

Bar frame B: long axis along Y, red end cap at +Y, blue end cap at -Y, origin at the geometric center.
Ledge frame L: Z up, origin on the support surface below the midpoint of the ledge edge; the ledge extends toward +Y,
the channel toward -Y, and the yellow line runs along X at Y = -line_overhang_m.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import isaaclab.sim as sim_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset


@dataclass(frozen=True)
class CantileverBarGeometry:
    """Dimensions (m), mass (kg) and center-of-mass offset (m) of a cantilever bar."""

    length_m: float = 0.24
    width_m: float = 0.03
    thickness_m: float = 0.02
    end_cap_m: float = 0.012
    """Length of the colored cap at each end."""

    mass_kg: float = 0.2
    com_offset_m: float = 0.0
    """Center-of-mass offset along the bar from its center; positive toward the red (+Y) end."""


@dataclass(frozen=True)
class ClampCounterweightGeometry:
    """Dimensions (m) and mass (kg) of the clamp-on counterweight, a saddle that straddles a bar."""

    mass_kg: float = 0.14
    inner_gap_m: float = 0.04
    """Gap between the two cheeks; wider than the bar so the saddle drops over it."""

    cheek_thickness_m: float = 0.008
    cheek_drop_m: float = 0.014
    """Length of the cheeks below the plate; shorter than the bar is thick, so only the plate rests on the bar."""

    plate_thickness_m: float = 0.016
    length_along_bar_m: float = 0.04


@dataclass(frozen=True)
class CantileverLedgeGeometry:
    """Dimensions (m) of the ledge-and-channel fixture."""

    width_m: float = 0.28
    """Extent along X of the ledge and the channel."""

    ledge_depth_m: float = 0.25
    """Extent of the ledge top from its edge toward +Y."""

    ledge_height_m: float = 0.05
    channel_length_m: float = 0.24
    """Extent of the channel from the ledge edge toward -Y."""

    channel_floor_thickness_m: float = 0.004
    wall_thickness_m: float = 0.012
    line_overhang_m: float = 0.14
    """Distance from the ledge edge to the yellow line a bar's outer end must pass."""


DEFAULT_BAR_GEOMETRY = CantileverBarGeometry()
DEFAULT_COUNTERWEIGHT_GEOMETRY = ClampCounterweightGeometry()
DEFAULT_LEDGE_GEOMETRY = CantileverLedgeGeometry()
BAR_COM_OFFSETS_M = {"red_heavy": 0.05, "blue_heavy": -0.05, "centered": 0.0}
"""Center-of-mass offset of each bar class, toward the red end."""

_BAR_BODY = (0.78, 0.76, 0.70)
_RED = (0.80, 0.10, 0.08)
_BLUE = (0.10, 0.25, 0.80)
_STEEL = (0.30, 0.32, 0.35)
_BRASS = (0.72, 0.55, 0.22)
_CONCRETE = (0.56, 0.56, 0.58)
_CHANNEL = (0.13, 0.14, 0.16)
_YELLOW = (0.96, 0.80, 0.05)


def _add_box(
    stage: Usd.Stage,
    path: str,
    center: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    color: tuple[float, float, float],
    collide: bool = True,
) -> None:
    """Add a box at a center expressed in its parent's frame."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _bind_physics_material(
    stage: Usd.Stage, prim: Usd.Prim, static_friction: float, dynamic_friction: float, restitution: float = 0.0
) -> None:
    """Bind a physics material to a prim and the colliders below it."""
    material = UsdShade.Material.Define(stage, prim.GetPath().AppendChild("physics_material"))
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(static_friction)
    material_api.CreateDynamicFrictionAttr(dynamic_friction)
    material_api.CreateRestitutionAttr(restitution)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _create_stage(path: Path, root_path: str) -> tuple[Usd.Stage, Usd.Prim]:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)
    return stage, root


def write_cantilever_bar_usd(path: Path, geometry: CantileverBarGeometry = DEFAULT_BAR_GEOMETRY) -> None:
    """Write a cantilever bar rigid body, with its center of mass authored explicitly, to a USD file."""
    g = geometry
    stage, root = _create_stage(path, "/CantileverBar")
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(g.mass_kg)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, g.com_offset_m, 0.0))
    # Inertia of a uniform box of the bar's size, taken about the shifted center of mass.
    w, length, t = g.width_m, g.length_m, g.thickness_m
    mass_api.CreateDiagonalInertiaAttr(
        Gf.Vec3f(
            g.mass_kg * (length**2 + t**2) / 12, g.mass_kg * (w**2 + t**2) / 12, g.mass_kg * (w**2 + length**2) / 12
        )
    )
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
    body_length = length - 2 * g.end_cap_m
    _add_box(stage, "/CantileverBar/body", (0.0, 0.0, 0.0), (w, body_length, t), _BAR_BODY)
    cap_center_y = length / 2 - g.end_cap_m / 2
    _add_box(stage, "/CantileverBar/red_end", (0.0, cap_center_y, 0.0), (w, g.end_cap_m, t), _RED)
    _add_box(stage, "/CantileverBar/blue_end", (0.0, -cap_center_y, 0.0), (w, g.end_cap_m, t), _BLUE)
    _bind_physics_material(stage, root, static_friction=0.7, dynamic_friction=0.6)
    stage.GetRootLayer().Save()


def write_clamp_counterweight_usd(
    path: Path, geometry: ClampCounterweightGeometry = DEFAULT_COUNTERWEIGHT_GEOMETRY
) -> None:
    """Write the clamp-on counterweight rigid body; its origin is the center of the plate's underside."""
    g = geometry
    stage, root = _create_stage(path, "/ClampCounterweight")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(g.mass_kg)
    plate_width = g.inner_gap_m + 2 * g.cheek_thickness_m
    _add_box(
        stage,
        "/ClampCounterweight/plate",
        (0.0, 0.0, g.plate_thickness_m / 2),
        (plate_width, g.length_along_bar_m, g.plate_thickness_m),
        _STEEL,
    )
    cheek_center_x = g.inner_gap_m / 2 + g.cheek_thickness_m / 2
    for name, sign in (("cheek_left", 1.0), ("cheek_right", -1.0)):
        _add_box(
            stage,
            f"/ClampCounterweight/{name}",
            (sign * cheek_center_x, 0.0, -g.cheek_drop_m / 2),
            (g.cheek_thickness_m, g.length_along_bar_m, g.cheek_drop_m),
            _STEEL,
        )
    knob = UsdGeom.Cylinder.Define(stage, "/ClampCounterweight/clamp_knob")
    knob.CreateRadiusAttr(0.007)
    knob.CreateHeightAttr(0.01)
    knob.CreateAxisAttr("Z")
    UsdGeom.Xformable(knob).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, g.plate_thickness_m + 0.005))
    knob.CreateDisplayColorAttr([Gf.Vec3f(*_BRASS)])
    _bind_physics_material(stage, root, static_friction=0.8, dynamic_friction=0.7)
    stage.GetRootLayer().Save()


def write_cantilever_ledge_usd(path: Path, geometry: CantileverLedgeGeometry = DEFAULT_LEDGE_GEOMETRY) -> None:
    """Write the static ledge-and-channel fixture (colliders without rigid bodies) to a USD file."""
    g = geometry
    stage, root = _create_stage(path, "/CantileverLedge")
    wall_height = g.ledge_height_m - 0.008
    _add_box(
        stage,
        "/CantileverLedge/ledge",
        (0.0, g.ledge_depth_m / 2, g.ledge_height_m / 2),
        (g.width_m, g.ledge_depth_m, g.ledge_height_m),
        _CONCRETE,
    )
    _add_box(
        stage,
        "/CantileverLedge/channel_floor",
        (0.0, -g.channel_length_m / 2, g.channel_floor_thickness_m / 2),
        (g.width_m, g.channel_length_m, g.channel_floor_thickness_m),
        _CHANNEL,
    )
    _add_box(
        stage,
        "/CantileverLedge/far_wall",
        (0.0, -g.channel_length_m - g.wall_thickness_m / 2, wall_height / 2),
        (g.width_m + 2 * g.wall_thickness_m, g.wall_thickness_m, wall_height),
        _CONCRETE,
    )
    end_wall_x = g.width_m / 2 + g.wall_thickness_m / 2
    for name, sign in (("end_wall_left", -1.0), ("end_wall_right", 1.0)):
        _add_box(
            stage,
            f"/CantileverLedge/{name}",
            (sign * end_wall_x, -g.channel_length_m / 2, wall_height / 2),
            (g.wall_thickness_m, g.channel_length_m, wall_height),
            _CONCRETE,
        )
    # The yellow line: a rod across the channel resting on the end walls, just below a bar lying on the ledge, so a
    # bar's outer end is seen against it without parallax, and its projection on the channel floor (visual only).
    _add_box(
        stage,
        "/CantileverLedge/yellow_line",
        (0.0, -g.line_overhang_m, wall_height + 0.002),
        (g.width_m + 2 * g.wall_thickness_m, 0.004, 0.004),
        _YELLOW,
        collide=False,
    )
    _add_box(
        stage,
        "/CantileverLedge/yellow_line_floor",
        (0.0, -g.line_overhang_m, g.channel_floor_thickness_m + 0.0005),
        (g.width_m, 0.006, 0.001),
        _YELLOW,
        collide=False,
    )
    _bind_physics_material(stage, root, static_friction=0.7, dynamic_friction=0.6)
    stage.GetRootLayer().Save()


def _get_cached_usd_path(prefix: str, geometry: object, writer: Callable[[Path, object], None]) -> str:
    """Return a cached USD path for a procedural asset geometry, writing the file on first use.

    The cache key covers the geometry and this module's source, so an edited writer never reuses a stale file.
    """
    cache_dir = Path(os.environ.get("ISAACLAB_ARENA_PROCEDURAL_CACHE", Path(tempfile.gettempdir()) / "isaaclab_arena"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(repr(geometry).encode() + Path(__file__).read_bytes()).hexdigest()[:16]
    path = cache_dir / f"{prefix}_{digest}.usda"
    if not path.exists():
        # Write to a temporary file first so parallel processes never read a partial file.
        tmp_path = path.with_suffix(f".{os.getpid()}.usda")
        writer(tmp_path, geometry)
        os.replace(tmp_path, path)
    return str(path)


_RIGID_SPAWN_ADDON = {
    "rigid_props": sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16, solver_velocity_iteration_count=1, sleep_threshold=0.0
    ),
    "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
}


@register_asset
class CantileverBar(LibraryObject):
    """Bar with a red and a blue end cap whose center of mass sits ``com_offset_m`` toward the red end.

    The USD is generated per offset and mass (cached by content hash); see the module docstring for the bar frame.
    Every offset looks the same; the registered subclasses fix the three classes of the cantilever task.
    """

    registry_name = "cantilever_bar"
    tags = ["object", "graspable", "procedural", "cantilever_ledge_bars"]
    spawn_cfg_addon = _RIGID_SPAWN_ADDON
    default_com_offset_m: float = 0.0

    def __init__(self, com_offset_m: float | None = None, mass_kg: float = DEFAULT_BAR_GEOMETRY.mass_kg, **kwargs):
        """Create a cantilever bar.

        Args:
            com_offset_m: Center-of-mass offset from the bar center toward the red end, in meters. Defaults to the
                class's offset.
            mass_kg: Mass of the bar in kilograms.
            **kwargs: Arguments passed to ``LibraryObject``.
        """
        offset_m = self.default_com_offset_m if com_offset_m is None else com_offset_m
        assert abs(offset_m) < DEFAULT_BAR_GEOMETRY.length_m / 2, "com_offset_m must lie inside the bar."
        self.geometry = replace(DEFAULT_BAR_GEOMETRY, com_offset_m=offset_m, mass_kg=mass_kg)
        self.usd_path = _get_cached_usd_path("cantilever_bar", self.geometry, write_cantilever_bar_usd)
        super().__init__(**kwargs)


@register_asset
class CantileverBarRedHeavy(CantileverBar):
    """Cantilever bar whose center of mass sits 5 cm toward the red end."""

    registry_name = "cantilever_bar_red_heavy"
    default_com_offset_m = BAR_COM_OFFSETS_M["red_heavy"]


@register_asset
class CantileverBarBlueHeavy(CantileverBar):
    """Cantilever bar whose center of mass sits 5 cm toward the blue end."""

    registry_name = "cantilever_bar_blue_heavy"
    default_com_offset_m = BAR_COM_OFFSETS_M["blue_heavy"]


@register_asset
class CantileverBarCentered(CantileverBar):
    """Cantilever bar whose center of mass is at its center."""

    registry_name = "cantilever_bar_centered"
    default_com_offset_m = BAR_COM_OFFSETS_M["centered"]


@register_asset
class ClampCounterweight(LibraryObject):
    """Saddle-shaped counterweight that drops over a bar and rests on it; its origin is the plate's underside."""

    registry_name = "clamp_counterweight"
    tags = ["object", "graspable", "procedural", "cantilever_ledge_bars"]
    spawn_cfg_addon = _RIGID_SPAWN_ADDON

    def __init__(self, mass_kg: float = DEFAULT_COUNTERWEIGHT_GEOMETRY.mass_kg, **kwargs):
        """Create the counterweight.

        Args:
            mass_kg: Mass in kilograms.
            **kwargs: Arguments passed to ``LibraryObject``.
        """
        self.geometry = replace(DEFAULT_COUNTERWEIGHT_GEOMETRY, mass_kg=mass_kg)
        self.usd_path = _get_cached_usd_path("clamp_counterweight", self.geometry, write_clamp_counterweight_usd)
        super().__init__(**kwargs)


@register_asset
class CantileverLedge(LibraryObject):
    """Static ledge with a channel beside it and a yellow line on the channel floor; see the module docstring frame."""

    registry_name = "cantilever_ledge"
    tags = ["object", "fixture", "procedural", "cantilever_ledge_bars"]
    object_type = ObjectType.BASE

    def __init__(self, line_overhang_m: float = DEFAULT_LEDGE_GEOMETRY.line_overhang_m, **kwargs):
        """Create the ledge fixture.

        Args:
            line_overhang_m: Distance from the ledge edge to the yellow line, in meters.
            **kwargs: Arguments passed to ``LibraryObject``.
        """
        assert 0.0 < line_overhang_m < DEFAULT_LEDGE_GEOMETRY.channel_length_m, "The line must lie over the channel."
        self.geometry = replace(DEFAULT_LEDGE_GEOMETRY, line_overhang_m=line_overhang_m)
        self.usd_path = _get_cached_usd_path("cantilever_ledge", self.geometry, write_cantilever_ledge_usd)
        super().__init__(**kwargs)
