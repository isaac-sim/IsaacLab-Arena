# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assets for the hinged-roof village task: look-alike houses with hinged roofs, toy animals, a pen and a mat.

Village frame B: Z up, origin at the bottom center of the board. The robot stands on the -X side, so "front" is -X and
"left" is +Y. Four identical houses sit in a 2 x 2 grid on one board: h1 front-left, h2 front-right, h3 back-left,
h4 back-right. Each house's flat roof turns about one hidden edge (front, back, left or right); every roof carries
identical tabs on all four sides, so nothing shows the hinge. Positive roof joint angles always open the roof.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass
from pxr import Gf, UsdPhysics

from isaaclab_arena.assets.hinged_lid_usd import (
    add_box,
    create_stage,
    define_link,
    define_revolute_joint,
    filter_collisions,
    get_cached_usd_path,
)
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose, compose_poses
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

HOUSE_NAMES = ("h1", "h2", "h3", "h4")
"""House names: h1 front-left, h2 front-right, h3 back-left, h4 back-right."""

HINGE_EDGES = ("front", "back", "left", "right")
BOARD_BODY_NAME = "board"

_HOUSE_GRID_SIGNS = {"h1": (-1.0, 1.0), "h2": (-1.0, -1.0), "h3": (1.0, 1.0), "h4": (1.0, -1.0)}
_EDGE_OUTWARD_NORMALS = {"front": (-1.0, 0.0), "back": (1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
# Hinge axis yaw per edge: the joint X axis d satisfies n x d = +Z, so positive angles lift the free edge.
_EDGE_AXIS_YAW_RAD = {"front": -math.pi / 2, "back": math.pi / 2, "left": math.pi, "right": 0.0}


def get_roof_body_name(house_name: str) -> str:
    """Return the body name of a house's roof."""
    return f"roof_{house_name}"


def get_roof_joint_name(house_name: str) -> str:
    """Return the joint name of a house's roof hinge."""
    return f"roof_{house_name}_hinge"


@dataclass(frozen=True)
class VillageGeometry:
    """Dimensions (m), masses (kg) and roof limit (deg) of the procedural village."""

    house_outer_m: float = 0.12
    house_pitch_m: float = 0.145
    """Distance between neighbouring house centers."""

    board_margin_m: float = 0.02
    board_thickness_m: float = 0.012
    wall_thickness_m: float = 0.008
    wall_height_m: float = 0.07
    """Height of the house walls above the board top."""

    roof_size_m: float = 0.124
    roof_thickness_m: float = 0.008
    roof_gap_m: float = 0.001
    tab_width_m: float = 0.03
    tab_length_m: float = 0.008
    tab_thickness_m: float = 0.005
    board_mass_kg: float = 3.0
    roof_mass_kg: float = 0.15
    roof_open_limit_deg: float = 115.0
    """Upper roof limit; past 90 degrees gravity holds an opened roof against it."""

    @property
    def board_size_m(self) -> float:
        return self.house_pitch_m + self.house_outer_m + 2 * self.board_margin_m

    @property
    def house_interior_half_m(self) -> float:
        return self.house_outer_m / 2 - self.wall_thickness_m

    @property
    def roof_bottom_z_m(self) -> float:
        return self.board_thickness_m + self.wall_height_m + self.roof_gap_m

    def get_house_center_B(self, house_name: str) -> tuple[float, float]:
        """Return the (x, y) center of a house in the village frame B."""
        sign_x, sign_y = _HOUSE_GRID_SIGNS[house_name]
        half_pitch = self.house_pitch_m / 2
        return sign_x * half_pitch, sign_y * half_pitch


DEFAULT_VILLAGE_GEOMETRY = VillageGeometry()

_BOARD = (0.55, 0.42, 0.28)
_WALL = (0.93, 0.88, 0.76)
_ROOF = (0.70, 0.20, 0.16)
_ROOF_TRIM = (0.55, 0.14, 0.12)
_DOOR = (0.40, 0.26, 0.16)
_WINDOW = (0.55, 0.72, 0.85)


def _assert_hinge_edges(hinge_edges: list[str] | tuple[str, ...]) -> None:
    assert len(hinge_edges) == len(HOUSE_NAMES), f"hinge_edges needs one edge per house {HOUSE_NAMES}"
    for edge in hinge_edges:
        assert edge in HINGE_EDGES, f"hinge edge must be one of {HINGE_EDGES}, got '{edge}'"


def write_village_usd(path: Path, hinge_edges: tuple[str, ...], geometry: VillageGeometry) -> None:
    """Write the village articulation (free board with four hinged roofs) to a USD file.

    Args:
        path: Output USD path; overwritten if it exists.
        hinge_edges: Hinge edge of each roof, in HOUSE_NAMES order.
        geometry: Dimensions, masses and roof limit.
    """
    g = geometry
    _assert_hinge_edges(hinge_edges)
    stage, root = create_stage(path, "HingedRoofVillage")
    UsdPhysics.ArticulationRootAPI.Apply(root)

    # Board link: the board slab and every house's walls. It is a free body that rests on the mat.
    board_path = f"/HingedRoofVillage/{BOARD_BODY_NAME}"
    define_link(stage, board_path, (0.0, 0.0, 0.0), g.board_mass_kg)
    board_size = g.board_size_m
    add_box(
        stage,
        f"{board_path}/slab",
        (0.0, 0.0, g.board_thickness_m / 2),
        (board_size, board_size, g.board_thickness_m),
        _BOARD,
    )
    half = g.house_outer_m / 2
    wall_t = g.wall_thickness_m
    wall_z = g.board_thickness_m + g.wall_height_m / 2
    for house_name in HOUSE_NAMES:
        center_x, center_y = g.get_house_center_B(house_name)
        for wall_name, offset, size in (
            ("front", (-half + wall_t / 2, 0.0), (wall_t, g.house_outer_m)),
            ("back", (half - wall_t / 2, 0.0), (wall_t, g.house_outer_m)),
            ("left", (0.0, half - wall_t / 2), (g.house_outer_m - 2 * wall_t, wall_t)),
            ("right", (0.0, -half + wall_t / 2), (g.house_outer_m - 2 * wall_t, wall_t)),
        ):
            add_box(
                stage,
                f"{board_path}/{house_name}_wall_{wall_name}",
                (center_x + offset[0], center_y + offset[1], wall_z),
                (*size, g.wall_height_m),
                _WALL,
            )
        # Identical door and window decals on every house's front wall.
        decal_x = center_x - half - 0.0006
        add_box(
            stage,
            f"{board_path}/{house_name}_door",
            (decal_x, center_y, g.board_thickness_m + 0.02),
            (0.001, 0.026, 0.04),
            _DOOR,
            collide=False,
        )
        for side in (-1.0, 1.0):
            add_box(
                stage,
                f"{board_path}/{house_name}_window_{'l' if side > 0 else 'r'}",
                (decal_x, center_y + side * 0.035, g.board_thickness_m + 0.048),
                (0.001, 0.018, 0.016),
                _WINDOW,
                collide=False,
            )

    # Roof links: frame R at the center of the roof's bottom face, aligned with the board when closed.
    roof_paths = []
    for house_name, hinge_edge in zip(HOUSE_NAMES, hinge_edges):
        center_x, center_y = g.get_house_center_B(house_name)
        roof_path = f"/HingedRoofVillage/{get_roof_body_name(house_name)}"
        roof_paths.append(roof_path)
        define_link(stage, roof_path, (center_x, center_y, g.roof_bottom_z_m), g.roof_mass_kg)
        add_box(
            stage,
            f"{roof_path}/plate",
            (0.0, 0.0, g.roof_thickness_m / 2),
            (g.roof_size_m, g.roof_size_m, g.roof_thickness_m),
            _ROOF,
        )
        add_box(
            stage,
            f"{roof_path}/ridge",
            (0.0, 0.0, g.roof_thickness_m + 0.004),
            (0.07, 0.07, 0.008),
            _ROOF_TRIM,
            collide=False,
        )
        tab_offset = g.roof_size_m / 2 + g.tab_length_m / 2
        for tab_name, (normal_x, normal_y) in _EDGE_OUTWARD_NORMALS.items():
            tab_size = (
                (g.tab_length_m, g.tab_width_m, g.tab_thickness_m)
                if normal_x != 0.0
                else (g.tab_width_m, g.tab_length_m, g.tab_thickness_m)
            )
            add_box(
                stage,
                f"{roof_path}/tab_{tab_name}",
                (normal_x * tab_offset, normal_y * tab_offset, g.tab_thickness_m / 2),
                tab_size,
                _ROOF_TRIM,
            )
        normal_x, normal_y = _EDGE_OUTWARD_NORMALS[hinge_edge]
        t_R_J = (normal_x * g.roof_size_m / 2, normal_y * g.roof_size_m / 2, 0.0)
        t_B_J = (center_x + t_R_J[0], center_y + t_R_J[1], g.roof_bottom_z_m)
        yaw = _EDGE_AXIS_YAW_RAD[hinge_edge]
        q_B_J = Gf.Quatf(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        define_revolute_joint(
            stage,
            f"/HingedRoofVillage/{get_roof_joint_name(house_name)}",
            board_path,
            roof_path,
            t_B_J,
            t_R_J,
            q_B_J,
            0.0,
            g.roof_open_limit_deg,
        )
    # Opened roofs may lean over a neighbour; roofs never collide with each other.
    filter_collisions(roof_paths, stage)
    stage.GetRootLayer().Save()


def get_village_usd_path(hinge_edges: tuple[str, ...], geometry: VillageGeometry = DEFAULT_VILLAGE_GEOMETRY) -> str:
    """Return a cached USD path for one hinge-edge pattern, writing the file on first use."""
    hinge_edges = tuple(hinge_edges)
    return get_cached_usd_path(
        "hinged_roof_village",
        (hinge_edges, geometry),
        lambda path: write_village_usd(path, hinge_edges, geometry),
    )


@register_asset
class HingedRoofVillage(LibraryObject):
    """Free-standing board with four identical houses whose flat roofs open about hidden edges.

    Set the hinge edges per environment with ``hinge_edges``, or per run with the ``roof_hinges`` variation, for
    example ``village.roof_hinges.enabled=true village.roof_hinges.edges=[left,back,front,right]``. The board is the
    articulation root and is not fixed, so careless handling can push it off its mat.
    """

    registry_name = "hinged_roof_village"
    tags = ["object", "articulation", "procedural", "hinged_roof_village"]
    object_type = ObjectType.ARTICULATION

    def __init__(
        self,
        hinge_edges: list[str] | None = None,
        roof_damping: float = 0.01,
        roof_armature: float = 0.002,
        **kwargs,
    ):
        """Create the village.

        Args:
            hinge_edges: Hinge edge of each roof (front, back, left or right), in h1..h4 order. Defaults to
                front, right, back, left.
            roof_damping: Viscous damping of each roof hinge, in N*m*s/rad.
            roof_armature: Armature of each roof hinge, in kg*m^2; keeps explicit joint efforts stable at coarse steps.
            **kwargs: Arguments passed to ``LibraryObject``.
        """
        self.geometry = DEFAULT_VILLAGE_GEOMETRY
        self.house_names = HOUSE_NAMES
        self.roof_damping = roof_damping
        self.roof_armature = roof_armature
        hinge_edges = tuple(hinge_edges) if hinge_edges is not None else ("front", "right", "back", "left")
        _assert_hinge_edges(hinge_edges)
        self.usd_path = get_village_usd_path(hinge_edges, self.geometry)
        super().__init__(**kwargs)
        self.object_cfg.init_state.joint_pos = {".*": 0.0}
        self.object_cfg.init_state.joint_vel = {".*": 0.0}
        self.set_hinge_edges(hinge_edges)
        self.add_variation(VillageRoofHingeVariation(self))

    @property
    def roof_joint_names(self) -> list[str]:
        """Roof hinge joint names in house order."""
        return [get_roof_joint_name(house_name) for house_name in self.house_names]

    def set_hinge_edges(self, hinge_edges: list[str] | tuple[str, ...]) -> None:
        """Select each roof's hinge edge, updating the spawned USD and the roof actuators."""
        from isaaclab.actuators import ImplicitActuatorCfg

        hinge_edges = tuple(hinge_edges)
        _assert_hinge_edges(hinge_edges)
        self.hinge_edges = hinge_edges
        self.usd_path = get_village_usd_path(hinge_edges, self.geometry)
        self.object_cfg.spawn = self.object_cfg.spawn.replace(usd_path=self.usd_path)
        self.object_cfg.actuators = {
            "roof_hinges": ImplicitActuatorCfg(
                joint_names_expr=self.roof_joint_names,
                stiffness=0.0,
                damping=self.roof_damping,
                armature=self.roof_armature,
            ),
        }

    def get_house_floor_pose(self, house_name: str, object_bottom_offset_m: float) -> Pose:
        """Return the world pose of an object standing in the middle of a house floor.

        Args:
            house_name: One of h1..h4.
            object_bottom_offset_m: Distance from the object's origin down to its bottom face.
        """
        assert house_name in self.house_names, f"house must be one of {self.house_names}, got '{house_name}'"
        T_W_B = self.get_initial_pose()
        assert isinstance(T_W_B, Pose), "The village needs a fixed initial pose."
        center_x, center_y = self.geometry.get_house_center_B(house_name)
        T_B_O = Pose(
            position_xyz=(center_x, center_y, self.geometry.board_thickness_m + object_bottom_offset_m + 0.001)
        )
        return compose_poses(T_W_B, T_B_O)


@configclass
class VillageRoofHingeVariationCfg(VariationBaseCfg):
    """Configuration for VillageRoofHingeVariation."""

    edges: list[str] | None = None
    """Exact hinge edge per roof in h1..h4 order. None draws each roof's edge from ``edge_choices``."""

    edge_choices: list[str] = field(default_factory=lambda: list(HINGE_EDGES))
    """Edges each roof draws from, independently, when ``edges`` is None."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Uniform draw of one edge per roof per build, seeded by the build seed."""


class VillageRoofHingeVariation(BuildTimeVariationBase):
    """Set the hidden hinge edge of every roof once per build, from an exact pattern or by random draws.

    Args:
        village: Village whose roof hinges the variation sets.
        cfg: Edge pattern or choices. By default the variation is off.
        name: Name of the variation on the village.
    """

    cfg: VillageRoofHingeVariationCfg

    def __init__(
        self, village: HingedRoofVillage, cfg: VillageRoofHingeVariationCfg | None = None, name: str = "roof_hinges"
    ):
        super().__init__(cfg=cfg if cfg is not None else VillageRoofHingeVariationCfg(), name=name)
        self._village = village

    def _realize_at_build_time(self) -> None:
        if self.cfg.edges is not None:
            hinge_edges = list(self.cfg.edges)
        else:
            choices = list(self.cfg.edge_choices)
            assert choices, "roof_hinges.edge_choices must not be empty."
            hinge_edges = self.sampler.sample(len(HOUSE_NAMES), choices)
        self._village.set_hinge_edges(hinge_edges)


ANIMAL_COLORS = {
    "blue": (0.10, 0.30, 0.90),
    "red": (0.85, 0.15, 0.10),
    "yellow": (0.95, 0.80, 0.10),
    "white": (0.92, 0.92, 0.88),
    "pink": (0.95, 0.55, 0.70),
    "orange": (0.95, 0.50, 0.10),
}


def write_toy_animal_usd(path: Path, color: tuple[float, float, float], mass_kg: float) -> None:
    """Write a four-legged toy animal rigid body to a USD file; its origin is the bottom center, facing +X."""
    stage, root = create_stage(path, "ToyAnimal")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass_kg)
    leg_height, body_height = 0.014, 0.026
    body_center_z = leg_height + body_height / 2
    add_box(stage, "/ToyAnimal/body", (0.0, 0.0, body_center_z), (0.058, 0.032, body_height), color)
    add_box(stage, "/ToyAnimal/head", (0.034, 0.0, body_center_z + 0.016), (0.024, 0.026, 0.024), color)
    for index, (leg_x, leg_y) in enumerate(((0.02, 0.011), (0.02, -0.011), (-0.02, 0.011), (-0.02, -0.011))):
        add_box(
            stage, f"/ToyAnimal/leg_{index}", (leg_x, leg_y, leg_height / 2 + 0.0005), (0.01, 0.01, leg_height), color
        )
    dark = (0.05, 0.05, 0.05)
    for side in (-1.0, 1.0):
        add_box(
            stage,
            f"/ToyAnimal/ear_{'l' if side > 0 else 'r'}",
            (0.03, side * 0.009, body_center_z + 0.031),
            (0.006, 0.006, 0.008),
            color,
            collide=False,
        )
        add_box(
            stage,
            f"/ToyAnimal/eye_{'l' if side > 0 else 'r'}",
            (0.0465, side * 0.006, body_center_z + 0.02),
            (0.002, 0.004, 0.004),
            dark,
            collide=False,
        )
    add_box(stage, "/ToyAnimal/tail", (-0.033, 0.0, body_center_z + 0.008), (0.012, 0.004, 0.004), color, collide=False)
    stage.GetRootLayer().Save()


@register_asset
class ToyAnimal(LibraryObject):
    """Small four-legged toy animal of a chosen color; all colors share one shape."""

    registry_name = "toy_animal"
    tags = ["object", "graspable", "procedural", "hinged_roof_village"]
    object_type = ObjectType.RIGID

    def __init__(self, color: str = "blue", mass_kg: float = 0.05, **kwargs):
        """Create a toy animal.

        Args:
            color: One of blue, red, yellow, white, pink, orange.
            mass_kg: Mass in kilograms.
            **kwargs: Arguments passed to ``LibraryObject``.
        """
        assert color in ANIMAL_COLORS, f"color must be one of {sorted(ANIMAL_COLORS)}, got '{color}'"
        rgb = ANIMAL_COLORS[color]
        self.color = color
        self.usd_path = get_cached_usd_path(
            "toy_animal", (rgb, mass_kg), lambda path: write_toy_animal_usd(path, rgb, mass_kg)
        )
        kwargs.setdefault("name", f"{color}_animal")
        super().__init__(**kwargs)


def write_playground_pen_usd(path: Path, outer_size_m: float, wall_height_m: float, wall_thickness_m: float) -> None:
    """Write a kinematic open-top square pen (floor and four low walls) to a USD file; origin at the bottom center."""
    stage, root = create_stage(path, "PlaygroundPen")
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body.CreateKinematicEnabledAttr(True)
    floor_thickness = 0.004
    grass = (0.30, 0.62, 0.25)
    fence = (0.96, 0.96, 0.94)
    add_box(
        stage,
        "/PlaygroundPen/floor",
        (0.0, 0.0, floor_thickness / 2),
        (outer_size_m, outer_size_m, floor_thickness),
        grass,
    )
    half = outer_size_m / 2
    wall_z = floor_thickness + wall_height_m / 2
    for wall_name, center, size in (
        ("front", (-half + wall_thickness_m / 2, 0.0), (wall_thickness_m, outer_size_m)),
        ("back", (half - wall_thickness_m / 2, 0.0), (wall_thickness_m, outer_size_m)),
        ("left", (0.0, half - wall_thickness_m / 2), (outer_size_m - 2 * wall_thickness_m, wall_thickness_m)),
        ("right", (0.0, -half + wall_thickness_m / 2), (outer_size_m - 2 * wall_thickness_m, wall_thickness_m)),
    ):
        add_box(
            stage, f"/PlaygroundPen/fence_{wall_name}", (center[0], center[1], wall_z), (*size, wall_height_m), fence
        )
    stage.GetRootLayer().Save()


@register_asset
class PlaygroundPen(LibraryObject):
    """Open-top square pen with a grass floor and a low white fence, fixed in place (kinematic)."""

    registry_name = "playground_pen"
    tags = ["object", "procedural", "hinged_roof_village"]
    object_type = ObjectType.RIGID

    outer_size_m = 0.14
    wall_height_m = 0.035
    wall_thickness_m = 0.006

    def __init__(self, **kwargs):
        """Create the pen.

        Args:
            **kwargs: Arguments passed to ``LibraryObject``.
        """
        self.usd_path = get_cached_usd_path(
            "playground_pen",
            (self.outer_size_m, self.wall_height_m, self.wall_thickness_m),
            lambda path: write_playground_pen_usd(path, self.outer_size_m, self.wall_height_m, self.wall_thickness_m),
        )
        super().__init__(**kwargs)

    @property
    def interior_half_size_m(self) -> float:
        return self.outer_size_m / 2 - self.wall_thickness_m


@register_asset
class VillageMat(Object):
    """Thin square mat fixed to the table (kinematic) that the village board stands on."""

    registry_name = "village_mat"
    tags = ["object", "procedural", "hinged_roof_village"]

    size_m = 0.34
    thickness_m = 0.004

    def __init__(
        self,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        """Create the mat.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
        """
        spawner_cfg = sim_utils.CuboidCfg(
            size=(self.size_m, self.size_m, self.thickness_m),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.36, 0.55), roughness=0.9),
        )
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawner_cfg,
            initial_pose=initial_pose,
            tags=list(self.tags),
        )
