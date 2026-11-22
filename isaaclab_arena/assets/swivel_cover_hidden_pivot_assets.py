# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assets for the swivel-cover retrieval task: a tray whose cover swivels on a concealed corner pivot, a core, a pad.

Fixture frame: Z up, origin at the bottom center of the tray. The robot stands on the -X side, so "front" is -X and
"left" is +Y. The cover lies flat on the tray rim and turns about a vertical axis through one of its corners; which
corner, which way it may swing (ccw or cw seen from above) and how strongly a spring returns it are hidden per variant.
Nothing on the cover marks the pivot: all four corners carry identical screw heads and the knob sits in the center.
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
    add_cylinder,
    create_stage,
    define_link,
    define_revolute_joint,
    fix_link_to_world,
    get_cached_usd_path,
)
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose, compose_poses
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

COVER_JOINT_NAME = "cover_pivot"
BASE_BODY_NAME = "base"
COVER_BODY_NAME = "cover"

PIVOT_CORNERS = ("front_left", "front_right", "back_left", "back_right")
"""Corners the cover can turn about, in the fixture frame (front is -X, left is +Y)."""

SWING_SIDES = ("ccw", "cw")
"""Free swing direction seen from above: ccw opens toward positive joint angles, cw toward negative ones."""

RETURN_SPRING_STIFFNESS_NM_PER_DEG = {"weak": 0.003, "strong": 0.006}
"""Return-spring stiffness per variant, in N*m/deg, as in the original task."""


@dataclass(frozen=True)
class SwivelFixtureGeometry:
    """Dimensions (m), masses (kg) and swing limit (deg) of the procedural swivel-cover fixture."""

    outer_size_m: float = 0.16
    """Side of the square tray and of the cover."""

    base_thickness_m: float = 0.006
    wall_thickness_m: float = 0.012
    rim_height_m: float = 0.075
    """Height of the tray rim above the fixture's bottom face."""

    cover_thickness_m: float = 0.008
    cover_gap_m: float = 0.001
    """Gap between the rim and the cover's bottom face."""

    knob_radius_m: float = 0.014
    knob_height_m: float = 0.014
    base_mass_kg: float = 1.5
    cover_mass_kg: float = 0.15
    open_limit_deg: float = 95.0
    """Joint limit on the free swing side, in degrees."""


DEFAULT_SWIVEL_FIXTURE_GEOMETRY = SwivelFixtureGeometry()

_GRAPHITE = (0.18, 0.19, 0.21)
_STEEL = (0.62, 0.64, 0.68)
_COVER = (0.80, 0.52, 0.16)
_CORE_BLUE = (0.08, 0.25, 0.85)
_PAD_GREEN = (0.15, 0.65, 0.22)


def get_pivot_corner_signs(pivot_corner: str) -> tuple[float, float]:
    """Return the (x, y) signs of a pivot corner in the fixture frame: front/back gives -1/+1, right/left -1/+1."""
    assert pivot_corner in PIVOT_CORNERS, f"pivot_corner must be one of {PIVOT_CORNERS}, got '{pivot_corner}'"
    front_back, left_right = pivot_corner.split("_")
    return (-1.0 if front_back == "front" else 1.0), (1.0 if left_right == "left" else -1.0)


def get_cover_joint_limits_deg(swing_side: str, geometry: SwivelFixtureGeometry) -> tuple[float, float]:
    """Return the (lower, upper) cover joint limits in degrees; the closed cover sits at zero."""
    assert swing_side in SWING_SIDES, f"swing_side must be one of {SWING_SIDES}, got '{swing_side}'"
    return (0.0, geometry.open_limit_deg) if swing_side == "ccw" else (-geometry.open_limit_deg, 0.0)


def write_swivel_fixture_usd(path: Path, pivot_corner: str, swing_side: str, geometry: SwivelFixtureGeometry) -> None:
    """Write the swivel-cover fixture articulation to a USD file.

    Args:
        path: Output USD path; overwritten if it exists.
        pivot_corner: Corner the cover turns about, one of PIVOT_CORNERS.
        swing_side: Free swing direction, one of SWING_SIDES.
        geometry: Dimensions, masses and swing limit.
    """
    g = geometry
    stage, root = create_stage(path, "SwivelCoverFixture")
    UsdPhysics.ArticulationRootAPI.Apply(root)

    # Tray: a floor and four walls. The cover is the base's joint child, so the two never collide.
    half = g.outer_size_m / 2
    wall_height = g.rim_height_m - g.base_thickness_m
    wall_center_z = g.base_thickness_m + wall_height / 2
    base_path = f"/SwivelCoverFixture/{BASE_BODY_NAME}"
    define_link(stage, base_path, (0.0, 0.0, 0.0), g.base_mass_kg)
    add_box(
        stage,
        f"{base_path}/floor",
        (0.0, 0.0, g.base_thickness_m / 2),
        (g.outer_size_m, g.outer_size_m, g.base_thickness_m),
        _GRAPHITE,
    )
    wall_t = g.wall_thickness_m
    for wall_name, center, size in (
        ("wall_front", (-half + wall_t / 2, 0.0), (wall_t, g.outer_size_m)),
        ("wall_back", (half - wall_t / 2, 0.0), (wall_t, g.outer_size_m)),
        ("wall_left", (0.0, half - wall_t / 2), (g.outer_size_m - 2 * wall_t, wall_t)),
        ("wall_right", (0.0, -half + wall_t / 2), (g.outer_size_m - 2 * wall_t, wall_t)),
    ):
        add_box(
            stage, f"{base_path}/{wall_name}", (center[0], center[1], wall_center_z), (*size, wall_height), _GRAPHITE
        )
    fix_link_to_world(stage, "/SwivelCoverFixture/root_joint", base_path)

    # Cover: its link frame sits on the pivot axis at the cover's bottom face, so the plate extends away from it.
    sign_x, sign_y = get_pivot_corner_signs(pivot_corner)
    t_A_pivot = (sign_x * half, sign_y * half, g.rim_height_m + g.cover_gap_m)
    cover_path = f"/SwivelCoverFixture/{COVER_BODY_NAME}"
    define_link(stage, cover_path, t_A_pivot, g.cover_mass_kg)
    center_x, center_y = -sign_x * half, -sign_y * half
    add_box(
        stage,
        f"{cover_path}/plate",
        (center_x, center_y, g.cover_thickness_m / 2),
        (g.outer_size_m, g.outer_size_m, g.cover_thickness_m),
        _COVER,
    )
    add_cylinder(
        stage,
        f"{cover_path}/knob",
        (center_x, center_y, g.cover_thickness_m + g.knob_height_m / 2),
        g.knob_radius_m,
        g.knob_height_m,
        "Z",
        _STEEL,
        collide=True,
    )
    # Identical screw heads on all four corners hide which one is the pivot.
    inset = 0.012
    for index, (corner_x, corner_y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        add_cylinder(
            stage,
            f"{cover_path}/screw_{index}",
            (center_x + corner_x * (half - inset), center_y + corner_y * (half - inset), g.cover_thickness_m + 0.001),
            0.005,
            0.002,
            "Z",
            _STEEL,
        )

    lower_deg, upper_deg = get_cover_joint_limits_deg(swing_side, g)
    # Joint frame X axis along +Z, so positive angles turn the cover counter-clockwise seen from above.
    q_B_J = Gf.Quatf(math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0)
    define_revolute_joint(
        stage,
        f"/SwivelCoverFixture/{COVER_JOINT_NAME}",
        base_path,
        cover_path,
        t_A_pivot,
        (0.0, 0.0, 0.0),
        q_B_J,
        lower_deg,
        upper_deg,
    )
    stage.GetRootLayer().Save()


def get_swivel_fixture_usd_path(
    pivot_corner: str, swing_side: str, geometry: SwivelFixtureGeometry = DEFAULT_SWIVEL_FIXTURE_GEOMETRY
) -> str:
    """Return a cached USD path for one pivot corner and swing side, writing the file on first use."""
    return get_cached_usd_path(
        "swivel_cover_fixture",
        (pivot_corner, swing_side, geometry),
        lambda path: write_swivel_fixture_usd(path, pivot_corner, swing_side, geometry),
    )


@register_asset
class SwivelCoverFixture(LibraryObject):
    """Fixed tray whose cover swivels on a concealed corner pivot against a return spring.

    The pivot corner, the free swing side and the return-spring strength are hidden variables. Set them per environment
    with the constructor arguments, or per run with the ``hidden_pivot`` variation, for example
    ``swivel_fixture.hidden_pivot.enabled=true swivel_fixture.hidden_pivot.pivot_corners=[back_right]
    swivel_fixture.hidden_pivot.swing_sides=[cw] swivel_fixture.hidden_pivot.return_springs=[strong]``.
    The USD is generated per pivot corner and swing side; see the module docstring for the frame.
    """

    registry_name = "swivel_cover_fixture"
    tags = ["object", "articulation", "procedural", "swivel_cover"]
    object_type = ObjectType.ARTICULATION

    def __init__(
        self,
        pivot_corner: str = "front_left",
        swing_side: str = "ccw",
        return_spring: str = "weak",
        cover_damping: float = 2.0,
        cover_armature: float = 0.002,
        **kwargs,
    ):
        """Create the fixture.

        Args:
            pivot_corner: Corner the cover turns about, one of front_left, front_right, back_left, back_right.
            swing_side: Free swing direction seen from above, ccw (limits [0, 95] deg) or cw (limits [-95, 0] deg).
            return_spring: Return-spring strength, weak (0.003 N*m/deg) or strong (0.006 N*m/deg).
            cover_damping: Viscous damping of the pivot, in N*m*s/rad. The default makes the cover close slowly
                enough that one arm can open it, let go and lift the core before it swings back.
            cover_armature: Armature of the pivot, in kg*m^2; keeps explicit joint efforts stable at coarse steps.
            **kwargs: Arguments passed to ``LibraryObject``.
        """
        self.geometry = DEFAULT_SWIVEL_FIXTURE_GEOMETRY
        self.cover_damping = cover_damping
        self.cover_armature = cover_armature
        self.usd_path = get_swivel_fixture_usd_path(pivot_corner, swing_side, self.geometry)
        super().__init__(**kwargs)
        self.object_cfg.init_state.joint_pos = {COVER_JOINT_NAME: 0.0}
        self.object_cfg.init_state.joint_vel = {".*": 0.0}
        self.set_hidden_pivot(pivot_corner, swing_side, return_spring)
        self.add_variation(SwivelCoverPivotVariation(self))

    def set_hidden_pivot(self, pivot_corner: str, swing_side: str, return_spring: str) -> None:
        """Select the pivot corner, swing side and return spring, updating the spawned USD and the pivot actuator."""
        from isaaclab.actuators import ImplicitActuatorCfg

        assert (
            return_spring in RETURN_SPRING_STIFFNESS_NM_PER_DEG
        ), f"return_spring must be one of {sorted(RETURN_SPRING_STIFFNESS_NM_PER_DEG)}, got '{return_spring}'"
        self.pivot_corner = pivot_corner
        self.swing_side = swing_side
        self.return_spring = return_spring
        self.usd_path = get_swivel_fixture_usd_path(pivot_corner, swing_side, self.geometry)
        self.object_cfg.spawn = self.object_cfg.spawn.replace(usd_path=self.usd_path)
        stiffness_nm_per_rad = RETURN_SPRING_STIFFNESS_NM_PER_DEG[return_spring] * 180.0 / math.pi
        self.object_cfg.actuators = {
            "cover_return_spring": ImplicitActuatorCfg(
                joint_names_expr=[COVER_JOINT_NAME],
                stiffness=stiffness_nm_per_rad,
                damping=self.cover_damping,
                armature=self.cover_armature,
            ),
        }

    def get_seat_pose(self, object_bottom_offset_m: float) -> Pose:
        """Return the world pose of an upright object standing on the tray floor center.

        Args:
            object_bottom_offset_m: Distance from the object's origin down to its bottom face.
        """
        T_W_F = self.get_initial_pose()
        assert isinstance(T_W_F, Pose), "The swivel-cover fixture needs a fixed initial pose."
        T_F_O = Pose(position_xyz=(0.0, 0.0, self.geometry.base_thickness_m + object_bottom_offset_m + 0.0005))
        return compose_poses(T_W_F, T_F_O)


@configclass
class SwivelCoverPivotVariationCfg(VariationBaseCfg):
    """Configuration for SwivelCoverPivotVariation."""

    pivot_corners: list[str] = field(default_factory=lambda: list(PIVOT_CORNERS))
    """Pivot corners to draw from."""

    swing_sides: list[str] = field(default_factory=lambda: list(SWING_SIDES))
    """Swing sides to draw from."""

    return_springs: list[str] = field(default_factory=lambda: list(RETURN_SPRING_STIFFNESS_NM_PER_DEG))
    """Return-spring strengths to draw from."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Uniform draw of one (corner, side, spring) tuple per build, seeded by the build seed."""


class SwivelCoverPivotVariation(BuildTimeVariationBase):
    """Draw the fixture's hidden pivot corner, swing side and return spring once per build.

    Every combination of the configured lists is equally likely; one-element lists select a fixed variant. The recorded
    sample is the selected ``corner:side:spring`` string.

    Args:
        fixture: Fixture whose hidden pivot the variation sets.
        cfg: Value lists to draw from. By default the variation is off.
        name: Name of the variation on the fixture.
    """

    cfg: SwivelCoverPivotVariationCfg

    def __init__(
        self, fixture: SwivelCoverFixture, cfg: SwivelCoverPivotVariationCfg | None = None, name: str = "hidden_pivot"
    ):
        super().__init__(cfg=cfg if cfg is not None else SwivelCoverPivotVariationCfg(), name=name)
        self._fixture = fixture

    def _realize_at_build_time(self) -> None:
        hidden_tuples = []
        for pivot_corner in self.cfg.pivot_corners:
            get_pivot_corner_signs(pivot_corner)
            for swing_side in self.cfg.swing_sides:
                assert swing_side in SWING_SIDES, f"swing side must be one of {SWING_SIDES}, got '{swing_side}'"
                for return_spring in self.cfg.return_springs:
                    hidden_tuples.append(f"{pivot_corner}:{swing_side}:{return_spring}")
        assert hidden_tuples, "hidden_pivot needs at least one pivot corner, swing side and return spring."
        pivot_corner, swing_side, return_spring = self.sampler.sample(1, hidden_tuples)[0].split(":")
        self._fixture.set_hidden_pivot(pivot_corner, swing_side, return_spring)


@register_asset
class SwivelCore(Object):
    """Blue cylindrical core that starts inside the swivel-cover fixture."""

    registry_name = "swivel_core"
    tags = ["object", "graspable", "procedural", "swivel_cover"]

    radius_m = 0.02
    height_m = 0.06

    def __init__(
        self,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        mass_kg: float = 0.08,
    ):
        """Create the core.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
            mass_kg: Mass in kilograms.
        """
        spawner_cfg = sim_utils.CylinderCfg(
            radius=self.radius_m,
            height=self.height_m,
            axis="Z",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, sleep_threshold=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_CORE_BLUE, roughness=0.4),
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
class SwivelTargetPad(Object):
    """Thin green square pad fixed to the table (kinematic), the target the core must stand on."""

    registry_name = "swivel_target_pad"
    tags = ["object", "procedural", "swivel_cover"]

    size_m = 0.09
    thickness_m = 0.006

    def __init__(
        self,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        """Create the pad.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
        """
        spawner_cfg = sim_utils.CuboidCfg(
            size=(self.size_m, self.size_m, self.thickness_m),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_PAD_GREEN, roughness=0.8),
        )
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawner_cfg,
            initial_pose=initial_pose,
            tags=list(self.tags),
        )
