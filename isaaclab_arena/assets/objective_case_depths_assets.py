# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Objective case with four look-alike sockets of hidden depth, a hinged lid, and the cylinders packed into it.

The case is a procedural fixed-base articulation: a body link whose sockets are built from separate wall and floor
boxes (one collider per box, so a cylinder can sink into a socket), and a lid on a revolute hinge along the back
edge. Each socket's depth below the rim is a parameter; the cylinders' lengths match the default depth set, so only
one assignment of cylinders to sockets leaves every cylinder flush with the rim.

Case frame C: Z up, origin at the bottom center of the case. Sockets 1-4 lie along Y, socket 1 at +Y (the robot's
left when the robot faces +X). The lid hinge runs along Y at the back (+X) edge of the rim; the lid opens toward +X.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

CASE_BODY_NAME = "body"
LID_BODY_NAME = "lid"
LID_JOINT_NAME = "lid_hinge"

DEFAULT_SOCKET_DEPTHS_MM = (42, 52, 62, 72)
"""Socket depths (mm) of the default case, socket 1 first; the cylinders' lengths are the same set."""


@dataclass(frozen=True)
class ObjectiveCaseGeometry:
    """Dimensions (m), masses (kg) and lid limit (deg) of the procedural objective case."""

    socket_depths_m: tuple[float, ...] = tuple(depth_mm / 1000.0 for depth_mm in DEFAULT_SOCKET_DEPTHS_MM)
    """Depth of each socket below the rim, socket 1 (+Y) first."""

    socket_pitch_m: float = 0.05
    """Center distance between neighbouring sockets along Y."""

    socket_width_m: float = 0.034
    """Inner side length of each square socket opening."""

    case_depth_m: float = 0.07
    """Outer size of the case along X."""

    end_wall_m: float = 0.013
    """Thickness of the case walls at both Y ends."""

    case_height_m: float = 0.082
    """Height of the rim above the case bottom; the deepest socket keeps a floor of at least 1 cm."""

    case_mass_kg: float = 1.5
    lid_thickness_m: float = 0.006
    lid_gap_m: float = 0.001
    """Gap between the rim and the underside of the closed lid, so a flush cylinder does not touch it."""

    lid_mass_kg: float = 0.15
    lid_open_limit_deg: float = 110.0
    """Upper lid limit; past 90 deg the open lid rests against it under gravity."""

    @property
    def num_sockets(self) -> int:
        return len(self.socket_depths_m)

    @property
    def case_length_m(self) -> float:
        """Outer size of the case along Y."""
        return (self.num_sockets - 1) * self.socket_pitch_m + self.socket_width_m + 2 * self.end_wall_m

    def socket_center_y_m(self, socket_index: int) -> float:
        """Return the Y coordinate, in the case frame, of a socket's center; index 0 is socket 1 (+Y)."""
        return ((self.num_sockets - 1) / 2 - socket_index) * self.socket_pitch_m


DEFAULT_CASE_GEOMETRY = ObjectiveCaseGeometry()

_GRAPHITE = (0.16, 0.17, 0.19)
_LID_GREY = (0.24, 0.26, 0.29)
_FELT = (0.42, 0.07, 0.08)
_BRASS = (0.72, 0.55, 0.22)
_WHITE = (0.9, 0.9, 0.88)


def _add_box(
    stage: Usd.Stage,
    path: str,
    center_L: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    color: tuple[float, float, float],
    collide: bool = True,
) -> None:
    """Add a box to a link at a center expressed in the link frame L."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center_L))
    xform.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _define_link(stage: Usd.Stage, path: str, t_C_L: tuple[float, float, float], mass_kg: float) -> Usd.Prim:
    """Define a rigid link whose frame L sits at t_C_L in the case frame C."""
    link = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdGeom.Xformable(link).AddTranslateOp().Set(Gf.Vec3d(*t_C_L))
    UsdPhysics.RigidBodyAPI.Apply(link)
    UsdPhysics.MassAPI.Apply(link).CreateMassAttr(mass_kg)
    return link


def _add_case_body_geometry(stage: Usd.Stage, body_path: str, g: ObjectiveCaseGeometry) -> None:
    """Add the case walls and the four socket floors, one box collider each, in the case frame."""
    half_depth = g.case_depth_m / 2
    half_length = g.case_length_m / 2
    half_socket = g.socket_width_m / 2
    height = g.case_height_m
    # Front (-X) and back (+X) walls span the full length, outside the socket band.
    wall_x_size = half_depth - half_socket
    for name, sign in (("front_wall", -1.0), ("back_wall", 1.0)):
        _add_box(
            stage,
            f"{body_path}/{name}",
            (sign * (half_socket + wall_x_size / 2), 0.0, height / 2),
            (wall_x_size, g.case_length_m, height),
            _GRAPHITE,
        )
    # Inside the socket band: end walls and dividers between the openings, then one floor per socket.
    opening_edges_y = [-half_length]
    for socket_index in reversed(range(g.num_sockets)):
        center_y = g.socket_center_y_m(socket_index)
        opening_edges_y.extend([center_y - half_socket, center_y + half_socket])
    opening_edges_y.append(half_length)
    for divider_index in range(g.num_sockets + 1):
        low_y, high_y = opening_edges_y[2 * divider_index], opening_edges_y[2 * divider_index + 1]
        _add_box(
            stage,
            f"{body_path}/divider_{divider_index}",
            (0.0, (low_y + high_y) / 2, height / 2),
            (g.socket_width_m, high_y - low_y, height),
            _GRAPHITE,
        )
    for socket_index, depth_m in enumerate(g.socket_depths_m):
        floor_height = height - depth_m
        assert floor_height >= 0.005, f"Socket {socket_index + 1} is deeper than the case allows."
        _add_box(
            stage,
            f"{body_path}/socket_{socket_index + 1}_floor",
            (0.0, g.socket_center_y_m(socket_index), floor_height / 2),
            (g.socket_width_m, g.socket_width_m, floor_height),
            _FELT,
        )
        # Socket numbers as white dots on the front face (visual only): socket k has k dots.
        for dot_index in range(socket_index + 1):
            dot_y = g.socket_center_y_m(socket_index) + (dot_index - socket_index / 2) * 0.006
            _add_box(
                stage,
                f"{body_path}/socket_{socket_index + 1}_mark_{dot_index}",
                (-half_depth - 0.0005, dot_y, height - 0.012),
                (0.001, 0.003, 0.003),
                _WHITE,
                collide=False,
            )


def write_objective_case_usd(path: Path, geometry: ObjectiveCaseGeometry = DEFAULT_CASE_GEOMETRY) -> None:
    """Write the objective case articulation to a USD file.

    Args:
        path: Output USD path; overwritten if it exists.
        geometry: Dimensions, socket depths, masses and lid limit of the case.
    """
    g = geometry
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/ObjectiveCase").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    body_path = f"/ObjectiveCase/{CASE_BODY_NAME}"
    _define_link(stage, body_path, (0.0, 0.0, 0.0), g.case_mass_kg)
    _add_case_body_geometry(stage, body_path, g)
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/ObjectiveCase/root_joint")
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])

    # Lid: its frame sits on the hinge line; the closed plate extends toward -X over the sockets.
    t_C_hinge = (g.case_depth_m / 2, 0.0, g.case_height_m)
    lid_path = f"/ObjectiveCase/{LID_BODY_NAME}"
    _define_link(stage, lid_path, t_C_hinge, g.lid_mass_kg)
    _add_box(
        stage,
        f"{lid_path}/plate",
        (-g.case_depth_m / 2, 0.0, g.lid_gap_m + g.lid_thickness_m / 2),
        (g.case_depth_m, g.case_length_m, g.lid_thickness_m),
        _LID_GREY,
    )
    _add_box(
        stage,
        f"{lid_path}/handle",
        (-g.case_depth_m + 0.004, 0.0, g.lid_gap_m + g.lid_thickness_m + 0.004),
        (0.008, 0.05, 0.008),
        _BRASS,
    )

    hinge = UsdPhysics.RevoluteJoint.Define(stage, f"/ObjectiveCase/{LID_JOINT_NAME}")
    hinge.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
    hinge.CreateBody1Rel().SetTargets([Sdf.Path(lid_path)])
    # A positive angle about +Y lifts the lid's front edge: the lid opens toward +X.
    hinge.CreateAxisAttr("Y")
    hinge.CreateLocalPos0Attr(Gf.Vec3f(*t_C_hinge))
    hinge.CreateLocalRot0Attr(Gf.Quatf(1.0))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    hinge.CreateLocalRot1Attr(Gf.Quatf(1.0))
    hinge.CreateLowerLimitAttr(0.0)
    hinge.CreateUpperLimitAttr(g.lid_open_limit_deg)
    # A drive with zero gains; the asset's actuator configuration supplies the hinge damping.
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)
    stage.GetRootLayer().Save()


def get_objective_case_usd_path(geometry: ObjectiveCaseGeometry = DEFAULT_CASE_GEOMETRY) -> str:
    """Return a cached USD path for an objective case geometry, writing the file on first use.

    The cache key covers the geometry and this module's source, so an edited writer never reuses a stale file.
    """
    cache_dir = Path(os.environ.get("ISAACLAB_ARENA_PROCEDURAL_CACHE", Path(tempfile.gettempdir()) / "isaaclab_arena"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(repr(geometry).encode() + Path(__file__).read_bytes()).hexdigest()[:16]
    path = cache_dir / f"objective_case_{digest}.usda"
    if not path.exists():
        # Write to a temporary file first so parallel processes never read a partial file.
        tmp_path = path.with_suffix(f".{os.getpid()}.usda")
        write_objective_case_usd(tmp_path, geometry)
        os.replace(tmp_path, path)
    return str(path)


@configclass
class SocketDepthsVariationCfg(VariationBaseCfg):
    """Configuration for SocketDepthsVariation."""

    depths_mm: list[int] = field(default_factory=list)
    """Socket depths in millimeters, socket 1 (+Y) first; a permutation of the case's depths. Empty draws one
    permutation uniformly at random, seeded by the build seed."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Uniform choice over the candidate depth assignments."""


class SocketDepthsVariation(BuildTimeVariationBase):
    """Assign the objective case's socket depths at build time, from a given permutation or a random one.

    The candidates are the permutations of the case's current depth set, so the cylinders still fit exactly one
    assignment. The case's USD is regenerated for the chosen depths (cached by content hash).

    Args:
        objective_case: The case whose socket depths the variation sets.
        cfg: Variation settings. By default the variation is off.
        name: Name of the variation on the case.
    """

    cfg: SocketDepthsVariationCfg

    def __init__(
        self,
        objective_case: ObjectiveCase,
        cfg: SocketDepthsVariationCfg | None = None,
        name: str = "socket_depths",
    ):
        super().__init__(cfg=cfg if cfg is not None else SocketDepthsVariationCfg(), name=name)
        self._objective_case = objective_case

    def _realize_at_build_time(self) -> None:
        current_depths_mm = self._objective_case.socket_depths_mm
        permutations = sorted(set(itertools.permutations(current_depths_mm)))
        if self.cfg.depths_mm:
            requested = tuple(int(depth_mm) for depth_mm in self.cfg.depths_mm)
            assert requested in permutations, (
                f"socket_depths.depths_mm={list(requested)} must be a permutation of the case's depths"
                f" {sorted(current_depths_mm)}."
            )
            choices = [requested]
        else:
            choices = permutations
        depths_mm = self.sampler.sample(num_samples=1, choices=choices)[0]
        self._objective_case.set_socket_depths_mm(depths_mm)


@register_asset
class ObjectiveCase(LibraryObject):
    """Objective case with four look-alike sockets of hidden depth and a hinged lid (fixed-base articulation).

    The USD is generated procedurally for the given socket depths; see the module docstring for the case frame. The
    lid joint has no position drive: its actuator only adds viscous damping, and gravity holds the lid open (resting
    on its upper limit) or closed (resting on its lower limit). The ``socket_depths`` variation sets the depths per
    build without editing the environment.
    """

    registry_name = "objective_case"
    tags = ["object", "articulation", "procedural", "objective_case_depths"]
    object_type = ObjectType.ARTICULATION

    def __init__(
        self,
        socket_depths_mm: Sequence[int] = DEFAULT_SOCKET_DEPTHS_MM,
        initial_lid_angle_deg: float = 108.0,
        lid_damping: float = 0.005,
        **kwargs,
    ):
        """Create the objective case.

        Args:
            socket_depths_mm: Depth of each socket below the rim in millimeters, socket 1 (+Y) first.
            initial_lid_angle_deg: Lid angle at reset in degrees; 0 is closed.
            lid_damping: Viscous damping of the lid hinge, in N*m*s/rad.
            **kwargs: Arguments passed to ``LibraryObject``.
        """
        import math

        from isaaclab.actuators import ImplicitActuatorCfg

        self.geometry = replace(
            DEFAULT_CASE_GEOMETRY, socket_depths_m=tuple(depth_mm / 1000.0 for depth_mm in socket_depths_mm)
        )
        self.usd_path = get_objective_case_usd_path(self.geometry)
        super().__init__(**kwargs)
        self.object_cfg.init_state.joint_pos = {LID_JOINT_NAME: math.radians(initial_lid_angle_deg)}
        self.object_cfg.init_state.joint_vel = {".*": 0.0}
        self.object_cfg.actuators = {
            "lid_bearing": ImplicitActuatorCfg(joint_names_expr=[LID_JOINT_NAME], stiffness=0.0, damping=lid_damping),
        }
        self.add_variation(SocketDepthsVariation(self))

    @property
    def socket_depths_mm(self) -> tuple[int, ...]:
        """Socket depths in millimeters, socket 1 (+Y) first."""
        return tuple(round(depth_m * 1000.0) for depth_m in self.geometry.socket_depths_m)

    def set_socket_depths_mm(self, socket_depths_mm: Sequence[int]) -> None:
        """Regenerate the case USD with new socket depths, socket 1 (+Y) first; the outer shape is unchanged."""
        self.geometry = replace(
            self.geometry, socket_depths_m=tuple(depth_mm / 1000.0 for depth_mm in socket_depths_mm)
        )
        self.usd_path = get_objective_case_usd_path(self.geometry)
        self.object_cfg.spawn.usd_path = self.usd_path
        print(f"[objective_case] '{self.name}' socket depths (mm, socket 1..4): {list(self.socket_depths_mm)}")


OBJECTIVE_CYLINDER_RADIUS_M = 0.013
OBJECTIVE_CYLINDER_COLORS = {
    # Length in mm: color band, following the microscope-objective magnification code (red, yellow, green, blue).
    42: (0.78, 0.14, 0.10),
    52: (0.88, 0.72, 0.12),
    62: (0.16, 0.58, 0.24),
    72: (0.14, 0.32, 0.78),
}


@register_asset
class ObjectiveCylinder(Object):
    """Upright cylinder ("objective") of a given length, packed into an objective case socket of equal depth."""

    registry_name = "objective_cylinder"
    tags = ["object", "graspable", "procedural", "objective_case_depths"]

    def __init__(
        self,
        length_mm: int = 42,
        mass_kg: float = 0.08,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        """Create an objective cylinder.

        Args:
            length_mm: Length in millimeters; one of 42, 52, 62 or 72. Sets the color band.
            mass_kg: Mass in kilograms.
            name: Name of the object in the scene. Defaults to ``objective_cylinder_<length_mm>mm``.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
        """
        assert length_mm in OBJECTIVE_CYLINDER_COLORS, f"length_mm must be one of {sorted(OBJECTIVE_CYLINDER_COLORS)}"
        self.length_m = length_mm / 1000.0
        spawner_cfg = sim_utils.CylinderCfg(
            radius=OBJECTIVE_CYLINDER_RADIUS_M,
            height=self.length_m,
            axis="Z",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=OBJECTIVE_CYLINDER_COLORS[length_mm], metallic=0.6, roughness=0.35
            ),
        )
        super().__init__(
            name=name if name is not None else f"objective_cylinder_{length_mm}mm",
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawner_cfg,
            initial_pose=initial_pose,
            tags=list(self.tags),
        )
