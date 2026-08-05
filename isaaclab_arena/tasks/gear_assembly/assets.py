# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scene assets for the Arena Gear Assembly task."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.tasks.gear_assembly.specs import (
    MAPLE_TABLE_TOP_COLLISION_SIZE,
    MAPLE_TABLE_TOP_COLLISION_THICKNESS,
    NEWTON_GEAR_CONTACT_OFFSET,
    NEWTON_GEAR_OFFSETS,
)
from isaaclab_arena.utils.pose import Pose

GEAR_ASSET_ROOT = f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets"
GEAR_GREEN_DIFFUSE_COLOR = (0.0, 0.8, 0.2)
GEAR_GREEN_VISUAL_MATERIAL_PATH = "green_material"
MAPLE_TABLE_TOP_COLLISION_COLOR = (0.43, 0.28, 0.15)
MAPLE_TABLE_LEG_RENDER_COLOR = (0.2, 0.22, 0.24)
NEWTON_GEAR_MAX_DEPENETRATION_VELOCITY = 1.0


class GearAssemblyRigidObject(Object):
    """Rigid object wrapper preserving the source task's contact-sensor setting."""

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=self._get_spawn_cfg(activate_contact_sensors=False),
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(object_cfg)


@clone
def spawn_newton_mesh_collision_usd(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a USD while making its mesh collision leaves visible to Newton."""
    from isaaclab.sim import schemas
    from isaaclab.sim.utils import (
        bind_visual_material,
        create_prim,
        get_current_stage,
        make_uninstanceable,
        select_usd_variants,
    )
    from isaaclab.utils.assets import check_file_path, retrieve_file_path
    from isaaclab.utils.version import has_kit

    usd_path = cfg.usd_path
    file_status = check_file_path(usd_path)
    if file_status == 0:
        raise FileNotFoundError(f"USD file not found at path: {usd_path}")
    if file_status == 2:
        usd_path = retrieve_file_path(usd_path, force_download=False)

    stage = get_current_stage()
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(
            prim_path,
            usd_path=usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
            stage=stage,
        )

    if cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)

    make_uninstanceable(prim_path, stage=stage)
    _author_newton_mesh_collision_leaves(stage, prim_path)

    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path, cfg.collision_props)
    if cfg.mass_props is not None:
        schemas.modify_mass_properties(prim_path, cfg.mass_props)

    if cfg.visual_material is not None and has_kit():
        material_path = (
            f"{prim_path}/{cfg.visual_material_path}"
            if not cfg.visual_material_path.startswith("/")
            else cfg.visual_material_path
        )
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(prim_path, material_path, stage=stage)

    return stage.GetPrimAtPath(prim_path)


def _author_newton_mesh_collision_leaves(stage, prim_path: str) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(prim_path)
    rigid_body = next((prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)), None)
    assert rigid_body is not None, f"No rigid body found below {prim_path}"
    source_meshes = [
        UsdGeom.Mesh(prim)
        for prim in Usd.PrimRange(root)
        if "/collisions" in str(prim.GetPath()) and prim.IsA(UsdGeom.Mesh)
    ]
    assert source_meshes, f"No collision meshes found below {prim_path}"

    xform_cache = UsdGeom.XformCache()
    body_to_world = xform_cache.GetLocalToWorldTransform(rigid_body)
    body_points = []
    for source_mesh in source_meshes:
        mesh_to_body = xform_cache.GetLocalToWorldTransform(source_mesh.GetPrim()) * body_to_world.GetInverse()
        body_points.extend(mesh_to_body.Transform(point) for point in source_mesh.GetPointsAttr().Get())
    body_center = Gf.Vec3d(*(
        (min(point[axis] for point in body_points) + max(point[axis] for point in body_points)) / 2.0
        for axis in range(3)
    ))
    for child_name in ("visuals", "collisions"):
        child = rigid_body.GetChild(child_name)
        if child.IsValid():
            UsdGeom.Xformable(child).AddTranslateOp().Set(-body_center)

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)

    is_base = "FactoryGearBase" in prim_path

    xform_cache.Clear()
    body_to_world = xform_cache.GetLocalToWorldTransform(rigid_body)
    collision_root = UsdGeom.Xform.Define(stage, rigid_body.GetPath().AppendChild("newton_collisions"))
    collision_points = []
    for source_mesh in source_meshes:
        mesh_to_body = xform_cache.GetLocalToWorldTransform(source_mesh.GetPrim()) * body_to_world.GetInverse()
        collision_points.extend(Gf.Vec3f(mesh_to_body.Transform(point)) for point in source_mesh.GetPointsAttr().Get())

    if is_base:
        _author_newton_base_collision(collision_root, collision_points)
    else:
        _author_newton_gear_collision(collision_root, collision_points)


def _author_newton_base_collision(collision_root, points) -> None:
    """Approximate the base as a platform and three separate convex pegs."""
    z_levels = sorted({round(float(point[2]), 6) for point in points})
    assert len(z_levels) >= 3, "Gear base must contain a platform and raised pegs"
    z_min, platform_top, z_max = z_levels[0], z_levels[1], z_levels[-1]
    x_min = min(float(point[0]) for point in points)
    x_max = max(float(point[0]) for point in points)
    y_min = min(float(point[1]) for point in points)
    y_max = max(float(point[1]) for point in points)
    _author_box(collision_root, "platform", x_min, x_max, y_min, y_max, z_min, platform_top)

    for gear_name, offset in NEWTON_GEAR_OFFSETS.items():
        center_x, center_y = offset[:2]
        peg_points = [
            point
            for point in points
            if float(point[2]) > platform_top + 1.0e-5 and abs(float(point[0]) - center_x) < 0.02
        ]
        assert peg_points, f"Could not find the {gear_name} base peg"
        radius = max(math.hypot(float(point[0]) - center_x, float(point[1]) - center_y) for point in peg_points)
        _author_convex_cylinder(
            collision_root,
            f"{gear_name}_peg",
            (center_x, center_y),
            radius,
            platform_top,
            z_max,
        )


def _author_newton_gear_collision(collision_root, points) -> None:
    """Approximate a movable gear as two convex annular tiers.

    Newton convexifies a concave mesh leaf, which fills both the center bore and
    the step between the gear plate and its hub. Separate convex leaves retain
    those features and provide a stable planar footprint.
    """
    z_min = min(float(point[2]) for point in points)
    z_max = max(float(point[2]) for point in points)
    radial_points = [(float(point[0]) ** 2 + float(point[1]) ** 2) ** 0.5 for point in points]
    bore_radius = min(radial_points)
    hub_radius = max(
        radius for point, radius in zip(points, radial_points, strict=True) if float(point[2]) > z_max - 1.0e-5
    )
    plate_top = max(
        float(point[2]) for point, radius in zip(points, radial_points, strict=True) if radius > hub_radius + 1.0e-4
    )
    segments = 6
    inner_radius = (bore_radius + 2.0 * NEWTON_GEAR_CONTACT_OFFSET) / math.cos(math.pi / segments)
    plate_radius = max(radial_points)

    _author_convex_annulus(
        collision_root,
        "plate",
        inner_radius,
        plate_radius,
        z_min,
        plate_top,
        segments,
    )
    _author_convex_annulus(collision_root, "hub", inner_radius, hub_radius, plate_top, z_max, segments)


def _author_box(
    collision_root,
    name: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> None:
    """Author a convex box mesh."""
    from pxr import Gf

    points = [
        Gf.Vec3f(x, y, z)
        for z in (z_min, z_max)
        for x, y in ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
    ]
    _author_convex_mesh(
        collision_root,
        name,
        points,
        [4] * 6,
        [3, 2, 1, 0, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7],
    )


def _author_convex_cylinder(
    collision_root,
    name: str,
    center,
    radius: float,
    z_min: float,
    z_max: float,
    segments: int = 12,
) -> None:
    """Author a convex polygonal cylinder mesh."""
    from pxr import Gf

    points = [
        Gf.Vec3f(
            center[0] + radius * math.cos(2.0 * math.pi * index / segments),
            center[1] + radius * math.sin(2.0 * math.pi * index / segments),
            z,
        )
        for z in (z_min, z_max)
        for index in range(segments)
    ]
    face_counts = [segments, segments, *([4] * segments)]
    face_indices = list(reversed(range(segments))) + list(range(segments, 2 * segments))
    for index in range(segments):
        next_index = (index + 1) % segments
        face_indices.extend((index, next_index, segments + next_index, segments + index))
    _author_convex_mesh(collision_root, name, points, face_counts, face_indices)


def _author_convex_annulus(
    collision_root,
    name: str,
    inner_radius: float,
    outer_radius: float,
    z_min: float,
    z_max: float,
    segments: int = 6,
) -> None:
    """Author an annular prism as convex mesh leaves while retaining its bore."""
    from pxr import Gf

    face_counts = [4] * 6
    face_indices = [3, 2, 1, 0, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
    angle_offset = math.pi / segments
    for index in range(segments):
        angles = (
            angle_offset + 2.0 * math.pi * index / segments,
            angle_offset + 2.0 * math.pi * (index + 1) / segments,
        )
        xy_points = [
            (inner_radius * math.cos(angles[0]), inner_radius * math.sin(angles[0])),
            (outer_radius * math.cos(angles[0]), outer_radius * math.sin(angles[0])),
            (outer_radius * math.cos(angles[1]), outer_radius * math.sin(angles[1])),
            (inner_radius * math.cos(angles[1]), inner_radius * math.sin(angles[1])),
        ]
        points = [Gf.Vec3f(x, y, z) for z in (z_min, z_max) for x, y in xy_points]
        _author_convex_mesh(collision_root, f"{name}_{index}", points, face_counts, face_indices)


def _author_convex_mesh(collision_root, name: str, points, face_counts, face_indices) -> None:
    """Author one Newton-readable convex mesh leaf."""
    from pxr import UsdGeom, UsdPhysics

    stage = collision_root.GetPrim().GetStage()
    mesh = UsdGeom.Mesh.Define(stage, collision_root.GetPath().AppendChild(name))
    mesh.CreatePointsAttr().Set(points)
    mesh.CreateFaceVertexCountsAttr().Set(face_counts)
    mesh.CreateFaceVertexIndicesAttr().Set(face_indices)
    mesh.CreateExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(points))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("convexHull")


@clone
def spawn_newton_maple_table_usd(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the maple table with Newton-readable render colors."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade

    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = prim.GetStage()
    for descendant in Usd.PrimRange(prim):
        if descendant.GetName() == "top" and descendant.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(descendant).CreateCollisionEnabledAttr().Set(False)
    _bind_newton_omnipbr_color(
        stage,
        f"{prim_path}/table/table_01/top",
        f"{prim_path}/Looks/newton_maple_top",
        MAPLE_TABLE_TOP_COLLISION_COLOR,
        Gf,
        Sdf,
        UsdShade,
    )
    for leg_index in range(4):
        _bind_newton_omnipbr_color(
            stage,
            f"{prim_path}/table/table_01/leg_{leg_index}",
            f"{prim_path}/Looks/newton_table_legs",
            MAPLE_TABLE_LEG_RENDER_COLOR,
            Gf,
            Sdf,
            UsdShade,
        )
    return prim


def _bind_newton_omnipbr_color(
    stage,
    shape_path: str,
    material_path: str,
    color: tuple[float, float, float],
    Gf,
    Sdf,
    UsdShade,
) -> None:
    shape_prim = stage.GetPrimAtPath(shape_path)
    if not shape_prim.IsValid():
        return

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/OmniPBRShader")
    shader_prim = shader.GetPrim()
    shader_prim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader_prim.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("OmniPBR")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
    UsdShade.MaterialBindingAPI.Apply(shape_prim)
    UsdShade.MaterialBindingAPI(shape_prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


def _author_display_color(stage, prim_path: str, color: tuple[float, float, float]) -> None:
    from pxr import Gf, Sdf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    display_color = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
    )
    display_color.Set([Gf.Vec3f(*color)])


@clone
def spawn_maple_table_top_collision(
    prim_path: str,
    cfg: sim_utils.CuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the tabletop collision proxy with a Newton-visible maple color."""
    from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid

    prim = spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = prim.GetStage()
    _author_display_color(stage, prim_path, MAPLE_TABLE_TOP_COLLISION_COLOR)
    _author_display_color(stage, f"{prim_path}/geometry/mesh", MAPLE_TABLE_TOP_COLLISION_COLOR)
    return prim


def _gear_spawn_cfg(
    kinematic_enabled: bool,
    newton_mesh_collisions: bool,
    visual_diffuse_color: tuple[float, float, float] | None = None,
) -> dict:
    max_depenetration_velocity = 5.0
    contact_offset = 0.02
    if newton_mesh_collisions:
        max_depenetration_velocity = NEWTON_GEAR_MAX_DEPENETRATION_VELOCITY
        contact_offset = NEWTON_GEAR_CONTACT_OFFSET

    cfg = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=kinematic_enabled,
            max_depenetration_velocity=max_depenetration_velocity,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        "mass_props": sim_utils.MassPropertiesCfg(mass=None),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=contact_offset, rest_offset=0.0),
    }
    if visual_diffuse_color is not None:
        cfg["visual_material"] = sim_utils.PreviewSurfaceCfg(diffuse_color=visual_diffuse_color, roughness=0.55)
        cfg["visual_material_path"] = GEAR_GREEN_VISUAL_MATERIAL_PATH
    if newton_mesh_collisions:
        cfg["func"] = spawn_newton_mesh_collision_usd
    return cfg


def make_factory_gear(
    name: str,
    prim_name: str,
    usd_leaf: str,
    pose: Pose,
    kinematic_enabled: bool = False,
    newton_mesh_collisions: bool = False,
    visual_diffuse_color: tuple[float, float, float] | None = None,
) -> Object:
    """Create one source-parity Factory gear rigid object."""
    gear = GearAssemblyRigidObject(
        name=name,
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        object_type=ObjectType.RIGID,
        usd_path=f"{GEAR_ASSET_ROOT}/{usd_leaf}/{usd_leaf}.usd",
        initial_pose=pose,
        spawn_cfg_addon=_gear_spawn_cfg(
            kinematic_enabled=kinematic_enabled,
            newton_mesh_collisions=newton_mesh_collisions,
            visual_diffuse_color=visual_diffuse_color,
        ),
    )
    gear.disable_reset_pose()
    return gear


def make_factory_gear_base(pose: Pose, newton_mesh_collisions: bool = False) -> Object:
    """Create the kinematic source Factory gear base."""
    return make_factory_gear(
        name="factory_gear_base",
        prim_name="FactoryGearBase",
        usd_leaf="factory_gear_base",
        pose=pose,
        kinematic_enabled=True,
        newton_mesh_collisions=newton_mesh_collisions,
    )


def make_factory_gear_small(pose: Pose, newton_mesh_collisions: bool = False) -> Object:
    """Create the source small Factory gear."""
    return make_factory_gear(
        "factory_gear_small",
        "FactoryGearSmall",
        "factory_gear_small",
        pose,
        newton_mesh_collisions=newton_mesh_collisions,
        visual_diffuse_color=GEAR_GREEN_DIFFUSE_COLOR,
    )


def make_factory_gear_medium(pose: Pose, newton_mesh_collisions: bool = False) -> Object:
    """Create the source medium Factory gear."""
    return make_factory_gear(
        "factory_gear_medium",
        "FactoryGearMedium",
        "factory_gear_medium",
        pose,
        newton_mesh_collisions=newton_mesh_collisions,
        visual_diffuse_color=GEAR_GREEN_DIFFUSE_COLOR,
    )


def make_factory_gear_large(pose: Pose, newton_mesh_collisions: bool = False) -> Object:
    """Create the source large Factory gear."""
    return make_factory_gear(
        "factory_gear_large",
        "FactoryGearLarge",
        "factory_gear_large",
        pose,
        newton_mesh_collisions=newton_mesh_collisions,
        visual_diffuse_color=GEAR_GREEN_DIFFUSE_COLOR,
    )


def make_ground() -> Object:
    """Create the source ground plane."""
    return Object(
        name="ground",
        prim_path="/World/ground",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.GroundPlaneCfg(),
        initial_pose=Pose(position_xyz=(0.0, 0.0, -1.05)),
    )


def make_maple_table_top_collision(pose: Pose) -> Object:
    """Create a Newton-safe collision surface for the maple tabletop."""
    collider_pose = Pose(
        position_xyz=(
            pose.position_xyz[0],
            pose.position_xyz[1],
            pose.position_xyz[2] - MAPLE_TABLE_TOP_COLLISION_THICKNESS / 2.0,
        ),
        rotation_xyzw=pose.rotation_xyzw,
    )
    return Object(
        name="maple_table_top_collision",
        prim_path="{ENV_REGEX_NS}/maple_table_top_collision",
        object_type=ObjectType.RIGID,
        spawner_cfg=sim_utils.CuboidCfg(
            func=spawn_maple_table_top_collision,
            size=(*MAPLE_TABLE_TOP_COLLISION_SIZE, MAPLE_TABLE_TOP_COLLISION_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=NEWTON_GEAR_CONTACT_OFFSET),
            visible=True,
        ),
        initial_pose=collider_pose,
        tags=["background", "collision"],
    )


def make_stand() -> Object:
    """Create the source vertical stand asset."""
    return Object(
        name="stand",
        prim_path="{ENV_REGEX_NS}/Stand",
        object_type=ObjectType.BASE,
        spawner_cfg=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )
