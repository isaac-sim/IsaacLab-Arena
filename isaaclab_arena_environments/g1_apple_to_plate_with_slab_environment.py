# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 apple-to-plate environment.

Floor is at env-local z = 0; all heights in this file are above the floor.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


TABLE_TOP_Z = 0.745  # m
TABLE_TOP_THICKNESS = 0.04  # m
TABLE_TOP_SIZE = (0.8, 1.5, TABLE_TOP_THICKNESS)
TABLE_TOP_CENTER = (0.74, 0.0, TABLE_TOP_Z - TABLE_TOP_THICKNESS / 2.0)
TABLE_AIRGAP = 0.003  # m; avoids spawn penetration

# Invisible slab: braces pelvis from behind, never underlies apple/plate.
TABLE_SUPPORT_PATCH_SIZE = (0.14, 1.4, 0.04)
TABLE_SUPPORT_PATCH_CENTER = (0.27, 0.0, 0.745)

BASELINE_MIDPOINT_Y = 0.0

# MIRROR_RIGHT=1: negates Y for right-hand teleop.
_MIRROR = bool(os.environ.get("MIRROR_RIGHT"))
PICK_UP_OBJECT_SPAWN_XY = (0.5967, -0.1472 if _MIRROR else 0.1472)
DESTINATION_SPAWN_XY = (0.60, 0.05 if _MIRROR else -0.05)

# Per-episode spawn randomization — three levels selectable via env var:
#
#   Level    | Env var    | Apple XY | Plate XY | Base XY | Base yaw | Arm joints
#   ---------|------------|----------|----------|---------|----------|-----------
#   FULLRAND | (default)  | ±7.5 cm  | ±5 cm    | ±1.5 cm | ±1.5°   | ±2.25°
#   MEDRAND  | MEDRAND=1  | ±2 cm    | ±2 cm    | ±1 cm   | ±0.75°  | ±1.0°
#   ZERODR   | ZERODR=1   | 0        | 0        | 0       | 0°      | 0°
_TRUTHY_ENV_VALUES = ("1", "true", "yes", "on")
_FULLRAND = os.environ.get("FULLRAND", "").strip().lower() in _TRUTHY_ENV_VALUES
_MEDRAND = os.environ.get("MEDRAND", "").strip().lower() in _TRUTHY_ENV_VALUES
_ZERODR = os.environ.get("ZERODR", "").strip().lower() in _TRUTHY_ENV_VALUES
assert (
    _FULLRAND + _MEDRAND + _ZERODR <= 1
), f"At most one DR level may be set, got FULLRAND={_FULLRAND} MEDRAND={_MEDRAND} ZERODR={_ZERODR}."

# APPLE_RANGE_OVERRIDE_M: optional float (m) overrides the apple XY half-range.
_apple_range_override = os.environ.get("APPLE_RANGE_OVERRIDE_M")
_apple_range_override_f = float(_apple_range_override) if _apple_range_override else None

APPLE_SPAWN_XY_RANGE_M = (
    0.0 if _ZERODR
    else (_apple_range_override_f if _apple_range_override_f is not None
          else (0.02 if _MEDRAND else 0.075))
)
PLATE_SPAWN_XY_RANGE_M = 0.0 if _ZERODR else (0.02 if _MEDRAND else 0.05)
BASE_SPAWN_XY_RANGE_M = 0.0 if _ZERODR else (0.01 if _MEDRAND else 0.015)
APPLE_RADIUS_M = 0.0375  # 7.5 cm diameter / 2
PLATE_RADIUS_M = 0.0951
BOUNDARY_GAP_M = 0.01
APPLE_CENTER_OFFSET_XY = (0.0013, -0.0004)  # apple root -> visual centre
PLATE_CENTER_OFFSET_XY = (-0.0080, 0.0118)  # plate root -> visual centre

# Z offset: USD origin to bottom face; added so asset bottom lands on table.
_USD_ORIGIN_ABOVE_BOTTOM_M: dict[str, float] = {
    "apple_01_objaverse_robolab": 0.0209,  # 7.5 cm Ø at scale 0.011
    "clay_plates_hot3d_robolab": 0.0,  # USD origin at plate bottom
}

TUNED_PICK_UP_OBJECT_NAME = "apple_01_objaverse_robolab"
TUNED_DESTINATION_NAME = "clay_plates_hot3d_robolab"

_TUNED_SCALES: dict[str, tuple[float, float, float]] = {
    TUNED_PICK_UP_OBJECT_NAME: (0.011, 0.011, 0.011),  # 7.5 cm diameter
    TUNED_DESTINATION_NAME: (0.63, 0.63, 0.63),  # 19.0 cm plate diameter
}


def _table_spawn_z(asset_name: str) -> float:
    """Return the env-local Z to spawn ``asset_name`` flush on the table surface."""
    if asset_name in _USD_ORIGIN_ABOVE_BOTTOM_M:
        return TABLE_TOP_Z + TABLE_AIRGAP + _USD_ORIGIN_ABOVE_BOTTOM_M[asset_name]
    warnings.warn(
        "g1_apple_to_plate_with_slab: no measured USD-origin offset for "
        f"'{asset_name}'; spawning at the table surface with no compensation. Verify "
        "the asset's bottom face actually lands on the table.",
        stacklevel=2,
    )
    return TABLE_TOP_Z + TABLE_AIRGAP


def _asset_scale(asset_name: str) -> tuple[float, float, float]:
    """Return the tuned uniform scale for ``asset_name``, or 1.0 with a warning."""
    if asset_name in _TUNED_SCALES:
        return _TUNED_SCALES[asset_name]
    warnings.warn(
        f"g1_apple_to_plate_with_slab: no measured scale for '{asset_name}'; spawning at scale=(1.0, 1.0, 1.0). "
        "Verify visually.",
        stacklevel=2,
    )
    return (1.0, 1.0, 1.0)


def _place_apple_clear_of_plate(
    env,
    env_ids,
    apple_cfg,
    plate_cfg,
    apple_range,
    plate_range,
    apple_radius,
    plate_radius,
    boundary_gap,
    apple_center_offset,
    plate_center_offset,
    max_tries=500,
):
    """Reset event: randomize the plate, then rejection-sample the apple until their
    boundaries are ``boundary_gap`` m apart, so the apple never spawns on the plate.

    Both are written at rest at their default table height, so the first recorded frame is
    already settled.
    """
    import torch

    import warp as wp

    apple = env.scene[apple_cfg.name]
    plate = env.scene[plate_cfg.name]
    dev = apple.device
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]  # (n, 3) world

    a_pose = wp.to_torch(apple.data.default_root_pose)[env_ids].clone()  # (n, 7)
    p_pose = wp.to_torch(plate.data.default_root_pose)[env_ids].clone()
    a_base = a_pose[:, :2]
    p_base = p_pose[:, :2]

    a_off = torch.tensor(apple_center_offset, device=dev)
    p_off = torch.tensor(plate_center_offset, device=dev)
    min_center = apple_radius + plate_radius + boundary_gap  # centre-to-centre minimum

    plate_xy = p_base + (torch.rand((n, 2), device=dev) * 2.0 - 1.0) * plate_range
    plate_center = plate_xy + p_off
    apple_xy = a_base.clone()
    for i in range(n):
        for _ in range(max_tries):
            cand = a_base[i] + (torch.rand(2, device=dev) * 2.0 - 1.0) * apple_range
            if torch.linalg.norm((cand + a_off) - plate_center[i]) >= min_center:
                apple_xy[i] = cand
                break

    # Debug placement log; remove when no longer needed.
    for i in range(n):
        print(
            f"[place_apple_clear] env={int(env_ids[i])} "
            f"a_new=[{apple_xy[i, 0]:.6f}, {apple_xy[i, 1]:.6f}] "
            f"p_new=[{plate_xy[i, 0]:.6f}, {plate_xy[i, 1]:.6f}] "
            f"centre_dist={torch.linalg.norm((apple_xy[i] + a_off) - plate_center[i]).item():.3f} "
            f"(min {min_center:.3f})",
            flush=True,
        )
    zeros_vel = torch.zeros((n, 6), device=dev)
    for asset, base_pose, xy in ((plate, p_pose, plate_xy), (apple, a_pose, apple_xy)):
        pose = base_pose.clone()
        pose[:, 0:2] = xy
        pose[:, 0:3] = pose[:, 0:3] + origins  # env-local -> world
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=zeros_vel, env_ids=env_ids)


# D435 mount in torso_link frame; z includes 10 mm USD/MJCF frame offset.
_REALSENSE_CAMERA_OFFSET_TORSO = {
    "position_xyz": (0.057, 0.04, 0.4407),
    "rotation_xyzw": (-0.6532815, 0.6532815, -0.2705981, 0.2705981),
}
_REALSENSE_FOCAL_LENGTH_MM = 15.0  # mm
_REALSENSE_RESOLUTION_PX = (640, 480)
_REALSENSE_INTRINSICS_PX = {"fx": 606.906, "fy": 606.209, "cx": 330.565, "cy": 253.434}


def _realsense_usd_apertures() -> tuple[float, float, float, float]:
    """Convert the measured D435 pixel intrinsics to USD aperture millimetres.

    Returns:
        ``(horizontal_aperture, vertical_aperture, offset_h, offset_v)``, where the two
        offsets carry the principal point. ``offset_v`` is negated because the USD film
        back is y-up while image ``v`` runs down.
    """
    focal = _REALSENSE_FOCAL_LENGTH_MM
    width, height = _REALSENSE_RESOLUTION_PX
    intrinsics = _REALSENSE_INTRINSICS_PX
    horizontal_aperture = focal * width / intrinsics["fx"]
    vertical_aperture = focal * height / intrinsics["fy"]
    offset_h = (intrinsics["cx"] - width / 2.0) * horizontal_aperture / width
    offset_v = -((intrinsics["cy"] - height / 2.0) * vertical_aperture / height)
    return horizontal_aperture, vertical_aperture, offset_h, offset_v


(
    _REALSENSE_HORIZONTAL_APERTURE,
    _REALSENSE_VERTICAL_APERTURE,
    _REALSENSE_APERTURE_OFFSET_H,
    _REALSENSE_APERTURE_OFFSET_V,
) = _realsense_usd_apertures()
_REALSENSE_CLIPPING_RANGE = (0.05, 100.0)


_GALILEO_ROOM = "galileo_locomanip"

_GALILEO_CLUTTER_PRIMS: tuple[str, ...] = (
    f"{_GALILEO_ROOM}/BackgroundAssets/office",
    f"{_GALILEO_ROOM}/BackgroundAssets/dollies",
    f"{_GALILEO_ROOM}/BackgroundAssets/rack",
    f"{_GALILEO_ROOM}/BackgroundAssets/pallets",
    f"{_GALILEO_ROOM}/BackgroundAssets/boxes",
    f"{_GALILEO_ROOM}/BackgroundAssets/crate",
    f"{_GALILEO_ROOM}/BackgroundAssets/closet",
    f"{_GALILEO_ROOM}/BackgroundAssets/bins",
    f"{_GALILEO_ROOM}/BackgroundAssets/wood_platform",
    f"{_GALILEO_ROOM}/TaskAssets/shelf",
    f"{_GALILEO_ROOM}/TaskAssets/power_drill_physics",
)


def _stage_galileo_room(env, env_ids) -> None:
    """Prestartup: hide room clutter (including the room's own table)."""
    if os.environ.get("SKIP_ROOM_STAGING") == "1":
        return
    del env_ids

    stage = env.sim.stage
    for env_prim_path in env.scene.env_prim_paths:
        for rel in _GALILEO_CLUTTER_PRIMS:
            prim_path = f"{env_prim_path}/{rel}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                stage.OverridePrim(prim_path).SetActive(False)
        room_table = f"{env_prim_path}/{_GALILEO_ROOM}/TaskAssets/table"
        if stage.GetPrimAtPath(room_table).IsValid():
            stage.OverridePrim(room_table).SetActive(False)


# Mirrors deploy stack bring-up home pose; intentionally asymmetric.
G1_DEPLOY_HOME_ARM_JOINT_POS: dict[str, float] = {
    "left_shoulder_yaw_joint": 0.524,
    "right_shoulder_yaw_joint": -0.524,
    "left_elbow_joint": -0.349,
}

# Selected with ARM_HOME=static_open.
_G1_STATIC_OPEN_ARM_JOINT_POS: dict[str, float] = {
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_shoulder_yaw_joint": 0.5,
    "right_shoulder_yaw_joint": -0.5,
}
# Selected with ARM_HOME=dataset_mean.
_G1_DATASET_MEAN_JOINT_POS: dict[str, float] = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    "waist_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "waist_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_knee_joint": 0.3,
    "right_knee_joint": 0.3,
    "left_shoulder_pitch_joint": 0.00086,
    "right_shoulder_pitch_joint": -0.00071,
    "left_ankle_pitch_joint": -0.2,
    "right_ankle_pitch_joint": -0.2,
    "left_shoulder_roll_joint": 0.00164,
    "right_shoulder_roll_joint": 0.00139,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.52566,
    "right_shoulder_yaw_joint": -0.522,
    "left_elbow_joint": -0.35207,
    "right_elbow_joint": 0.00086,
    "left_wrist_roll_joint": 0.00206,
    "right_wrist_roll_joint": -0.00157,
    "left_wrist_pitch_joint": 0.00035,
    "right_wrist_pitch_joint": -0.00271,
    "left_wrist_yaw_joint": 0.00125,
    "right_wrist_yaw_joint": 0.0007,
    "left_hand_index_0_joint": 0.0,
    "left_hand_middle_0_joint": 0.0,
    "left_hand_thumb_0_joint": 0.0,
    "right_hand_index_0_joint": 0.0,
    "right_hand_middle_0_joint": 0.0,
    "right_hand_thumb_0_joint": 0.0,
    "left_hand_index_1_joint": 0.0,
    "left_hand_middle_1_joint": 0.0,
    "left_hand_thumb_1_joint": 0.0,
    "right_hand_index_1_joint": 0.0,
    "right_hand_middle_1_joint": 0.0,
    "right_hand_thumb_1_joint": 0.0,
    "left_hand_thumb_2_joint": 0.0,
    "right_hand_thumb_2_joint": 0.0,
}

_arm_home_sel = os.environ.get("ARM_HOME", "deploy")
if _arm_home_sel == "dataset_mean":
    _ARM_HOME = _G1_DATASET_MEAN_JOINT_POS
elif _arm_home_sel == "static_open":
    _ARM_HOME = _G1_STATIC_OPEN_ARM_JOINT_POS
elif _arm_home_sel == "zero":
    # Arms straight down, clear of the table.
    _ARM_HOME = {
        f"{side}_{j}_joint": 0.0
        for side in ("left", "right")
        for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    }
else:
    _ARM_HOME = G1_DEPLOY_HOME_ARM_JOINT_POS


# Pale-sage plate albedo (linear), matched to the real plate.
# Texture connection on the clay asset must be broken before applying.
_CALIB_PLATE_ALBEDO_LINEAR = (0.41, 0.41, 0.355)

# Charcoal table top; default procedural cuboid renders near-white.
_TABLE_ALBEDO_LINEAR = (0.06, 0.06, 0.06)
_TABLE_ROUGHNESS = 0.92


def _apply_realsense_calibration(env, env_ids) -> None:
    """Startup: author the D435 calibration onto the head-cam prim.

    The spawner cfg carries only the horizontal aperture; the vertical aperture and both
    principal-point offsets are authored here so the sim K matrix matches the real unit.
    """
    del env_ids
    from pxr import Sdf

    stage = env.sim.stage
    for prim in stage.Traverse():
        if not prim.GetPath().pathString.endswith("RobotHeadCam"):
            continue
        prim.CreateAttribute("horizontalAperture", Sdf.ValueTypeNames.Float).Set(_REALSENSE_HORIZONTAL_APERTURE)
        prim.CreateAttribute("verticalAperture", Sdf.ValueTypeNames.Float).Set(_REALSENSE_VERTICAL_APERTURE)
        prim.CreateAttribute("horizontalApertureOffset", Sdf.ValueTypeNames.Float).Set(_REALSENSE_APERTURE_OFFSET_H)
        prim.CreateAttribute("verticalApertureOffset", Sdf.ValueTypeNames.Float).Set(_REALSENSE_APERTURE_OFFSET_V)


def _recolor_plate_to_real(env, env_ids) -> None:
    from pxr import Gf, Sdf, UsdShade

    stage = env.scene.stage
    rgb = Gf.Vec3f(*_CALIB_PLATE_ALBEDO_LINEAR)
    for env_prim_path in env.scene.env_prim_paths:
        root = stage.GetPrimAtPath(f"{env_prim_path}/{TUNED_DESTINATION_NAME}")
        if not root:
            continue
        mat_path = f"{env_prim_path}/{TUNED_DESTINATION_NAME}/CalibPlateMaterial"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgb)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(root).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


@dataclass
class G1AppleToPlateWithSlabEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the G1 apple-to-plate environment."""

    enable_cameras: bool = False
    object: str = TUNED_PICK_UP_OBJECT_NAME
    destination: str = TUNED_DESTINATION_NAME
    embodiment: str = "g1_wbc_agile_pink"
    """Use AGILE whole-body control by default; ``g1_wbc_pink`` selects HOMIE instead."""
    teleop_device: str | None = None
    task_description: str = "move the apple to the plate"
    lock_waist: bool = True
    """Keep the waist out of Pink IK unless extended arm reach is required."""


@register_environment
class G1AppleToPlateWithSlabEnvironment(ArenaEnvironmentFactory[G1AppleToPlateWithSlabEnvironmentCfg]):
    """G1 (WBC-balanced, no nav) apple-to-plate pick and place.

    The scene holds the apple and plate, ground, dome light, a procedural table, and an
    invisible collision slab bracing the pelvis from behind.
    """

    name: str = "g1_apple_to_plate_with_slab"
    _legacy_argparse_cfg_type = G1AppleToPlateWithSlabEnvironmentCfg

    def build(self, cfg: G1AppleToPlateWithSlabEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab import sim as sim_utils

        from isaaclab_arena.assets.background import Background
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.mdp.galileo_g1_static_pick_and_place.robot_configs import (
            G1_STATIC_FINGER_DYNAMIC_FRICTION,
            G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
            G1_STATIC_FINGER_PRIM_NAME_MARKERS,
            G1_STATIC_FINGER_STATIC_FRICTION,
        )

        background_room = self.asset_registry.get_asset_by_name(_GALILEO_ROOM)()
        # Room floor registered at z=-0.795; raise so floor lands at z=0.
        from isaaclab_arena.utils.pose import Pose as _RoomPose

        background_room.set_initial_pose(
            _RoomPose(position_xyz=(4.420, 1.408, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )

        class Table(Background):
            """Plain procedural table: a collidable slab whose top face is the worksurface.

            Doubles as the task's ``background_scene`` so the dropped-object failure
            uses its ``object_min_z``: anything below 2 cm has fallen to the floor.
            """

            def __init__(self):
                spawner_cfg = sim_utils.CuboidCfg(
                    size=TABLE_TOP_SIZE,
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=_TABLE_ALBEDO_LINEAR,
                        roughness=_TABLE_ROUGHNESS,
                    ),
                )
                super().__init__(
                    name="table",
                    usd_path=None,
                    object_min_z=0.02,
                    prim_path="{ENV_REGEX_NS}/table",
                    initial_pose=Pose(
                        position_xyz=TABLE_TOP_CENTER,
                        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    spawner_cfg=spawner_cfg,
                    tags=["background", "procedural"],
                )

        table = Table()

        class TableSupport(Object):
            """Invisible static collision slab that braces the pelvis from behind.

            A ``visible=False`` cuboid with collision props and ``ObjectType.BASE`` (static
            collider, no rigid-body dynamics). It sits behind the table's near edge and
            never underlies the apple or plate.
            """

            def __init__(self):
                spawner_cfg = sim_utils.CuboidCfg(
                    size=TABLE_SUPPORT_PATCH_SIZE,
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
                    visible=False,
                )
                super().__init__(
                    name="apple_to_plate_table_support",
                    prim_path="{ENV_REGEX_NS}/apple_to_plate_table_support",
                    object_type=ObjectType.BASE,
                    spawner_cfg=spawner_cfg,
                    initial_pose=Pose(
                        position_xyz=TABLE_SUPPORT_PATCH_CENTER,
                        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    tags=["background", "procedural"],
                )

        table_support = TableSupport()
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)(scale=_asset_scale(cfg.object))
        destination = self.asset_registry.get_asset_by_name(cfg.destination)(scale=_asset_scale(cfg.destination))
        from isaaclab_arena.embodiments.g1.g1 import G1CameraCfg
        from isaaclab_arena.utils.pose import Pose as _Pose

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
            lock_waist=cfg.lock_waist,
        )

        class _RealsenseG1CameraCfg(G1CameraCfg):
            """RealSense-aligned head camera for this env only.

            These must be class attributes: ``G1CameraCfg.__post_init__`` reads them via
            getattr at construction, and the combined scene cfg copies the already-built
            ``robot_head_cam`` field, so instance attributes set afterwards are too late.
            The parent is ``torso_link`` because the menagerie G1 nests ``head_link`` under
            it, leaving the stock ``Robot/head_link`` path unspawned.
            """

            # TiledCamera avoids all-black frames on reset_to kinematic re-render.
            _is_tiled_camera = True
            _camera_offset = _Pose(**_REALSENSE_CAMERA_OFFSET_TORSO)
            _parent_link = "torso_link"
            _horizontal_aperture = _REALSENSE_HORIZONTAL_APERTURE
            _clipping_range = _REALSENSE_CLIPPING_RANGE

        embodiment.camera_config = _RealsenseG1CameraCfg()
        embodiment.set_finger_contact_friction(
            material_path=G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
            static_friction=G1_STATIC_FINGER_STATIC_FRICTION,
            dynamic_friction=G1_STATIC_FINGER_DYNAMIC_FRICTION,
            prim_name_markers=G1_STATIC_FINGER_PRIM_NAME_MARKERS,
        )

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        # AGILE policy has no get-up skill; pelvis must start at standing height.
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(0.25, BASELINE_MIDPOINT_Y, float(os.environ.get("PELVIS_Z", "0.75"))),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        embodiment.set_joint_initial_pos(_ARM_HOME)
        pick_up_object_x, pick_up_object_y = PICK_UP_OBJECT_SPAWN_XY
        destination_x, destination_y = DESTINATION_SPAWN_XY
        pick_up_object_z = _table_spawn_z(cfg.object)
        _dest_z = _table_spawn_z(cfg.destination)
        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(pick_up_object_x, pick_up_object_y, pick_up_object_z),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        destination.set_initial_pose(
            Pose(position_xyz=(destination_x, destination_y, _dest_z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )

        # WORLD_SINK=<m>: diagnostic; shifts scene down to world z=0.
        _sink = float(os.environ.get("WORLD_SINK", "0") or 0)
        if _sink:
            from isaaclab_arena.utils.pose import Pose as _P
            from isaaclab_arena.utils.pose import PoseRange as _PR

            def _shift(pose):
                if isinstance(pose, _PR):
                    lo = list(pose.position_xyz_min)
                    hi = list(pose.position_xyz_max)
                    lo[2] -= _sink
                    hi[2] -= _sink
                    return _PR(
                        position_xyz_min=tuple(lo),
                        position_xyz_max=tuple(hi),
                        rpy_min=pose.rpy_min,
                        rpy_max=pose.rpy_max,
                    )
                p = list(pose.position_xyz)
                p[2] -= _sink
                return _P(position_xyz=tuple(p), rotation_xyzw=pose.rotation_xyzw)

            for _a in (background_room, table, table_support, pick_up_object, destination, embodiment):
                _ip = _a.get_initial_pose() if hasattr(_a, "get_initial_pose") else getattr(_a, "initial_pose", None)
                if _ip is None:
                    _a.set_initial_pose(_P(position_xyz=(0.0, 0.0, -_sink), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
                else:
                    _a.set_initial_pose(_shift(_ip))
            table.object_min_z = 0.02 - _sink

        task_description = cfg.task_description

        def env_cfg_callback(env_cfg):
            import math

            from isaaclab.envs import mdp as base_mdp
            from isaaclab.managers import EventTermCfg, SceneEntityCfg

            env_cfg.events.randomize_robot_base = EventTermCfg(
                func=base_mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-BASE_SPAWN_XY_RANGE_M, BASE_SPAWN_XY_RANGE_M),
                        "y": (-BASE_SPAWN_XY_RANGE_M, BASE_SPAWN_XY_RANGE_M),
                        "yaw": (
                            (0.0, 0.0)
                            if _ZERODR
                            else (
                                (-math.radians(0.75), math.radians(0.75))
                                if _MEDRAND
                                else (-math.radians(1.5), math.radians(1.5))
                            )
                        ),
                    },
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            )
            env_cfg.events.randomize_arm_joints = EventTermCfg(
                func=base_mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (
                        (0.0, 0.0)
                        if _ZERODR
                        else (
                            (-math.radians(1.0), math.radians(1.0))
                            if _MEDRAND
                            else (-math.radians(2.25), math.radians(2.25))
                        )
                    ),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=[
                            ".*shoulder_(pitch|roll|yaw)_joint",
                            ".*elbow_joint",
                            ".*wrist_(roll|pitch|yaw)_joint",
                        ],
                    ),
                },
            )
            # NO_RANDOMIZE=1: diagnostic; drops both reset events.
            if os.environ.get("NO_RANDOMIZE") == "1":
                env_cfg.events.randomize_robot_base = None
                env_cfg.events.randomize_arm_joints = None

            if not _ZERODR:
                env_cfg.events.place_apple_clear = EventTermCfg(
                    func=_place_apple_clear_of_plate,
                    mode="reset",
                    params={
                        "apple_cfg": SceneEntityCfg(TUNED_PICK_UP_OBJECT_NAME),
                        "plate_cfg": SceneEntityCfg(TUNED_DESTINATION_NAME),
                        "apple_range": APPLE_SPAWN_XY_RANGE_M,
                        "plate_range": PLATE_SPAWN_XY_RANGE_M,
                        "apple_radius": APPLE_RADIUS_M,
                        "plate_radius": PLATE_RADIUS_M,
                        "boundary_gap": BOUNDARY_GAP_M,
                        "apple_center_offset": APPLE_CENTER_OFFSET_XY,
                        "plate_center_offset": PLATE_CENTER_OFFSET_XY,
                    },
                )

            env_cfg.events.stage_galileo_room = EventTermCfg(
                func=_stage_galileo_room,
                mode="prestartup",
            )
            env_cfg.events.recolor_plate_to_real = EventTermCfg(
                func=_recolor_plate_to_real,
                mode="startup",
            )
            env_cfg.events.apply_realsense_calibration = EventTermCfg(
                func=_apply_realsense_calibration,
                mode="startup",
            )
            # One RTX refresh so policy sees reset-frame, not prior episode.
            env_cfg.num_rerenders_on_reset = 1
            return env_cfg

        from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
        from isaaclab.utils import configclass as _configclass

        _pick_name, _dest_name = pick_up_object.name, destination.name

        @_configclass
        class AppleToPlateG1MimicEnvCfg(MimicEnvCfg):
            def __post_init__(self):
                super().__post_init__()
                self.datagen_config.name = f"apple_to_plate_{_pick_name}_to_{_dest_name}_D0"
                self.datagen_config.generation_guarantee = True
                self.datagen_config.generation_keep_failed = bool(os.environ.get("MIMIC_KEEP_FAILED", ""))
                self.datagen_config.generation_num_trials = 100
                self.datagen_config.generation_select_src_per_subtask = True
                self.datagen_config.generation_select_src_per_arm = False
                self.datagen_config.generation_relative = False
                self.datagen_config.generation_joint_pos = False
                self.datagen_config.generation_transform_first_robot_pose = False
                self.datagen_config.generation_interpolate_from_last_target_pose = True
                self.datagen_config.max_num_failures = 600
                self.datagen_config.seed = int(os.environ.get("MIMIC_SEED", "1"))
                self.datagen_config.use_navigation_controller = False

                _common = dict(
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.003,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                )
                self.subtask_configs["left"] = [
                    SubTaskConfig(
                        object_ref=_pick_name,
                        subtask_term_signal="grasp_left",
                        subtask_term_offset_range=(10, 20),
                        num_interpolation_steps=25,
                        **_common,
                    ),
                    SubTaskConfig(
                        object_ref=_dest_name,
                        subtask_term_signal=None,
                        subtask_term_offset_range=(0, 0),
                        num_interpolation_steps=25,
                        **_common,
                    ),
                ]
                self.subtask_configs["right"] = [
                    SubTaskConfig(
                        object_ref=_pick_name,
                        subtask_term_signal=None,
                        subtask_term_offset_range=(0, 0),
                        num_interpolation_steps=0,
                        **_common,
                    ),
                ]
                self.subtask_configs["body"] = [
                    SubTaskConfig(
                        object_ref=_pick_name,
                        subtask_term_signal=None,
                        subtask_term_offset_range=(0, 0),
                        num_interpolation_steps=0,
                        **_common,
                    ),
                ]

        def _build_g1_pick_and_place_mimic_cfg(arm_mode):
            return AppleToPlateG1MimicEnvCfg()

        # FLOOR_ONLY=1: diagnostic; moves table/objects 100 m away.
        if os.environ.get("FLOOR_ONLY") == "1":
            from isaaclab_arena.utils.pose import Pose as _FarPose

            table.set_initial_pose(
                _FarPose(position_xyz=(100.0, 100.0, TABLE_TOP_CENTER[2]), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
            )
            table_support.set_initial_pose(
                _FarPose(position_xyz=(100.0, 100.0, TABLE_SUPPORT_PATCH_CENTER[2]), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
            )
            pick_up_object.set_initial_pose(
                _FarPose(position_xyz=(100.0, 100.0, pick_up_object_z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
            )
            destination.set_initial_pose(
                _FarPose(position_xyz=(100.0, 100.0, _dest_z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
            )

        scene = Scene(assets=[background_room, table, table_support, pick_up_object, destination])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(
                pick_up_object=pick_up_object,
                destination_location=destination,
                background_scene=table,
                episode_length_s=35.0,
                task_description=task_description,
                force_threshold=0.5,
                velocity_threshold=0.1,
                # Apple must settle within plate footprint, not just touch.
                max_horizontal_distance=PLATE_RADIUS_M - APPLE_RADIUS_M,
                mimic_env_cfg_factory=_build_g1_pick_and_place_mimic_cfg,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
