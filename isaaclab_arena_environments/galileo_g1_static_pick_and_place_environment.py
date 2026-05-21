# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static-base G1 pick-and-place environment (WBC stands the robot in place; no nav).

This is a same-shelf-only variant of ``galileo_g1_locomanip_pick_and_place``: same
``galileo_locomanip`` background, same OpenXR retargeter, same 23-D action layout. The
default embodiment is ``g1_wbc_agile_pink`` (AGILE end-to-end velocity policy) instead of
``g1_wbc_pink`` (HOMIE stand+walk pair) -- the static task never walks, so the AGILE
single-policy backend is a better fit than HOMIE's stand/walk model split. The
``g1_wbc_pink`` embodiment is still accepted via ``--embodiment`` for users who want
HOMIE behaviour. The other differences from the locomanip env are:

1. The destination plate sits on the *same* shelf as the apple (within arm's reach), so
   the robot never needs to drive its base anywhere -- WBC just holds the standing pose.
2. The apple and destination plate spawn at fixed same-shelf poses, so recorded demos
   and closed-loop evaluation start from the same object layout.
"""

from __future__ import annotations

import argparse
import warnings
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# Pose tuning constants (all values empirically validated -- see commit history for the
# manual procedure: spawn the destination plate at the locomanip apple z (0.0707) and
# read the final z after gravity-settle; spawn the apple and read its USD AABB):
#
# - SHELF_SURFACE_Z: measured by gravity-settling the plate (whose USD origin sits at
#   its bottom, i.e. BBox min_z = 0); the settled z = -0.030 in env-local frame is the
#   actual shelf-top z in the galileo_locomanip scene.
# - SHELF_AIRGAP: keeps PhysX from spawning objects in collider penetration with the
#   shelf on the first sim tick (which would otherwise launch them upward).
SHELF_SURFACE_Z = -0.030
SHELF_AIRGAP = 0.005
SHELF_SUPPORT_PATCH_SIZE = (0.8, 1.5, 0.04)
# Cuboid center: top face = SHELF_SURFACE_Z.
SHELF_SUPPORT_PATCH_CENTER = (0.62, 0.0, SHELF_SURFACE_Z - SHELF_SUPPORT_PATCH_SIZE[2] / 2.0)

# Object XY spawn pose (env-local frame, shelf-relative). X mirrors the locomanip env
# (the only on-shelf X we have ground-truth data for via the brown_box flow). Y values
# are tuned for the static same-shelf apple-to-plate setup: both objects sit left on the
# table, and the plate is close enough to reduce unnecessary reach while still clearing
# the apple.
PICK_UP_OBJECT_SPAWN_XY = (0.5785, 0.27)
DESTINATION_SPAWN_XY = (0.5785, 0.06)

# Per-asset Z offset from the asset's USD origin to its bottom face. Added on top of
# ``SHELF_SURFACE_Z + SHELF_AIRGAP`` so the asset's *bottom* lands on the shelf rather
# than its USD origin (which may sit anywhere inside the AABB depending on how the
# asset was authored). Measured from each asset's USD AABB. Assets not in this table
# are spawned with no Z compensation -- callers passing arbitrary ``--object`` /
# ``--destination`` values are expected to verify the resulting spawn pose visually.
_USD_ORIGIN_ABOVE_BOTTOM_M: dict[str, float] = {
    "apple_01_objaverse_robolab": 0.0171,  # BBox min_z = -0.019, max_z = 0.049
    "clay_plates_hot3d_robolab": 0.0,  # USD origin at plate bottom (BBox min_z = 0)
}

TUNED_PICK_UP_OBJECT_NAME = "apple_01_objaverse_robolab"
TUNED_DESTINATION_NAME = "clay_plates_hot3d_robolab"

# Per-asset uniform scale matching the tuned pick-up / destination pair. Assets not in
# this table spawn at scale=(1.0, 1.0, 1.0) with a one-shot warning so the user knows
# the resulting size may need visual verification.
_TUNED_SCALES: dict[str, tuple[float, float, float]] = {
    TUNED_PICK_UP_OBJECT_NAME: (0.009, 0.009, 0.009),
    TUNED_DESTINATION_NAME: (0.5, 0.5, 0.5),
}

# The sim scene includes these boxes on the shelf/table workspace used by
# the static pick-and-place task, where they can block or clutter the
# apple-to-plate interaction area.
_BACKGROUND_PRIMS_TO_DEACTIVATE: tuple[str, ...] = (
    "galileo_locomanip/BackgroundAssets/boxes/jetson_orin_06",
    "galileo_locomanip/BackgroundAssets/boxes/jetson_orin_03",
    "galileo_locomanip/BackgroundAssets/boxes/hesai_box_06",
)

# Mild open-arm posture using shoulder joints only. Shoulder yaw keeps the
# forearms in the liked orientation; shoulder roll moves the arms away from the torso.
_G1_STATIC_OPEN_ARM_JOINT_POS: dict[str, float] = {
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_shoulder_yaw_joint": 0.5,
    "right_shoulder_yaw_joint": -0.5,
}

_G1_FINGER_FRICTION_MATERIAL_PATH = "/World/Materials/g1_static_pick_place_high_friction_fingers"
_G1_FINGER_STATIC_FRICTION = 6.0
_G1_FINGER_DYNAMIC_FRICTION = 5.0
_G1_FINGER_PRIM_NAME_MARKERS: tuple[str, ...] = (
    "hand",
    "thumb",
    "index",
    "middle",
)


def _deactivate_background_prims(env, env_ids, prim_relative_paths: tuple[str, ...]) -> None:
    """Deactivate selected referenced background prims before simulation starts."""
    del env_ids
    stage = env.sim.stage
    for env_prim_path in env.scene.env_prim_paths:
        for prim_relative_path in prim_relative_paths:
            prim_path = f"{env_prim_path}/{prim_relative_path}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                stage.OverridePrim(prim_path).SetActive(False)
            else:
                warnings.warn(
                    f"_deactivate_background_prims: prim not found at '{prim_path}'; "
                    "the background asset may still be visible.",
                    stacklevel=1,
                )


def _apply_high_friction_to_g1_fingers(env, env_ids) -> None:
    """Bind a high-friction contact material to G1 hand/finger collision prims."""
    del env_ids
    from isaaclab.sim import bind_physics_material
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
    from pxr import UsdPhysics, UsdShade

    stage = env.sim.stage
    material_cfg = RigidBodyMaterialCfg(
        static_friction=_G1_FINGER_STATIC_FRICTION,
        dynamic_friction=_G1_FINGER_DYNAMIC_FRICTION,
        friction_combine_mode="max",
    )
    material_cfg.func(_G1_FINGER_FRICTION_MATERIAL_PATH, material_cfg)

    bound_count = 0
    for env_prim_path in env.scene.env_prim_paths:
        robot_prim_path = f"{env_prim_path}/Robot"
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not prim_path.startswith(robot_prim_path):
                continue
            prim_path_lower = prim_path.lower()
            if not any(marker in prim_path_lower for marker in _G1_FINGER_PRIM_NAME_MARKERS):
                continue
            applied_schemas = set(prim.GetAppliedSchemas())
            has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI) or "PhysicsCollisionAPI" in applied_schemas
            if not has_collision_api:
                continue

            bind_physics_material(prim_path, _G1_FINGER_FRICTION_MATERIAL_PATH, stage=stage)
            # bind_physics_material is decorated with apply_nested and returns None, so
            # verify the resulting direct physics material binding explicitly.
            material_binding_api = UsdShade.MaterialBindingAPI(prim)
            direct_binding = material_binding_api.GetDirectBinding("physics")
            if str(direct_binding.GetMaterialPath()) == _G1_FINGER_FRICTION_MATERIAL_PATH:
                bound_count += 1

    if bound_count == 0:
        warnings.warn(
            "_apply_high_friction_to_g1_fingers: no G1 hand/finger collision prims were found; "
            "apple grip friction may be unchanged.",
            stacklevel=1,
        )


def _shelf_spawn_z(asset_name: str) -> float:
    """Return the env-local Z to spawn ``asset_name`` flush on the shelf surface.

    Falls back to ``SHELF_SURFACE_Z + SHELF_AIRGAP`` (no USD-origin compensation) for
    assets we have not measured, with a one-shot warning so the user knows the spawn
    pose may need visual verification.
    """
    if asset_name in _USD_ORIGIN_ABOVE_BOTTOM_M:
        return SHELF_SURFACE_Z + SHELF_AIRGAP + _USD_ORIGIN_ABOVE_BOTTOM_M[asset_name]
    warnings.warn(
        "galileo_g1_static_pick_and_place: no measured USD-origin offset for "
        f"'{asset_name}'; spawning at shelf surface with no compensation. Verify "
        "the asset's bottom face actually lands on the shelf.",
        stacklevel=2,
    )
    return SHELF_SURFACE_Z + SHELF_AIRGAP


def _asset_scale(asset_name: str) -> tuple[float, float, float]:
    """Return the tuned uniform scale for ``asset_name``, or 1.0 with a warning."""
    if asset_name in _TUNED_SCALES:
        return _TUNED_SCALES[asset_name]
    warnings.warn(
        "galileo_g1_static_pick_and_place: no measured scale for "
        f"'{asset_name}'; spawning at scale=(1.0, 1.0, 1.0). Verify visually.",
        stacklevel=2,
    )
    return (1.0, 1.0, 1.0)


@register_environment
class GalileoG1StaticPickAndPlaceEnvironment(ExampleEnvironmentBase):
    """G1 (WBC-balanced, no nav) pick-and-place on the locomanip warehouse shelf.

    Defaults to the apple-to-plate pairing so this env composes cleanly into the existing
    apple-to-plate workflow (record_demos -> replay -> eval) without requiring locomotion.
    """

    name: str = "galileo_g1_static_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab import sim as sim_utils

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        # Reuse the locomanip background USD: it bakes in lighting and provides the same
        # shelf-in-front-of-robot geometry the locomanip env was tuned against.
        background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
        # The imported shelf mesh has uneven/perforated collision in the task region:
        # small objects can fall through parts of the visible shelf. Add an invisible
        # static cuboid flush with the shelf top so task objects see a clean support.
        # This has only reproduced in GPU simulation; CPU runs have not shown the issue.
        shelf_support = Object(
            name="static_pick_place_shelf_support",
            prim_path="{ENV_REGEX_NS}/static_pick_place_shelf_support",
            object_type=ObjectType.BASE,
            spawner_cfg=sim_utils.CuboidCfg(
                size=SHELF_SUPPORT_PATCH_SIZE,
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
                visible=False,
            ),
        )
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)(scale=_asset_scale(args_cli.object))
        destination = self.asset_registry.get_asset_by_name(args_cli.destination)(
            scale=_asset_scale(args_cli.destination)
        )
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        # Lock the 3 waist joints (yaw/roll/pitch) for this static task. The
        # ``g1_wbc_agile_pink`` / ``g1_wbc_pink`` embodiments add the waist joints to
        # the upper-body Pink IK active set so the IK can use the torso to extend arm
        # reach. For the static apple-to-plate workflow that's undesirable: data
        # collection is supposed to keep the torso fixed (see step_2_teleoperation.rst:
        # "keep the robot torso and body fixed during this static task"), and the
        # IK-driven waist motion gets baked into recorded ``robot_joint_pos`` /
        # ``robot_joint_vel`` observations and bleeds into downstream training.
        # Removing the waist from the IK's active set lets the lower-body AGILE
        # policy's waist=0 command stay (see ``G1AgilePolicy._build_joint_mappings``:
        # "AGILE only drives the 12 leg slots; the 3 waist slots stay at zero"), and
        # the stiff waist PD (~k=300 in ``G1_AGILE_CFG``) holds the torso at the
        # default neutral pose. Flag is opt-out so anyone wanting the IK-extended
        # reach can disable it without editing the env.
        if args_cli.lock_waist:
            g1_action_cfg = getattr(embodiment.action_config, "g1_action", None)
            if g1_action_cfg is not None and hasattr(g1_action_cfg, "upperbody_extra_active_joints"):
                g1_action_cfg.upperbody_extra_active_joints = []
            else:
                warnings.warn(
                    f"galileo_g1_static_pick_and_place: --lock_waist=True but embodiment "
                    f"'{args_cli.embodiment}' has no 'upperbody_extra_active_joints' on its "
                    "action config; the waist may still move via the IK.",
                    stacklevel=1,
                )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Robot pose mirrors the locomanip env exactly so the WBC controller stands the
        # robot up in the same shelf-relative spot. The controller dynamically lifts the
        # pelvis to ~z=0.74 at runtime; init_state.pos.z=0 is correct.
        embodiment.set_initial_pose(Pose(position_xyz=(0.20, 0.08, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        embodiment.scene_config.robot.init_state.joint_pos.update(_G1_STATIC_OPEN_ARM_JOINT_POS)
        shelf_support.set_initial_pose(
            Pose(position_xyz=SHELF_SUPPORT_PATCH_CENTER, rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
        pick_up_object_x, pick_up_object_y = PICK_UP_OBJECT_SPAWN_XY
        destination_x, destination_y = DESTINATION_SPAWN_XY
        pick_up_object_z = _shelf_spawn_z(args_cli.object)
        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(pick_up_object_x, pick_up_object_y, pick_up_object_z),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        destination.set_initial_pose(
            Pose(
                position_xyz=(destination_x, destination_y, _shelf_spawn_z(args_cli.destination)),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        if args_cli.task_description is not None:
            task_description = args_cli.task_description
        else:
            object_label = args_cli.object.replace("_", " ")
            destination_label = args_cli.destination.replace("_", " ")
            task_description = (
                f"Pick up the {object_label} from the shelf and place it onto the "
                f"{destination_label} on the same shelf next to it."
            )

        def env_cfg_callback(env_cfg):
            from isaaclab.managers import EventTermCfg

            # The source galileo_locomanip USD includes boxes that sit in the
            # static task workspace. Deactivate the referenced prims at startup so
            # they are absent from the composed scene for every cloned environment.
            env_cfg.events.deactivate_static_pick_place_background_prims = EventTermCfg(
                func=_deactivate_background_prims,
                mode="prestartup",
                params={"prim_relative_paths": _BACKGROUND_PRIMS_TO_DEACTIVATE},
            )
            env_cfg.events.apply_high_friction_to_g1_fingers = EventTermCfg(
                func=_apply_high_friction_to_g1_fingers,
                mode="prestartup",
            )
            # The GR00T policy consumes camera observations. Force one RTX sensor
            # refresh after reset so the next policy query does not see the
            # previous episode's final rendered frame.
            env_cfg.num_rerenders_on_reset = 1
            return env_cfg

        scene = Scene(assets=[background, shelf_support, pick_up_object, destination])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(
                pick_up_object=pick_up_object,
                destination_location=destination,
                background_scene=background,
                episode_length_s=6.0,
                task_description=task_description,
                # Mirror the locomanip env's success thresholds so metrics are comparable.
                force_threshold=0.5,
                velocity_threshold=0.1,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default=TUNED_PICK_UP_OBJECT_NAME)
        parser.add_argument("--destination", type=str, default=TUNED_DESTINATION_NAME)
        # Default embodiment is g1_wbc_agile_pink: AGILE end-to-end velocity policy for
        # whole-body balance + PinkIK upper body. The static task never walks, so AGILE's
        # single-policy backend is a better fit than HOMIE's stand+walk split (which
        # ``g1_wbc_pink`` ships). Same 23-D action layout and OpenXR retargeter as the
        # locomanip env -- the only knob that flips is which lower-body ONNX policy gets
        # loaded by the WBC factory. ``g1_wbc_pink`` is still accepted as an override
        # for users who specifically want HOMIE.
        parser.add_argument("--embodiment", type=str, default="g1_wbc_agile_pink")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument(
            "--task_description",
            type=str,
            default=None,
            help=(
                "Override the natural-language task description. Defaults to a template "
                "derived from --object and --destination."
            ),
        )
        # The static task is upper-body-only by design, so we lock the 3 waist
        # joints by default. Pass ``--no-lock_waist`` to fall back to the default
        # AGILE-pink behaviour (waist active in Pink IK for extended arm reach).
        parser.add_argument(
            "--lock_waist",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Remove waist_yaw/roll/pitch from the upper-body Pink IK active set so "
                "the torso stays fixed during teleoperation and recorded observations. "
                "On by default for this static task; pass --no-lock_waist to allow the "
                "IK to use the waist for extended arm reach (the production AGILE-pink "
                "default)."
            ),
        )
