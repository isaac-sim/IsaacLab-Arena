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
2. The apple's spawn pose is configured through ``APPLE_SPAWN_XY_RANGE_M`` (XY only);
   the branch sets that range to zero for deterministic demos. The destination plate
   stays at a fixed pose so the place target is identical across episodes.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase
from isaaclab_arena_environments.galileo_static_apple_scene import (
    TUNED_DESTINATION_NAME,
    TUNED_PICK_UP_OBJECT_NAME,
    build_static_apple_scene_assets,
    make_static_apple_env_cfg_callback,
    make_static_apple_task_description,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@register_environment
class GalileoG1StaticPickAndPlaceEnvironment(ExampleEnvironmentBase):
    """G1 (WBC-balanced, no nav) pick-and-place on the locomanip warehouse shelf.

    Defaults to the apple-to-plate pairing so this env composes cleanly into the existing
    apple-to-plate workflow (record_demos -> replay -> eval) without requiring locomotion.
    """

    name: str = "galileo_g1_static_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.mdp.galileo_g1_static_pick_and_place.robot_configs import (
            G1_STATIC_FINGER_DYNAMIC_FRICTION,
            G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
            G1_STATIC_FINGER_PRIM_NAME_MARKERS,
            G1_STATIC_FINGER_STATIC_FRICTION,
            G1_STATIC_OPEN_ARM_JOINT_POS,
        )

        scene_assets = build_static_apple_scene_assets(
            self.asset_registry,
            pick_up_object_name=args_cli.object,
            destination_name=args_cli.destination,
            warning_prefix=self.name,
        )
        background = scene_assets.background
        shelf_support = scene_assets.shelf_support
        pick_up_object = scene_assets.pick_up_object
        destination = scene_assets.destination

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
            lock_waist=args_cli.lock_waist,
        )
        embodiment.set_finger_contact_friction(
            material_path=G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
            static_friction=G1_STATIC_FINGER_STATIC_FRICTION,
            dynamic_friction=G1_STATIC_FINGER_DYNAMIC_FRICTION,
            prim_name_markers=G1_STATIC_FINGER_PRIM_NAME_MARKERS,
        )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Robot pose is tuned for the same-shelf static task: slightly forward toward
        # the table while preserving the lateral offset that keeps both arms usable.
        # The controller dynamically lifts the pelvis to ~z=0.74 at runtime;
        # init_state.pos.z=0 is correct.
        embodiment.set_initial_pose(Pose(position_xyz=(0.25, 0.08, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        embodiment.set_joint_initial_pos(G1_STATIC_OPEN_ARM_JOINT_POS)

        if args_cli.task_description is not None:
            task_description = args_cli.task_description
        else:
            task_description = make_static_apple_task_description(args_cli.object, args_cli.destination)

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
            # The GR00T policy consumes camera observations. Force one RTX sensor
            # refresh after reset so the next policy query does not see the
            # previous episode's final rendered frame.
            env_cfg_callback=make_static_apple_env_cfg_callback(num_rerenders_on_reset=1),
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
