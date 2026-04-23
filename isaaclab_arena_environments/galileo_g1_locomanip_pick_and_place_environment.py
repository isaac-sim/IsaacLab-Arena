# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import math
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


logger = logging.getLogger(__name__)


# The v0.2 brown-box-to-blue-bin workflow was SQA'd against this exact task description, Mimic
# datagen name, and (object, destination) pair. We preserve them verbatim for that specific pair so
# the pretrained gr00t model and existing Mimic datasets / policy checkpoints keyed on them keep
# resolving bit-identically. Any other pair -- including ``brown_box`` against a non-default
# destination -- uses the templated behavior from the base task / environment.
_LEGACY_PICK_UP_OBJECT_NAME = "brown_box"
_LEGACY_DESTINATION_NAME = "blue_sorting_bin"
_LEGACY_DATAGEN_NAME = "locomanip_pick_and_place_D0"
_LEGACY_BROWN_BOX_TO_BLUE_BIN_DESCRIPTION = (
    "Pick up the brown box from the shelf, and place it into the blue bin on the table located at the"
    " right of the shelf."
)


def _is_legacy_pair(pick_up_object_name: str, destination_name: str) -> bool:
    return pick_up_object_name == _LEGACY_PICK_UP_OBJECT_NAME and destination_name == _LEGACY_DESTINATION_NAME


def _apply_legacy_datagen_name_override(
    env_cfg: Any,
    pick_up_object_name: str,
    destination_name: str,
) -> Any:
    """Rewrite the Mimic ``datagen_config.name`` to the legacy value for the v0.2 workflow.

    Only applies to Mimic configs (where ``datagen_config`` exists) and only to the exact
    ``(brown_box, blue_sorting_bin)`` pair that was SQA'd against this datagen key. All other
    pairs keep the templated name produced by ``LocomanipPickAndPlaceMimicEnvCfg``.
    """
    if not _is_legacy_pair(pick_up_object_name, destination_name):
        return env_cfg

    datagen_config = getattr(env_cfg, "datagen_config", None)
    if datagen_config is None:
        return env_cfg

    print(
        f"Overriding Mimic datagen_config.name from {datagen_config.name} to the legacy {_LEGACY_DATAGEN_NAME}"
        "This preserves identical behavior with existing Mimic datasets"
        "Remove this in the future when checkpoints are retrained."
    )
    datagen_config.name = _LEGACY_DATAGEN_NAME
    return env_cfg


@register_environment
class GalileoG1LocomanipPickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "galileo_g1_locomanip_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.locomanip_pick_and_place_task import LocomanipPickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose, PoseRange

        background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        destination = self.asset_registry.get_asset_by_name(args_cli.destination)()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        XY_RANGE_M = 0.025
        pick_up_object.set_initial_pose(
            PoseRange(
                position_xyz_min=(0.5785 - XY_RANGE_M, 0.18 - XY_RANGE_M, 0.0707),
                position_xyz_max=(0.5785 + XY_RANGE_M, 0.18 + XY_RANGE_M, 0.0707),
                rpy_min=(math.pi, 0.0, math.pi),
                rpy_max=(math.pi, 0.0, math.pi),
            )
        )

        destination.set_initial_pose(
            Pose(
                position_xyz=(-0.2450, -1.6272, -0.2641),
                rotation_xyzw=(0.0, 0.0, 1.0, 0.0),
            )
        )
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.18, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        if (
            args_cli.embodiment == "g1_wbc_pink"
            and hasattr(args_cli, "mimic")
            and args_cli.mimic
            and not hasattr(args_cli, "auto")
        ):
            # Patch the Mimic generate function for locomanip use case
            from isaaclab_arena.utils.locomanip_mimic_patch import patch_g1_locomanip_mimic

            patch_g1_locomanip_mimic()

            # Set navigation p-controller for locomanip use case
            action_cfg = embodiment.get_action_cfg()
            action_cfg.g1_action.use_p_control = True
            # Set nav subgoals (x,y,heading) and turning_in_place flag for G1 WBC Pink navigation p-controller
            action_cfg.g1_action.navigation_subgoals = [
                ([0.18, 0.18, 0.0], False),
                ([0.18, 0.18, -1.78], True),
                ([-0.0955, -1.1070, -1.78], False),
                ([-0.0955, -1.1070, -1.78], False),
            ]

        if args_cli.task_description is not None:
            task_description = args_cli.task_description
        elif _is_legacy_pair(args_cli.object, args_cli.destination):
            task_description = _LEGACY_BROWN_BOX_TO_BLUE_BIN_DESCRIPTION
        else:
            object_label = args_cli.object.replace("_", " ")
            destination_label = args_cli.destination.replace("_", " ")
            task_description = (
                f"Pick up the {object_label} from the shelf, and place it on the {destination_label} on the table"
                " located at the right of the shelf."
            )

        def env_cfg_callback(env_cfg):
            return _apply_legacy_datagen_name_override(
                env_cfg,
                pick_up_object_name=pick_up_object.name,
                destination_name=destination.name,
            )

        scene = Scene(assets=[background, pick_up_object, destination])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=LocomanipPickAndPlaceTask(
                pick_up_object,
                destination,
                background,
                episode_length_s=30.0,
                task_description=task_description,
                force_threshold=0.5,
                velocity_threshold=0.1,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="brown_box")
        parser.add_argument("--destination", type=str, default="blue_sorting_bin")
        parser.add_argument("--embodiment", type=str, default="g1_wbc_pink")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument(
            "--task_description",
            type=str,
            default=None,
            help=(
                "Override the natural-language task description. Defaults to a template derived from --object "
                "and --destination."
            ),
        )
