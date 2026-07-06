# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_cfg import ArenaEnvironmentCfg
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


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
    pairs keep the templated name produced by ``G1PickAndPlaceMimicEnvCfg``.
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


@dataclass
class GalileoG1LocomanipPickAndPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Galileo G1 locomotion-and-manipulation environment."""

    enable_cameras: bool = False
    object: str = "brown_box"
    destination: str = "blue_sorting_bin"
    embodiment: str = "g1_wbc_pink"
    teleop_device: str | None = None
    task_description: str | None = None
    mimic: bool = False
    auto: bool | None = None
    """Legacy auto-annotation flag; ``None`` records that the caller did not define it."""


@register_environment
class GalileoG1LocomanipPickAndPlaceEnvironment(ExampleEnvironmentBase[GalileoG1LocomanipPickAndPlaceEnvironmentCfg]):

    name: str = "galileo_g1_locomanip_pick_and_place"
    _legacy_argparse_cfg_type = GalileoG1LocomanipPickAndPlaceEnvironmentCfg

    def build(self, cfg: GalileoG1LocomanipPickAndPlaceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import G1PickAndPlaceMimicEnvCfg, PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose, PoseRange

        background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)()
        destination = self.asset_registry.get_asset_by_name(cfg.destination)()
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
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

        if cfg.embodiment == "g1_wbc_pink" and cfg.mimic and cfg.auto is None:
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

        if cfg.task_description is not None:
            task_description = cfg.task_description
        elif _is_legacy_pair(cfg.object, cfg.destination):
            task_description = _LEGACY_BROWN_BOX_TO_BLUE_BIN_DESCRIPTION
        else:
            object_label = cfg.object.replace("_", " ")
            destination_label = cfg.destination.replace("_", " ")
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

        def _build_g1_pick_and_place_mimic_cfg(arm_mode):
            return G1PickAndPlaceMimicEnvCfg(
                pick_up_object_name=pick_up_object.name,
                destination_location_name=destination.name,
                arm_mode=arm_mode,
            )

        scene = Scene(assets=[background, pick_up_object, destination])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(
                pick_up_object,
                destination,
                background,
                episode_length_s=30.0,
                task_description=task_description,
                force_threshold=0.5,
                velocity_threshold=0.1,
                mimic_env_cfg_factory=_build_g1_pick_and_place_mimic_cfg,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
        return isaaclab_arena_environment
