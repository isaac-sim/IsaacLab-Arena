# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FR3 embodiment matching Isaac Cap easy tool-sorting tasks."""

from __future__ import annotations

from typing import ClassVar

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.isaac_cap.embodiments.insertion_task import (
    IndustrialFr3Robotiq2f85DifferentialIKEmbodiment,
    IndustrialFr3Robotiq2f85Embodiment,
)
from isaaclab_arena_environments.isaac_cap.embodiments.insertion_task.config import GRIPPER_JOINT_NAME

# AUTOLab easy tool-sort close command (radians), not the authored USD driver limit.
_TOOL_SORT_GRIPPER_CLOSED_RAD = 0.8
_TOOL_SORT_GRIPPER_STIFFNESS = 20.0
_TOOL_SORT_GRIPPER_DAMPING = 1.0


def _apply_tool_sort_camera_and_gripper(
    embodiment: IndustrialFr3Robotiq2f85Embodiment | IndustrialFr3Robotiq2f85DifferentialIKEmbodiment,
) -> None:
    """Apply easy tool-sort overhead camera and Robotiq actuator tuning."""
    embodiment.camera_config.configure_tool_sorting_overhead()
    driver = embodiment.scene_config.robot.actuators["robotiq_driver"]
    driver.stiffness = _TOOL_SORT_GRIPPER_STIFFNESS
    driver.damping = _TOOL_SORT_GRIPPER_DAMPING
    embodiment.action_config.gripper_action.close_command_expr = {
        GRIPPER_JOINT_NAME: _TOOL_SORT_GRIPPER_CLOSED_RAD,
    }


@register_asset
class ToolSortingFr3Robotiq2f85Embodiment(IndustrialFr3Robotiq2f85Embodiment):
    """Insertion-task FR3 stack with easy tool-sort camera and Robotiq tuning."""

    name = "tool_sorting_fr3_robotiq_2f85"
    tags: ClassVar[list[str]] = ["embodiment"]

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        use_tiled_cameras: bool = False,
    ):
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            initial_joint_pose=initial_joint_pose,
            concatenate_observation_terms=concatenate_observation_terms,
            arm_mode=arm_mode,
            use_tiled_cameras=use_tiled_cameras,
        )
        _apply_tool_sort_camera_and_gripper(self)


class ToolSortingFr3Robotiq2f85DifferentialIKEmbodiment(IndustrialFr3Robotiq2f85DifferentialIKEmbodiment):
    """Tool-sort FR3 stack with relative Cartesian commands for scripted validation."""

    name = "tool_sorting_fr3_robotiq_2f85_differential_ik"
    tags: ClassVar[list[str]] = ["embodiment"]

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        use_tiled_cameras: bool = False,
    ):
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            initial_joint_pose=initial_joint_pose,
            concatenate_observation_terms=concatenate_observation_terms,
            arm_mode=arm_mode,
            use_tiled_cameras=use_tiled_cameras,
        )
        _apply_tool_sort_camera_and_gripper(self)
