# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID absolute-pose action, motion-generation, and prompt defaults."""

from isaaclab_arena_vlm_agent.absolute_action_adapter import (
    AbsoluteChunkActionAdapter,
    AbsoluteGoalActionAdapter,
    AbsoluteStepActionAdapter,
)
from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter
from isaaclab_arena_vlm_agent.pose_target_motion_generator import PoseTargetMotionGenerator


class DroidVLMAdapter:
    """Provide VLM defaults for the droid_abs_ik embodiment."""

    def make_step_action_adapter(self) -> AbsoluteStepActionAdapter:
        """Create a decoder for one absolute pose command."""
        return AbsoluteStepActionAdapter()

    def make_chunk_action_adapter(self, chunk_size: int) -> AbsoluteChunkActionAdapter:
        """Create a decoder for the requested number of absolute pose commands."""
        return AbsoluteChunkActionAdapter(chunk_size)

    def make_goal_action_adapter(
        self, position_tolerance_m: float, rotation_tolerance_rad: float
    ) -> AbsoluteGoalActionAdapter:
        """Create pose-goal execution with DROID's Robotiq gripper measurements."""
        generator = PoseTargetMotionGenerator(
            grippers={"gripper_action": _make_gripper()},
            position_tolerance_m=position_tolerance_m,
            rotation_tolerance_rad=rotation_tolerance_rad,
        )
        return AbsoluteGoalActionAdapter(generator)

    def get_system_prompt(self, action_adapter: AgentActionAdapter) -> str:
        """Describe the DROID pose contract and the selected response schema."""
        if isinstance(action_adapter, AbsoluteChunkActionAdapter):
            response = (
                f"Return a JSON object with a commands list of exactly {action_adapter.chunk_size} commands, "
                "executed one per simulation step. Each command has a values list of eight numbers: "
            )
        else:
            assert isinstance(
                action_adapter, (AbsoluteStepActionAdapter, AbsoluteGoalActionAdapter)
            ), "Provide system_prompt when injecting a custom action adapter"
            response = "Return one command as a JSON object with a values list of eight numbers: "
        return (
            "Control DROID to complete the supplied task using the current camera images and robot state. "
            + response
            + "absolute XYZ in meters and a unit XYZW quaternion for the gripper base_link body, expressed in "
            "the robot root frame, followed by the gripper command (0 opens, 1 closes). Read the current "
            "commanded-body pose from command_body_poses.arm_action when state is provided. This body is not "
            "the finger midpoint. Predict nearby waypoints using small pose changes. Approach with the gripper "
            "open, close around the object, lift it, move above the destination, and open to release. "
            "Follow the response schema exactly, with no explanatory text."
        )


def _make_gripper():
    """Create the measurements used by the DROID embodiment after simulation startup."""
    from isaaclab_arena.embodiments.gripper import RobotiqGripper

    return RobotiqGripper()
