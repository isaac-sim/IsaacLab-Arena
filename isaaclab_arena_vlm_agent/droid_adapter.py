# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID absolute-pose action, motion-generation, and prompt defaults."""

from isaaclab_arena_vlm_agent.absolute_action_adapter import AbsoluteChunkActionAdapter, AbsoluteGoalActionAdapter
from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter
from isaaclab_arena_vlm_agent.command_indices import AbsoluteCommandIndices, GripperCommandIndex, PoseCommandIndices
from isaaclab_arena_vlm_agent.pose_target_motion_generator import PoseTargetMotionGenerator


class _DroidPoseActionAdapter:
    """Require the native pose and gripper layout described by the DROID prompt."""

    def is_robot_action_space_supported(self, env) -> bool:
        return super().is_robot_action_space_supported(env) and self._command_indices == AbsoluteCommandIndices(
            action_dim=8,
            poses=(PoseCommandIndices(slice(0, 7), "robot", "base_link"),),
            joints=(),
            grippers=(GripperCommandIndex(7, "gripper_action", 0.0, 1.0),),
        )


class _DroidChunkActionAdapter(_DroidPoseActionAdapter, AbsoluteChunkActionAdapter):
    """Decode a chunk of DROID pose commands."""


class _DroidGoalActionAdapter(_DroidPoseActionAdapter, AbsoluteGoalActionAdapter):
    """Decode a DROID pose goal."""


class DroidVLMAdapter:
    """Provide VLM defaults for the droid_abs_ik embodiment."""

    def make_chunk_action_adapter(self, chunk_size: int) -> AbsoluteChunkActionAdapter:
        """Create a decoder for the requested number of absolute pose commands."""
        return _DroidChunkActionAdapter(chunk_size)

    def make_goal_action_adapter(
        self, position_tolerance_m: float, rotation_tolerance_rad: float
    ) -> AbsoluteGoalActionAdapter:
        """Create pose-goal execution with DROID's Robotiq gripper measurements."""
        generator = PoseTargetMotionGenerator(
            grippers={"gripper_action": _make_gripper()},
            position_tolerance_m=position_tolerance_m,
            rotation_tolerance_rad=rotation_tolerance_rad,
        )
        return _DroidGoalActionAdapter(generator)

    def get_system_prompt(self, action_adapter: AgentActionAdapter) -> str:
        """Describe the DROID pose contract and the selected response schema."""
        if isinstance(action_adapter, AbsoluteChunkActionAdapter):
            response = (
                f"Return a JSON object with a commands list of exactly {action_adapter.chunk_size} commands, "
                "executed one per simulation step. Each command has a values list of eight numbers: "
            )
        else:
            assert isinstance(
                action_adapter, AbsoluteGoalActionAdapter
            ), "Provide system_prompt when injecting a custom action adapter"
            response = "Return one command as a JSON object with a values list of eight numbers: "
        return (
            "Control DROID to complete the supplied task using the current camera images and robot state. "
            + response
            + "absolute XYZ in meters and a unit XYZW quaternion for the gripper base_link body, expressed in "
            "the robot root frame, followed by the gripper command (0 opens, 1 closes). Read the current "
            "camera_geometry for each named image: image_size_px is [width, height], intrinsic_matrix is K "
            "in the transmitted image's pixels, and camera poses map optical coordinates (+X right, +Y down, "
            "+Z forward) into the named world or robot root frame. For a camera point p_C, "
            "p_R = rotate(quaternion_root_xyzw, p_C) + position_root_m; pixels project as K p_C / p_C.z. "
            "All quaternions in input and output use XYZW. policy.eef_pos is in world meters and "
            "policy.eef_quat is the gripper's world orientation in XYZW. Read the current "
            "commanded-body pose from command_body_poses.arm_action when state is provided. This body is not "
            "the finger midpoint. Predict nearby waypoints using small pose changes. "
            "Follow the response schema exactly, with no explanatory text."
        )


def _make_gripper():
    """Create the measurements used by the DROID embodiment after simulation startup."""
    from isaaclab_arena.embodiments.gripper import RobotiqGripper

    return RobotiqGripper()
