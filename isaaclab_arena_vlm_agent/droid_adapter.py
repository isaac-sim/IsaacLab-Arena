# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID absolute-pose action, motion-generation, and prompt defaults."""

from isaaclab_arena_vlm_agent.absolute_action_adapter import AbsoluteChunkActionAdapter, AbsoluteGoalActionAdapter
from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter
from isaaclab_arena_vlm_agent.command_indices import AbsoluteCommandIndices, GripperCommandIndex, PoseCommandIndices
from isaaclab_arena_vlm_agent.pose_target_motion_generator import PoseTargetMotionGenerator

GRASP_POINT_OFFSET_M = 0.131
"""Distance along Robotiq base_link +X to the open finger-pad midpoint, matching DroidSceneCfg.ee_frame."""


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
        guidance = ""
        if isinstance(action_adapter, AbsoluteChunkActionAdapter):
            response = (
                f"Return a JSON object with a commands list of exactly {action_adapter.chunk_size} commands, "
                "executed one per simulation step. Each command has a values list of eight numbers: "
            )
            # Each request is decided from the current state alone, so describe grasp height and holding.
            guidance = (
                f"The grasp point between the open finger pads is {GRASP_POINT_OFFSET_M} m along base_link's "
                "local +X axis, toward the fingertips. With the fingers pointing down, place base_link "
                f"{GRASP_POINT_OFFSET_M} m above the height where the object should sit between the pads; lower "
                "targets push the fingers into the table or object and cannot be reached, so close the gripper "
                "once base_link stops descending near that height. Closing takes several steps. "
                "policy.gripper_pos is the measured finger closure: 0 is open, it rises toward 1 as the fingers "
                "close, and it stops short of 1 when the fingers hold an object. The last value of "
                "policy.actions is the previous gripper command. Once you command 1 to grasp, keep commanding 1 "
                "while the fingers close, lift, and carry; command 0 only when the object is above the "
                "destination. "
            )
        else:
            assert isinstance(
                action_adapter, AbsoluteGoalActionAdapter
            ), "Provide system_prompt when injecting a custom action adapter"
            response = (
                "Return one goal as a JSON object with a values list of eight numbers. The goal is held until it "
                "is reached, blocked, or its step budget runs out. After the first goal, previous_goal reports "
                "the last goal's values, executed_steps, and outcome: reached (within tolerance), blocked (the "
                "arm and gripper stopped moving short of the goal, for example because the fingers touch the "
                "table or an object), or timeout (the budget ran out while still moving). Do not resend a "
                "blocked goal unchanged; correct it from the images and current pose, or close the gripper if "
                "the object is already between the fingers. The grasp point between the open finger pads is "
                f"{GRASP_POINT_OFFSET_M} m along base_link's local +X axis, toward the fingertips, and the fingers "
                f"close along local Y. With the fingers pointing down, place base_link {GRASP_POINT_OFFSET_M} m "
                "above the height where the object should sit between the pads; lower goals push the fingers "
                "into the table or object and are blocked. The values are "
            )
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
            + guidance
            + "Follow the response schema exactly, with no explanatory text."
        )


def _make_gripper():
    """Create the measurements used by the DROID embodiment after simulation startup."""
    from isaaclab_arena.embodiments.gripper import RobotiqGripper

    return RobotiqGripper()
