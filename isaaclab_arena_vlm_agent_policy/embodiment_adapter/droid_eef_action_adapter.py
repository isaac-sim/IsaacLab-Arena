# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID observation and absolute EEF action adapters for VLM agents."""

import numpy as np
import torch

from isaaclab_arena_vlm_agent_policy.policy.vlm_agent_policy import AgentActionAdapter, VLMObservationAdapter
from isaaclab_arena_vlm_agent_policy.utils import compute_next_reference_pose, encoded_intrinsics


class DroidProprioceptionAdapter(VLMObservationAdapter):
    """Expose DROID cameras, arm joints, gripper opening, and EEF pose in the root frame."""

    camera_keys = ("external_camera_rgb", "external_camera_2_rgb", "wrist_camera_rgb")

    def extract_proprioception(self, env, observation) -> list[dict]:
        from isaaclab.utils.math import subtract_frame_transforms

        base = env.unwrapped
        robot = base.scene["robot"]
        proprio = observation["policy"]
        position, quaternion = subtract_frame_transforms(
            robot.data.root_pos_w.torch,
            robot.data.root_quat_w.torch,
            proprio["eef_pos"],
            proprio["eef_quat"],
        )
        poses = torch.cat([position, quaternion], dim=-1).detach().cpu().tolist()
        joints = proprio["joint_pos"].detach().cpu().tolist()
        grippers = proprio["gripper_pos"].detach().cpu().tolist()
        return [
            {
                "eef_pose_root_xyz_xyzw": poses[i],
                "joint_positions_rad": joints[i],
                "gripper_position": grippers[i],
                "control_dt_s": base.step_dt,
            }
            for i in range(base.num_envs)
        ]


class DroidEEFActionAdapter(AgentActionAdapter):
    """Check the absolute DROID action contract."""

    def validate_environment(self, env) -> None:
        arm = env.unwrapped.action_manager.get_term("arm_action")
        assert arm.cfg.body_name == "base_link" and arm.action_dim == 7, "Use absolute EEF pose actions"
        assert not arm.cfg.controller.use_relative_mode and arm.cfg.scale == 1.0 and arm.cfg.body_offset is None


class CalibratedDroidObservationAdapter(DroidProprioceptionAdapter):
    """Add calibrated views and measured robot geometry, without object poses."""

    def __init__(self, image_max_edge):
        self.image_max_edge = image_max_edge

    def extract_tracking_state(self, env, observation):
        return super().extract_proprioception(env, observation)

    def extract_proprioception(self, env, observation):
        states = self.extract_tracking_state(env, observation)
        base = env.unwrapped
        robot = base.scene["robot"]
        for env_id, state in enumerate(states):
            state["robot_position_world"] = robot.data.root_pos_w.torch[env_id].tolist()
            state["robot_quaternion_world_xyzw"] = robot.data.root_quat_w.torch[env_id].tolist()
            state["finger_links_world"] = {}
            for name in ("left_inner_finger", "right_inner_finger"):
                index = robot.data.body_names.index(name)
                state["finger_links_world"][name] = {
                    "position": robot.data.body_pos_w.torch[env_id, index].tolist(),
                    "quaternion_xyzw": robot.data.body_quat_w.torch[env_id, index].tolist(),
                }
            state["camera_calibration"] = {}
            for key in self.camera_keys:
                sensor = base.scene.sensors[key.removesuffix("_rgb")]
                assert sensor.cfg.update_latest_camera_pose, "Enable fresh camera poses for calibrated control"
                matrix, image_hw = encoded_intrinsics(
                    sensor.data.intrinsic_matrices[env_id].cpu().numpy(),
                    observation["camera_obs"][key].shape[1:3],
                    self.image_max_edge,
                )
                state["camera_calibration"][key] = {
                    "intrinsic_matrix_encoded": matrix,
                    "image_shape_hw": image_hw,
                    "position_world": sensor.data.pos_w[env_id].tolist(),
                    "quaternion_world_ros_xyzw": sensor.data.quat_w_ros[env_id].tolist(),
                }
        return states


class DroidGoalActionAdapter(DroidEEFActionAdapter):
    """Validate one goal command for execution through absolute EEF IK."""

    def __init__(self, max_position_step_m: float = 0.004, max_rotation_step_rad: float = 0.04):
        assert (
            np.isfinite(max_position_step_m) and max_position_step_m > 0
        ), "max_position_step_m must be finite and > 0"
        assert (
            np.isfinite(max_rotation_step_rad) and max_rotation_step_rad > 0
        ), "max_rotation_step_rad must be finite and > 0"
        self.max_position_step_m = max_position_step_m
        self.max_rotation_step_rad = max_rotation_step_rad

    def compute_next_reference_pose(self, reference_pose: np.ndarray, goal_pose: np.ndarray) -> np.ndarray:
        """Compute the next EEF reference toward the goal with DROID's configured motion limits."""
        return compute_next_reference_pose(
            reference_pose,
            goal_pose,
            max_position_step_m=self.max_position_step_m,
            max_rotation_step_rad=self.max_rotation_step_rad,
        )

    response_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "enum": ["move_to", "set_gripper", "wait"]},
            "position": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "quaternion_xyzw": {
                "type": ["array", "null"],
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
            "gripper": {"type": ["number", "null"], "enum": [0, 1, None]},
            "steps": {"type": "integer", "minimum": 1, "maximum": 240},
            "note": {"type": "string"},
        },
        "required": ["command", "position", "quaternion_xyzw", "gripper", "steps", "note"],
        "additionalProperties": False,
    }

    def decode_command(self, payload, proprioception):
        assert set(payload) == set(self.response_schema["required"]), "Unexpected or missing command fields"
        command = payload["command"]
        assert command in ("move_to", "set_gripper", "wait"), "Unknown command"
        assert isinstance(payload["note"], str)
        steps = payload["steps"]
        assert type(steps) is int and 1 <= steps <= 240, "steps must be an integer in 1..240"
        gripper = payload["gripper"]
        assert gripper is None or (type(gripper) in (int, float) and gripper in (0, 1)), "Gripper must be binary"
        if command == "set_gripper":
            assert gripper is not None, "set_gripper requires a gripper target"
        if command == "wait":
            assert gripper is None, "wait preserves the gripper target"
        pose = np.array(proprioception["eef_pose_root_xyz_xyzw"], dtype=float)
        if command == "move_to":
            position = np.asarray(payload["position"], dtype=float)
            assert position.shape == (3,) and np.isfinite(position).all(), "Invalid position"
            assert np.all(position >= [0.05, -0.65, 0.10]) and np.all(
                position <= [0.85, 0.65, 0.80]
            ), "Outside workspace"
            pose[:3] = position
            if payload["quaternion_xyzw"] is not None:
                quat = np.asarray(payload["quaternion_xyzw"], dtype=float)
                assert quat.shape == (4,) and np.isfinite(quat).all(), "Invalid quaternion"
                assert abs(np.linalg.norm(quat) - 1) < 1e-3, "Quaternion must have unit norm"
                pose[3:7] = quat / np.linalg.norm(quat)
        else:
            assert payload["position"] is None and payload["quaternion_xyzw"] is None, "Hold commands take no pose"
        gripper = proprioception["last_gripper_command"] if gripper is None else gripper
        return np.array([*pose, gripper, steps, int(command == "move_to")], dtype=np.float32)

    def command_to_action(self, env, command: np.ndarray) -> torch.Tensor:
        """Convert the pose and gripper to an action, excluding goal execution metadata."""
        return super().command_to_action(env, command[:8])


def chunk_action_schema(chunk_size: int) -> dict:
    """JSON schema requiring ``chunk_size`` absolute EEF pose-and-gripper commands."""
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": chunk_size,
                "maxItems": chunk_size,
                "items": {"type": "array", "minItems": 8, "maxItems": 8, "items": {"type": "number"}},
            }
        },
        "required": ["actions"],
        "additionalProperties": False,
    }


def validate_actions(payload: dict, current_pose: np.ndarray, chunk_size: int = 15) -> np.ndarray:
    """Validate absolute poses and binary gripper commands before executing a chunk.

    Args:
        payload: Parsed model response containing actions.
        current_pose: Current xyz and xyzw quaternion in the robot root frame.
        chunk_size: Number of pose-and-gripper commands in the chunk.

    Returns:
        A float32 array of shape ``(chunk_size, 8)``, with normalized quaternions.
    """
    actions = np.asarray(payload["actions"], dtype=np.float64)
    assert actions.shape == (chunk_size, 8), f"Expected ({chunk_size}, 8) actions, got {actions.shape}"
    assert np.isfinite(actions).all(), "Actions must be finite"
    assert np.isin(actions[:, 7], [0, 1]).all(), "Gripper must be 0 (open) or 1 (closed)"
    norms = np.linalg.norm(actions[:, 3:7], axis=1)
    assert np.all(np.abs(norms - 1) < 0.05), "EEF quaternions must have unit norm"
    actions[:, 3:7] /= norms[:, None]
    poses = np.vstack([current_pose, actions[:, :7]])
    translations = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    assert np.all(translations <= 0.05 + 1e-6), "EEF translation exceeds 0.05 m per step"
    dots = np.abs(np.sum(poses[1:, 3:7] * poses[:-1, 3:7], axis=1))
    rotations = 2 * np.arccos(np.clip(dots, 0, 1))
    assert np.all(rotations <= 0.35 + 1e-6), "EEF rotation exceeds 0.35 rad per step"
    return actions.astype(np.float32)


class DroidCalibratedChunkActionAdapter(DroidEEFActionAdapter):
    """Validate a chunk of absolute EEF poses using the chunk policy's limits."""

    def __init__(self, chunk_size: int = 15):
        assert chunk_size >= 1, f"chunk_size must be >= 1, got {chunk_size}"
        self.chunk_size = chunk_size

    @property
    def response_schema(self) -> dict:
        return chunk_action_schema(self.chunk_size)

    def decode_command(self, payload, proprioception):
        """Return a validated chunk flattened for the shared inference interface."""
        return validate_actions(
            payload, np.asarray(proprioception["eef_pose_root_xyz_xyzw"]), chunk_size=self.chunk_size
        ).reshape(-1)

    def tracking_error(self, previous_command, proprioception):
        """Return the original chunk-interruption feedback when measured tracking falls behind."""
        pose = np.asarray(proprioception["eef_pose_root_xyz_xyzw"])
        position_error = np.linalg.norm(previous_command[:3] - pose[:3])
        rotation_error = 2 * np.arccos(np.clip(abs(np.dot(previous_command[3:7], pose[3:7])), 0, 1))
        if position_error > 0.05 or rotation_error > 0.3:
            return (
                f"Previous EEF target was not tracked: {position_error:.3f} m, {rotation_error:.3f} rad. "
                "Remaining actions were discarded. Replan from the measured pose with smaller motions; "
                "do not close the gripper until the fingers reach the object."
            )
        return None
