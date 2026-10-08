# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read DROID state and translate root-frame pose references into relative IK actions."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DroidState:
    """Snapshot batched DROID state on the simulation device."""

    joint_position: torch.Tensor
    """Seven arm joint positions from the policy observation, shaped (N, 7)."""

    gripper_closed: torch.Tensor
    """Measured normalized closure, shaped (N, 1): zero open, one closed."""

    T_W_B: torch.Tensor
    """Robot root B in simulation world W, shaped (N, 7), meters and XYZW."""

    T_B_G: torch.Tensor
    """Robotiq base_link G in robot root B; excludes the grasp-point offset."""


def extract_droid_state(env: Any, observation: dict) -> DroidState:
    """Read measured poses and copy the current DROID policy observation.

    Args:
        env: Arena environment, optionally gym-wrapped.
        observation: Current observation with unconcatenated DROID policy terms.

    Returns:
        Independent tensors for all environments. Gripper normalization follows
        the configured observation term, including the Newton-specific close target.
    """
    arena_env = env.unwrapped
    proprioception = observation["policy"]
    joint_position = proprioception["joint_pos"]
    gripper_closed = proprioception["gripper_pos"]
    assert joint_position.shape == (arena_env.num_envs, 7), "Expected seven DROID arm joints per environment"
    assert gripper_closed.shape == (arena_env.num_envs, 1), "Expected one gripper observation per environment"
    T_W_B, T_B_G = _read_droid_poses(arena_env)
    return DroidState(
        joint_position=joint_position.detach().clone(),
        gripper_closed=gripper_closed.detach().clone(),
        T_W_B=T_W_B.detach().clone(),
        T_B_G=T_B_G.detach().clone(),
    )


def droid_pose_to_action(env: Any, T_B_G_target: torch.Tensor, gripper_closed: torch.Tensor) -> torch.Tensor:
    """Convert absolute Robotiq base-link targets into DROID relative pose actions.

    Args:
        env: Arena environment with DROID relative pose IK and binary gripper actions.
        T_B_G_target: Desired base_link G in robot root B, shaped (N, 7), meters and XYZW.
        gripper_closed: Normalized closure shaped (N, 1): zero open, one closed.

    Returns:
        Actions on the environment device in action-manager term order. Controller
        offsets and scale are accounted for. The caller supplies bounded references;
        this function does not plan trajectories, limit speed, or avoid collisions.
    """
    from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

    from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction

    arena_env = env.unwrapped
    manager = arena_env.action_manager
    assert set(manager.active_terms) == {"arm_action", "gripper_action"}, "Expected only DROID arm and gripper actions"
    arm = manager.get_term("arm_action")
    gripper = manager.get_term("gripper_action")
    cfg = arm.cfg
    assert cfg.asset_name == "robot" and cfg.body_name == "base_link", "Expected DROID robot/base_link IK"
    assert cfg.controller.command_type == "pose" and cfg.controller.use_relative_mode, "Expected relative pose IK"
    assert arm.action_dim == 6 and gripper.action_dim == 1, "Expected six pose actions and one gripper action"
    assert cfg.clip is None and gripper.cfg.clip is None, "Clipped action terms are not supported"
    assert isinstance(gripper, BinaryJointPositionZeroToOneAction), "Expected DROID zero-to-one gripper action"

    _, T_B_G = _read_droid_poses(arena_env)
    target = T_B_G_target.to(device=T_B_G.device, dtype=T_B_G.dtype)
    assert target.shape == T_B_G.shape, "Expected one target pose per environment"
    assert torch.isfinite(target).all(), "Target poses must be finite"
    quaternion_norm = torch.linalg.vector_norm(target[:, 3:], dim=-1)
    assert torch.allclose(
        quaternion_norm, torch.ones_like(quaternion_norm), atol=1e-4
    ), "Expected unit XYZW quaternions"
    closure = gripper_closed.to(device=T_B_G.device, dtype=T_B_G.dtype)
    assert closure.shape == (arena_env.num_envs, 1), "Expected one gripper target per environment"
    assert (
        torch.isfinite(closure).all() and ((closure >= 0) & (closure <= 1)).all()
    ), "Gripper closure must be in [0, 1]"

    # C is the controller frame; its optional offset is fixed relative to G.
    t_B_C, q_B_C = T_B_G[:, :3], T_B_G[:, 3:]
    t_B_C_target, q_B_C_target = target[:, :3], target[:, 3:]
    if cfg.body_offset is not None:
        t_G_C = T_B_G.new_tensor(cfg.body_offset.pos).expand(arena_env.num_envs, -1)
        q_G_C = T_B_G.new_tensor(cfg.body_offset.rot).expand(arena_env.num_envs, -1)
        t_B_C, q_B_C = combine_frame_transforms(t_B_C, q_B_C, t_G_C, q_G_C)
        t_B_C_target, q_B_C_target = combine_frame_transforms(t_B_C_target, q_B_C_target, t_G_C, q_G_C)
    translation, rotation = compute_pose_error(t_B_C, q_B_C, t_B_C_target, q_B_C_target, rot_error_type="axis_angle")
    scale = T_B_G.new_tensor(cfg.scale)
    assert scale.ndim == 0 or scale.shape == (6,), "IK scale must be a scalar or six components"
    assert torch.isfinite(scale).all() and (scale != 0).all(), "IK scale must be finite and nonzero"
    arm_action = torch.cat((translation, rotation), dim=-1) / scale
    terms = {"arm_action": arm_action, "gripper_action": closure}
    return torch.cat([terms[name] for name in manager.active_terms], dim=-1)


def _read_droid_poses(arena_env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Read world/root and root/gripper transforms from live articulation buffers."""
    from isaaclab.utils.math import subtract_frame_transforms

    robot = arena_env.scene["robot"]
    body_index = robot.data.body_names.index("base_link")
    T_W_B = robot.data.root_pose_w.torch
    T_W_G = robot.data.body_link_pose_w.torch[:, body_index]
    t_B_G, q_B_G = subtract_frame_transforms(T_W_B[:, :3], T_W_B[:, 3:], T_W_G[:, :3], T_W_G[:, 3:])
    return T_W_B, torch.cat((t_B_G, q_B_G), dim=-1)
