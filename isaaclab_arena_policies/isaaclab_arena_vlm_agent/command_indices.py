# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read absolute command indices and semantics from active action terms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PoseCommandIndices:
    """Identify an absolute root-relative XYZ/XYZW command and its controlled body."""

    indices: slice
    """Seven action indices containing root-relative XYZ followed by XYZW."""
    asset_name: str
    """Scene key of the controlled articulation."""
    body_name: str
    """Controlled body whose pose is measured."""


@dataclass(frozen=True)
class GripperCommandIndex:
    """Identify a binary command and its native open and close inputs."""

    index: int
    """Action index containing the binary input."""
    term_name: str
    """Active action term associated with the gripper."""
    open_value: float
    """Native input requesting an open gripper."""
    close_value: float
    """Native input requesting a closed gripper."""


@dataclass(frozen=True)
class AbsoluteCommandIndices:
    """Describe absolute pose, joint-position, and binary-gripper action indices."""

    action_dim: int
    """Total number of native action indices."""
    poses: tuple[PoseCommandIndices, ...]
    """Absolute end-effector pose commands in action order."""
    joints: tuple[slice, ...]
    """Absolute joint-position indices in controller joint order."""
    grippers: tuple[GripperCommandIndex, ...]
    """Binary gripper inputs in action order."""


def read_absolute_command_indices(env) -> AbsoluteCommandIndices | None:
    """Read supported absolute action indices, or return None for incompatible controllers.

    Args:
        env: Arena environment, optionally gym-wrapped, with initialized action terms.

    Returns:
        Command indices for unscaled absolute targets without offsets or clipping, or None.
    """
    from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

    from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction

    manager = env.unwrapped.action_manager
    poses, joints, grippers = [], [], []
    first_index = 0
    # Diff controller could have diff indices for pose, joint, and gripper
    for name, dimension in zip(manager.active_terms, manager.action_term_dim, strict=True):
        term = manager.get_term(name)
        cfg = term.cfg
        if dimension <= 0 or cfg.clip is not None:
            return None
        indices = slice(first_index, first_index + dimension)
        if type(term) is DifferentialInverseKinematicsAction:
            pose: PoseCommandIndices | None = _read_pose_indices(cfg, indices)
            if pose is None:
                return None
            poses.append(pose)
        elif type(term) is JointPositionAction:
            joint_indices: slice | None = _read_joint_indices(cfg, indices)
            if joint_indices is None:
                return None
            joints.append(joint_indices)
        elif type(term) in (BinaryJointPositionZeroToOneAction, BinaryJointPositionAction):
            gripper = _read_binary_gripper_index(term, name, indices)
            if gripper is None:
                return None
            grippers.append(gripper)
        else:
            return None
        first_index += dimension
    if first_index == 0 or first_index != manager.total_action_dim:
        return None
    return AbsoluteCommandIndices(first_index, tuple(poses), tuple(joints), tuple(grippers))


def _read_pose_indices(cfg, indices: slice) -> PoseCommandIndices | None:
    """Return pose indices when the controller accepts unscaled absolute XYZ/XYZW targets."""
    if (
        indices.stop - indices.start != 7  # xyz, xyzw
        or cfg.controller.command_type != "pose"
        or cfg.controller.use_relative_mode
        or not _all_equal(cfg.scale, 1.0)
        or cfg.body_offset is not None
    ):
        return None
    return PoseCommandIndices(indices, cfg.asset_name, cfg.body_name)


def _read_joint_indices(cfg, indices: slice) -> slice | None:
    """Return joint indices when targets are applied without scaling or offsets."""
    # joint_target = command * scale + offset
    # Adapter is supposed to forward the command to the controller exactly, so it should not apply any scaling or offsets
    if cfg.use_default_offset or not _all_equal(cfg.scale, 1.0) or not _all_equal(cfg.offset, 0.0):
        return None
    return indices


def _read_binary_gripper_index(term, name: str, indices: slice) -> GripperCommandIndex | None:
    """Return one binary gripper index with the controller's native open and close inputs."""
    from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction

    # gripper command is binary, so it should have 1 index
    if indices.stop - indices.start != 1:
        return None
    values = (0.0, 1.0) if type(term) is BinaryJointPositionZeroToOneAction else (1.0, -1.0)
    return GripperCommandIndex(indices.start, name, *values)


def _all_equal(value, expected: float) -> bool:
    """Check scalar, per-axis, or regex-mapped controller scale and offset settings."""
    if isinstance(value, dict):
        value = tuple(value.values())
    if isinstance(value, (tuple, list)):
        return all(item == expected for item in value)
    return value == expected
