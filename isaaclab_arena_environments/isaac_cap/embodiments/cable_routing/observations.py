# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observations for the bimanual YAM cable-routing embodiment."""

from __future__ import annotations

import torch

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .config import (
    ARM_JOINT_NAMES,
    END_EFFECTOR_BODY_NAME,
    GRIPPER_CLOSED_POSITION,
    GRIPPER_JOINT_NAME,
    GRIPPER_OPEN_POSITION,
)


def _robot(env, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name]


def _index(names: list[str], expected: str, kind: str, side: str) -> int:
    try:
        return names.index(expected)
    except ValueError as error:
        raise ValueError(f"{side} is missing required {kind} {expected!r}") from error


def arm_joint_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return arm positions in declared action order for one YAM."""
    robot = _robot(env, asset_cfg)
    indices = [_index(robot.data.joint_names, name, "joint", asset_cfg.name) for name in ARM_JOINT_NAMES]
    return robot.data.joint_pos.torch[:, indices]


def gripper_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return zero-open, one-closed state for one YAM."""
    robot = _robot(env, asset_cfg)
    index = _index(robot.data.joint_names, GRIPPER_JOINT_NAME, "joint", asset_cfg.name)
    position = robot.data.joint_pos.torch[:, index : index + 1]
    travel = GRIPPER_CLOSED_POSITION - GRIPPER_OPEN_POSITION
    return torch.clamp((position - GRIPPER_OPEN_POSITION) / travel, min=0.0, max=1.0)


def ee_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return one YAM's end-effector world position."""
    robot = _robot(env, asset_cfg)
    index = _index(robot.data.body_names, END_EFFECTOR_BODY_NAME, "body", asset_cfg.name)
    return robot.data.body_pos_w.torch[:, index, :]


def ee_quat(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return one YAM's end-effector world quaternion."""
    robot = _robot(env, asset_cfg)
    index = _index(robot.data.body_names, END_EFFECTOR_BODY_NAME, "body", asset_cfg.name)
    return robot.data.body_quat_w.torch[:, index, :]


@configclass
class BimanualYamObservationsCfg:
    """Ordered, side-qualified policy observations for both YAMs."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            for side in ("left", "right"):
                asset_cfg = SceneEntityCfg(f"{side}_robot")
                setattr(self, f"{side}_joint_pos", ObsTerm(func=arm_joint_pos, params={"asset_cfg": asset_cfg}))
                setattr(self, f"{side}_gripper_pos", ObsTerm(func=gripper_pos, params={"asset_cfg": asset_cfg}))
                setattr(self, f"{side}_eef_pos", ObsTerm(func=ee_pos, params={"asset_cfg": asset_cfg}))
                setattr(self, f"{side}_eef_quat", ObsTerm(func=ee_quat, params={"asset_cfg": asset_cfg}))
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
