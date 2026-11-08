# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Virtual force actuators attached directly to existing scene assets."""

from __future__ import annotations

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_apply_inverse


class ForcePoseActionTerm(ActionTerm):
    """Apply gravity-compensated PD wrenches at every physics substep.

    Actions are world-frame position (xyz), orientation (xyzw), and enabled (> 0).
    Disabled actions release the asset; no extra body or constraint is created.
    Controller gains can be changed through the term's cfg between environment steps.
    """

    def __init__(self, cfg: ForcePoseActionCfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((env.num_envs, 8), device=env.device)
        self._processed_actions = self._raw_actions.clone()
        self.last_wrench = None

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        released = (self._processed_actions[:, 7] > 0) & (actions[:, 7] <= 0)
        if released.any():
            self._asset.permanent_wrench_composer.reset(env_ids=released.nonzero().flatten())
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(actions)
        enabled = actions[:, 7] > 0
        if enabled.any():
            q_W_goal = actions[enabled, 3:7]
            norms = q_W_goal.norm(dim=-1, keepdim=True)
            assert (norms > 0).all(), "Enabled force pose actions require a nonzero quaternion"
            self._processed_actions[enabled, 3:7] = q_W_goal / norms
        if not enabled.any():
            self.last_wrench = None

    def apply_actions(self) -> None:
        env_ids = (self._processed_actions[:, 7] > 0).nonzero().flatten()
        if env_ids.numel() == 0:
            return
        cfg = self.cfg
        data = self._asset.data
        T_W_O = data.root_pose_w.torch[env_ids]
        v_W = data.root_lin_vel_w.torch[env_ids]
        omega_W = data.root_ang_vel_w.torch[env_ids]
        T_W_goal = self._processed_actions[env_ids, :7]
        _, rotation_error_W = compute_pose_error(T_W_O[:, :3], T_W_O[:, 3:], T_W_goal[:, :3], T_W_goal[:, 3:])
        masses = data.body_mass.torch[env_ids]
        mass = masses.sum(dim=1, keepdim=True)
        stiffness = (cfg.linear_gain * mass).clamp_min(cfg.minimum_stiffness)
        damping = 2 * (stiffness * mass).sqrt() if cfg.critical_damping else cfg.linear_damping * mass
        gravity_W = torch.tensor(self._env.cfg.sim.gravity, device=self.device)
        force_W = stiffness * (T_W_goal[:, :3] - T_W_O[:, :3]) - damping * v_W - mass * gravity_W
        force_W = force_W.clamp(-cfg.max_force, cfg.max_force)
        angular_command_W = cfg.rotation_gain * rotation_error_W - cfg.angular_damping * omega_W
        if cfg.rotation_mode == "torque":
            torque_W = angular_command_W
        else:
            assert cfg.rotation_mode == "acceleration", f"Unknown rotation mode: {cfg.rotation_mode}"
            q_W_B = T_W_O[:, 3:]
            acceleration_B = quat_apply_inverse(q_W_B, angular_command_W)
            inertia_B = data.body_inertia.torch[env_ids].reshape(len(env_ids), -1, 3, 3).sum(dim=1)
            torque_B = torch.bmm(inertia_B, acceleration_B[:, :, None]).squeeze(-1)
            torque_W = quat_apply(q_W_B, torque_B)
        torque_W = torque_W.clamp(-cfg.max_torque, cfg.max_torque)
        self.last_wrench = {
            "goal": T_W_goal[:, :3].clone(),
            "linear_velocity": v_W.clone(),
            "angular_velocity": omega_W.clone(),
            "force": force_W,
            "torque": torque_W,
            "mass": mass.squeeze(-1),
        }
        forces_W = masses[:, :, None] / mass[:, :, None] * force_W[:, None, :]
        torques_W = torch.zeros_like(forces_W)
        torques_W[:, 0, :] = torque_W
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces_W, torques=torques_W, env_ids=env_ids, is_global=True
        )

    def reset(self, env_ids=None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0
        self._processed_actions[env_ids] = 0
        self._asset.permanent_wrench_composer.reset(env_ids=env_ids)
        self.last_wrench = None


@configclass
class ForcePoseActionCfg(ActionTermCfg):
    """Configure a force-limited pose controller."""

    class_type: type[ActionTerm] = ForcePoseActionTerm
    linear_gain: float = 150.0
    """Mass-scaled translational stiffness gain, in inverse seconds squared."""

    linear_damping: float = 24.0
    """Mass-scaled translational damping gain, in inverse seconds."""

    critical_damping: bool = False
    """Use critical translational damping instead of ``linear_damping``."""

    minimum_stiffness: float = 0.0
    """Minimum translational stiffness, in newtons per meter."""

    max_force: float = 100.0
    """Maximum absolute force along each world-frame axis, in newtons."""

    rotation_mode: str = "acceleration"
    """Interpret the rotational command as angular acceleration or torque."""

    rotation_gain: float = 100.0
    """Rotational proportional gain in the units selected by ``rotation_mode``."""

    angular_damping: float = 20.0
    """Rotational damping gain in the units selected by ``rotation_mode``."""

    max_torque: float = 5.0
    """Maximum absolute torque along each world-frame axis, in newton-meters."""


class ForceJointActionTerm(ActionTerm):
    """Apply joint PD efforts each physics substep, retaining passive actuator drives.

    Actions contain (goal, gain, damping, effort limit, enabled) for each joint,
    in the articulation's joint order. Disabled joints keep their existing drives.
    """

    def __init__(self, cfg: ForceJointActionCfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((env.num_envs, self.action_dim), device=env.device)
        self._processed_actions = self._raw_actions.clone()

    @property
    def action_dim(self) -> int:
        return self._asset.num_joints * 5

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        previous = self._processed_actions.reshape(self.num_envs, -1, 5)
        current = actions.reshape(self.num_envs, -1, 5)
        released = (previous[:, :, 4] > 0) & (current[:, :, 4] <= 0)
        for joint_id in released.any(dim=0).nonzero().flatten().tolist():
            env_ids = released[:, joint_id].nonzero().flatten()
            self._asset.actuators.target_command.set_effort_index(
                value=torch.zeros((len(env_ids), 1), device=self.device),
                joint_ids=[joint_id],
                env_ids=env_ids,
            )
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(actions)

    def apply_actions(self) -> None:
        commands = self._processed_actions.reshape(self.num_envs, -1, 5)
        for joint_id in (commands[:, :, 4] > 0).any(dim=0).nonzero().flatten().tolist():
            env_ids = (commands[:, joint_id, 4] > 0).nonzero().flatten()
            goal, gain, damping, limit, _ = commands[env_ids, joint_id].unbind(dim=-1)
            q = self._asset.data.joint_pos.torch[env_ids, joint_id]
            qd = self._asset.data.joint_vel.torch[env_ids, joint_id]
            effort = (gain * (goal - q) - damping * qd).clamp(-limit, limit)
            self._asset.actuators.target_command.set_effort_index(
                value=effort[:, None], joint_ids=[joint_id], env_ids=env_ids
            )

    def reset(self, env_ids=None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0
        self._processed_actions[env_ids] = 0
        self._asset.actuators.target_command.set_effort_index(
            value=torch.zeros_like(self._asset.data.joint_pos.torch[env_ids]), env_ids=env_ids
        )


@configclass
class ForceJointActionCfg(ActionTermCfg):
    """Configure direct effort control on an existing articulation."""

    class_type: type[ActionTerm] = ForceJointActionTerm
