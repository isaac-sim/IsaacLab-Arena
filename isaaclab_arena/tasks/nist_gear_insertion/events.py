# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reset event terms for NIST gear insertion tasks."""

from __future__ import annotations

import torch
from collections.abc import Callable, Sequence
from dataclasses import field
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.automate import factory_control as fc

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class GraspCfg:
    """Embodiment-specific reset grasp parameters.

    The task owns the reset event, but the embodiment supplies the hand body,
    finger joints, offsets, and gripper-width setter because those details are
    robot-specific.
    """

    hand_grasp_width: float = 0.03
    """Opening width used before flushing the reset grasp state."""

    hand_close_width: float = 0.0
    """Final gripper target width after the gear is placed."""

    gripper_joint_setter_func: Callable[..., None] | None = None
    """Callable that maps a total opening width to embodiment-specific joints."""

    end_effector_body_name: str = "panda_hand"
    """Robot body used as the IK end-effector for reset grasp placement."""

    finger_joint_names: str = "panda_finger_joint[1-2]"
    """Joint-name expression for the gripper fingers."""

    grasp_rot_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    """Gear-frame to end-effector grasp rotation in Isaac Lab xyzw convention."""

    grasp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Gear-frame translation from the gear root to the end-effector grasp pose."""

    arm_joint_names: str = "panda_joint.*"
    """Arm joints affected by task reset randomization."""

    max_ik_iterations: int = 10
    """Maximum DLS IK updates used to place the hand at the grasp frame."""

    pos_threshold: float = 1e-6
    """Position-error threshold for stopping reset-time IK."""

    rot_threshold: float = 1e-6
    """Axis-angle error threshold for stopping reset-time IK."""


class place_gear_in_gripper(ManagerTermBase):
    """Place the held gear in the robot gripper during reset.

    The event follows the Factory/Forge reset pattern: solve IK to move the hand
    to a configured grasp frame on the gear, open to the grasp width, flush the
    simulator state, then set the final closed target. The final target is not
    written as a teleport so the controller maintains the grasp after reset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        gear_cfg = cfg.params["gear_cfg"]
        grasp_cfg = cfg.params["grasp_cfg"]
        robot_cfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.robot: Articulation = env.scene[robot_cfg.name]

        self.gear = env.scene[gear_cfg.name]

        self.hand_grasp_width = grasp_cfg.hand_grasp_width
        self.hand_close_width = grasp_cfg.hand_close_width
        assert grasp_cfg.gripper_joint_setter_func is not None, "A gripper joint setter is required for grasp reset."
        self.gripper_joint_setter_func = grasp_cfg.gripper_joint_setter_func

        # Cache default grasp offsets as tensors because this event runs every
        # reset and often across thousands of environments.
        self.grasp_rot_offset_tensor = (
            torch.tensor(grasp_cfg.grasp_rot_offset, device=env.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(env.num_envs, 1)
        )

        self.grasp_offset_tensor = torch.tensor(grasp_cfg.grasp_offset, device=env.device, dtype=torch.float32)

        self._resolve_robot_indices(grasp_cfg.end_effector_body_name, grasp_cfg.finger_joint_names)

        self._max_ik_iterations = grasp_cfg.max_ik_iterations
        self._pos_threshold = grasp_cfg.pos_threshold
        self._rot_threshold = grasp_cfg.rot_threshold

    def _resolve_robot_indices(self, end_effector_body_name: str, finger_joint_names: str) -> None:
        """Resolve the body and joint indices needed by reset-time IK."""
        eef_indices, _ = self.robot.find_bodies([end_effector_body_name])
        if not eef_indices:
            raise ValueError(f"End-effector body '{end_effector_body_name}' not found in robot")
        self.eef_idx = eef_indices[0]
        # Body index is shifted by one when indexing ``root_view.get_jacobians()``.
        self.jacobi_body_idx = self.eef_idx - 1
        assert self.jacobi_body_idx >= 0, "End-effector body must not be the articulation root."

        self.all_joints, _ = self.robot.find_joints([".*"])
        self.finger_joints, _ = self.robot.find_joints([finger_joint_names])
        if not self.finger_joints:
            raise ValueError(f"Finger joints '{finger_joint_names}' not found in robot")

    def _set_gripper_width(
        self,
        joint_pos: torch.Tensor,
        width: float,
        gripper_joint_setter_func: Callable[..., None],
    ) -> None:
        """Update the selected finger joints for the requested opening width."""
        row_indices = torch.arange(joint_pos.shape[0], device=joint_pos.device)
        gripper_joint_setter_func(joint_pos, row_indices, self.finger_joints, width)

    def _sync_sim_state(self, env: ManagerBasedEnv) -> None:
        """Flush written joint state before reading body poses again."""
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)

    def _run_grasp_ik(
        self,
        env: ManagerBasedEnv,
        env_ids: Sequence[int] | slice | torch.Tensor,
        num_envs: int,
        grasp_rot_offset: torch.Tensor,
        grasp_offset: torch.Tensor,
    ) -> torch.Tensor:
        """Move the end-effector to the gear grasp pose using iterative DLS IK.

        Each iteration computes the world-frame grasp pose from the current gear pose,
        applies one damped-least-squares joint update, and flushes the simulation state
        so the next iteration sees updated body transforms.
        """
        joint_vel = torch.zeros(num_envs, len(self.all_joints), device=env.device)
        for _ in range(self._max_ik_iterations):
            joint_pos, joint_vel = self._read_joint_state(env_ids)
            target_pos, target_quat = self._compute_grasp_target_pose(env_ids, grasp_rot_offset, grasp_offset)
            delta_hand_pose, pos_error, aa_error = self._compute_pose_error(env_ids, target_pos, target_quat)

            if self._has_converged(pos_error, aa_error):
                break

            delta_dof_pos = self._compute_dls_joint_delta(env, env_ids, delta_hand_pose)
            joint_pos = joint_pos + delta_dof_pos
            joint_vel = torch.zeros_like(joint_pos)
            self._write_ik_joint_state(env, env_ids, joint_pos, joint_vel)

        return joint_vel

    def _read_joint_state(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current joint positions and velocities for the reset IK solve."""
        joint_pos = wp.to_torch(self.robot.data.joint_pos)[env_ids].clone()
        joint_vel = wp.to_torch(self.robot.data.joint_vel)[env_ids].clone()
        return joint_pos, joint_vel

    def _compute_grasp_target_pose(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor,
        grasp_rot_offset: torch.Tensor,
        grasp_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the world-frame hand pose for the configured gear grasp."""
        gear_pos_w = wp.to_torch(self.gear.data.root_link_pos_w)[env_ids].clone()
        gear_quat_w = wp.to_torch(self.gear.data.root_link_quat_w)[env_ids].clone()
        target_quat = math_utils.quat_mul(gear_quat_w, grasp_rot_offset)
        target_pos = gear_pos_w + math_utils.quat_apply(target_quat, grasp_offset)
        return target_pos, target_quat

    def _compute_pose_error(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Cartesian pose error used by the Factory DLS IK routine."""
        eef_pos = wp.to_torch(self.robot.data.body_pos_w)[env_ids, self.eef_idx]
        eef_quat = wp.to_torch(self.robot.data.body_quat_w)[env_ids, self.eef_idx]
        pos_error, aa_error = fc.get_pose_error(
            fingertip_midpoint_pos=eef_pos,
            fingertip_midpoint_quat=eef_quat,
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        return torch.cat((pos_error, aa_error), dim=-1), pos_error, aa_error

    def _has_converged(self, pos_error: torch.Tensor, aa_error: torch.Tensor) -> bool:
        """Return whether the reset IK solve is within configured tolerances."""
        return (
            torch.norm(pos_error, dim=-1).max() < self._pos_threshold
            and torch.norm(aa_error, dim=-1).max() < self._rot_threshold
        )

    def _compute_dls_joint_delta(
        self,
        env: ManagerBasedEnv,
        env_ids: Sequence[int] | slice | torch.Tensor,
        delta_hand_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Return one damped-least-squares joint update for the hand pose error."""
        # Isaac Lab articulation Jacobians are indexed by non-root body id;
        # the shifted index is resolved once in ``_resolve_robot_indices``.
        jacobians = wp.to_torch(self.robot.root_view.get_jacobians()).clone()
        jacobian = jacobians[env_ids, self.jacobi_body_idx, :, :]
        return fc._get_delta_dof_pos(
            delta_pose=delta_hand_pose,
            ik_method="dls",
            jacobian=jacobian,
            device=env.device,
        )

    def _write_ik_joint_state(
        self,
        env: ManagerBasedEnv,
        env_ids: Sequence[int] | slice | torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        """Write one IK update and refresh tensors before the next iteration."""
        self.robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
        self.robot.set_joint_velocity_target_index(target=joint_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self._sync_sim_state(env)

    def _write_gripper_width(
        self,
        env: ManagerBasedEnv,
        env_ids: Sequence[int] | slice | torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        width: float,
        gripper_joint_setter_func: Callable[..., None],
    ) -> None:
        """Write one gripper width and flush the reset state to the simulator."""
        self._set_gripper_width(joint_pos, width, gripper_joint_setter_func)
        self.robot.set_joint_position_target_index(
            target=joint_pos,
            joint_ids=self.all_joints,
            env_ids=env_ids,
        )
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self._sync_sim_state(env)

    def _set_gripper_width_target(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor,
        joint_pos: torch.Tensor,
        width: float,
        gripper_joint_setter_func: Callable[..., None],
    ) -> None:
        """Set the final gripper target without teleporting the sim joint state."""
        self._set_gripper_width(joint_pos, width, gripper_joint_setter_func)
        self.robot.set_joint_position_target_index(
            target=joint_pos,
            joint_ids=self.all_joints,
            env_ids=env_ids,
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: Sequence[int] | slice | torch.Tensor,
        gear_cfg: SceneEntityCfg,
        grasp_cfg: GraspCfg,
        robot_cfg: SceneEntityCfg | None = None,
    ) -> None:
        """Run the reset grasp sequence for ``env_ids``.

        The event manager passes ``gear_cfg``, ``grasp_cfg``, and ``robot_cfg``
        on every call; this term resolves and caches them in ``__init__``.
        """
        grasp_rot_offset_tensor = self.grasp_rot_offset_tensor[env_ids]
        num_envs = grasp_rot_offset_tensor.shape[0]
        grasp_offset_batch = self.grasp_offset_tensor.unsqueeze(0).expand(num_envs, -1)
        joint_vel = self._run_grasp_ik(
            env=env,
            env_ids=env_ids,
            num_envs=num_envs,
            grasp_rot_offset=grasp_rot_offset_tensor,
            grasp_offset=grasp_offset_batch,
        )

        joint_pos = wp.to_torch(self.robot.data.joint_pos)[env_ids].clone()

        self._write_gripper_width(
            env,
            env_ids,
            joint_pos,
            joint_vel,
            self.hand_grasp_width,
            self.gripper_joint_setter_func,
        )
        self._set_gripper_width_target(env_ids, joint_pos, self.hand_close_width, self.gripper_joint_setter_func)
