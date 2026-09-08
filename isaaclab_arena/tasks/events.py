# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab_tasks.contrib.stack.mdp.franka_stack_events import sample_object_poses

if TYPE_CHECKING:
    from collections.abc import Sequence


class ResetRobotToObjectGraspPose(ManagerTermBase):
    """Reset an arm and object to a calibrated, closed-gripper grasp."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        robot_cfg: SceneEntityCfg = cfg.params["robot_cfg"]
        object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        self.robot: Articulation = env.scene[robot_cfg.name]
        self.target_object: RigidObject = env.scene[object_cfg.name]

        self.arm_joint_ids, arm_joint_names = self.robot.find_joints(cfg.params["arm_joint_names"], preserve_order=True)
        assert len(self.arm_joint_ids) == len(
            cfg.params["arm_joint_names"]
        ), f"Expected arm joints {cfg.params['arm_joint_names']}, found {arm_joint_names}."

        end_effector_body_name = cfg.params["end_effector_body_name"]
        end_effector_indices, _ = self.robot.find_bodies([end_effector_body_name])
        assert (
            len(end_effector_indices) == 1
        ), f"Expected one '{end_effector_body_name}' body, found {end_effector_indices}."
        self.end_effector_body_index = end_effector_indices[0]
        self.jacobian_body_index = (
            self.end_effector_body_index - 1 if self.robot.is_fixed_base else self.end_effector_body_index
        )

        gripper_close_command: dict[str, float] = cfg.params["gripper_close_command"]
        self.gripper_joint_ids, gripper_joint_names = self.robot.find_joints(
            list(gripper_close_command), preserve_order=True
        )
        assert len(self.gripper_joint_ids) == len(
            gripper_close_command
        ), f"Expected gripper joints {list(gripper_close_command)}, found {gripper_joint_names}."
        self.gripper_close_position = torch.tensor(
            list(gripper_close_command.values()), device=env.device, dtype=torch.float32
        )
        self.grasp_offset = torch.tensor(
            cfg.params["grasp_offset_xyz"], device=env.device, dtype=torch.float32
        ).unsqueeze(0)
        self.grasp_rotation = torch.tensor(
            cfg.params["grasp_rotation_xyzw"], device=env.device, dtype=torch.float32
        ).unsqueeze(0)
        self._ik_controllers: dict[int, DifferentialIKController] = {}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        arm_joint_names: Sequence[str],
        end_effector_body_name: str,
        grasp_offset_xyz: tuple[float, float, float],
        grasp_rotation_xyzw: tuple[float, float, float, float],
        gripper_close_command: dict[str, float],
        max_iterations: int = 150,
        position_threshold: float = 1.0e-5,
        rotation_threshold: float = 1.0e-5,
    ) -> None:
        """Solve the pre-grasp arm pose, align the object, and close the gripper."""
        del env, robot_cfg, object_cfg, arm_joint_names, end_effector_body_name
        del grasp_offset_xyz, grasp_rotation_xyzw, gripper_close_command

        target_position, target_orientation, grasp_offset, grasp_rotation = self._target_grasp_pose(env_ids)
        self._solve_arm_ik(
            env_ids,
            target_position,
            target_orientation,
            max_iterations=max_iterations,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
        )
        self._align_object_to_gripper(env_ids, grasp_offset, grasp_rotation)
        self._set_closed_gripper_state(env_ids)

    def _target_grasp_pose(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the desired hand pose and batched object-to-hand transform."""
        grasp_offset = self.grasp_offset.expand(len(env_ids), -1)
        grasp_rotation = self.grasp_rotation.expand(len(env_ids), -1)
        object_position = self.target_object.data.root_link_pos_w.torch[env_ids]
        object_orientation = self.target_object.data.root_link_quat_w.torch[env_ids]
        target_orientation = math_utils.quat_mul(object_orientation, grasp_rotation)
        target_position = object_position + math_utils.quat_apply(target_orientation, grasp_offset)
        return target_position, target_orientation, grasp_offset, grasp_rotation

    def _solve_arm_ik(
        self,
        env_ids: torch.Tensor,
        target_position: torch.Tensor,
        target_orientation: torch.Tensor,
        max_iterations: int,
        position_threshold: float,
        rotation_threshold: float,
    ) -> None:
        """Iteratively write an inverse-kinematics solution into the articulation state."""
        controller = self._get_ik_controller(len(env_ids))
        controller.set_command(torch.cat((target_position, target_orientation), dim=-1))

        joint_limits = self.robot.data.joint_pos_limits.torch[env_ids][:, self.arm_joint_ids]
        joint_lower = joint_limits[:, :, 0]
        joint_upper = joint_limits[:, :, 1]
        zero_velocity = torch.zeros((len(env_ids), len(self.arm_joint_ids)), device=self.device, dtype=torch.float32)

        for _ in range(max_iterations):
            joint_position = self.robot.data.joint_pos.torch[env_ids][:, self.arm_joint_ids].clone()
            end_effector_position = self.robot.data.body_pos_w.torch[env_ids, self.end_effector_body_index]
            end_effector_orientation = self.robot.data.body_quat_w.torch[env_ids, self.end_effector_body_index]
            position_error, rotation_error = math_utils.compute_pose_error(
                end_effector_position,
                end_effector_orientation,
                target_position,
                target_orientation,
                rot_error_type="axis_angle",
            )
            if torch.all(torch.linalg.vector_norm(position_error, dim=-1) < position_threshold) and torch.all(
                torch.linalg.vector_norm(rotation_error, dim=-1) < rotation_threshold
            ):
                break

            jacobian = self.robot.data.body_link_jacobian_w.torch[env_ids, self.jacobian_body_index][
                :, :, self.arm_joint_ids
            ]
            joint_position = controller.compute(
                end_effector_position,
                end_effector_orientation,
                jacobian,
                joint_position,
            )
            joint_position = torch.maximum(torch.minimum(joint_position, joint_upper), joint_lower)
            self.robot.actuators.target_command.set_position_index(
                value=joint_position, joint_ids=self.arm_joint_ids, env_ids=env_ids
            )
            self.robot.actuators.target_command.set_velocity_index(
                value=zero_velocity, joint_ids=self.arm_joint_ids, env_ids=env_ids
            )
            self.robot.write_joint_position_to_sim_index(
                position=joint_position, joint_ids=self.arm_joint_ids, env_ids=env_ids
            )
            self.robot.write_joint_velocity_to_sim_index(
                velocity=zero_velocity, joint_ids=self.arm_joint_ids, env_ids=env_ids
            )

    def _get_ik_controller(self, num_envs: int) -> DifferentialIKController:
        """Return a DLS controller sized for the reset batch."""
        controller = self._ik_controllers.get(num_envs)
        if controller is None:
            controller = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                    ik_params={"lambda_val": 0.1},
                ),
                num_envs=num_envs,
                device=self.device,
            )
            self._ik_controllers[num_envs] = controller
        return controller

    def _align_object_to_gripper(
        self,
        env_ids: torch.Tensor,
        grasp_offset: torch.Tensor,
        grasp_rotation: torch.Tensor,
    ) -> None:
        """Place the target object at the grasp transform reached by the hand."""
        hand_position = self.robot.data.body_pos_w.torch[env_ids, self.end_effector_body_index].clone()
        hand_orientation = self.robot.data.body_quat_w.torch[env_ids, self.end_effector_body_index].clone()
        object_orientation = math_utils.quat_mul(hand_orientation, math_utils.quat_conjugate(grasp_rotation))
        object_position = hand_position - math_utils.quat_apply(hand_orientation, grasp_offset)
        self.target_object.write_root_pose_to_sim_index(
            root_pose=torch.cat((object_position, object_orientation), dim=-1), env_ids=env_ids
        )
        self.target_object.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )

    def _set_closed_gripper_state(self, env_ids: torch.Tensor) -> None:
        """Write the configured closed pose to the gripper joints and targets."""
        gripper_position = self.gripper_close_position.expand(len(env_ids), -1)
        gripper_velocity = torch.zeros_like(gripper_position)
        self.robot.actuators.target_command.set_position_index(
            value=gripper_position, joint_ids=self.gripper_joint_ids, env_ids=env_ids
        )
        self.robot.actuators.target_command.set_velocity_index(
            value=gripper_velocity, joint_ids=self.gripper_joint_ids, env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(
            position=gripper_position, joint_ids=self.gripper_joint_ids, env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=gripper_velocity, joint_ids=self.gripper_joint_ids, env_ids=env_ids
        )


def randomize_poses_and_align_auxiliary_assets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
    fixed_asset_cfg: SceneEntityCfg | None = None,
    auxiliary_asset_cfgs: list[SceneEntityCfg] | None = None,
    randomization_mode: Literal["held_and_fixed_only", "held_fixed_and_auxiliary"] = "held_and_fixed_only",
):
    """
    Randomize object poses and update the poses of related assets accordingly.

    Args:
        randomization_mode:
            - "held_and_fixed_only": Randomize only the fixed and held assets independently.
            - "held_fixed_and_auxiliary": Randomize fixed, held, and auxiliary assets, with auxiliary
              assets positioned relative to the fixed asset.
    """
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])

            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )

            if (
                randomization_mode == "held_fixed_and_auxiliary"
                and auxiliary_asset_cfgs is not None
                and fixed_asset_cfg is not None
                and asset_cfg.name == fixed_asset_cfg.name
            ):
                # Place auxiliary assets at exactly the same pose as the fixed asset (zero offset).
                # NOTE: This assumes the asset USD files have base frames defined such that zero offset creates a valid scene.
                # Currently designed for gear mesh task where all gears share the same center point.
                # For other assets, this may cause geometry intersections. Customers need to adjust it accordingly.
                for j in range(len(auxiliary_asset_cfgs)):
                    rel_asset_cfg = auxiliary_asset_cfgs[j]
                    rel_asset = env.scene[rel_asset_cfg.name]
                    rel_asset.write_root_pose_to_sim(
                        torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
                    )
                    rel_asset.write_root_velocity_to_sim(
                        torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
                    )
