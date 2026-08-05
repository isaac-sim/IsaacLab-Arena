# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Event terms for Arena Gear Assembly."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.deploy.mdp.events import set_robot_to_grasp_pose

from isaaclab_arena.tasks.gear_assembly.specs import GEAR_TABLETOP_ORIENTATION_XYZW, GEAR_TABLETOP_PARKING_POSITIONS

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


class set_robot_to_grasp_pose_with_finite_difference_ik(ManagerTermBase):
    """Reset the robot with IK when Newton cannot expose the end-effector Jacobian."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._source_term = set_robot_to_grasp_pose(cfg, env)

    @staticmethod
    def _write_joint_state(env, robot, env_ids, joint_position) -> None:
        joint_velocity = torch.zeros_like(joint_position)
        robot.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
        env.sim.forward()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        pos_threshold: float = 1.0e-3,
        rot_threshold: float = 1.0e-3,
        max_iterations: int = 50,
        pos_randomization_range: dict | None = None,
        gear_offsets_grasp: dict | None = None,
        end_effector_body_name: str | None = None,
        num_arm_joints: int | None = None,
        grasp_rot_offset: list | None = None,
        gripper_joint_setter_func=None,
    ) -> None:
        source = self._source_term
        robot = source.robot_asset
        gear_type_indices = env._gear_type_manager.get_all_gear_type_indices()[env_ids]
        gear_positions = torch.stack(
            [
                env.scene["factory_gear_small"].data.root_link_pos_w.torch,
                env.scene["factory_gear_medium"].data.root_link_pos_w.torch,
                env.scene["factory_gear_large"].data.root_link_pos_w.torch,
            ],
            dim=1,
        )[env_ids, gear_type_indices]
        gear_orientations = torch.stack(
            [
                env.scene["factory_gear_small"].data.root_link_quat_w.torch,
                env.scene["factory_gear_medium"].data.root_link_quat_w.torch,
                env.scene["factory_gear_large"].data.root_link_quat_w.torch,
            ],
            dim=1,
        )[env_ids, gear_type_indices]
        target_orientation = math_utils.quat_mul(gear_orientations, source.grasp_rot_offset_tensor[env_ids])
        grasp_offset = source.gear_grasp_offsets_stacked[gear_type_indices].clone()
        if pos_randomization_range is not None:
            ranges = torch.tensor(
                [pos_randomization_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")],
                device=env.device,
            )
            grasp_offset += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.device)
        target_position = gear_positions + math_utils.quat_apply(target_orientation, grasp_offset)

        joint_position = robot.data.joint_pos.torch[env_ids].clone()
        self._write_joint_state(env, robot, env_ids, joint_position)
        for _ in range(max_iterations):
            end_effector_position = robot.data.body_pos_w.torch[env_ids, source.eef_idx].clone()
            end_effector_orientation = robot.data.body_quat_w.torch[env_ids, source.eef_idx].clone()
            position_error, orientation_error = math_utils.compute_pose_error(
                end_effector_position,
                end_effector_orientation,
                target_position,
                target_orientation,
            )
            if torch.all(torch.linalg.norm(position_error, dim=-1) < pos_threshold) and torch.all(
                torch.linalg.norm(orientation_error, dim=-1) < rot_threshold
            ):
                break

            perturbation = 1.0e-3
            jacobian = torch.empty(
                len(env_ids), 6, source.num_arm_joints, device=env.device, dtype=joint_position.dtype
            )
            for joint_index in range(source.num_arm_joints):
                perturbed_position = joint_position.clone()
                perturbed_position[:, joint_index] += perturbation
                self._write_joint_state(env, robot, env_ids, perturbed_position)
                jacobian[:, :3, joint_index] = (
                    robot.data.body_pos_w.torch[env_ids, source.eef_idx] - end_effector_position
                ) / perturbation
                _, orientation_delta = math_utils.compute_pose_error(
                    end_effector_position,
                    end_effector_orientation,
                    robot.data.body_pos_w.torch[env_ids, source.eef_idx],
                    robot.data.body_quat_w.torch[env_ids, source.eef_idx],
                )
                jacobian[:, 3:, joint_index] = orientation_delta / perturbation

            self._write_joint_state(env, robot, env_ids, joint_position)
            jacobian_transpose = jacobian.transpose(1, 2)
            damping = 0.01 * torch.eye(6, device=env.device, dtype=joint_position.dtype)
            pose_error = torch.cat((position_error, orientation_error), dim=-1).unsqueeze(-1)
            joint_delta = (
                jacobian_transpose @ torch.linalg.solve(jacobian @ jacobian_transpose + damping, pose_error)
            ).squeeze(-1)
            joint_limits = robot.data.joint_pos_limits.torch[env_ids, : source.num_arm_joints]
            joint_position[:, : source.num_arm_joints] = torch.clamp(
                joint_position[:, : source.num_arm_joints] + torch.clamp(joint_delta, -0.2, 0.2),
                min=joint_limits[:, :, 0],
                max=joint_limits[:, :, 1],
            )
            self._write_joint_state(env, robot, env_ids, joint_position)

        robot.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
        robot.set_joint_velocity_target_index(target=torch.zeros_like(joint_position), env_ids=env_ids)

        source(
            env,
            env_ids,
            robot_asset_cfg=robot_asset_cfg,
            max_iterations=0,
            pos_randomization_range=None,
            gear_offsets_grasp=gear_offsets_grasp,
            end_effector_body_name=end_effector_body_name,
            num_arm_joints=num_arm_joints,
            grasp_rot_offset=grasp_rot_offset,
            gripper_joint_setter_func=gripper_joint_setter_func,
        )


class randomize_gears_and_base_pose_with_inactive_gear_parking(ManagerTermBase):
    """Randomize Gear Assembly poses while parking inactive gear variants on the tabletop."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize cached gear asset names and type indices."""
        super().__init__(cfg, env)
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.gear_asset_names = ["factory_gear_small", "factory_gear_medium", "factory_gear_large"]
        self.base_asset_name = "factory_gear_base"

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict | None = None,
        velocity_range: dict | None = None,
        gear_pos_range: dict | None = None,
        parking_positions: dict[str, tuple[float, float, float]] | None = None,
        parking_orientation_xyzw: tuple[float, float, float, float] | None = None,
        selected_parking_positions: dict[str, tuple[float, float, float]] | None = None,
        selected_orientation_xyzw: tuple[float, float, float, float] | None = None,
        parking_offsets: dict[str, tuple[float, float, float]] | None = None,
    ):
        """Randomize the active gear and place inactive gear variants in non-overlapping tabletop positions."""
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type is configured before this event."
            )

        pose_range = pose_range or {}
        velocity_range = velocity_range or {}
        gear_pos_range = gear_pos_range or {}
        parking_positions = parking_positions or GEAR_TABLETOP_PARKING_POSITIONS
        parking_orientation_xyzw = parking_orientation_xyzw or GEAR_TABLETOP_ORIENTATION_XYZW
        selected_orientation_xyzw = selected_orientation_xyzw or parking_orientation_xyzw

        gear_type_manager = env._gear_type_manager
        device = env.device

        pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        ranges_pose = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in pose_keys], device=device)
        rand_pose_samples = math_utils.sample_uniform(
            ranges_pose[:, 0], ranges_pose[:, 1], (len(env_ids), 6), device=device
        )
        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_pose_samples[:, 3], rand_pose_samples[:, 4], rand_pose_samples[:, 5]
        )

        ranges_vel = torch.tensor([velocity_range.get(key, (0.0, 0.0)) for key in pose_keys], device=device)
        rand_vel_samples = math_utils.sample_uniform(
            ranges_vel[:, 0], ranges_vel[:, 1], (len(env_ids), 6), device=device
        )

        positions_by_asset = {}
        default_positions_by_asset = {}
        default_orientations_by_asset = {}
        orientations_by_asset = {}
        velocities_by_asset = {}
        for asset_name in [self.base_asset_name, *self.gear_asset_names]:
            asset: RigidObject | Articulation = env.scene[asset_name]
            default_root_pose = asset.data.default_root_pose.torch[env_ids].clone()
            default_root_vel = asset.data.default_root_vel.torch[env_ids].clone()
            default_positions_by_asset[asset_name] = default_root_pose[:, 0:3] + env.scene.env_origins[env_ids]
            default_orientations_by_asset[asset_name] = default_root_pose[:, 3:7]
            positions_by_asset[asset_name] = default_positions_by_asset[asset_name] + rand_pose_samples[:, 0:3]
            orientations_by_asset[asset_name] = math_utils.quat_mul(default_root_pose[:, 3:7], orientations_delta)
            velocities_by_asset[asset_name] = default_root_vel + rand_vel_samples

        ranges_gear = torch.tensor(
            [gear_pos_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]],
            device=device,
        )
        rand_gear_offsets = math_utils.sample_uniform(
            ranges_gear[:, 0], ranges_gear[:, 1], (len(env_ids), 3), device=device
        )

        num_reset_envs = len(env_ids)
        gear_type_indices = self.gear_type_indices[:num_reset_envs]
        gear_type_indices[:] = gear_type_manager.get_all_gear_type_indices()[env_ids]
        parking_position_tensor = torch.tensor(
            [parking_positions[gear_key] for gear_key in ["gear_small", "gear_medium", "gear_large"]],
            device=device,
            dtype=torch.float32,
        )
        parking_orientation_tensor = torch.tensor(parking_orientation_xyzw, device=device, dtype=torch.float32)
        selected_position_tensor = None
        if selected_parking_positions is not None:
            selected_position_tensor = torch.tensor(
                [selected_parking_positions[gear_key] for gear_key in ["gear_small", "gear_medium", "gear_large"]],
                device=device,
                dtype=torch.float32,
            )
        selected_orientation_tensor = torch.tensor(selected_orientation_xyzw, device=device, dtype=torch.float32)
        parking_offset_tensor = None
        if parking_offsets is not None:
            parking_offset_tensor = torch.tensor(
                [parking_offsets[gear_key] for gear_key in ["gear_small", "gear_medium", "gear_large"]],
                device=device,
                dtype=torch.float32,
            )

        for gear_idx, asset_name in enumerate(self.gear_asset_names):
            selected_mask = gear_type_indices == gear_idx
            if selected_position_tensor is None:
                positions_by_asset[asset_name][selected_mask] += rand_gear_offsets[selected_mask]
            else:
                positions_by_asset[asset_name][selected_mask] = (
                    env.scene.env_origins[env_ids][selected_mask] + selected_position_tensor[gear_idx]
                )
                orientations_by_asset[asset_name][selected_mask] = selected_orientation_tensor
                velocities_by_asset[asset_name][selected_mask] = torch.zeros_like(
                    velocities_by_asset[asset_name][selected_mask]
                )
            if parking_offset_tensor is None:
                positions_by_asset[asset_name][~selected_mask] = (
                    env.scene.env_origins[env_ids][~selected_mask] + parking_position_tensor[gear_idx]
                )
            else:
                positions_by_asset[asset_name][~selected_mask] = (
                    default_positions_by_asset[asset_name][~selected_mask] + parking_offset_tensor[gear_idx]
                )
            orientations_by_asset[asset_name][~selected_mask] = parking_orientation_tensor
            velocities_by_asset[asset_name][~selected_mask] = torch.zeros_like(
                velocities_by_asset[asset_name][~selected_mask]
            )

        for asset_name, positions in positions_by_asset.items():
            asset = env.scene[asset_name]
            asset.write_root_pose_to_sim_index(
                root_pose=torch.cat([positions, orientations_by_asset[asset_name]], dim=-1),
                env_ids=env_ids,
            )
            asset.write_root_velocity_to_sim_index(root_velocity=velocities_by_asset[asset_name], env_ids=env_ids)
