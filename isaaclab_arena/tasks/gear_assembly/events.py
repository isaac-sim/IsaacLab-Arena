# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Event terms for Arena Gear Assembly."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase

from isaaclab_arena.tasks.gear_assembly.specs import GEAR_TABLETOP_ORIENTATION_XYZW, GEAR_TABLETOP_PARKING_POSITIONS

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


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
