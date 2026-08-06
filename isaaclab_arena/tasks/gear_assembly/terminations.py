# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Termination terms for Arena Gear Assembly."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from pxr import Usd


class selected_gear_on_base(ManagerTermBase):
    """Terminate when the active gear is seated and settled on the gear base."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        """Cache the active gear assets and base asset."""
        super().__init__(cfg, env)
        self.base_asset_cfg: SceneEntityCfg = cfg.params.get("base_asset_cfg", SceneEntityCfg("factory_gear_base"))
        self.base_asset = env.scene[self.base_asset_cfg.name]
        self.gear_assets = {
            "gear_small": env.scene["factory_gear_small"],
            "gear_medium": env.scene["factory_gear_medium"],
            "gear_large": env.scene["factory_gear_large"],
        }
        self.gear_names = ["gear_small", "gear_medium", "gear_large"]
        self.env_indices = torch.arange(env.num_envs, device=env.device)
        self.up_axis = torch.tensor([[0.0, 0.0, 1.0]], device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
        self.consecutive_success_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        self.base_support_prim_name = cfg.params.get("base_support_prim_name")
        self.enabled_colliders_only = cfg.params.get("enabled_colliders_only", False)
        self.base_collision_corners = self._collision_corners(
            self.base_asset,
            env.device,
            self.base_support_prim_name,
            self.enabled_colliders_only,
        )
        self.gear_collision_corners = {
            gear_name: self._collision_corners(asset, env.device, enabled_only=self.enabled_colliders_only)
            for gear_name, asset in self.gear_assets.items()
        }

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the consecutive success counter."""
        if env_ids is None:
            env_ids = slice(None)
        self.consecutive_success_count[env_ids] = 0

    @staticmethod
    def _collision_corners(
        asset: RigidObject,
        device: str,
        collision_prim_name: str | None = None,
        enabled_only: bool = False,
    ) -> torch.Tensor:
        from pxr import Usd, UsdGeom, UsdPhysics

        root_prims = sim_utils.find_matching_prims(asset.cfg.prim_path)
        assert root_prims, f"{asset.cfg.prim_path} has no matching prims"
        root_prim = root_prims[0]
        rigid_prim = selected_gear_on_base._rigid_body_prim(root_prim)
        assert rigid_prim is not None, f"{asset.cfg.prim_path} has no rigid-body prim"

        bbox_cache = UsdGeom.BBoxCache(
            0,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
            useExtentsHint=True,
        )
        corners = []
        for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Boundable):
                continue
            collision_prim = prim
            while collision_prim != root_prim and not collision_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_prim = collision_prim.GetParent()
            if not collision_prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if enabled_only and UsdPhysics.CollisionAPI(collision_prim).GetCollisionEnabledAttr().Get() is False:
                continue
            if collision_prim_name is not None and collision_prim.GetName() != collision_prim_name:
                continue
            local_box = bbox_cache.ComputeRelativeBound(prim, rigid_prim).ComputeAlignedBox()
            box_min = local_box.GetMin()
            box_max = local_box.GetMax()
            corners.extend(
                [x, y, z]
                for x in (box_min[0], box_max[0])
                for y in (box_min[1], box_max[1])
                for z in (box_min[2], box_max[2])
            )
        assert corners, f"{asset.cfg.prim_path} has no collision geometry"
        return torch.tensor(corners, device=device, dtype=torch.float32)

    @staticmethod
    def _rigid_body_prim(root_prim: Usd.Prim) -> Usd.Prim | None:
        from pxr import Usd, UsdPhysics

        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim
        return None

    @staticmethod
    def _world_collision_z_bounds(
        local_corners: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = root_pos.shape[0]
        num_corners = local_corners.shape[0]
        corners = local_corners.unsqueeze(0).expand(num_envs, num_corners, 3).reshape(-1, 3)
        quats = root_quat.unsqueeze(1).expand(num_envs, num_corners, 4).reshape(-1, 4)
        positions = root_pos.unsqueeze(1).expand(num_envs, num_corners, 3).reshape(-1, 3)
        world_z = (positions + math_utils.quat_apply(quats, corners))[:, 2].reshape(num_envs, num_corners)
        return world_z.min(dim=1).values, world_z.max(dim=1).values

    def _selected_param(
        self, value: float | dict[str, float], gear_type_indices: torch.Tensor, env: ManagerBasedEnv
    ) -> torch.Tensor:
        if isinstance(value, dict):
            values = torch.tensor(
                [value[gear_name] for gear_name in self.gear_names], device=env.device, dtype=torch.float32
            )
            return values[gear_type_indices]
        return torch.full((env.num_envs,), float(value), device=env.device, dtype=torch.float32)

    def _selected_offset(
        self, value: dict[str, list[float]] | None, gear_type_indices: torch.Tensor, env: ManagerBasedEnv
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
        offsets = torch.tensor([value[gear_name] for gear_name in self.gear_names], device=env.device)
        return offsets[gear_type_indices]

    def __call__(
        self,
        env: ManagerBasedEnv,
        base_asset_cfg: SceneEntityCfg = SceneEntityCfg("factory_gear_base"),
        root_z_above_base: float | dict[str, float] = 0.03,
        root_xy_offset_from_base: dict[str, list[float]] | None = None,
        xy_threshold: float = 0.015,
        z_threshold: float = 0.01,
        upright_axis_threshold_deg: float = 15.0,
        linear_velocity_threshold: float = 0.05,
        angular_velocity_threshold: float = 0.5,
        support_z_offset: float | dict[str, float] = 0.0,
        base_support_prim_name: str | None = None,
        enabled_colliders_only: bool = False,
        support_z_threshold: float = 0.005,
        consecutive_success_steps: int = 10,
    ) -> torch.Tensor:
        """Return true when the selected gear is aligned with and settled on the base.

        Args:
            env: Environment instance.
            base_asset_cfg: Configuration of the gear-base asset.
            root_z_above_base: Expected selected gear root height above the base root.
            root_xy_offset_from_base: Expected gear-root offset in the base frame for each gear type.
            xy_threshold: Maximum allowed root XY error relative to the base root.
            z_threshold: Maximum allowed root height error relative to ``root_z_above_base``.
            upright_axis_threshold_deg: Maximum allowed local-Z axis angle between gear and base.
            linear_velocity_threshold: Maximum selected gear linear speed.
            angular_velocity_threshold: Maximum selected gear angular speed.
            support_z_offset: Expected gear-bottom height relative to the base-top collision surface.
            base_support_prim_name: Optional base collision prim used as the support surface.
            enabled_colliders_only: Whether support bounds exclude disabled collision geometry.
            support_z_threshold: Maximum allowed error around ``support_z_offset``.
            consecutive_success_steps: Number of consecutive checks required before terminating.

        Returns:
            Boolean tensor indicating which environments have completed the assembly.
        """
        if not hasattr(env, "_gear_type_manager"):
            self.consecutive_success_count.zero_()
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        assert (
            base_asset_cfg.name == self.base_asset_cfg.name
        ), "selected_gear_on_base does not support changing base_asset_cfg after initialization"
        assert (
            base_support_prim_name == self.base_support_prim_name
        ), "selected_gear_on_base does not support changing base_support_prim_name after initialization"
        assert (
            enabled_colliders_only == self.enabled_colliders_only
        ), "selected_gear_on_base does not support changing enabled_colliders_only after initialization"

        gear_type_indices = env._gear_type_manager.get_all_gear_type_indices()

        all_gear_pos = torch.stack(
            [
                self.gear_assets["gear_small"].data.root_link_pos_w.torch,
                self.gear_assets["gear_medium"].data.root_link_pos_w.torch,
                self.gear_assets["gear_large"].data.root_link_pos_w.torch,
            ],
            dim=1,
        )
        all_gear_quat = torch.stack(
            [
                self.gear_assets["gear_small"].data.root_link_quat_w.torch,
                self.gear_assets["gear_medium"].data.root_link_quat_w.torch,
                self.gear_assets["gear_large"].data.root_link_quat_w.torch,
            ],
            dim=1,
        )
        all_gear_vel = torch.stack(
            [
                self.gear_assets["gear_small"].data.root_com_vel_w.torch,
                self.gear_assets["gear_medium"].data.root_com_vel_w.torch,
                self.gear_assets["gear_large"].data.root_com_vel_w.torch,
            ],
            dim=1,
        )

        gear_pos = all_gear_pos[self.env_indices, gear_type_indices]
        gear_quat = all_gear_quat[self.env_indices, gear_type_indices]
        gear_vel = all_gear_vel[self.env_indices, gear_type_indices]

        base_pos = self.base_asset.data.root_link_pos_w.torch
        base_quat = self.base_asset.data.root_link_quat_w.torch

        root_offsets = self._selected_offset(root_xy_offset_from_base, gear_type_indices, env)
        root_targets = base_pos + math_utils.quat_apply(base_quat, root_offsets)
        xy_error = torch.linalg.norm(gear_pos[:, :2] - root_targets[:, :2], dim=-1)
        root_z_targets = self._selected_param(root_z_above_base, gear_type_indices, env)
        z_error = torch.abs((gear_pos[:, 2] - base_pos[:, 2]) - root_z_targets)
        _, base_top_z = self._world_collision_z_bounds(self.base_collision_corners, base_pos, base_quat)
        gear_bottom_z = torch.empty(env.num_envs, dtype=torch.float32, device=env.device)
        for gear_idx, gear_name in enumerate(self.gear_names):
            mask = gear_type_indices == gear_idx
            if not mask.any():
                continue
            selected_gear_bottom_z, _ = self._world_collision_z_bounds(
                self.gear_collision_corners[gear_name],
                self.gear_assets[gear_name].data.root_link_pos_w.torch,
                self.gear_assets[gear_name].data.root_link_quat_w.torch,
            )
            gear_bottom_z[mask] = selected_gear_bottom_z[mask]
        support_targets = self._selected_param(support_z_offset, gear_type_indices, env)
        support_error = torch.abs((gear_bottom_z - base_top_z) - support_targets)

        gear_up = math_utils.quat_apply(gear_quat, self.up_axis)
        base_up = math_utils.quat_apply(base_quat, self.up_axis)
        min_upright_cos = torch.cos(torch.deg2rad(torch.tensor(upright_axis_threshold_deg, device=env.device)))
        upright = torch.sum(gear_up * base_up, dim=-1) >= min_upright_cos

        linear_speed = torch.linalg.norm(gear_vel[:, :3], dim=-1)
        angular_speed = torch.linalg.norm(gear_vel[:, 3:], dim=-1)

        success_now = (
            (xy_error <= xy_threshold)
            & (z_error <= z_threshold)
            & (support_error <= support_z_threshold)
            & upright
            & (linear_speed <= linear_velocity_threshold)
            & (angular_speed <= angular_velocity_threshold)
        )
        self.consecutive_success_count = torch.where(
            success_now,
            self.consecutive_success_count + 1,
            torch.zeros_like(self.consecutive_success_count),
        )
        return self.consecutive_success_count >= consecutive_success_steps
