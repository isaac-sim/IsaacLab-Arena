# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Termination predicates for gear insertion."""

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


class all_gears_seated(ManagerTermBase):
    """Terminate after all configured gears remain seated and settled."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.plate_asset_cfg: SceneEntityCfg = cfg.params["plate_asset_cfg"]
        self.gear_asset_cfgs: tuple[SceneEntityCfg, ...] = tuple(cfg.params["gear_asset_cfgs"])
        self.plate_asset = env.scene[self.plate_asset_cfg.name]
        self.gear_assets = tuple(env.scene[gear_cfg.name] for gear_cfg in self.gear_asset_cfgs)
        self.up_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)
        self.consecutive_success_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        self.success_per_gear = torch.zeros(
            (env.num_envs, len(self.gear_assets)),
            device=env.device,
            dtype=torch.bool,
        )
        self.diagnostics_per_gear: dict[str, torch.Tensor] = {}
        self.plate_collision_corners = self._collision_corners(
            self.plate_asset,
            env.device,
            collision_prim_name="platform",
            enabled_only=True,
        )
        self.gear_collision_corners = tuple(
            self._collision_corners(asset, env.device, enabled_only=True) for asset in self.gear_assets
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the consecutive-success counter for the selected environments."""
        if env_ids is None:
            env_ids = slice(None)
        self.consecutive_success_count[env_ids] = 0
        self.success_per_gear[env_ids] = False

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
        rigid_prim = all_gears_seated._rigid_body_prim(root_prim)
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
        local_corners: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = root_pos.shape[0]
        num_corners = local_corners.shape[0]
        corners = local_corners.unsqueeze(0).expand(num_envs, num_corners, 3).reshape(-1, 3)
        quats = root_quat.unsqueeze(1).expand(num_envs, num_corners, 4).reshape(-1, 4)
        positions = root_pos.unsqueeze(1).expand(num_envs, num_corners, 3).reshape(-1, 3)
        world_z = (positions + math_utils.quat_apply(quats, corners))[:, 2].reshape(num_envs, num_corners)
        return world_z.min(dim=1).values, world_z.max(dim=1).values

    def __call__(
        self,
        env: ManagerBasedEnv,
        plate_asset_cfg: SceneEntityCfg,
        gear_asset_cfgs: Sequence[SceneEntityCfg],
        target_offsets_xyz: Sequence[Sequence[float]],
        xy_threshold: float = 0.015,
        z_threshold: float = 0.01,
        upright_axis_threshold_deg: float = 15.0,
        linear_velocity_threshold: float = 0.05,
        angular_velocity_threshold: float = 0.5,
        support_z_threshold: float = 0.005,
        consecutive_success_steps: int = 10,
    ) -> torch.Tensor:
        """Return true after every gear is aligned, supported, upright, and still."""
        assert plate_asset_cfg.name == self.plate_asset_cfg.name
        assert tuple(cfg.name for cfg in gear_asset_cfgs) == tuple(cfg.name for cfg in self.gear_asset_cfgs)

        plate_pos = self.plate_asset.data.root_link_pos_w.torch
        plate_quat = self.plate_asset.data.root_link_quat_w.torch
        gear_pos = torch.stack([asset.data.root_link_pos_w.torch for asset in self.gear_assets], dim=1)
        gear_quat = torch.stack([asset.data.root_link_quat_w.torch for asset in self.gear_assets], dim=1)
        gear_vel = torch.stack([asset.data.root_com_vel_w.torch for asset in self.gear_assets], dim=1)

        offsets = torch.as_tensor(target_offsets_xyz, device=env.device, dtype=plate_pos.dtype)
        expanded_plate_quat = plate_quat.unsqueeze(1).expand(-1, len(self.gear_assets), -1)
        target_pos = plate_pos.unsqueeze(1) + math_utils.quat_apply(
            expanded_plate_quat.reshape(-1, 4),
            offsets.unsqueeze(0).expand(env.num_envs, -1, -1).reshape(-1, 3),
        ).reshape(env.num_envs, len(self.gear_assets), 3)
        position_error = gear_pos - target_pos
        xy_error = torch.linalg.norm(position_error[..., :2], dim=-1)
        z_error = torch.abs(position_error[..., 2])

        _, plate_top_z = self._world_collision_z_bounds(self.plate_collision_corners, plate_pos, plate_quat)
        gear_bottom_z = torch.stack(
            [
                self._world_collision_z_bounds(corners, pos, quat)[0]
                for corners, pos, quat in zip(
                    self.gear_collision_corners,
                    gear_pos.unbind(dim=1),
                    gear_quat.unbind(dim=1),
                    strict=True,
                )
            ],
            dim=1,
        )
        support_error = torch.abs(gear_bottom_z - plate_top_z.unsqueeze(1))

        up_axis = self.up_axis.expand(env.num_envs * len(self.gear_assets), -1)
        gear_up = math_utils.quat_apply(gear_quat.reshape(-1, 4), up_axis).reshape(
            env.num_envs, len(self.gear_assets), 3
        )
        plate_up = math_utils.quat_apply(plate_quat, self.up_axis.expand(env.num_envs, -1)).unsqueeze(1)
        min_upright_cos = torch.cos(torch.deg2rad(torch.tensor(upright_axis_threshold_deg, device=env.device)))
        upright = torch.sum(gear_up * plate_up, dim=-1) >= min_upright_cos

        linear_speed = torch.linalg.norm(gear_vel[..., :3], dim=-1)
        angular_speed = torch.linalg.norm(gear_vel[..., 3:], dim=-1)
        self.diagnostics_per_gear = {
            "xy_error_m": xy_error,
            "z_error_m": z_error,
            "support_error_m": support_error,
            "upright": upright,
            "linear_speed_m_s": linear_speed,
            "angular_speed_rad_s": angular_speed,
        }
        self.success_per_gear = (
            (xy_error <= xy_threshold)
            & (z_error <= z_threshold)
            & (support_error <= support_z_threshold)
            & upright
            & (linear_speed <= linear_velocity_threshold)
            & (angular_speed <= angular_velocity_threshold)
        )
        success_now = torch.all(self.success_per_gear, dim=1)
        self.consecutive_success_count = torch.where(
            success_now,
            self.consecutive_success_count + 1,
            torch.zeros_like(self.consecutive_success_count),
        )
        return self.consecutive_success_count >= consecutive_success_steps
