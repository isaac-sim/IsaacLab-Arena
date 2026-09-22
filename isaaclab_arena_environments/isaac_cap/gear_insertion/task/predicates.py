# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gear-specific insertion predicates."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

from isaaclab_arena.tasks.predicates.spatial import (
    depth_in_range,
    lateral_in_proximity,
    tilt_axis_aligned,
    velocity_below_threshold,
)

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from pxr import Usd


def reset_gear_insertion_diagnostics(env: ManagerBasedEnv, env_ids=None) -> None:
    """Clear GearInsertionConditions diagnostics for standalone or composite tasks."""
    # TODO(cvolk): Temporary CAP workaround while this predicate caches diagnostics.
    # Move those caches out of the predicate, then remove this reset callback.
    progress_tracker = env.progress_tracker
    found_matching_objective = False
    for objective in progress_tracker.progress_objectives:
        # CompositeTaskBase prefixes objective names with the subtask index.
        if objective.name.rsplit("/", 1)[-1] == "gear_insertion":
            gear_insertion_conditions = progress_tracker.get_predicate(objective.name)
            gear_insertion_conditions.reset(env_ids)
            found_matching_objective = True
    # A renamed objective must not leave diagnostics from the previous episode.
    assert found_matching_objective, "reset_gear_insertion_diagnostics found no gear_insertion objective to reset."


class GearInsertionConditions:
    """Check every gear's current placement and cache named diagnostics.

    ProgressObjectiveRunner owns the consecutive-step counters.
    GearInsertionTask clears diagnostics through reset_gear_insertion_diagnostics.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        self._support_checks = {}
        self.per_gear_results: dict[str, torch.Tensor] = {}
        self.per_gear_gate_results: dict[str, dict[str, torch.Tensor]] = {}
        for gear_name in cfg.params["gear_names"]:
            support_cfg = TerminationTermCfg(
                func=GearIsSupported,
                params={
                    "plate_asset_cfg": SceneEntityCfg(cfg.params["plate_name"]),
                    "gear_asset_cfg": SceneEntityCfg(gear_name),
                    "support_z_threshold": cfg.params["support_z_threshold"],
                },
            )
            self._support_checks[gear_name] = GearIsSupported(support_cfg, env)
            self.per_gear_results[gear_name] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            self.per_gear_gate_results[gear_name] = {
                gate_name: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                for gate_name in ("xy", "z", "upright", "support", "velocity")
            }

    def __call__(
        self,
        env: ManagerBasedEnv,
        plate_name: str,
        gear_names: tuple[str, ...],
        target_offsets_xyz: tuple[tuple[float, float, float], ...],
        xy_threshold: float,
        z_threshold: float,
        upright_axis_threshold_deg: float,
        linear_velocity_threshold: float,
        angular_velocity_threshold: float,
        support_z_threshold: float,
    ) -> torch.Tensor:
        for gear_name, target_offset_xyz in zip(gear_names, target_offsets_xyz, strict=True):
            relative_position_params = {
                "subject_name": gear_name,
                "receiver_name": plate_name,
                "target_offset_xyz": target_offset_xyz,
            }
            support_check = self._support_checks[gear_name]
            gate_results = {
                "xy": lateral_in_proximity(env, **relative_position_params, tolerance_lateral=xy_threshold),
                "z": depth_in_range(env, **relative_position_params, depth_min=-z_threshold, depth_max=z_threshold),
                "upright": tilt_axis_aligned(
                    env,
                    subject_name=gear_name,
                    receiver_name=plate_name,
                    max_tilt_rad=math.radians(upright_axis_threshold_deg),
                ),
                "support": support_check(env, support_z_threshold=support_z_threshold),
                "velocity": velocity_below_threshold(
                    env,
                    subject_name=gear_name,
                    linear_velocity_threshold=linear_velocity_threshold,
                    angular_velocity_threshold=angular_velocity_threshold,
                ),
            }
            self.per_gear_gate_results[gear_name] = gate_results
            self.per_gear_results[gear_name] = torch.stack(list(gate_results.values())).all(dim=0)
        return torch.stack(list(self.per_gear_results.values())).all(dim=0)

    def reset(self, env_ids=None) -> None:
        """Clear cached diagnostics for the environments whose episodes restart."""
        if env_ids is None:
            env_ids = slice(None)
        for gear_name, gate_results in self.per_gear_gate_results.items():
            self.per_gear_results[gear_name][env_ids] = False
            for results in gate_results.values():
                results[env_ids] = False


class GearIsSupported:
    """Check that one gear bottom remains near the plate support surface."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        self.plate_asset_cfg: SceneEntityCfg = cfg.params["plate_asset_cfg"]
        self.gear_asset_cfg: SceneEntityCfg = cfg.params["gear_asset_cfg"]
        plate_asset = env.scene[self.plate_asset_cfg.name]
        gear_asset = env.scene[self.gear_asset_cfg.name]
        self.plate_collision_corners = self._collision_corners(
            plate_asset,
            env.device,
            collision_prim_name="platform",
            enabled_only=True,
        )
        self.gear_collision_corners = self._collision_corners(gear_asset, env.device, enabled_only=True)

    def __call__(
        self,
        env: ManagerBasedEnv,
        support_z_threshold: float,
    ) -> torch.Tensor:
        T_W_P = env.arena_world.get_pose_w(self.plate_asset_cfg.name)
        T_W_G = env.arena_world.get_pose_w(self.gear_asset_cfg.name)
        _, plate_top_z = self._world_collision_z_bounds(
            self.plate_collision_corners,
            T_W_P[:, :3],
            T_W_P[:, 3:],
        )
        gear_bottom_z, _ = self._world_collision_z_bounds(
            self.gear_collision_corners,
            T_W_G[:, :3],
            T_W_G[:, 3:],
        )
        support_error = torch.abs(gear_bottom_z - plate_top_z)
        return support_error <= support_z_threshold

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
        rigid_prim = GearIsSupported._rigid_body_prim(root_prim)
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
