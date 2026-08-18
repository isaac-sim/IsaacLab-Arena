# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import warp as wp
from isaaclab.managers import EventTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from pxr import Sdf


@dataclass(frozen=True)
class _RigidBodyReset:
    """A rigid-body view paired with its first-reset transform snapshot."""

    view: Any
    transforms: torch.Tensor

    def restore(self, indices: wp.array, zero_velocities: torch.Tensor) -> None:
        self.view.set_transforms(wp.from_torch(self.transforms), indices=indices)
        self.view.set_velocities(wp.from_torch(zero_velocities), indices=indices)


@dataclass(frozen=True)
class _ArticulationReset:
    """An articulation view paired with its first-reset state snapshot."""

    view: Any
    root_transforms: torch.Tensor
    dof_positions: torch.Tensor

    def restore(self, indices: wp.array, zero_root_velocities: torch.Tensor) -> None:
        self.view.set_root_transforms(wp.from_torch(self.root_transforms), indices=indices)
        self.view.set_root_velocities(wp.from_torch(zero_root_velocities), indices=indices)
        self.view.set_dof_positions(wp.from_torch(self.dof_positions), indices=indices)
        self.view.set_dof_velocities(wp.from_torch(torch.zeros_like(self.dof_positions)), indices=indices)


def _exclude_articulation_owned_bodies(
    body_paths: tuple[str, ...], articulation_root_paths: tuple[Sdf.Path, ...]
) -> tuple[str, ...]:
    """Return bodies outside subtrees owned by an articulation root."""
    from pxr import Sdf

    return tuple(
        body_path
        for body_path in body_paths
        if not any(Sdf.Path(body_path).HasPrefix(root_path) for root_path in articulation_root_paths)
    )


class ResetBackgroundBodies(ManagerTermBase):
    """Restore embedded background bodies to their state at the first environment reset."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._background_name = cast(str, cfg.params["background_name"])
        self._rigid_body_resets: list[_RigidBodyReset] = []
        self._articulation_resets: list[_ArticulationReset] = []
        # EventManager constructs non-prestartup class terms immediately before their first
        # invocation. Capture once here, before this term writes any reset state.
        self._initialize(env)

    def _initialize(self, env: ManagerBasedEnv) -> None:
        from pxr import Usd, UsdPhysics

        root_path = f"{env.scene.env_prim_paths[0]}/{self._background_name}"
        root_prim = env.sim.stage.GetPrimAtPath(root_path)
        assert root_prim.IsValid(), f"Missing background prim at '{root_path}'"

        body_paths = tuple(
            str(prim.GetPath()) for prim in Usd.PrimRange(root_prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        assert body_paths, f"No rigid bodies found under '{root_path}'"
        physics_view = env.sim.physics_manager.get_physics_sim_view()
        articulation_root_paths = tuple(
            prim.GetPath() for prim in Usd.PrimRange(root_prim) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        for articulation_root_path in articulation_root_paths:
            path_pattern = str(articulation_root_path).replace("/env_0/", "/env_*/", 1)
            view = physics_view.create_articulation_view(path_pattern)
            assert view.count == env.num_envs, (
                f"Expected one '{articulation_root_path}' articulation per environment, "
                f"got {view.count} for {env.num_envs} envs"
            )
            self._articulation_resets.append(
                _ArticulationReset(
                    view=view,
                    root_transforms=wp.to_torch(view.get_root_transforms()).clone(),
                    dof_positions=wp.to_torch(view.get_dof_positions()).clone(),
                )
            )

        # Articulation links must be restored through their articulation view; PhysX rejects
        # transform writes through a rigid-body view. Replicator assets keep those links below
        # the prim carrying ArticulationRootAPI, so subtree membership partitions the two APIs.
        rigid_body_paths = _exclude_articulation_owned_bodies(body_paths, articulation_root_paths)
        for body_path in rigid_body_paths:
            path_pattern = body_path.replace("/env_0/", "/env_*/", 1)
            view = physics_view.create_rigid_body_view(path_pattern)
            assert (
                view.count == env.num_envs
            ), f"Expected one '{body_path}' rigid body per environment, got {view.count} for {env.num_envs} envs"
            self._rigid_body_resets.append(
                _RigidBodyReset(view=view, transforms=wp.to_torch(view.get_transforms()).clone())
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        background_name: str,  # noqa: ARG002
    ) -> None:
        if env_ids is None:
            reset_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        else:
            reset_env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
        view_indices = wp.from_torch(reset_env_ids.to(dtype=torch.int32))
        zero_velocities = torch.zeros((env.num_envs, 6), device=env.device)

        # Background bodies stay dynamic for settling and interaction, so reset both state
        # and residual momentum between episodes.
        for reset in self._rigid_body_resets:
            reset.restore(view_indices, zero_velocities)
        for reset in self._articulation_resets:
            reset.restore(view_indices, zero_velocities)
