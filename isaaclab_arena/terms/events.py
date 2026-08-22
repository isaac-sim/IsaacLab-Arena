# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from dataclasses import dataclass
from typing import Any

import carb
import warp as wp
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_prim_tree import exclude_referenced_physics_roots, find_nested_physics_roots
from isaaclab_arena.utils.velocity import Velocity


@dataclass(frozen=True)
class _RigidReset:
    """A private rigid asset and its env-local initial pose."""

    asset: Any
    root_pose_local: torch.Tensor
    """Root pose with shape ``(7,)``."""

    def restore(self, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
        root_pose = self.root_pose_local.unsqueeze(0).repeat(len(env_ids), 1)
        root_pose[:, :3] += env_origins[env_ids]
        self.asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        root_velocity = torch.zeros_like(self.asset.data.root_vel_w.torch[env_ids])
        self.asset.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)


@dataclass(frozen=True)
class _ArticulationReset:
    """A private articulation asset and its env-local initial state."""

    asset: Any
    root_pose_local: torch.Tensor
    """Root pose with shape ``(7,)``."""
    joint_position: torch.Tensor
    """Joint positions with shape ``(num_joints,)``."""

    def restore(self, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
        root_pose = self.root_pose_local.unsqueeze(0).repeat(len(env_ids), 1)
        root_pose[:, :3] += env_origins[env_ids]
        self.asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        root_velocity = torch.zeros_like(self.asset.data.root_vel_w.torch[env_ids])
        joint_velocity = torch.zeros_like(self.asset.data.joint_vel.torch[env_ids])
        self.asset.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        joint_position = self.joint_position.unsqueeze(0).repeat(len(env_ids), 1)
        self.asset.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)


class ResetBackgroundPhysics(ManagerTermBase):
    """Restore unreferenced background physics roots through deferred runtime views."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._background_prim_paths: dict[str, str] = cfg.params["background_prim_paths"]
        self._physics_paths: dict[str, dict[str, ObjectType]] = cfg.params["physics_paths"]
        self._referenced_paths: dict[str, dict[str, ObjectType]] = cfg.params["referenced_paths"]
        self._is_initialized = False
        self._rigid_resets: list[_RigidReset] = []
        self._articulation_resets: list[_ArticulationReset] = []
        self._all_env_ids = torch.arange(env.scene.num_envs, device=env.device)

    @staticmethod
    def _runtime_path(path_template: str, env_prim_path: str) -> str:
        return path_template.format(ENV_REGEX_NS=env_prim_path)

    @staticmethod
    def _is_unavailable_backend_error(exc: Exception) -> bool:
        """Return whether initialization failed because no live backend view exists."""
        if isinstance(exc, RuntimeError):
            message = str(exc)
            return message.endswith("Please check PhysX logs.") and message.startswith(
                ("Failed to create rigid body at:", "Failed to create articulation at:")
            )
        if isinstance(exc, KeyError) and len(exc.args) == 1:
            message = str(exc.args[0])
            return message.startswith("No articulations matching pattern '") and message.endswith("'")
        return False

    @staticmethod
    def _initialize_asset(asset_cfg: Any, prim_path: str, asset_kind: str) -> Any | None:
        """Initialize a private asset, returning None when no backend object exists."""
        asset = asset_cfg.class_type(asset_cfg)
        try:
            asset._initialize_callback(None)
        except Exception as exc:
            asset._clear_callbacks()
            # Composed USD APIs may expose authored physics roots that the active
            # backend does not materialize. Backends do not share an exception
            # type for that case, so accept only their known error signatures.
            if not ResetBackgroundPhysics._is_unavailable_backend_error(exc):
                raise
            carb.log_warn(f"Skipping unavailable background {asset_kind} '{prim_path}': {exc}")
            return None
        return asset

    def _validate_runtime_composition(self, env: ManagerBasedEnv) -> None:
        """Verify source-stage discovery covers every live background-owned physics root."""
        env_prim_path = env.scene.env_prim_paths[0]
        for background_name, background_path_template in self._background_prim_paths.items():
            background_path = self._runtime_path(background_path_template, env_prim_path)
            background_prim = env.scene.stage.GetPrimAtPath(background_path)
            assert background_prim.IsValid(), f"Missing opted-in background prim at '{background_path}'"
            referenced_paths = {
                self._runtime_path(path, env_prim_path): object_type
                for path, object_type in self._referenced_paths[background_name].items()
            }
            runtime_paths = set(
                exclude_referenced_physics_roots(
                    find_nested_physics_roots(background_prim),
                    referenced_paths,
                )
            )
            expected_paths = {self._runtime_path(path, env_prim_path) for path in self._physics_paths[background_name]}
            unregistered_runtime_paths = sorted(runtime_paths - expected_paths)
            missing_runtime_paths = sorted(expected_paths - runtime_paths)
            assert not unregistered_runtime_paths and not missing_runtime_paths, (
                f"Nested physics discovery for background '{background_name}' differs from the live composed stage. "
                f"Unregistered runtime roots: {unregistered_runtime_paths}. "
                f"Missing runtime roots for generated entities: {missing_runtime_paths}."
            )

    def _capture_initial_state(self, env: ManagerBasedEnv) -> None:
        """Create private assets after simulation initialization and snapshot their state."""
        # Source-USD discovery happens before runtime composition. Validate it once
        # here, after deferred RTX/backend initialization has completed.
        self._validate_runtime_composition(env)
        for physics_paths in self._physics_paths.values():
            for path_template, object_type in physics_paths.items():
                prim_path = self._runtime_path(path_template, env.scene.env_regex_ns)
                if object_type == ObjectType.ARTICULATION:
                    asset_cfg = ArticulationCfg(
                        prim_path=prim_path,
                        actuators={},
                        init_state=ArticulationCfg.InitialStateCfg(joint_pos={}, joint_vel={}),
                    )
                    asset = self._initialize_asset(asset_cfg, prim_path, "articulation")
                    if asset is None:
                        continue
                    self._articulation_resets.append(
                        _ArticulationReset(
                            asset=asset,
                            root_pose_local=self._env_local_root_pose(asset, env),
                            joint_position=asset.data.joint_pos.torch[0].clone(),
                        )
                    )
                else:
                    asset_cfg = RigidObjectCfg(prim_path=prim_path)
                    asset = self._initialize_asset(asset_cfg, prim_path, "rigid body")
                    if asset is None:
                        continue
                    self._rigid_resets.append(
                        _RigidReset(
                            asset=asset,
                            root_pose_local=self._env_local_root_pose(asset, env),
                        )
                    )
        self._is_initialized = True

    @staticmethod
    def _env_local_root_pose(asset: Any, env: ManagerBasedEnv) -> torch.Tensor:
        """Return env 0's root pose in its environment-local frame."""
        root_pose_local = asset.data.root_pose_w.torch[0].clone()
        root_pose_local[:3] -= env.scene.env_origins[0]
        return root_pose_local

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        background_prim_paths: dict[str, str],  # noqa: ARG002
        physics_paths: dict[str, dict[str, ObjectType]],  # noqa: ARG002
        referenced_paths: dict[str, dict[str, ObjectType]],  # noqa: ARG002
    ) -> None:
        if not self._is_initialized:
            self._capture_initial_state(env)
        if env_ids is None:
            env_ids = self._all_env_ids
        else:
            env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
        for reset in self._rigid_resets:
            reset.restore(env_ids, env.scene.env_origins)
        for reset in self._articulation_resets:
            reset.restore(env_ids, env.scene.env_origins)


def set_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
    velocity: Velocity | None = None,
) -> None:
    if env_ids is None:
        return
    # Grab the object
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    # Convert the pose to the env frame
    pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).repeat(num_envs, 1)
    pose_t_xyz_q_xyzw[:, :3] += env.scene.env_origins[env_ids]
    # Set the pose and velocity
    asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=env_ids)
    if velocity is not None:
        vel = velocity.to_tensor(device=env.device).unsqueeze(0).expand(num_envs, -1)
        asset.write_root_velocity_to_sim(vel, env_ids=env_ids)
    else:
        asset.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=env.device), env_ids=env_ids)


def reset_articulation_pose_and_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
    velocity: Velocity | None = None,
) -> None:
    """Restore an articulation's root state and default joint state."""
    if env_ids is None:
        return
    set_object_pose(env, env_ids, asset_cfg, pose, velocity)
    asset = env.scene[asset_cfg.name]
    joint_position = asset.data.default_joint_pos.torch[env_ids].clone()
    joint_velocity = asset.data.default_joint_vel.torch[env_ids].clone()
    asset.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)


def set_object_pose_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_list: list[Pose],
) -> None:
    if env_ids is None:
        return

    # Grab the object
    asset = env.scene[asset_cfg.name]

    # Set the objects pose in each environment independently
    assert env_ids.ndim == 1
    for cur_env in env_ids.tolist():
        # Convert the pose to the env frame
        pose = pose_list[cur_env]
        pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).unsqueeze(0)
        pose_t_xyz_q_xyzw[0, :3] += env.scene.env_origins[cur_env, :]
        # Set the pose and velocity
        asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=torch.tensor([cur_env], device=env.device))
        asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
        )


def _write_scene_pose(env: ManagerBasedEnv, scene_name: str, pose: Pose, env_ids: torch.Tensor) -> None:
    """Write one env-local ``pose`` to a scene entity's root across ``env_ids`` (env origins added)."""
    asset = env.scene[scene_name]
    num_envs = len(env_ids)
    pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).repeat(num_envs, 1)
    pose_t_xyz_q_xyzw[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=env.device), env_ids=env_ids)


def reset_placement_asset_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    scene_writes: list[tuple[str, Pose]],
) -> None:
    """Restore a placement asset to fixed env-local poses on reset.

    Each ``(scene entity name, pose)`` in ``scene_writes`` is written to every resetting env,
    letting a compound asset place several prims (e.g. a robot and its stand) together.
    """
    if env_ids is None:
        return
    for scene_name, pose in scene_writes:
        _write_scene_pose(env, scene_name, pose, env_ids)


def reset_placement_asset_pose_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    write_pose_list: list[list[tuple[str, Pose]]],
) -> None:
    """Restore a placement asset to a distinct per-env pose on reset.

    ``write_pose_list[env]`` holds that env's ``(scene entity name, pose)`` writes, so different
    environments can hold different solved layouts for the same asset (and its auxiliary prims).
    """
    if env_ids is None:
        return
    assert env_ids.ndim == 1, "env_ids must be a 1-D tensor of environment indices"
    num_scene_envs = env.scene.env_origins.shape[0]
    assert len(write_pose_list) == num_scene_envs, (
        f"per-env pose writes cover {len(write_pose_list)} envs, but the scene has {num_scene_envs}; "
        "write_pose_list is indexed by absolute env id and must span every environment."
    )
    for cur_env in env_ids.tolist():
        single_env = torch.tensor([cur_env], device=env.device)
        for scene_name, pose in write_pose_list[cur_env]:
            _write_scene_pose(env, scene_name, pose, single_env)


def reset_all_articulation_joints(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset the articulation joints to the initial state."""
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = wp.to_torch(articulation_asset.data.default_root_state)[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = wp.to_torch(articulation_asset.data.default_joint_pos)[env_ids].clone()
        default_joint_vel = wp.to_torch(articulation_asset.data.default_joint_vel)[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
