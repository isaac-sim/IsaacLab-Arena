# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, ManagerTermBase

from isaaclab_arena.relations.relations import RotateAroundSolution, get_anchor_objects
from isaaclab_arena.utils.pose import Pose, PosePerEnv
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw, yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer

IDENTITY_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)

# Name of the reset event term that owns the pooled object placer.
PLACEMENT_RESET_EVENT_NAME = "placement_reset"
CACHED_PLACEMENT_RESET_EVENT_NAME = "cached_placement_reset"


class PlacementPoolHandle:
    """Opaque holder for a runtime placement pool to bypass EventTermCfg param deepcopy/validation errors.

    PooledObjectPlacer is used as an EventTermCfg param to set the initial spawn pose. Isaac Lab deep-copies
    and validates the configclass param, leading to two crashes: deepcopy fails for the Warp GPU cache
    wp.Mesh BVHs ("ctypes objects containing pointers cannot be pickled"); validation hits RecursionError
    when recursively walking all dicts and reaches placement assets (including embodiments with cyclic
    scene configs).

    This handle wraps PooledObjectPlacer with overrides for deepcopy and validation, while
    PooledObjectPlacer itself stays a normal class. EventTermCfg params use this handle.
    """

    __slots__ = ("pool",)
    """Store pool in a slot instead of an instance dictionary; ``hasattr(handle, "__dict__")`` is false,
    so ``_validate(handle)`` stops traversing into PooledObjectPlacer ."""

    def __init__(self, pool: PooledObjectPlacer) -> None:
        self.pool = pool

    def __deepcopy__(self, memo: dict[int, object]) -> PlacementPoolHandle:
        """Share the live pool across ``copy.deepcopy`` to avoid deep-copying the Warp cache BVHs."""
        memo[id(self)] = self
        return self


def get_placement_pool(env) -> PooledObjectPlacer | None:
    """Return the pooled placer stored on the env reset event, or ``None`` when absent.

    Lets a runtime caller reach the pool (e.g. to run the post-reset settle check) from the env alone,
    without holding the builder. The pool is reached through the env's event manager.

    Args:
        env: The gym-wrapped Isaac Lab env; the base env is reached via ``env.unwrapped``.
    """
    try:
        term_cfg = env.unwrapped.event_manager.get_term_cfg(PLACEMENT_RESET_EVENT_NAME)
    except ValueError:
        return None
    handle = term_cfg.params.get("placement_pool")
    assert handle is not None, f"'{PLACEMENT_RESET_EVENT_NAME}' event is missing its placement_pool parameter."
    return handle.pool


def get_rotation_xyzw(asset: PlaceableAsset) -> tuple[float, float, float, float]:
    """Return the RotateAroundSolution rotation for an asset, or identity if none."""
    rotate_marker = next((r for r in asset.get_relations() if isinstance(r, RotateAroundSolution)), None)
    return rotate_marker.get_rotation_xyzw() if rotate_marker else IDENTITY_ROTATION_XYZW


def get_base_rotation_per_asset(
    assets: list[PlaceableAsset],
) -> dict[PlaceableAsset, tuple[float, float, float, float]]:
    """Return the base rotation for each asset."""
    return {asset: get_rotation_xyzw(asset) for asset in assets}


def get_pose_from_layout(asset: PlaceableAsset, layout: PlacementResult) -> Pose:
    """Return an asset pose from a solved layout."""
    assert asset in layout.positions, f"Placement layout is missing non-anchor asset '{asset.name}'"
    base_rotation = get_rotation_xyzw(asset)
    marker_yaw = yaw_from_quat_xyzw(base_rotation)
    total_yaw = layout.orientations.get(asset, marker_yaw)
    rotation = rotate_quat_by_yaw(base_rotation, total_yaw - marker_yaw)
    return Pose(position_xyz=layout.positions[asset], rotation_xyzw=rotation)


def get_movable_asset_names(
    assets: list[PlaceableAsset],
    anchor_assets: set[PlaceableAsset],
) -> list[str]:
    """Return scene names for non-anchor placement assets."""
    return [asset.get_scene_key() for asset in assets if asset not in anchor_assets]


def validate_scene_poses(poses: dict[str, torch.Tensor]) -> None:
    """Require finite xyz/xyzw pose tensors of shape (N, 7) with unit quaternions."""
    for name, pose in poses.items():
        assert pose.ndim == 2 and pose.shape[1] == 7, f"Root poses for '{name}' must have shape (N, 7)"
        assert torch.isfinite(pose).all(), f"Root poses for '{name}' must be finite"
        assert torch.allclose(
            pose[:, 3:].square().sum(dim=-1), torch.ones_like(pose[:, 0]), atol=1e-4, rtol=0
        ), f"Root poses for '{name}' require unit quaternions"


def write_scene_poses_to_sim(env: ManagerBasedEnv, env_ids: torch.Tensor, poses: dict[str, torch.Tensor]) -> None:
    """Apply environment-local root poses and zero velocities for the selected environments.

    Frames E, W, and O denote the environment, simulation world, and object.

    Args:
        env: Constructed simulation environment.
        env_ids: Absolute indices of the N resetting environments, shape (N,).
        poses: Scene entity names mapped to xyz/xyzw tensors, each shaped (N, 7).
            Compound asset poses must first be expanded with layout_pose_to_scene_writes().
            Call validate_scene_poses() before writing unvalidated external poses.
    """
    for name, pose in poses.items():
        assert pose.shape == (len(env_ids), 7), f"Root poses for '{name}' must have shape (N, 7)"
    env_origins = env.scene.env_origins[env_ids]
    zero_velocity = torch.zeros((len(env_ids), 6), device=env.device)
    for name, T_E_O in poses.items():
        T_W_O = T_E_O.clone()
        T_W_O[:, :3] += env_origins
        scene_asset = env.scene[name]
        scene_asset.write_root_pose_to_sim(T_W_O, env_ids=env_ids)
        scene_asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def write_layout_to_sim(
    env: ManagerBasedEnv,
    env_id: int,
    result: PlacementResult,
    anchor_assets: set[PlaceableAsset],
    base_rotations: dict[PlaceableAsset, tuple[float, float, float, float]],
) -> None:
    """Write one env's solved layout into the sim.

    Even writing zero velocity, the sim will still apply gravity and other forces from collisions,
    so collided assets will still be subject to move.

    Args:
        env: The Isaac Lab ManagerBasedEnv environment.
        env_id: The environment index.
        result: The placement result to write to the sim.
        anchor_assets: The set of anchor assets.
        base_rotations: The base rotations for all assets.
    """
    missing_assets = [
        asset.name for asset in base_rotations if asset not in anchor_assets and asset not in result.positions
    ]
    assert not missing_assets, f"Placement layout is missing non-anchor assets: {missing_assets}"
    for asset in result.positions:
        if asset in anchor_assets:
            continue
        layout_pose = get_pose_from_layout(asset, result)
        asset.write_layout_pose_to_sim(env, env_id, layout_pose)


def solve_and_place_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    placement_pool: PlacementPoolHandle,
) -> None:
    """Coordinated reset event that draws layouts from the pool and writes poses.

    Registered as a single EventTermCfg(mode="reset"). Layouts are env-indexed:
    one layout is consumed for each requested absolute env id, so partial resets
    only advance the pools of the resetting envs.

    Args:
        env: The Isaac Lab environment.
        env_ids: 1-D tensor of environment indices being reset.
        placement_pool: Opaque handle to the runtime pool of solved placement layouts.
            Layout assets come from ``placement_pool.pool.objects``.
    """
    pool = placement_pool.pool
    if env_ids is None or len(env_ids) == 0:
        return
    assets = pool.objects
    reset_env_ids = env_ids.tolist()
    num_scene_envs = env.scene.env_origins.shape[0]
    assert (
        pool.num_envs == num_scene_envs
    ), f"Placement pool has {pool.num_envs} envs, but scene has {num_scene_envs} env origins."
    results_by_env = pool.sample_for_envs(reset_env_ids)
    anchor_assets = set(get_anchor_objects(assets))
    base_rotations = get_base_rotation_per_asset(assets)

    for cur_env in reset_env_ids:
        result = results_by_env[cur_env]
        if not result.success:
            print(
                "Warning: Writing best-loss fallback placement for "
                f"env {cur_env}; failed checks: {result.validation_results.get_failed_validation_check_names}."
            )
        # Only write non-anchor assets to the sim.
        write_layout_to_sim(env, cur_env, result, anchor_assets, base_rotations)


class ResetPlacementLayouts(ManagerTermBase):
    """Complete cached layouts drawn from one shared queue for N environments.

    L is the layout count; each pose contains xyz position and xyzw rotation.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._poses = {
            name: torch.tensor(poses, device=env.device, dtype=torch.float32)
            for name, poses in cfg.params["poses"].items()
        }
        """Object-to-pose tensors, each shaped (L, 7); L is the number of layouts."""
        assert self._poses, "Cached reset requires at least one object"
        shapes = {tuple(poses.shape) for poses in self._poses.values()}
        assert len(shapes) == 1, "Cached reset objects must have equal layout counts"
        shape = next(iter(shapes))
        assert len(shape) == 2 and shape[0] > 0 and shape[1] == 7, "Cached reset poses must have shape (L, 7), L > 0"
        validate_scene_poses(self._poses)
        self._num_layouts = shape[0]
        self._all_env_ids = torch.arange(env.num_envs, device=env.device)
        """Absolute environment indices, shape (N,)."""
        self._next_layout = 0
        """Next index in the shared layout queue; wraps after all L layouts are consumed."""
        for name in self._poses:
            assert (
                name in env.scene.rigid_objects or name in env.scene.articulations
            ), f"Cached object '{name}' must have a writable physics root"

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None, poses: dict[str, list[list[float]]]) -> None:
        """Apply the next complete layout to each resetting environment.

        Args:
            env: Environment whose root poses are reset.
            env_ids: Environments to reset, or None for all environments.
            poses: Layout configuration required by the event-manager calling contract.
                Pose tensors are built once in __init__; this argument is unused here.
        """
        env_ids = self._all_env_ids if env_ids is None else env_ids
        if len(env_ids) == 0:
            return
        selected_poses = self.draw(env_ids)
        write_scene_poses_to_sim(env, env_ids, selected_poses)

    def draw(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Draw complete layouts in env_ids order, wrapping the shared queue on exhaustion.

        Args:
            env_ids: Absolute indices of the M resetting environments, shape (M,).

        Returns:
            Scene entity poses in the environment frame, each shaped (M, 7).
        """
        layout_ids = (self._next_layout + torch.arange(len(env_ids), device=env_ids.device)) % self._num_layouts
        poses = {name: values[layout_ids] for name, values in self._poses.items()}
        self._next_layout = (self._next_layout + len(env_ids)) % self._num_layouts
        return poses


def make_cached_placement_event(
    layouts: PlacementLayouts, placement_assets: list[PlaceableAsset], num_envs: int
) -> EventTermCfg:
    """Replace cached assets' initial poses and pose-reset events with one reset writer."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectBase

    layouts.validate_assets(placement_assets)
    assets = {asset.get_scene_key(): asset for asset in placement_assets}
    for name in layouts.poses:
        asset = assets[name]
        if isinstance(asset, Object):
            assert asset.reset_pose, f"Cached asset '{name}' has pose resets disabled"
        if isinstance(asset, ObjectBase) and asset.initial_velocity is not None:
            velocity = asset.initial_velocity
            assert all(
                value == 0 for value in (*velocity.linear_xyz, *velocity.angular_xyz)
            ), f"Cached asset '{name}' has nonzero initial velocity; replay resets velocity to zero"
        assert not asset.has_pose_reset_event() or isinstance(
            asset.get_initial_pose(), Pose
        ), f"Cached asset '{name}' has a non-fixed pose-reset policy"
    scene_poses: dict[str, list[list[float]]] = {}
    for name, poses in layouts.poses.items():
        asset = assets[name]
        asset.clear_pose_reset_event()
        asset.set_initial_pose(
            PosePerEnv([poses[i % layouts.num_layouts] for i in range(num_envs)]), create_reset_event=False
        )
        for pose in poses:
            for scene_name, scene_pose in asset.layout_pose_to_scene_writes(pose):
                scene_poses.setdefault(scene_name, []).append(list(scene_pose.position_xyz + scene_pose.rotation_xyzw))
    assert scene_poses and all(
        len(poses) == layouts.num_layouts for poses in scene_poses.values()
    ), "Cached assets must write distinct scene entities in every layout"
    return EventTermCfg(func=ResetPlacementLayouts, mode="reset", params={"poses": scene_poses})
