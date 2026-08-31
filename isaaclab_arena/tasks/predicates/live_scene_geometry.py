# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read current scene poses and build geometry used by spatial predicates."""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.views import FrameView
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


# TODO(cvolk, 2026-08-24): [arena-world-migration] Move current-pose lookup, spawned-geometry extraction,
# ObjectSet variant mapping, and geometry caching behind env.arena_world, then delete this predicate-owned module.
def _get_spawned_entity_groups(
    env: ManagerBasedEnv,
    entity_name: str,
    geometry_prim_path: str,
) -> list[tuple[Usd.Prim, tuple[int, ...]]]:
    """Return one representative prim and its cloned environments per asset variant."""
    scene = env.scene
    assert scene.clone_plan is not None, f"Scene entity '{entity_name}' has no clone plan."

    clone_matches = tuple(cloner.query.iter_sources(scene.clone_plan, geometry_prim_path))
    assert (
        clone_matches
    ), f"Scene entity '{entity_name}' geometry path '{geometry_prim_path}' does not match the scene clone plan."

    spawned_entity_groups = []
    for _source_root, _destination_template, representative_path, environment_ids in clone_matches:
        representative_prim = scene.stage.GetPrimAtPath(representative_path)
        assert (
            representative_prim.IsValid()
        ), f"Scene entity '{entity_name}' clone source '{representative_path}' is not a valid USD prim."
        spawned_entity_groups.append((representative_prim, environment_ids))
    return spawned_entity_groups


def _find_single_rigid_body_prim_in_subtree(root_prim: Usd.Prim, entity_name: str) -> Usd.Prim:
    """Return the only rigid-body prim in the root prim's subtree."""
    rigid_body_prims_in_subtree = sim_utils.get_all_matching_child_prims(
        root_prim.GetPath(),
        predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
        stage=root_prim.GetStage(),
        traverse_instance_prims=False,
    )
    assert len(rigid_body_prims_in_subtree) == 1, (
        f"Rigid scene entity '{entity_name}' source '{root_prim.GetPath()}' contains "
        f"{len(rigid_body_prims_in_subtree)} rigid bodies; expected exactly one."
    )
    return rigid_body_prims_in_subtree[0]


def _compute_geometry_bounds_in_prim_frame(prim: Usd.Prim) -> AxisAlignedBoundingBox:
    """Compute descendant geometry bounds in the prim frame.

    The bounds are aligned with the prim's axes and measured from its origin.
    Initial world translation and rotation are excluded, while authored scale
    remains in the bounds.
    """
    assert prim.IsValid(), "Prim must be valid."

    time_code = Usd.TimeCode.Default()
    transform_cache = UsdGeom.XformCache(time_code)
    T_W_P = transform_cache.GetLocalToWorldTransform(prim).RemoveScaleShear()
    T_P_W = T_W_P.GetInverse()
    bounding_box_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])

    lower_P = np.full(3, np.inf, dtype=np.float64)
    upper_P = np.full(3, -np.inf, dtype=np.float64)
    found_geometry = False
    for geometry_prim in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not geometry_prim.IsA(UsdGeom.Gprim):
            continue
        if UsdGeom.Imageable(geometry_prim).ComputePurpose() != UsdGeom.Tokens.default_:
            continue

        geometry_bounds = bounding_box_cache.ComputeWorldBound(geometry_prim)
        # Remove the prim's initial world translation and rotation while leaving spawned scale in the bounds.
        geometry_bounds.Transform(T_P_W)
        geometry_range = geometry_bounds.ComputeAlignedRange()
        if geometry_range.IsEmpty():
            continue

        lower_P = np.minimum(lower_P, np.asarray(geometry_range.GetMin(), dtype=np.float64))
        upper_P = np.maximum(upper_P, np.asarray(geometry_range.GetMax(), dtype=np.float64))
        found_geometry = True

    prim_path = prim.GetPath()
    assert found_geometry, f"Prim '{prim_path}' has no default-purpose geometry."
    return AxisAlignedBoundingBox(
        min_point=tuple(float(value) for value in lower_P),
        max_point=tuple(float(value) for value in upper_P),
    )


def compute_spawned_geometry_bounds_in_entity_frame(
    env: ManagerBasedEnv,
    entity_cfg: SceneEntityCfg,
) -> AxisAlignedBoundingBox:
    """Build spawned geometry bounds in the entity frame.

    This reads the USD stage. Callers should cache the result and combine it
    with current poses rather than rebuilding it on each simulation step.

    Args:
        env: The runtime manager-based environment.
        entity_cfg: Scene entity whose runtime pose defines frame ``E``.

    Returns:
        One batched AABB aligned with frame ``E`` and measured from its origin.
        Initial world translation and rotation are excluded, while spawned
        scale remains in the bounds.
    """
    scene = env.scene
    entity_name = entity_cfg.name
    assert (
        entity_name in scene.rigid_objects or entity_name in scene.extras
    ), f"Scene entity '{entity_name}' must be a rigid object or an AssetBaseCfg scene entry."
    is_rigid_object = entity_name in scene.rigid_objects
    resolved_geometry_prim_path = getattr(scene.cfg, entity_name).prim_path.format(ENV_REGEX_NS=scene.env_regex_ns)

    lower_E_by_environment = torch.empty((env.num_envs, 3), dtype=torch.float32, device=env.device)
    upper_E_by_environment = torch.empty_like(lower_E_by_environment)
    coverage_count = [0] * env.num_envs

    for representative_prim, environment_ids in _get_spawned_entity_groups(
        env,
        entity_name,
        resolved_geometry_prim_path,
    ):
        entity_frame_prim = (
            _find_single_rigid_body_prim_in_subtree(representative_prim, entity_name)
            if is_rigid_object
            else representative_prim
        )
        geometry_bounds_E = _compute_geometry_bounds_in_prim_frame(entity_frame_prim).to(env.device)
        environment_indices = torch.tensor(environment_ids, dtype=torch.long, device=env.device)
        lower_E_by_environment[environment_indices] = geometry_bounds_E.min_point[0]
        upper_E_by_environment[environment_indices] = geometry_bounds_E.max_point[0]
        for environment_id in environment_ids:
            coverage_count[environment_id] += 1

    assert all(count == 1 for count in coverage_count), (
        f"Scene entity '{entity_name}' geometry must cover every environment exactly once; got coverage"
        f" {coverage_count}."
    )
    return AxisAlignedBoundingBox(min_point=lower_E_by_environment, max_point=upper_E_by_environment)


# TODO(cvolk, 2026-08-25): Remove this workaround when
# test_scene_extra_frame_view_covers_cloned_environments starts reporting XPASS.
class AssetBaseCfgPoseReader:
    """Read current poses for one named AssetBaseCfg scene entry.

    InteractiveScene creates ``scene.extras`` before cloning. This reader is
    created after cloning so its FrameView covers every parallel environment.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        entity_name: str,
    ):
        scene = env.scene
        assert entity_name in scene.extras, f"Scene entity '{entity_name}' must be an AssetBaseCfg scene entry."

        self._entity_name = entity_name
        self._num_envs = env.num_envs
        entity_prim_path = getattr(scene.cfg, entity_name).prim_path.format(ENV_REGEX_NS=scene.env_regex_ns)
        self._frame_view: FrameView | None = FrameView(
            entity_prim_path,
            device=env.device,
            stage=scene.stage,
        )
        # Ensure pose rows use the same environment order as other scene tensors.
        entity_prim_paths = self._frame_view.prim_paths
        assert len(entity_prim_paths) == env.num_envs, (
            f"AssetBaseCfg scene entry '{entity_name}' resolved to {len(entity_prim_paths)} prims; "
            f"expected {env.num_envs}."
        )
        for environment_id, prim_path in enumerate(entity_prim_paths):
            environment_prim_path = scene.env_prim_paths[environment_id]
            assert str(prim_path).startswith(f"{environment_prim_path}/"), (
                f"AssetBaseCfg scene entry '{entity_name}' pose row {environment_id} belongs to '{prim_path}', "
                f"not environment '{environment_prim_path}'."
            )

    def close(self) -> None:
        """Close the term-owned frame view. Safe to call more than once."""
        if self._frame_view is not None:
            self._frame_view.close()
            self._frame_view = None

    def get_pose_in_world_frame(self) -> torch.Tensor:
        """Return ``T_W_R``, taking points from reference frame ``R`` to world frame ``W``.

        Poses are encoded as ``(x, y, z, qx, qy, qz, qw)``.
        """
        assert self._frame_view is not None, f"Pose reader for '{self._entity_name}' is closed."
        position_w_buffer, orientation_w_buffer = self._frame_view.get_world_poses()
        T_W_R = torch.cat((position_w_buffer.torch, orientation_w_buffer.torch), dim=-1)
        assert T_W_R.shape == (self._num_envs, 7), (
            f"AssetBaseCfg scene entry '{self._entity_name}' returned pose shape {tuple(T_W_R.shape)}; "
            f"expected ({self._num_envs}, 7)."
        )
        return T_W_R
