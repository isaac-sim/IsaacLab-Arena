# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read current scene poses and build geometry used by spatial predicates."""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.cloner.cloner_utils import iter_clone_plan_matches
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

    clone_matches = tuple(iter_clone_plan_matches(scene.clone_plan, geometry_prim_path))
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


def _find_rigid_body_prim(representative_prim: Usd.Prim, entity_name: str) -> Usd.Prim:
    """Find the rigid-body prim represented by the entity's root pose."""
    rigid_body_prims = sim_utils.get_all_matching_child_prims(
        representative_prim.GetPath(),
        predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
        stage=representative_prim.GetStage(),
        traverse_instance_prims=False,
    )
    assert len(rigid_body_prims) == 1, (
        f"Rigid scene entity '{entity_name}' source '{representative_prim.GetPath()}' contains "
        f"{len(rigid_body_prims)} rigid bodies; expected exactly one."
    )
    return rigid_body_prims[0]


def _compute_aabb_relative_to_prim(prim: Usd.Prim) -> AxisAlignedBoundingBox:
    """Compute descendant geometry bounds relative to a prim's origin and axes."""
    assert prim.IsValid(), "Prim must be valid."

    time_code = Usd.TimeCode.Default()
    transform_cache = UsdGeom.XformCache(time_code)
    prim_to_world = transform_cache.GetLocalToWorldTransform(prim).RemoveScaleShear()
    world_to_prim = prim_to_world.GetInverse()
    bounding_box_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])

    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    found_geometry = False
    for geometry_prim in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not geometry_prim.IsA(UsdGeom.Gprim):
            continue
        if UsdGeom.Imageable(geometry_prim).ComputePurpose() != UsdGeom.Tokens.default_:
            continue

        geometry_bounds = bounding_box_cache.ComputeWorldBound(geometry_prim)
        geometry_bounds.Transform(world_to_prim)
        geometry_range = geometry_bounds.ComputeAlignedRange()
        if geometry_range.IsEmpty():
            continue

        lower = np.minimum(lower, np.asarray(geometry_range.GetMin(), dtype=np.float64))
        upper = np.maximum(upper, np.asarray(geometry_range.GetMax(), dtype=np.float64))
        found_geometry = True

    prim_path = prim.GetPath()
    assert found_geometry, f"Prim '{prim_path}' has no default-purpose geometry."
    return AxisAlignedBoundingBox(
        min_point=tuple(float(value) for value in lower),
        max_point=tuple(float(value) for value in upper),
    )


def compute_spawned_geometry_aabbs_relative_to_pose(
    env: ManagerBasedEnv,
    entity_cfg: SceneEntityCfg,
) -> AxisAlignedBoundingBox:
    """Build one local AABB per environment from the geometry that was spawned.

    This reads the USD stage. Callers should cache the result and combine it
    with current poses rather than rebuilding it on each simulation step.

    Args:
        env: The runtime manager-based environment.
        entity_cfg: Scene entity whose pose defines the returned coordinates.

    Returns:
        One AABB per environment, expressed relative to the entity pose used by
        the predicate.
    """
    scene = env.scene
    entity_name = entity_cfg.name
    assert (
        entity_name in scene.rigid_objects or entity_name in scene.extras
    ), f"Scene entity '{entity_name}' must be a rigid object or read-only reference."
    is_rigid_object = entity_name in scene.rigid_objects
    resolved_geometry_prim_path = getattr(scene.cfg, entity_name).prim_path.format(ENV_REGEX_NS=scene.env_regex_ns)

    lower_by_environment = torch.empty((env.num_envs, 3), dtype=torch.float32, device=env.device)
    upper_by_environment = torch.empty_like(lower_by_environment)
    coverage_count = [0] * env.num_envs

    for representative_prim, environment_ids in _get_spawned_entity_groups(
        env,
        entity_name,
        resolved_geometry_prim_path,
    ):
        pose_prim = _find_rigid_body_prim(representative_prim, entity_name) if is_rigid_object else representative_prim
        local_aabb = _compute_aabb_relative_to_prim(pose_prim).to(env.device)
        environment_indices = torch.tensor(environment_ids, dtype=torch.long, device=env.device)
        lower_by_environment[environment_indices] = local_aabb.min_point[0]
        upper_by_environment[environment_indices] = local_aabb.max_point[0]
        for environment_id in environment_ids:
            coverage_count[environment_id] += 1

    assert all(count == 1 for count in coverage_count), (
        f"Scene entity '{entity_name}' geometry must cover every environment exactly once; got coverage"
        f" {coverage_count}."
    )
    return AxisAlignedBoundingBox(min_point=lower_by_environment, max_point=upper_by_environment)


class SceneReferencePoseReader:
    """Read current poses for one named scene reference."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        reference_name: str,
    ):
        scene = env.scene
        assert reference_name in scene.extras, f"Scene entity '{reference_name}' must be a read-only reference."

        self._reference_name = reference_name
        self._num_envs = env.num_envs
        reference_prim_path = getattr(scene.cfg, reference_name).prim_path.format(ENV_REGEX_NS=scene.env_regex_ns)
        self._frame_view = FrameView(
            reference_prim_path,
            device=env.device,
            stage=scene.stage,
            validate_xform_ops=False,
        )
        # Ensure pose rows use the same environment order as other scene tensors.
        reference_prim_paths = self._frame_view.prim_paths
        assert len(reference_prim_paths) == env.num_envs, (
            f"Read-only scene reference '{reference_name}' resolved to {len(reference_prim_paths)} prims; "
            f"expected {env.num_envs}."
        )
        for environment_id, prim_path in enumerate(reference_prim_paths):
            environment_prim_path = scene.env_prim_paths[environment_id]
            assert str(prim_path).startswith(f"{environment_prim_path}/"), (
                f"Read-only scene reference '{reference_name}' pose row {environment_id} belongs to '{prim_path}', "
                f"not environment '{environment_prim_path}'."
            )

    def get_pose_w(self) -> torch.Tensor:
        """Return current poses as ``(x, y, z, qx, qy, qz, qw)``."""
        position_w_buffer, orientation_w_buffer = self._frame_view.get_world_poses()
        position_w = position_w_buffer.torch
        orientation_w = orientation_w_buffer.torch
        assert position_w.shape == (self._num_envs, 3), (
            f"Read-only scene reference '{self._reference_name}' returned position shape {tuple(position_w.shape)}; "
            f"expected ({self._num_envs}, 3)."
        )
        assert orientation_w.shape == (self._num_envs, 4), (
            f"Read-only scene reference '{self._reference_name}' returned orientation shape "
            f"{tuple(orientation_w.shape)}; expected ({self._num_envs}, 4)."
        )
        return torch.cat((position_w, orientation_w), dim=-1)
