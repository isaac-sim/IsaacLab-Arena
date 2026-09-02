# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read live poses and derive geometry for ArenaWorld scene entities."""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.cloner.cloner_utils import iter_clone_plan_matches
from isaaclab.scene import InteractiveScene
from isaaclab.sim.views import FrameView
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def _get_spawned_entity_groups(
    scene: InteractiveScene,
    entity_name: str,
    geometry_prim_path: str,
) -> list[tuple[Usd.Prim, tuple[int, ...]]]:
    """Return one representative prim and its cloned environments per asset variant."""
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
    """Compute descendant geometry bounds expressed in the prim's local frame ``P``."""
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
        geometry_range_P = geometry_bounds.ComputeAlignedRange()
        if geometry_range_P.IsEmpty():
            continue

        lower_P = np.minimum(lower_P, np.asarray(geometry_range_P.GetMin(), dtype=np.float64))
        upper_P = np.maximum(upper_P, np.asarray(geometry_range_P.GetMax(), dtype=np.float64))
        found_geometry = True

    prim_path = prim.GetPath()
    assert found_geometry, f"Prim '{prim_path}' has no default-purpose geometry."
    return AxisAlignedBoundingBox(
        min_point=tuple(float(value) for value in lower_P),
        max_point=tuple(float(value) for value in upper_P),
    )


def compute_spawned_geometry_bounds_in_entity_frame(
    scene: InteractiveScene,
    entity_name: str,
) -> AxisAlignedBoundingBox:
    """Build spawned geometry bounds expressed in entity frame ``E``.

    Args:
        scene: Interactive scene containing the spawned entity.
        entity_name: Scene entity whose prim-local coordinate system defines frame ``E``.

    Returns:
        One batched AABB aligned with frame ``E`` and measured from its origin.
        The bounds do not change when the entity's live pose changes.
    """
    assert entity_name in scene.rigid_objects or entity_name in scene.extras, (
        "ArenaWorld geometry queries support only entities registered in InteractiveScene.rigid_objects or "
        f"InteractiveScene.extras; '{entity_name}' is registered in neither."
    )
    is_rigid_object = entity_name in scene.rigid_objects
    resolved_geometry_prim_path = getattr(scene.cfg, entity_name).prim_path.format(ENV_REGEX_NS=scene.env_regex_ns)

    minimum_points_E_by_environment = torch.empty((scene.num_envs, 3), dtype=torch.float32, device=scene.device)
    maximum_points_E_by_environment = torch.empty_like(minimum_points_E_by_environment)
    coverage_count = [0] * scene.num_envs

    for representative_prim, environment_ids in _get_spawned_entity_groups(
        scene,
        entity_name,
        resolved_geometry_prim_path,
    ):
        entity_frame_prim = (
            _find_single_rigid_body_prim_in_subtree(representative_prim, entity_name)
            if is_rigid_object
            else representative_prim
        )
        geometry_bounds_E = _compute_geometry_bounds_in_prim_frame(entity_frame_prim).to(scene.device)
        environment_indices = torch.tensor(environment_ids, dtype=torch.long, device=scene.device)
        minimum_points_E_by_environment[environment_indices] = geometry_bounds_E.min_point[0]
        maximum_points_E_by_environment[environment_indices] = geometry_bounds_E.max_point[0]
        for environment_id in environment_ids:
            coverage_count[environment_id] += 1

    assert all(count == 1 for count in coverage_count), (
        f"Scene entity '{entity_name}' geometry must cover every environment exactly once; got coverage"
        f" {coverage_count}."
    )
    return AxisAlignedBoundingBox(
        min_point=minimum_points_E_by_environment,
        max_point=maximum_points_E_by_environment,
    )


class SceneExtraPoseReader:
    """Read current ``T_W_E`` poses for an ``InteractiveScene.extras`` entry defining frame ``E``."""

    def __init__(self, scene: InteractiveScene, entity_name: str):
        assert (
            entity_name in scene.extras
        ), f"Scene entity '{entity_name}' must be registered in InteractiveScene.extras."

        self._entity_name = entity_name
        self._num_envs = scene.num_envs
        entity_prim_path = getattr(scene.cfg, entity_name).prim_path.format(ENV_REGEX_NS=scene.env_regex_ns)
        self._frame_view = FrameView(
            entity_prim_path,
            device=scene.device,
            stage=scene.stage,
            validate_xform_ops=False,
        )
        # InteractiveScene creates extras before cloning. This post-clone view must cover every environment.
        entity_prim_paths = self._frame_view.prim_paths
        assert (
            len(entity_prim_paths) == scene.num_envs
        ), f"Scene extra '{entity_name}' resolved to {len(entity_prim_paths)} prims; expected {scene.num_envs}."
        for environment_id, prim_path in enumerate(entity_prim_paths):
            environment_prim_path = scene.env_prim_paths[environment_id]
            assert str(prim_path).startswith(f"{environment_prim_path}/"), (
                f"Scene extra '{entity_name}' pose row {environment_id} belongs to '{prim_path}', "
                f"not environment '{environment_prim_path}'."
            )

    def get_pose_w(self) -> torch.Tensor:
        """Return ``T_W_E`` as ``(x, y, z, qx, qy, qz, qw)``."""
        position_w_buffer, orientation_w_buffer = self._frame_view.get_world_poses()
        t_W_E = position_w_buffer.torch
        q_W_E = orientation_w_buffer.torch
        T_W_E = torch.cat((t_W_E, q_W_E), dim=-1)
        assert T_W_E.shape == (self._num_envs, 7), (
            f"Scene extra '{self._entity_name}' returned pose shape {tuple(T_W_E.shape)}; "
            f"expected ({self._num_envs}, 7)."
        )
        return T_W_E
