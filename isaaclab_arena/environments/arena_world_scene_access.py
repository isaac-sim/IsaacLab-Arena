# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read live scene-extra poses and derive local geometry bounds for ArenaWorld."""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.scene import InteractiveScene
from isaaclab.sim.views import FrameView
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def _get_clone_source_environment_ids(clone_plan: cloner.ClonePlan, prim_path: str) -> dict[str, tuple[int, ...]]:
    """Return representative clone source paths mapped to their environment IDs."""
    asset_templates, prototype_starts, world_ids, world_id_starts = cloner.path.get_world_prototype_asset_templates(
        clone_plan, include_world_indices=True
    )
    source_paths = cloner.path.get_asset_prototype_paths(clone_plan)
    topology = clone_plan.topology
    candidates_by_environment: dict[int, tuple[int, str]] = {}
    for world_group, (prototype_start, prototype_end) in enumerate(
        zip(prototype_starts[:-1], prototype_starts[1:], strict=True)
    ):
        environment_ids = world_ids[world_id_starts[world_group] : world_id_starts[world_group + 1]]
        for occurrence_index in range(prototype_start, prototype_end):
            asset_template = asset_templates[occurrence_index]
            if asset_template.count("{}") != 1:
                continue
            template_match = cloner.path.match(prim_path, asset_template)
            asset_prototype_id = topology.world_prototypes[occurrence_index]
            source_root = source_paths[asset_prototype_id]
            if template_match is None or source_root is None:
                continue
            source_path = f"{source_root.rstrip('/')}{template_match.suffix}"
            specificity = len(asset_template)
            for environment_id in environment_ids:
                environment_id = int(environment_id)
                previous = candidates_by_environment.get(environment_id)
                if previous is None or specificity > previous[0]:
                    candidates_by_environment[environment_id] = (specificity, source_path)

    environment_ids_by_source_path: dict[str, list[int]] = {}
    for environment_id, (_specificity, source_path) in candidates_by_environment.items():
        environment_ids_by_source_path.setdefault(source_path, []).append(environment_id)
    return {
        source_path: tuple(sorted(environment_ids))
        for source_path, environment_ids in environment_ids_by_source_path.items()
    }


def get_representative_geometry_prim_groups(
    scene: InteractiveScene,
    scene_key: str,
) -> list[tuple[Usd.Prim, tuple[int, ...]]]:
    """Return one geometry root and its environment IDs per spawned asset variant.

    Args:
        scene: Scene containing the cloned assets.
        scene_key: Asset key in the scene configuration.
    """
    geometry_prim_path = getattr(scene.cfg, scene_key).prim_path.format(ENV_REGEX_NS=scene.env_regex_ns)
    assert scene.clone_plan is not None, f"Cannot resolve geometry for scene key '{scene_key}' without a clone plan."

    environment_ids_by_source_path = _get_clone_source_environment_ids(scene.clone_plan, geometry_prim_path)
    resolved_environment_count = sum(
        len(environment_ids) for environment_ids in environment_ids_by_source_path.values()
    )
    assert resolved_environment_count == scene.num_envs, (
        f"Geometry path '{geometry_prim_path}' for scene key '{scene_key}' must resolve one populated clone source "
        f"per environment; resolved {resolved_environment_count} of {scene.num_envs}."
    )

    representative_prim_groups = []
    for representative_path, environment_ids in environment_ids_by_source_path.items():
        representative_prim = scene.stage.GetPrimAtPath(representative_path)
        assert (
            representative_prim.IsValid()
        ), f"Clone source '{representative_path}' for scene key '{scene_key}' is not a valid USD prim."
        representative_prim_groups.append((representative_prim, environment_ids))
    return representative_prim_groups


def _find_single_rigid_body_prim_in_subtree(root_prim: Usd.Prim, rigid_object_name: str) -> Usd.Prim:
    """Return the only rigid-body prim in the root prim's subtree."""
    rigid_body_prims_in_subtree = sim_utils.get_all_matching_child_prims(
        root_prim.GetPath(),
        predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
        stage=root_prim.GetStage(),
        traverse_instance_prims=False,
    )
    assert len(rigid_body_prims_in_subtree) == 1, (
        f"Rigid object '{rigid_object_name}' source '{root_prim.GetPath()}' contains "
        f"{len(rigid_body_prims_in_subtree)} rigid bodies; expected exactly one."
    )
    return rigid_body_prims_in_subtree[0]


def get_representative_rigid_body_prims(scene: InteractiveScene, scene_key: str) -> list[Usd.Prim]:
    """Return the single rigid body of each spawned variant of a rigid object.

    Args:
        scene: Scene containing the cloned rigid objects.
        scene_key: Key in scene.rigid_objects.
    """
    assert scene_key in scene.rigid_objects, f"Scene key {scene_key!r} is not a rigid object"
    return [
        _find_single_rigid_body_prim_in_subtree(prim, scene_key)
        for prim, _ in get_representative_geometry_prim_groups(scene, scene_key)
    ]


def _compute_geometry_bounds_in_prim_frame(prim: Usd.Prim) -> AxisAlignedBoundingBox:
    """Compute descendant geometry bounds expressed in the prim's local frame P."""
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


def compute_spawned_geometry_bounds_in_local_frame(
    scene: InteractiveScene,
    scene_key: str,
) -> AxisAlignedBoundingBox:
    """Build local-frame bounds from default-purpose geometry in the cloned prim hierarchy.

    Args:
        scene: Interactive scene containing the rigid object or scene extra.
        scene_key: Key whose corresponding local coordinate system defines frame F.

    Returns:
        One batched AABB aligned with frame F and measured from its origin.
        Physics properties do not affect which geometry is included. The bounds remain
        valid under whole-subtree motion, but not when descendants move relative
        to frame F.
    """
    assert scene_key in scene.rigid_objects or scene_key in scene.deformable_objects or scene_key in scene.extras, (
        "ArenaWorld geometry queries require a scene key registered in InteractiveScene.rigid_objects, "
        f"InteractiveScene.deformable_objects, or InteractiveScene.extras; '{scene_key}' is registered in none."
    )
    is_rigid_object = scene_key in scene.rigid_objects

    minimum_points_F_by_environment = torch.empty((scene.num_envs, 3), dtype=torch.float32, device=scene.device)
    maximum_points_F_by_environment = torch.empty_like(minimum_points_F_by_environment)
    coverage_count = [0] * scene.num_envs

    for representative_prim, environment_ids in get_representative_geometry_prim_groups(scene, scene_key):
        local_frame_prim = (
            _find_single_rigid_body_prim_in_subtree(representative_prim, scene_key)
            if is_rigid_object
            else representative_prim
        )
        geometry_bounds_F = _compute_geometry_bounds_in_prim_frame(local_frame_prim).to(scene.device)
        environment_indices = torch.tensor(environment_ids, dtype=torch.long, device=scene.device)
        minimum_points_F_by_environment[environment_indices] = geometry_bounds_F.min_point[0]
        maximum_points_F_by_environment[environment_indices] = geometry_bounds_F.max_point[0]
        for environment_id in environment_ids:
            coverage_count[environment_id] += 1

    assert all(
        count == 1 for count in coverage_count
    ), f"Geometry for scene key '{scene_key}' must cover every environment exactly once; got coverage {coverage_count}."
    return AxisAlignedBoundingBox(
        min_point=minimum_points_F_by_environment,
        max_point=maximum_points_F_by_environment,
    )


class SceneExtraPoseReader:
    """Read current T_W_F poses for an InteractiveScene.extras entry defining frame F."""

    def __init__(self, scene: InteractiveScene, scene_extra_key: str):
        assert (
            scene_extra_key in scene.extras
        ), f"Scene key '{scene_extra_key}' must be registered in InteractiveScene.extras."

        self._scene_extra_key = scene_extra_key
        self._num_envs = scene.num_envs
        scene_extra_prim_path = getattr(scene.cfg, scene_extra_key).prim_path.format(ENV_REGEX_NS=scene.env_regex_ns)
        prims = sim_utils.find_matching_prims(scene_extra_prim_path, stage=scene.stage)
        # Reauthoring an already valid collider frame can invalidate live physics views.
        needs_standardization = False
        for prim in prims:
            operations = [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
            if operations != ["xformOp:translate", "xformOp:orient", "xformOp:scale"]:
                needs_standardization = True
                break
        self._frame_view = FrameView(
            scene_extra_prim_path,
            device=scene.device,
            stage=scene.stage,
            validate_xform_ops=needs_standardization,
        )
        # InteractiveScene creates extras before cloning. This post-clone view must cover every environment.
        assert (
            self._frame_view.count == scene.num_envs
        ), f"Scene extra '{scene_extra_key}' resolved to {self._frame_view.count} frames; expected {scene.num_envs}."

        # Newton exposes no USD prim handles, so this check only verifies USD-backed
        # views. Newton row ordering is supplied by FrameView.
        for environment_id, prim in enumerate(self._frame_view.prims):
            prim_path = str(prim.GetPath())
            environment_prim_path = scene.env_prim_paths[environment_id]
            assert prim_path.startswith(f"{environment_prim_path}/"), (
                f"Scene extra '{scene_extra_key}' pose row {environment_id} belongs to '{prim_path}', "
                f"not environment '{environment_prim_path}'."
            )

    def get_pose_w(self) -> torch.Tensor:
        """Return T_W_F with shape (num_envs, 7), ordered as (x, y, z, qx, qy, qz, qw)."""
        position_w_buffer, orientation_w_buffer = self._frame_view.get_world_poses()
        t_W_F = position_w_buffer.torch
        q_W_F = orientation_w_buffer.torch
        T_W_F = torch.cat((t_W_F, q_W_F), dim=-1)
        assert T_W_F.shape == (self._num_envs, 7), (
            f"Scene extra '{self._scene_extra_key}' returned pose shape {tuple(T_W_F.shape)}; "
            f"expected ({self._num_envs}, 7)."
        )
        return T_W_F
