# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read current scene poses and build geometry used by spatial predicates."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.cloner.cloner_utils import iter_clone_plan_matches
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.views import FrameView
from pxr import Usd, UsdPhysics

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.usd_helpers import compute_bounding_box_relative_to_prim


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


def build_spawned_entity_local_aabbs(
    env: ManagerBasedEnv,
    pose_entity_cfg: SceneEntityCfg,
    geometry_prim_path: str | None = None,
) -> AxisAlignedBoundingBox:
    """Build one local AABB per environment from the geometry that was spawned.

    This reads the USD stage. Callers should cache the result and combine it
    with current poses rather than rebuilding it on each simulation step.

    Args:
        env: The runtime manager-based environment.
        pose_entity_cfg: Scene entity and optional articulation body whose pose defines the returned coordinates.
        geometry_prim_path: Prim path whose spawned geometry should be bounded. Defaults to the entity's
            configured prim path. An articulation body requires its explicit prim path.

    Returns:
        One AABB per environment. Its coordinates are relative to the origin and
        axes of the pose returned by :meth:`LiveSceneEntityGeometry.get_pose_w`.
    """
    scene = env.scene
    entity_name = pose_entity_cfg.name
    assert (
        entity_name in scene.rigid_objects or entity_name in scene.articulations or entity_name in scene.extras
    ), f"Scene entity '{entity_name}' must be a rigid object, articulation, or read-only reference."
    is_rigid_object = entity_name in scene.rigid_objects
    is_articulation = entity_name in scene.articulations
    if is_articulation:
        assert (
            geometry_prim_path is not None
        ), f"Articulation '{entity_name}' requires the selected body's explicit geometry prim path."
    resolved_geometry_prim_path = (geometry_prim_path or getattr(scene.cfg, entity_name).prim_path).format(
        ENV_REGEX_NS=scene.env_regex_ns
    )

    lower_by_environment = torch.empty((env.num_envs, 3), dtype=torch.float32, device=env.device)
    upper_by_environment = torch.empty_like(lower_by_environment)
    coverage_count = [0] * env.num_envs

    for representative_prim, environment_ids in _get_spawned_entity_groups(
        env,
        entity_name,
        resolved_geometry_prim_path,
    ):
        pose_prim = _find_rigid_body_prim(representative_prim, entity_name) if is_rigid_object else representative_prim
        if is_articulation:
            assert pose_prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ), f"Articulation '{entity_name}' geometry path '{resolved_geometry_prim_path}' must identify a rigid body."
        local_aabb = compute_bounding_box_relative_to_prim(pose_prim).to(env.device)
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


def _get_reference_pose_row_order(
    env: ManagerBasedEnv,
    entity_name: str,
    reference_prim_path: str,
    reference_frame_view,
) -> torch.Tensor:
    """Return FrameView row indices that put reference poses in environment order.

    FrameView returns poses in backend-defined row order. Predicates require row
    ``i`` of every input to describe the same environment.
    """
    scene = env.scene
    prim_paths = getattr(reference_frame_view, "prim_paths", None)
    if prim_paths is not None:
        environment_id_by_path = {
            environment_prim_path: environment_id
            for environment_id, environment_prim_path in enumerate(scene.env_prim_paths)
        }
        environment_path_depth = len(scene.env_prim_paths[0].split("/"))
        environment_ids_by_pose_row = []
        for prim_path in prim_paths:
            environment_prim_path = "/".join(str(prim_path).split("/")[:environment_path_depth])
            assert (
                environment_prim_path in environment_id_by_path
            ), f"Read-only scene reference '{entity_name}' pose path '{prim_path}' is not inside a scene environment."
            environment_ids_by_pose_row.append(environment_id_by_path[environment_prim_path])
    else:
        assert scene.clone_plan is not None, f"Read-only scene reference '{entity_name}' has no clone plan."
        clone_matches = tuple(iter_clone_plan_matches(scene.clone_plan, reference_prim_path))
        assert clone_matches, (
            f"Read-only scene reference '{entity_name}' path '{reference_prim_path}' does not match the "
            "scene clone plan."
        )
        environment_ids_by_pose_row = [
            environment_id
            for _source_root, _destination_template, _representative_path, environment_ids in clone_matches
            for environment_id in environment_ids
        ]

    assert len(environment_ids_by_pose_row) == reference_frame_view.count == env.num_envs, (
        f"Read-only scene reference '{entity_name}' resolved to {reference_frame_view.count} poses for "
        f"{len(environment_ids_by_pose_row)} environment rows; expected {env.num_envs}."
    )
    pose_row_by_environment = [-1] * env.num_envs
    for pose_row, environment_id in enumerate(environment_ids_by_pose_row):
        assert (
            pose_row_by_environment[environment_id] == -1
        ), f"Read-only scene reference '{entity_name}' has more than one pose for environment {environment_id}."
        pose_row_by_environment[environment_id] = pose_row
    assert all(
        pose_row >= 0 for pose_row in pose_row_by_environment
    ), f"Read-only scene reference '{entity_name}' does not provide exactly one pose per environment."
    return torch.tensor(pose_row_by_environment, dtype=torch.long, device=env.device)


class LiveSceneEntityGeometry:
    """Cache spawned local AABBs and read current poses for one scene entity."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        pose_entity_cfg: SceneEntityCfg,
        geometry_prim_path: str | None = None,
    ):
        scene = env.scene
        entity_name = pose_entity_cfg.name
        assert (
            entity_name in scene.rigid_objects or entity_name in scene.articulations or entity_name in scene.extras
        ), f"Scene entity '{entity_name}' must be a rigid object, articulation, or read-only reference."

        self._entity_name = entity_name
        self._num_envs = env.num_envs
        self.local_aabbs = build_spawned_entity_local_aabbs(
            env,
            pose_entity_cfg,
            geometry_prim_path,
        )
        """One cached local AABB for each environment."""
        self._rigid_object: RigidObject | None = None
        self._articulation: Articulation | None = None
        self._articulation_body_id: int | None = None
        self._reference_frame_view = None

        if entity_name in scene.rigid_objects:
            self._rigid_object = scene[entity_name]
        elif entity_name in scene.articulations:
            self._articulation = scene[entity_name]
            selected_body_ids = (
                list(range(self._articulation.num_bodies))[pose_entity_cfg.body_ids]
                if isinstance(pose_entity_cfg.body_ids, slice)
                else list(pose_entity_cfg.body_ids)
            )
            assert len(selected_body_ids) == 1, (
                f"Articulation '{entity_name}' must select exactly one body for runtime geometry; "
                f"got body ids {selected_body_ids}."
            )
            self._articulation_body_id = selected_body_ids[0]
        else:
            reference_prim_path = (geometry_prim_path or getattr(scene.cfg, entity_name).prim_path).format(
                ENV_REGEX_NS=scene.env_regex_ns
            )
            self._reference_frame_view = FrameView(
                reference_prim_path,
                device=env.device,
                stage=scene.stage,
                validate_xform_ops=False,
            )
            self._reference_pose_row_order = _get_reference_pose_row_order(
                env,
                entity_name,
                reference_prim_path,
                self._reference_frame_view,
            )

    def get_pose_w(self) -> torch.Tensor:
        """Return current poses as ``(x, y, z, qx, qy, qz, qw)``."""
        if self._rigid_object is not None:
            pose_w = self._rigid_object.data.root_pose_w.torch
        elif self._articulation is not None:
            pose_w = self._articulation.data.body_pose_w.torch[:, self._articulation_body_id, :]
        else:
            position_w_buffer, orientation_w_buffer = self._reference_frame_view.get_world_poses()
            position_w = position_w_buffer.torch
            orientation_w = orientation_w_buffer.torch
            assert position_w.shape == (self._num_envs, 3), (
                f"Read-only scene reference '{self._entity_name}' returned position shape {tuple(position_w.shape)}; "
                f"expected ({self._num_envs}, 3)."
            )
            assert orientation_w.shape == (self._num_envs, 4), (
                f"Read-only scene reference '{self._entity_name}' returned orientation shape "
                f"{tuple(orientation_w.shape)}; expected ({self._num_envs}, 4)."
            )
            pose_w = torch.cat((position_w, orientation_w), dim=-1).index_select(
                0,
                self._reference_pose_row_order,
            )

        assert pose_w.shape == (self._num_envs, 7), (
            f"Scene entity '{self._entity_name}' returned pose shape {tuple(pose_w.shape)}; "
            f"expected ({self._num_envs}, 7)."
        )
        return pose_w
