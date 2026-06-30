# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import torch
import trimesh
import trimesh.sample
import trimesh.util
from dataclasses import dataclass
from typing import Any

import isaacsim.core.utils.prims as prim_utils
from isaaclab.sim.utils import get_all_matching_child_prims
from pxr import UsdGeom, UsdPhysics

from isaaclab_arena_datagen.object_registry import InstanceKey, ObjectInstanceRegistry, ObjectType
from isaaclab_arena_datagen.utils.constants import (
    DEFAULT_ROTATION_EPS_RAD,
    DEFAULT_TRANSLATION_EPS_M,
    MAX_MESH_SAMPLE_POINTS,
)
from isaaclab_arena_datagen.utils.isaac_data import to_torch
from isaaclab_arena_datagen.utils.mesh_utils import create_primitive_mesh, triangulate_usd_faces
from isaaclab_arena_datagen.utils.transform_utils import compute_se3_origin_from_surface, se3_from_pos_quat


@dataclass
class MeshSamplesResult:
    """Result bundle from :meth:`DynamicObjectTracker.sample_dynamic_object_meshes`.

    Contains per-object/link relative SE(3) arrays encoding sampled surface
    points and outward normals.  Keys match
    :attr:`DynamicObjectResult.T_W_from_localbody_arrays`.
    """

    se3_localbody_from_point_arrays: dict[str, np.ndarray]
    """Per-object (or per-body-link) arrays of shape ``(N, 3, 4)`` float32.
    Each row is the 3x4 portion of the relative SE(3) from the object/link
    centre to the sampled point.  The z-column of the rotation encodes the
    outward surface normal."""


@dataclass
class DynamicObjectResult:
    """Result from :meth:`DynamicObjectTracker.filter_and_collect_moving_object_poses`.

    Contains JSON-serialisable metadata and per-object NumPy pose arrays.
    """

    metadata: dict[str, Any]
    """Top-level metadata dict (num_steps, thresholds, conventions)."""
    objects_metadata: dict[str, dict[str, Any]]
    """Per-object metadata keyed by display name (type, asset_name, body info).
    Does **not** contain the pose data itself -- that lives in *T_W_from_localbody_arrays*."""
    T_W_from_localbody_arrays: dict[str, np.ndarray]
    """Per-object (or per-body-link) pose arrays, keyed by a sanitised name
    suitable for use as an ``.npz`` entry.  Each array has shape
    ``(num_steps, 3, 4)`` float32."""


class DynamicObjectTracker:
    """Tracks dynamic (moving) rigid and articulated objects across cameras and time steps.

    Collects which RIGID / ARTICULATION objects are visible in any camera at
    any step, records their world-frame SE(3) poses every step using
    pre-allocated contiguous tensors, and after the sequence filters down to
    objects that actually moved.

    Args:
        registry: The shared :class:`ObjectInstanceRegistry` used by camera
            handlers so that object names are consistent.
        num_steps: Total number of simulation steps (used to pre-allocate
            pose tensors on first encounter).
    """

    def __init__(self, registry: ObjectInstanceRegistry, num_steps: int) -> None:
        """Initialize the tracker with a shared registry and step count."""
        self._registry = registry
        self._num_steps = num_steps
        self._seen_assets: set[InstanceKey] = set()
        self._seen_articulation_body_indices: dict[str, set[int]] = {}

        self._rigid_T_W_from_localbody: dict[str, torch.Tensor] = {}
        self._articulation_T_W_from_localbody: dict[str, torch.Tensor] = {}
        self._articulation_body_names: dict[str, list[str]] = {}

    def trim(self, num_valid_steps: int) -> None:
        """Shrink all per-step pose buffers to *num_valid_steps* actual steps.

        Used when the tracker was pre-allocated to an upper bound (e.g. the
        environment's max episode length) but the episode finished earlier, so
        the trailing pre-initialised identity poses must be dropped before
        filtering.
        """
        assert (
            0 < num_valid_steps <= self._num_steps
        ), f"num_valid_steps={num_valid_steps} out of range (1, {self._num_steps}]"
        for name, buf in self._rigid_T_W_from_localbody.items():
            self._rigid_T_W_from_localbody[name] = buf[:num_valid_steps]
        for name, buf in self._articulation_T_W_from_localbody.items():
            self._articulation_T_W_from_localbody[name] = buf[:num_valid_steps]
        self._num_steps = num_valid_steps

    def register_visible_objects(self, semantic_info: list[dict[str, Any]]) -> None:
        """Record which dynamic objects are visible in the current frame.

        For articulated objects, tracks which individual body links had
        visible pixels so that only those links are considered for dynamic
        classification.

        Call once per camera per step, passing the ``semantic_info`` list
        returned by :meth:`IsaacLabArenaCameraHandler.get_object_instance_segmentation`.
        """
        for obj in semantic_info:
            track_type = obj.get("track_type", "")
            if track_type == ObjectType.RIGID.name:
                self._seen_assets.add(InstanceKey(ObjectType.RIGID, obj["asset_name"]))
            elif track_type == ObjectType.ARTICULATION.name:
                asset_name = obj["asset_name"]
                self._seen_assets.add(InstanceKey(ObjectType.ARTICULATION, asset_name))
                visible_bodies = obj.get("visible_body_indices", [])
                if visible_bodies:
                    if asset_name not in self._seen_articulation_body_indices:
                        self._seen_articulation_body_indices[asset_name] = set()
                    self._seen_articulation_body_indices[asset_name].update(visible_bodies)

    def record_step_poses(self, environment: Any, step_idx: int) -> None:
        """Record world-frame poses of all rigid objects and articulations.

        On first encounter each object gets a pre-allocated pose tensor
        (rigid) or multi-body pose tensor (articulation).  Subsequent calls
        fill in ``step_idx`` without allocating new memory.
        """
        scene = environment.unwrapped.scene
        self._record_rigid_poses(scene, step_idx)
        self._record_articulation_poses(scene, step_idx)

    def _record_rigid_poses(self, scene: Any, step_idx: int) -> None:
        """Record world-frame poses for all rigid objects at *step_idx*.

        On first encounter each rigid object gets a pre-allocated pose
        tensor.  Subsequent calls fill in the slice at ``step_idx``
        without allocating new memory.
        """
        for name in scene.rigid_objects.keys():
            obj = scene[name]
            T_W_from_localbody = se3_from_pos_quat(
                to_torch(obj.data.root_link_pos_w)[0],
                to_torch(obj.data.root_link_quat_w)[0],
            )
            if name not in self._rigid_T_W_from_localbody:
                pose_buffer = torch.zeros(self._num_steps, 4, 4, dtype=torch.float32)
                for s in range(self._num_steps):
                    pose_buffer[s] = torch.eye(4)
                self._rigid_T_W_from_localbody[name] = pose_buffer
            self._rigid_T_W_from_localbody[name][step_idx] = T_W_from_localbody.to_matrix().squeeze(0).cpu()

    def _record_articulation_poses(self, scene: Any, step_idx: int) -> None:
        """Record world-frame poses for all articulation body links at *step_idx*.

        On first encounter each articulation gets a pre-allocated multi-body
        pose tensor.  Subsequent calls fill in the slice at ``step_idx``
        without allocating new memory.
        """
        for name, articulation in scene.articulations.items():
            num_bodies = articulation.data.body_link_pos_w.shape[1]
            body_names = articulation.data.body_names

            if name not in self._articulation_T_W_from_localbody:
                pose_buffer = torch.zeros(self._num_steps, num_bodies, 4, 4, dtype=torch.float32)
                for s in range(self._num_steps):
                    for body_index in range(num_bodies):
                        pose_buffer[s, body_index] = torch.eye(4)
                self._articulation_T_W_from_localbody[name] = pose_buffer
                self._articulation_body_names[name] = [
                    body_names[body_index] if body_index < len(body_names) else f"body_{body_index}"
                    for body_index in range(num_bodies)
                ]

            for body_index in range(num_bodies):
                T_W_from_localbody = se3_from_pos_quat(
                    to_torch(articulation.data.body_link_pos_w)[0, body_index],
                    to_torch(articulation.data.body_link_quat_w)[0, body_index],
                )
                self._articulation_T_W_from_localbody[name][step_idx, body_index] = (
                    T_W_from_localbody.to_matrix().squeeze(0).cpu()
                )

    def filter_and_collect_moving_object_poses(
        self,
        translation_eps_m: float = DEFAULT_TRANSLATION_EPS_M,
        rotation_eps_rad: float = DEFAULT_ROTATION_EPS_RAD,
    ) -> DynamicObjectResult:
        """Filter and collect pose data for objects that were seen and actually moved.

        An object is classified as dynamic when its per-step translation
        exceeds *translation_eps_m* OR its per-step rotation exceeds
        *rotation_eps_rad*.  The two thresholds are independent so a
        rotation-only motion is still detected even when translation is
        zero.  Both defaults are sized to filter out physics-solver jitter
        on truly static bodies.

        For articulated objects, only links that individually exceed at
        least one threshold are included.

        Args:
            translation_eps_m: Minimum translation (metres) between adjacent
                frames to count as dynamic.
            rotation_eps_rad: Minimum rotation (radians) between adjacent
                frames to count as dynamic.

        Returns:
            A :class:`DynamicObjectResult` containing JSON metadata and
            NumPy pose arrays ready to be written with
            :meth:`IsaacLabSyntheticDatasetWriter.write_dynamic_object_poses`.
        """
        objects_meta: dict[str, dict[str, Any]] = {}
        T_W_from_localbody_arrays: dict[str, np.ndarray] = {}

        for instance_key in self._seen_assets:
            asset_name = instance_key.asset_name
            if instance_key.kind is ObjectType.RIGID:
                T_W_from_localbody_poses = self._rigid_T_W_from_localbody.get(asset_name)
                if T_W_from_localbody_poses is None:
                    continue
                if not self._has_motion_tensor(
                    T_W_from_localbody_poses,
                    translation_eps_m,
                    rotation_eps_rad=rotation_eps_rad,
                ):
                    continue

                display_name = self._registry.instance_key_to_display_name(instance_key)
                objects_meta[display_name] = {
                    "type": ObjectType.RIGID.label,
                    "asset_name": asset_name,
                    "pose_array_key": display_name,
                }
                T_W_from_localbody_arrays[display_name] = T_W_from_localbody_poses[:, :3, :].numpy()

            elif instance_key.kind is ObjectType.ARTICULATION:
                T_W_from_localbody_poses = self._articulation_T_W_from_localbody.get(asset_name)
                if T_W_from_localbody_poses is None:
                    continue
                body_names = self._articulation_body_names.get(asset_name, [])

                display_name = self._registry.instance_key_to_display_name(instance_key)

                visible_indices = self._seen_articulation_body_indices.get(asset_name)

                parts_meta: dict[str, dict[str, Any]] = {}
                for body_index in visible_indices if visible_indices else range(T_W_from_localbody_poses.shape[1]):
                    if not self._has_motion_tensor(
                        T_W_from_localbody_poses[:, body_index],
                        translation_eps_m,
                        rotation_eps_rad=rotation_eps_rad,
                    ):
                        continue
                    body_name = body_names[body_index] if body_index < len(body_names) else f"body_{body_index}"
                    array_key = f"{display_name}/{body_name}"
                    parts_meta[body_name] = {
                        "body_index": body_index,
                        "pose_array_key": array_key,
                    }
                    T_W_from_localbody_arrays[array_key] = T_W_from_localbody_poses[:, body_index, :3, :].numpy()

                if not parts_meta:
                    continue

                objects_meta[display_name] = {
                    "type": ObjectType.ARTICULATION.label,
                    "asset_name": asset_name,
                    "parts": parts_meta,
                }

        metadata = {
            "num_steps": self._num_steps,
            "translation_threshold": translation_eps_m,
            "rotation_threshold": rotation_eps_rad,
            "coordinate_frame": "world",
            "pose_format": "(num_steps, 3, 4) float32 row-major [R|t] per .npy array",
        }

        return DynamicObjectResult(
            metadata=metadata,
            objects_metadata=objects_meta,
            T_W_from_localbody_arrays=T_W_from_localbody_arrays,
        )

    # ---- internal helpers ------------------------------------------------

    @staticmethod
    def _has_motion_tensor(
        poses: torch.Tensor,
        eps_m: float = 0.0,
        rotation_eps_rad: float | None = None,
    ) -> bool:
        """Check adjacent-frame motion on a contiguous stack of SE(3) poses.

        Args:
            poses: Stack of SE(3) matrices.
            eps_m: Threshold for translation (metres).
            rotation_eps_rad: Threshold for rotation (radians).  Defaults to
                *eps_m* when ``None``.  Set to ``float('inf')`` to ignore
                rotation entirely.

        Returns:
            ``True`` if any adjacent frame pair exceeds the thresholds.
        """
        if poses.shape[0] < 2:
            return False
        if rotation_eps_rad is None:
            rotation_eps_rad = eps_m
        prev = poses[:-1]
        cur = poses[1:]

        t_diffs = (cur[:, :3, 3] - prev[:, :3, 3]).norm(dim=-1)
        if t_diffs.max().item() > eps_m:
            return True

        if rotation_eps_rad < float("inf"):
            R_rel = prev[:, :3, :3].transpose(-1, -2) @ cur[:, :3, :3]
            cos_angles = ((R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
            angles_rad = torch.acos(cos_angles)
            if angles_rad.max().item() > rotation_eps_rad:
                return True

        return False

    # ---- mesh surface sampling -------------------------------------------

    def sample_dynamic_object_meshes(
        self,
        environment: Any,
        dynamic_object_result: DynamicObjectResult,
        spacing_m: float = 0.01,
        translation_eps_m: float = DEFAULT_TRANSLATION_EPS_M,
        rotation_eps_rad: float = DEFAULT_ROTATION_EPS_RAD,
    ) -> MeshSamplesResult:
        """Sample surface points with normals on moving objects' meshes only.

        For each object/link in *dynamic_object_result* whose pose actually
        changes between at least two adjacent steps (above
        *translation_eps_m* in translation or *rotation_eps_rad* in rotation),
        resolves the USD mesh geometry, uniformly samples
        ``area / spacing_m**2`` points on the surface, and computes the
        relative SE(3) from the object/link centre (step-0 pose) to each
        point.  The z-column of each relative rotation encodes the outward
        surface normal.

        Objects and individual articulation links that do not exhibit motion
        above either threshold are skipped.  The thresholds must match those
        passed to :meth:`filter_and_collect_moving_object_poses` so the two
        passes agree on which bodies are dynamic.

        Must be called while the simulation is still running (USD stage
        accessible).

        Args:
            environment: The gymnasium-wrapped IsaacLab environment.
            dynamic_object_result: The :class:`DynamicObjectResult` from
                :meth:`filter_and_collect_moving_object_poses`.
            spacing_m: Target distance (metres) between sampled points.
                Point count is ``max(1, int(area / spacing_m**2))``.
            translation_eps_m: Minimum adjacent-frame translation (m) for a
                link to be considered moving.  Links below this threshold
                in both translation and rotation are skipped.
            rotation_eps_rad: Minimum adjacent-frame rotation (rad) for a
                link to be considered moving.  Links below this threshold
                in both translation and rotation are skipped.

        Returns:
            A :class:`MeshSamplesResult` with keys only for objects/links
            that actually moved.
        """
        scene = environment.unwrapped.scene
        se3_localbody_from_point: dict[str, np.ndarray] = {}

        for _, meta in dynamic_object_result.objects_metadata.items():
            if meta["type"] == ObjectType.RIGID.label:
                asset_name = meta["asset_name"]
                pose_key = meta["pose_array_key"]

                T_W_from_localbody_poses = self._rigid_T_W_from_localbody.get(asset_name)
                if T_W_from_localbody_poses is not None and not self._has_motion_tensor(
                    T_W_from_localbody_poses,
                    translation_eps_m,
                    rotation_eps_rad=rotation_eps_rad,
                ):
                    continue

                prim_path = scene[asset_name].cfg.prim_path.replace(".*", "0")
                combined = self._collect_mesh_from_prim(prim_path)
                if combined is None or combined.area < 1e-12:
                    continue

                surface_points_localbody_n3, normals_localbody_n3 = self._sample_on_mesh(combined, spacing_m)
                T_localbody_from_mesh = np.eye(4, dtype=np.float64)
                se3_localbody_from_point[pose_key] = compute_se3_origin_from_surface(
                    surface_points_localbody_n3,
                    normals_localbody_n3,
                    T_localbody_from_mesh,
                )

            elif meta["type"] == ObjectType.ARTICULATION.label:
                asset_name = meta["asset_name"]
                articulation_obj = scene[asset_name]
                all_link_paths = list(articulation_obj.root_physx_view.link_paths[0])

                for _, part_meta in meta["parts"].items():
                    body_index = part_meta["body_index"]
                    pose_key = part_meta["pose_array_key"]

                    T_W_from_localbody_poses = self._articulation_T_W_from_localbody.get(asset_name)
                    if T_W_from_localbody_poses is not None and not self._has_motion_tensor(
                        T_W_from_localbody_poses[:, body_index],
                        translation_eps_m,
                        rotation_eps_rad=rotation_eps_rad,
                    ):
                        continue

                    link_path = all_link_paths[body_index]
                    child_link_paths = [
                        all_link_paths[j]
                        for j in range(len(all_link_paths))
                        if j != body_index and all_link_paths[j].startswith(link_path + "/")
                    ]
                    combined = self._collect_mesh_from_prim(link_path, exclude_subtrees=child_link_paths or None)
                    if combined is None or combined.area < 1e-12:
                        continue

                    surface_points_localbody_n3, normals_localbody_n3 = self._sample_on_mesh(combined, spacing_m)
                    T_localbody_from_mesh = np.eye(4, dtype=np.float64)
                    se3_localbody_from_point[pose_key] = compute_se3_origin_from_surface(
                        surface_points_localbody_n3,
                        normals_localbody_n3,
                        T_localbody_from_mesh,
                    )

        return MeshSamplesResult(se3_localbody_from_point_arrays=se3_localbody_from_point)

    # ---- mesh helpers (private) ------------------------------------------

    @staticmethod
    def _usd_to_trimesh_matrix(m: Any) -> np.ndarray:
        """Convert a USD GfMatrix4d (row-vector) to a numpy matrix for trimesh.apply_transform."""
        return np.array([[m[r][c] for c in range(4)] for r in range(4)], dtype=np.float64).T

    @staticmethod
    def _rigid_part(matrix: np.ndarray) -> np.ndarray:
        """Return *matrix* with per-axis scale removed, keeping rotation and translation.

        The body pose used to reconstruct world points is a scale-free rigid transform, so
        a prim's scale must stay baked into the sampled vertices rather than be divided out.
        Assumes axis-aligned scale (no shear), which holds for USD xform scale ops.
        """
        rigid = matrix.copy()
        linear = rigid[:3, :3]
        scale = np.linalg.norm(linear, axis=0)
        scale[scale == 0] = 1.0
        rigid[:3, :3] = linear / scale
        return rigid

    @staticmethod
    def _collect_mesh_from_prim(
        prim_path: str,
        exclude_subtrees: list[str] | None = None,
    ) -> Any | None:
        """Walk *prim_path* and combine all child meshes / primitives into one trimesh.

        Vertices are returned in the **body-local frame** relative to
        *prim_path*, with the root prim's scale retained (only its rotation
        and translation are removed).  The runtime body pose applied during
        reconstruction is scale-free, so keeping the root scale baked in is
        what makes points reconstruct at metric world scale.  Physics-driven
        transforms (applied by PhysX at runtime) are NOT baked in, because
        ``UsdGeom.XformCache`` does not see them.

        Args:
            prim_path: USD prim path to collect meshes from.
            exclude_subtrees: Prim paths whose subtrees should be excluded
                from collection.  Used for articulations to prevent a
                parent link from collecting meshes of descendant links.
        """
        geometry_type_names = frozenset({"Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone"})

        def _is_under_excluded_subtree(prim_obj: Any) -> bool:
            if exclude_subtrees is None:
                return False
            prim_str = prim_obj.GetPath().pathString
            return any(prim_str == excl or prim_str.startswith(excl + "/") for excl in exclude_subtrees)

        prims = get_all_matching_child_prims(
            prim_path,
            predicate=lambda p: (
                p.GetTypeName() in geometry_type_names
                and not p.HasAPI(UsdPhysics.CollisionAPI)
                and not _is_under_excluded_subtree(p)
            ),
        )
        if not prims:
            prims = get_all_matching_child_prims(
                prim_path,
                predicate=lambda p: (p.GetTypeName() in geometry_type_names and not _is_under_excluded_subtree(p)),
            )
        if not prims:
            return None

        xform_cache = UsdGeom.XformCache()
        root_prim = prim_utils.get_prim_at_path(prim_path)
        T_W_from_root_trimesh = DynamicObjectTracker._usd_to_trimesh_matrix(
            xform_cache.GetLocalToWorldTransform(root_prim)
        )
        # Strip only the root's rotation/translation, not its scale: the runtime body pose
        # applied during reconstruction is scale-free, so the root's scale (e.g. a cm->m unit
        # factor) must remain baked into the vertices or the points reconstruct 1/scale too big.
        T_root_from_W_trimesh = np.linalg.inv(DynamicObjectTracker._rigid_part(T_W_from_root_trimesh))

        meshes: list = []

        for prim in prims:
            prim_type = prim.GetTypeName()
            if prim_type == "Mesh":
                mesh_geom = UsdGeom.Mesh(prim)
                verts_localbody_n3 = np.asarray(mesh_geom.GetPointsAttr().Get(), dtype=np.float32)
                faces_n3 = triangulate_usd_faces(prim)
                tm = trimesh.Trimesh(vertices=verts_localbody_n3, faces=faces_n3, process=False)
            else:
                tm = create_primitive_mesh(prim)

            T_W_from_prim_trimesh = DynamicObjectTracker._usd_to_trimesh_matrix(
                xform_cache.GetLocalToWorldTransform(prim)
            )
            T_root_from_prim_trimesh = T_root_from_W_trimesh @ T_W_from_prim_trimesh
            tm.apply_transform(T_root_from_prim_trimesh)
            meshes.append(tm)

        if not meshes:
            return None
        return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    @staticmethod
    def _sample_on_mesh(
        mesh: Any,
        spacing_m: float,
        oversample_factor: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample surface points and face normals at the given spacing.

        Oversamples randomly then applies **farthest point sampling** (FPS) to
        select a maximally-spaced subset, guaranteeing uniform coverage
        regardless of mesh tessellation quality.

        The raw point count is ``mesh.area / spacing_m ** 2`` but is capped at
        :data:`MAX_MESH_SAMPLE_POINTS` so pathologically large meshes (certain
        Objaverse-derived USDs report tens of m^2 of surface area) do not
        stall the O(n * k) FPS below.
        """
        num_points_raw = max(1, int(mesh.area / (spacing_m**2)))
        num_points = min(num_points_raw, MAX_MESH_SAMPLE_POINTS)
        if num_points_raw > MAX_MESH_SAMPLE_POINTS:
            print(
                f"[DynamicObjectTracker] mesh area {mesh.area:.2f} m^2 would "
                f"require {num_points_raw} samples at spacing {spacing_m} m; "
                f"capping at {MAX_MESH_SAMPLE_POINTS}.",
                flush=True,
            )
        num_candidates = num_points * oversample_factor
        candidate_points_localbody_n3, candidate_face_indices = trimesh.sample.sample_surface(mesh, num_candidates)

        if candidate_points_localbody_n3.shape[0] <= num_points:
            normals_localbody_n3 = mesh.face_normals[candidate_face_indices]
            return candidate_points_localbody_n3.astype(np.float32), normals_localbody_n3.astype(np.float32)

        selected_indices = DynamicObjectTracker._farthest_point_sampling(candidate_points_localbody_n3, num_points)
        np.random.shuffle(selected_indices)
        points_localbody_n3 = candidate_points_localbody_n3[selected_indices]
        normals_localbody_n3 = mesh.face_normals[candidate_face_indices[selected_indices]]
        return points_localbody_n3.astype(np.float32), normals_localbody_n3.astype(np.float32)

    @staticmethod
    def _farthest_point_sampling(points_n3: np.ndarray, num_samples: int) -> np.ndarray:
        """Select *num_samples* points from *points_n3* maximising the minimum pairwise distance."""
        n = points_n3.shape[0]
        num_samples = min(num_samples, n)
        selected = np.empty(num_samples, dtype=np.int64)
        selected[0] = np.random.randint(n)
        min_dists = np.full(n, np.inf, dtype=np.float64)
        for i in range(1, num_samples):
            diff_n3 = points_n3 - points_n3[selected[i - 1]]
            dists_n = np.einsum("ij,ij->i", diff_n3, diff_n3)
            np.minimum(min_dists, dists_n, out=min_dists)
            selected[i] = np.argmax(min_dists)
        return selected
