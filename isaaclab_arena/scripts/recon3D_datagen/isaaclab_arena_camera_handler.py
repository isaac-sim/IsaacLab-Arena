# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import colorsys
import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_from_matrix

ALL_DATA_TYPES = [
    "rgb",
    "distance_to_image_plane",
    "normals",
    "motion_vectors",
    "semantic_segmentation",
    "instance_id_segmentation_fast",
]


class TrackType(enum.IntEnum):
    """Tracking category for each pixel in the scene-flow output."""

    STATIC = 0
    RIGID = 1
    ARTICULATION = 2
    UNSUPPORTED = 255


@dataclass
class SceneFlowResult:
    """Bundle returned by the exact adjacent-frame scene-flow computation."""

    scene_flow_3d: torch.Tensor
    """(H, W, 3) float32 — world-space displacement from frame t to t+1.
    Use :meth:`~IsaacLabArenaCameraHandler.world_to_camera_scene_flow` to
    obtain camera-relative flow that also captures camera ego-motion."""
    scene_flow_valid_mask: torch.Tensor
    """(H, W) bool — True where the flow value is trustworthy ground truth."""
    scene_flow_track_type: torch.Tensor
    """(H, W) uint8 — per-pixel :class:`TrackType` enum value."""


@dataclass
class FirstFrameFlowResult:
    """Bundle returned by :meth:`IsaacLabArenaCameraHandler.compute_anchor_frame_flow`.

    All tensors are defined over the frame-0 pixel grid ``(H, W)``.
    """

    flow3d_from_first: torch.Tensor
    """(H, W, 3) float32 — world-space displacement from anchor to
    frame-k reconstructed position: ``p_k - p_anchor``.
    Use :meth:`~IsaacLabArenaCameraHandler.world_to_camera_anchor_flow`
    to obtain camera-relative flow that also captures camera ego-motion."""
    trackable_mask: torch.Tensor
    """(H, W) bool — True for frame-0 pixels that are STATIC, RIGID, or
    ARTICULATION (constant across the sequence)."""
    in_frame_mask: torch.Tensor
    """(H, W) bool — True when the projected ``p_k`` falls inside the current
    image bounds **and** has ``z_cam > 0``."""
    visible_now_mask: torch.Tensor
    """(H, W) bool — True when the point is in-frame **and** depth-consistent
    with the current depth map (not occluded)."""
    points_world_k: torch.Tensor
    """(H, W, 3) float32 — reconstructed world positions ``p_k`` for every
    trackable anchor (debug / trajectory visualisation)."""


@dataclass
class _AnchorFrameData:
    """Internal storage for per-anchor-frame tracking data."""

    p0_world: torch.Tensor
    """(H, W, 3) float32 — world positions at the anchor frame."""
    extrinsics: torch.Tensor
    """(4, 4) float32 — camera-to-world matrix at the anchor frame."""
    trackable_mask: torch.Tensor
    """(H, W) bool — True for trackable pixels (not UNSUPPORTED)."""
    track_type: torch.Tensor
    """(H, W) uint8 — per-pixel TrackType."""
    local_points: torch.Tensor
    """(H, W, 3) float32 — body-local anchor coordinates."""
    rigid_keys: torch.Tensor
    """(H, W) int64 — hash key identifying the rigid object."""
    artic_keys: torch.Tensor
    """(H, W) int64 — hash key identifying the articulation."""
    artic_body_idx: torch.Tensor
    """(H, W) int64 — body-link index within the articulation."""
    rigid_key_to_name: Dict[int, str]
    artic_key_to_name: Dict[int, str]


class ObjectInstanceRegistry:
    """Shared registry that assigns temporally consistent object IDs, names, and colors.

    Pass a single instance to every :class:`IsaacLabArenaCameraHandler` so that
    the same physical object receives the same ``object_id``, ``object_name``,
    and RGBA color regardless of which camera observes it first.
    """

    def __init__(self) -> None:
        self._color_to_object_id: Dict[tuple, int] = {}
        self._next_object_id: int = 0

        self._instance_key_to_object_id: Dict[Tuple[str, str], int] = {}
        self._instance_key_to_name: Dict[Tuple[str, str], str] = {}
        self._next_instance_object_id: int = 0
        self._next_rigid_instance_idx: int = 1
        self._next_articulated_instance_idx: int = 1
        self._next_static_instance_idx: int = 1
        self._next_unsupported_instance_idx: int = 1

    # -- colour-based legacy ID (used by get_semantic_info) ----------------

    def get_object_id(self, rgba: tuple) -> int:
        """Return a stable integer object ID for a given RGBA colour."""
        if rgba not in self._color_to_object_id:
            self._color_to_object_id[rgba] = self._next_object_id
            self._next_object_id += 1
        return self._color_to_object_id[rgba]

    # -- instance-key-based identity (used by get_object_instance_segmentation) --

    @staticmethod
    def _safe_name_token(raw: str) -> str:
        """Convert an arbitrary label/path to an ASCII-safe token."""
        token = "".join(ch if ch.isalnum() else "_" for ch in raw.strip("/"))
        token = token.strip("_")
        return token or "unknown"

    def instance_key_to_display_name(self, instance_key: Tuple[str, str]) -> str:
        """Get or create a temporally consistent display name for an object key."""
        if instance_key in self._instance_key_to_name:
            return self._instance_key_to_name[instance_key]

        kind, source = instance_key
        if kind == "RIGID":
            prefix = f"rigid_object_{self._next_rigid_instance_idx}"
            self._next_rigid_instance_idx += 1
            suffix = self._safe_name_token(source)
            name = f"{prefix}_{suffix}"
        elif kind == "ARTICULATION":
            prefix = f"articulated_object_{self._next_articulated_instance_idx}"
            self._next_articulated_instance_idx += 1
            suffix = self._safe_name_token(source)
            name = f"{prefix}_{suffix}"
        elif kind == "STATIC":
            if source == "background":
                name = "background"
            else:
                prefix = f"static_object_{self._next_static_instance_idx}"
                self._next_static_instance_idx += 1
                suffix = self._safe_name_token(source.split("/")[-1])
                name = f"{prefix}_{suffix}"
        else:
            prefix = f"unsupported_object_{self._next_unsupported_instance_idx}"
            self._next_unsupported_instance_idx += 1
            suffix = self._safe_name_token(source.split("/")[-1])
            name = f"{prefix}_{suffix}"

        self._instance_key_to_name[instance_key] = name
        return name

    @staticmethod
    def _instance_id_to_rgba(object_id: int) -> Tuple[int, int, int, int]:
        """Deterministically map an object ID to a vivid RGBA color."""
        hue = (0.17 + 0.6180339887498948 * object_id) % 1.0
        sat = 0.75
        val = 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), 255)

    def get_or_create_instance_identity(
        self, instance_key: Tuple[str, str]
    ) -> Tuple[int, str, Tuple[int, int, int, int]]:
        """Allocate/retrieve stable object-id, name, and color for an instance key."""
        if instance_key not in self._instance_key_to_object_id:
            self._instance_key_to_object_id[instance_key] = self._next_instance_object_id
            self._next_instance_object_id += 1
        object_id = self._instance_key_to_object_id[instance_key]
        object_name = self.instance_key_to_display_name(instance_key)
        rgba = self._instance_id_to_rgba(object_id)
        return object_id, object_name, rgba


@dataclass
class MeshSamplesResult:
    """Result bundle from :meth:`DynamicObjectTracker.sample_dynamic_object_meshes`.

    Contains per-object/link relative SE(3) arrays encoding sampled surface
    points and outward normals.  Keys match
    :attr:`DynamicObjectResult.pose_arrays`.
    """

    relative_se3_arrays: Dict[str, np.ndarray]
    """Per-object (or per-body-link) arrays of shape ``(N, 3, 4)`` float32.
    Each row is the 3x4 portion of the relative SE(3) from the object/link
    centre to the sampled point.  The z-column of the rotation encodes the
    outward surface normal."""


@dataclass
class DynamicObjectResult:
    """Result bundle returned by :meth:`DynamicObjectTracker.get_dynamic_object_data`.

    Contains JSON-serialisable metadata and per-object NumPy pose arrays.
    """

    metadata: Dict[str, Any]
    """Top-level metadata dict (num_steps, thresholds, conventions)."""
    objects_metadata: Dict[str, Dict[str, Any]]
    """Per-object metadata keyed by display name (type, asset_name, body info).
    Does **not** contain the pose data itself — that lives in *pose_arrays*."""
    pose_arrays: Dict[str, np.ndarray]
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
        self._registry = registry
        self._num_steps = num_steps
        self._seen_assets: set[tuple[str, str]] = set()

        self._rigid_poses: Dict[str, torch.Tensor] = {}
        self._artic_poses: Dict[str, torch.Tensor] = {}
        self._artic_body_names: Dict[str, List[str]] = {}

    def register_visible_objects(self, semantic_info: List[Dict[str, Any]]) -> None:
        """Record which dynamic objects are visible in the current frame.

        Call once per camera per step, passing the ``semantic_info`` list
        returned by :meth:`IsaacLabArenaCameraHandler.get_object_instance_segmentation`.
        """
        for obj in semantic_info:
            track_type = obj.get("track_type", "")
            if track_type in ("RIGID", "ARTICULATION"):
                self._seen_assets.add((track_type, obj["asset_name"]))

    @staticmethod
    def _se3_from_pos_quat(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """Build a 4x4 SE(3) matrix from position (3,) and quaternion (w,x,y,z) (4,)."""
        T = torch.eye(4, dtype=torch.float32, device=pos.device)
        T[:3, :3] = matrix_from_quat(quat.reshape(4))
        T[:3, 3] = pos
        return T

    def record_step_poses(self, env: Any, step_idx: int) -> None:
        """Record world-frame poses of all rigid objects and articulations.

        On first encounter each object gets a pre-allocated tensor of shape
        ``(num_steps, 4, 4)`` (rigid) or ``(num_steps, num_bodies, 4, 4)``
        (articulation).  Subsequent calls fill in ``step_idx`` without
        allocating new memory.
        """
        scene = env.unwrapped.scene

        for name in scene.rigid_objects.keys():
            obj = scene[name]
            T = self._se3_from_pos_quat(
                obj.data.root_link_pos_w[0],
                obj.data.root_link_quat_w[0],
            )
            if name not in self._rigid_poses:
                buf = torch.zeros(self._num_steps, 4, 4, dtype=torch.float32)
                for s in range(self._num_steps):
                    buf[s] = torch.eye(4)
                self._rigid_poses[name] = buf
            self._rigid_poses[name][step_idx] = T.cpu()

        for name, artic in scene.articulations.items():
            num_bodies = artic.data.body_link_pos_w.shape[1]
            body_names = artic.data.body_names

            if name not in self._artic_poses:
                buf = torch.zeros(self._num_steps, num_bodies, 4, 4, dtype=torch.float32)
                for s in range(self._num_steps):
                    for bi in range(num_bodies):
                        buf[s, bi] = torch.eye(4)
                self._artic_poses[name] = buf
                self._artic_body_names[name] = [
                    body_names[bi] if bi < len(body_names) else f"body_{bi}"
                    for bi in range(num_bodies)
                ]

            for bi in range(num_bodies):
                T = self._se3_from_pos_quat(
                    artic.data.body_link_pos_w[0, bi],
                    artic.data.body_link_quat_w[0, bi],
                )
                self._artic_poses[name][step_idx, bi] = T.cpu()

    def get_dynamic_object_data(
        self,
        motion_eps: float = 1e-4,
    ) -> DynamicObjectResult:
        """Return filtered data for objects that were seen and actually moved.

        Uses the same lenient rotation threshold as
        :meth:`sample_dynamic_object_meshes` so results are consistent:
        ``rotation_eps = max(motion_eps * 100, 1e-2)`` (~0.57 deg).  This
        avoids classifying stationary objects as dynamic due to physics
        solver jitter in rotations.

        For articulated objects, only links that individually exceed the
        motion threshold are included.

        Args:
            motion_eps: Minimum translation (metres) between adjacent
                frames to count as dynamic.

        Returns:
            A :class:`DynamicObjectResult` containing JSON metadata and
            NumPy pose arrays ready to be written with
            :meth:`IsaacLabArenaWriter.write_dynamic_object_poses`.
        """
        objects_meta: Dict[str, Dict[str, Any]] = {}
        pose_arrays: Dict[str, np.ndarray] = {}
        rot_eps = max(motion_eps * 100.0, 1e-2)

        for kind, asset_name in self._seen_assets:
            if kind == "RIGID":
                poses_4x4 = self._rigid_poses.get(asset_name)
                if poses_4x4 is None:
                    continue
                if not self._has_motion_tensor(
                    poses_4x4, motion_eps, rotation_eps=rot_eps,
                ):
                    continue

                instance_key = ("RIGID", asset_name)
                display_name = self._registry.instance_key_to_display_name(instance_key)
                objects_meta[display_name] = {
                    "type": "rigid",
                    "asset_name": asset_name,
                    "pose_array_key": display_name,
                }
                pose_arrays[display_name] = poses_4x4[:, :3, :].numpy()

            elif kind == "ARTICULATION":
                poses_4x4 = self._artic_poses.get(asset_name)
                if poses_4x4 is None:
                    continue
                body_names = self._artic_body_names.get(asset_name, [])
                num_bodies = poses_4x4.shape[1]

                instance_key = ("ARTICULATION", asset_name)
                display_name = self._registry.instance_key_to_display_name(instance_key)

                parts_meta: Dict[str, Dict[str, Any]] = {}
                for bi in range(num_bodies):
                    if not self._has_motion_tensor(
                        poses_4x4[:, bi], motion_eps, rotation_eps=rot_eps,
                    ):
                        continue
                    bname = body_names[bi] if bi < len(body_names) else f"body_{bi}"
                    array_key = f"{display_name}/{bname}"
                    parts_meta[bname] = {
                        "body_index": bi,
                        "pose_array_key": array_key,
                    }
                    pose_arrays[array_key] = poses_4x4[:, bi, :3, :].numpy()

                if not parts_meta:
                    continue

                objects_meta[display_name] = {
                    "type": "articulation",
                    "asset_name": asset_name,
                    "parts": parts_meta,
                }

        metadata = {
            "num_steps": self._num_steps,
            "motion_threshold": motion_eps,
            "coordinate_frame": "world",
            "pose_format": "(num_steps, 3, 4) float32 row-major [R|t] per .npy array",
        }

        return DynamicObjectResult(
            metadata=metadata,
            objects_metadata=objects_meta,
            pose_arrays=pose_arrays,
        )

    # ---- internal helpers ------------------------------------------------

    @staticmethod
    def _has_motion_tensor(
        poses_4x4: torch.Tensor,
        eps: float = 0.0,
        rotation_eps: Optional[float] = None,
    ) -> bool:
        """Check adjacent-frame motion on a contiguous ``(N, 4, 4)`` tensor.

        Args:
            poses_4x4: ``(N, 4, 4)`` tensor of SE(3) matrices.
            eps: Threshold for translation (metres).
            rotation_eps: Threshold for rotation (radians).  Defaults to
                *eps* when ``None``.  Set to ``float('inf')`` to ignore
                rotation entirely.
        """
        if poses_4x4.shape[0] < 2:
            return False
        if rotation_eps is None:
            rotation_eps = eps
        prev = poses_4x4[:-1]
        cur = poses_4x4[1:]

        t_diffs = (cur[:, :3, 3] - prev[:, :3, 3]).norm(dim=-1)
        if t_diffs.max().item() > eps:
            return True

        if rotation_eps < float("inf"):
            R_rel = prev[:, :3, :3].transpose(-1, -2) @ cur[:, :3, :3]
            cos_angles = ((R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
            angles = torch.acos(cos_angles)
            if angles.max().item() > rotation_eps:
                return True

        return False

    # ---- mesh surface sampling -------------------------------------------

    def sample_dynamic_object_meshes(
        self,
        env: Any,
        result: DynamicObjectResult,
        spacing: float = 0.01,
        motion_eps: float = 1e-4,
    ) -> MeshSamplesResult:
        """Sample surface points with normals on moving objects' meshes only.

        For each object/link in *result* whose pose actually changes between
        at least two adjacent steps (above *motion_eps*), resolves the USD
        mesh geometry, uniformly samples ``area / spacing**2`` points on the
        surface, and computes the relative SE(3) from the object/link centre
        (step-0 pose) to each point.  The z-column of each relative rotation
        encodes the outward surface normal.

        Objects and individual articulation links that do not exhibit motion
        above the threshold are skipped.

        Must be called while the simulation is still running (USD stage
        accessible).

        Args:
            env: The gymnasium-wrapped IsaacLab environment.
            result: The :class:`DynamicObjectResult` from
                :meth:`get_dynamic_object_data`.
            spacing: Target distance (metres) between sampled points.
                Point count is ``max(1, int(area / spacing**2))``.
            motion_eps: Minimum adjacent-frame translation (m) or rotation
                (rad) for a link to be considered moving.  Links below this
                threshold are skipped.

        Returns:
            A :class:`MeshSamplesResult` with keys only for objects/links
            that actually moved.
        """
        import trimesh

        scene = env.unwrapped.scene
        relative_se3: Dict[str, np.ndarray] = {}
        rot_eps = max(motion_eps * 100.0, 1e-2)

        for display_name, meta in result.objects_metadata.items():
            if meta["type"] == "rigid":
                asset_name = meta["asset_name"]
                pose_key = meta["pose_array_key"]

                poses_4x4 = self._rigid_poses.get(asset_name)
                if poses_4x4 is not None and not self._has_motion_tensor(
                    poses_4x4, motion_eps, rotation_eps=rot_eps,
                ):
                    continue

                prim_path = scene[asset_name].cfg.prim_path.replace(".*", "0")
                combined = self._collect_mesh_from_prim(prim_path)
                if combined is None or combined.area < 1e-12:
                    continue

                pts, normals = self._sample_on_mesh(combined, spacing)
                T_local = np.eye(4, dtype=np.float64)
                relative_se3[pose_key] = self._compute_relative_se3(
                    pts, normals, T_local,
                )

            elif meta["type"] == "articulation":
                asset_name = meta["asset_name"]
                artic_obj = scene[asset_name]
                root_path = artic_obj.cfg.prim_path.replace(".*", "0")

                for bname, part_meta in meta["parts"].items():
                    bi = part_meta["body_index"]
                    pose_key = part_meta["pose_array_key"]

                    link_poses = self._artic_poses.get(asset_name)
                    if link_poses is not None and not self._has_motion_tensor(
                        link_poses[:, bi], motion_eps, rotation_eps=rot_eps,
                    ):
                        continue

                    link_path = f"{root_path}/{bname}"
                    combined = self._collect_mesh_from_prim(link_path)
                    if combined is None or combined.area < 1e-12:
                        continue

                    pts, normals = self._sample_on_mesh(combined, spacing)
                    T_local = np.eye(4, dtype=np.float64)
                    relative_se3[pose_key] = self._compute_relative_se3(
                        pts, normals, T_local,
                    )

        return MeshSamplesResult(relative_se3_arrays=relative_se3)

    # ---- mesh helpers (private) ------------------------------------------

    @staticmethod
    def _collect_mesh_from_prim(prim_path: str) -> Optional[Any]:
        """Walk *prim_path* and combine all child meshes / primitives into one trimesh.

        Vertices are returned in the **body-local frame** relative to
        *prim_path*.  Physics-driven transforms (applied by PhysX at
        runtime) are NOT baked in, because ``UsdGeom.XformCache`` does not
        see them.  Only the static USD xform hierarchy *below* the root
        prim is applied so that sub-mesh parts are correctly assembled.
        """
        import trimesh

        try:
            import isaacsim.core.utils.prims as prim_utils
            from pxr import UsdGeom

            from isaaclab.sim.utils import get_all_matching_child_prims
        except Exception:
            return None

        prims = get_all_matching_child_prims(
            prim_path,
            predicate=lambda p: p.GetTypeName() in (
                "Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone",
            ),
        )
        if not prims:
            return None

        xform_cache = UsdGeom.XformCache()
        root_prim = prim_utils.get_prim_at_path(prim_path)
        root_world = xform_cache.GetLocalToWorldTransform(root_prim)
        root_world_inv = root_world.GetInverse()

        meshes: list = []

        for prim in prims:
            prim_type = prim.GetTypeName()
            if prim_type == "Mesh":
                mesh_geom = UsdGeom.Mesh(prim)
                verts = np.asarray(mesh_geom.GetPointsAttr().Get(), dtype=np.float32)
                faces = _triangulate_usd_faces(prim)
                tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            else:
                tm = _create_primitive_mesh(prim)

            prim_to_root = root_world_inv * xform_cache.GetLocalToWorldTransform(prim)
            mat_np = np.array(
                [[prim_to_root[r][c] for c in range(4)] for r in range(4)],
                dtype=np.float64,
            )
            tm.apply_transform(mat_np)
            meshes.append(tm)

        if not meshes:
            return None
        return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    @staticmethod
    def _sample_on_mesh(
        mesh: Any, spacing: float, oversample_factor: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample surface points and face normals at the given spacing.

        Oversamples randomly then applies **farthest point sampling** (FPS) to
        select a maximally-spaced subset, guaranteeing uniform coverage
        regardless of mesh tessellation quality.
        """
        from trimesh.sample import sample_surface

        num_points = max(1, int(mesh.area / (spacing ** 2)))
        n_candidates = num_points * oversample_factor
        cand_pts, cand_fi = sample_surface(mesh, n_candidates)

        if cand_pts.shape[0] <= num_points:
            normals = mesh.face_normals[cand_fi]
            return cand_pts.astype(np.float32), normals.astype(np.float32)

        sel = DynamicObjectTracker._farthest_point_sampling(cand_pts, num_points)
        np.random.shuffle(sel)
        fps_pts = cand_pts[sel]
        normals = mesh.face_normals[cand_fi[sel]]
        return fps_pts.astype(np.float32), normals.astype(np.float32)

    @staticmethod
    def _farthest_point_sampling(pts: np.ndarray, k: int) -> np.ndarray:
        """Select *k* points from *pts* maximising the minimum pairwise distance."""
        n = pts.shape[0]
        k = min(k, n)
        selected = np.empty(k, dtype=np.int64)
        selected[0] = np.random.randint(n)
        min_dists = np.full(n, np.inf, dtype=np.float64)
        for i in range(1, k):
            diff = pts - pts[selected[i - 1]]
            dists = np.einsum("ij,ij->i", diff, diff)
            np.minimum(min_dists, dists, out=min_dists)
            selected[i] = np.argmax(min_dists)
        return selected

    @staticmethod
    def _compute_relative_se3(
        points_world: np.ndarray,
        normals_world: np.ndarray,
        T_obj_0: np.ndarray,
    ) -> np.ndarray:
        """Compute per-point relative SE(3) w.r.t. the object/link pose at step 0.

        Returns:
            ``(N, 3, 4)`` float32 array of relative SE(3) transforms.
        """
        N = points_world.shape[0]
        T_obj_inv = np.eye(4, dtype=np.float64)
        R = T_obj_0[:3, :3].astype(np.float64)
        t = T_obj_0[:3, 3].astype(np.float64)
        T_obj_inv[:3, :3] = R.T
        T_obj_inv[:3, 3] = -R.T @ t

        result = np.zeros((N, 3, 4), dtype=np.float32)

        for i in range(N):
            n = normals_world[i].astype(np.float64)
            n_len = np.linalg.norm(n)
            if n_len < 1e-12:
                n = np.array([0.0, 0.0, 1.0])
            else:
                n = n / n_len

            R_point = _rotation_from_normal(n)

            T_point = np.eye(4, dtype=np.float64)
            T_point[:3, :3] = R_point
            T_point[:3, 3] = points_world[i].astype(np.float64)

            T_rel = T_obj_inv @ T_point
            result[i] = T_rel[:3, :].astype(np.float32)

        return result


def _rotation_from_normal(n: np.ndarray) -> np.ndarray:
    """Build a 3x3 rotation matrix whose z-column equals the unit normal *n*."""
    z = n / np.linalg.norm(n)
    ref = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _triangulate_usd_faces(prim: Any) -> np.ndarray:
    """Convert a USD Mesh prim into triangulated face indices ``(F, 3)``."""
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces: list = []
    it = iter(indices)
    for cnt in counts:
        poly = [next(it) for _ in range(cnt)]
        for k in range(1, cnt - 1):
            faces.append([poly[0], poly[k], poly[k + 1]])
    return np.asarray(faces, dtype=np.int64)


def _create_primitive_mesh(prim: Any) -> Any:
    """Create a trimesh mesh from a USD geometric primitive."""
    import trimesh
    from pxr import UsdGeom

    prim_type = prim.GetTypeName()
    if prim_type == "Cube":
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        return trimesh.creation.box(extents=(size, size, size))
    elif prim_type == "Sphere":
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        return trimesh.creation.icosphere(subdivisions=3, radius=r)
    elif prim_type == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        return trimesh.creation.cylinder(
            radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get(),
        )
    elif prim_type == "Capsule":
        c = UsdGeom.Capsule(prim)
        return trimesh.creation.capsule(
            radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get(),
        )
    elif prim_type == "Cone":
        c = UsdGeom.Cone(prim)
        return trimesh.creation.cone(
            radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get(),
        )
    raise KeyError(f"{prim_type} is not a valid primitive mesh type")


def reconstruct_mesh_points_at_step(
    mesh_samples: MeshSamplesResult,
    pose_arrays: Dict[str, np.ndarray],
    step_idx: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Reconstruct world-space positions and normals of sampled mesh points.

    For each object/link, applies the SE(3) pose at *step_idx* to the stored
    relative SE(3) per point to recover world-space positions and normals.

    Args:
        mesh_samples: The :class:`MeshSamplesResult` from
            :meth:`DynamicObjectTracker.sample_dynamic_object_meshes`.
        pose_arrays: Dict of ``(num_steps, 3, 4)`` float32 arrays, loaded
            from ``dynamic_objects_poses.npz``.
        step_idx: The simulation step to reconstruct at.

    Returns:
        Dict mapping each pose-array key to a tuple
        ``(points_world, normals_world)`` where each is ``(N, 3)`` float32.
    """
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for key, rel_se3 in mesh_samples.relative_se3_arrays.items():
        if key not in pose_arrays:
            continue
        N = rel_se3.shape[0]

        pose_3x4 = pose_arrays[key][step_idx]  # (3, 4)
        T_obj = np.eye(4, dtype=np.float64)
        T_obj[:3, :] = pose_3x4.astype(np.float64)

        T_rel_4x4 = np.zeros((N, 4, 4), dtype=np.float64)
        T_rel_4x4[:, :3, :] = rel_se3.astype(np.float64)
        T_rel_4x4[:, 3, 3] = 1.0

        T_world = T_obj[np.newaxis, :, :] @ T_rel_4x4  # (N, 4, 4)

        points = T_world[:, :3, 3].astype(np.float32)
        normals = T_world[:, :3, 2].astype(np.float32)

        result[key] = (points, normals)

    return result


class IsaacLabArenaCameraHandler:
    """Handles a static camera sensor in an IsaacLab Arena environment.

    Wraps an IsaacLab Camera sensor and provides methods to extract
    RGB, depth, normals, optical flow, semantic segmentation,
    intrinsic matrix, and camera-to-world extrinsic matrix.

    Args:
        camera: An isaaclab.sensors.Camera instance.
        camera_name: Identifier used for file naming when writing data.
        instance_registry: Optional shared :class:`ObjectInstanceRegistry`.
            When provided, object IDs / names / colors are consistent across
            all handlers that share the same registry.  When ``None`` a
            private registry is created (backward-compatible behaviour).
    """

    def __init__(
        self,
        camera: Any,
        camera_name: str = "static_camera",
        instance_registry: Optional[ObjectInstanceRegistry] = None,
    ):
        self._camera = camera
        self._camera_name = camera_name
        self._registry = instance_registry or ObjectInstanceRegistry()

        # Override for get_extrinsics() when camera is moved via set_world_pose
        # (the sensor's pos_w / quat_w_ros do not update after USD prim changes)
        self._override_extrinsics: Optional[torch.Tensor] = None

        # Cached data for one-frame-lag exact scene flow
        self._prev_points_world: Optional[torch.Tensor] = None  # (H, W, 3)
        self._prev_extrinsics: Optional[torch.Tensor] = None  # (4, 4)
        self._prev_track_type: Optional[torch.Tensor] = None  # (H, W) uint8
        self._prev_valid_mask: Optional[torch.Tensor] = None  # (H, W) bool
        self._prev_local_points: Optional[torch.Tensor] = None  # (H, W, 3)
        self._prev_rigid_keys: Optional[torch.Tensor] = None  # (H, W) int64
        self._prev_artic_keys: Optional[torch.Tensor] = None  # (H, W) int64
        self._prev_artic_body_idx: Optional[torch.Tensor] = None  # (H, W) int64

        # Mapping tables filled per cache_frame call
        self._rigid_key_to_name: Dict[int, str] = {}
        self._artic_key_to_name: Dict[int, str] = {}

        # Anchor-frame data (one entry per anchor, set by init_anchor_frame)
        self._ff_anchors: Dict[int, _AnchorFrameData] = {}

    @property
    def camera_name(self) -> str:
        return self._camera_name

    def _get_camera_output(self) -> Dict[str, torch.Tensor]:
        from isaaclab.utils import convert_dict_to_backend

        return convert_dict_to_backend(self._camera.data.output, backend="torch")

    def update(self, dt: float) -> None:
        """Read the latest render data from the camera sensor."""
        self._camera.update(dt)

    def set_world_pose(
        self,
        position: Tuple[float, float, float],
        target: Tuple[float, float, float],
    ) -> None:
        """Move the camera to *position* and orient it to look at *target*.

        Updates the USD prim transform so the next :meth:`update` call
        renders from the new viewpoint.  Call once per step *before*
        :meth:`update` for each step where the camera should move.
        """
        import omni.usd
        from pxr import Gf, UsdGeom

        eye = np.array(position, dtype=np.float64)
        tgt = np.array(target, dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        forward = tgt - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        cam_up = np.cross(right, forward)

        # USD camera convention: x = right, y = up, z = backward.
        # Row-vector convention: v_world = v_local * M, so the rows of M
        # are the camera basis vectors expressed in the world frame.
        mat = Gf.Matrix4d(
            float(right[0]),    float(right[1]),    float(right[2]),    0.0,
            float(cam_up[0]),   float(cam_up[1]),   float(cam_up[2]),   0.0,
            float(-forward[0]), float(-forward[1]), float(-forward[2]), 0.0,
            float(eye[0]),      float(eye[1]),      float(eye[2]),      1.0,
        )

        prim_path = self._camera.cfg.prim_path
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)

        if not hasattr(self, "_xform_op"):
            xformable.ClearXformOpOrder()
            self._xform_op = xformable.AddTransformOp()

        self._xform_op.Set(mat)

        # Cache ROS-convention cam-to-world extrinsics so that
        # get_extrinsics() returns the correct pose even though
        # the camera sensor's internal pos_w / quat_w_ros do not
        # update after direct USD prim changes.
        # ROS camera axes: x = right, y = down, z = forward.
        T = torch.eye(4, dtype=torch.float32)
        T[0, 0], T[1, 0], T[2, 0] = float(right[0]), float(right[1]), float(right[2])
        T[0, 1], T[1, 1], T[2, 1] = float(-cam_up[0]), float(-cam_up[1]), float(-cam_up[2])
        T[0, 2], T[1, 2], T[2, 2] = float(forward[0]), float(forward[1]), float(forward[2])
        T[0, 3], T[1, 3], T[2, 3] = float(eye[0]), float(eye[1]), float(eye[2])
        self._override_extrinsics = T

    # ------------------------------------------------------------------
    # Core outputs
    # ------------------------------------------------------------------

    def get_rgb(self) -> torch.Tensor:
        """Returns RGB image as a (H, W, 3) uint8 tensor."""
        rgb = self._get_camera_output()["rgb"].clone()
        assert rgb.shape[0] == 1, "Expected single environment"
        assert rgb.shape[-1] == 3, "Expected 3-channel RGB"
        return rgb.squeeze(0)

    def get_depth(self) -> torch.Tensor:
        """Returns depth map as a (H, W) float32 tensor (distance to image plane)."""
        depth = self._get_camera_output()["distance_to_image_plane"].clone()
        assert depth.shape[0] == 1, "Expected single environment"
        return depth.squeeze()

    def get_intrinsics(self) -> torch.Tensor:
        """Returns the 3x3 camera intrinsic matrix."""
        assert self._camera.data.intrinsic_matrices.shape[0] == 1, "Expected single environment"
        return self._camera.data.intrinsic_matrices.data[0].clone()

    def get_extrinsics(self) -> torch.Tensor:
        """Returns the 4x4 camera-to-world homogeneous transformation matrix.

        If the camera was repositioned via :meth:`set_world_pose`, the
        override extrinsics are returned (the sensor's internal ``pos_w``
        / ``quat_w_ros`` do not update after direct USD prim changes).
        """
        if self._override_extrinsics is not None:
            device = self._camera.data.pos_w.device
            return self._override_extrinsics.clone().to(device)

        assert self._camera.data.pos_w.shape[0] == 1, "Expected single environment"
        translation = self._camera.data.pos_w.data[0].clone()
        rotation_quat = self._camera.data.quat_w_ros.data[0].clone()

        T = torch.eye(4, dtype=torch.float32, device=rotation_quat.device)
        quat_flat = rotation_quat.reshape(4)
        T[:3, :3] = matrix_from_quat(quat_flat)
        T[:3, 3] = translation
        return T

    # ------------------------------------------------------------------
    # Normals
    # ------------------------------------------------------------------

    def get_normals(self) -> torch.Tensor:
        """Returns surface normals as a (H, W, 3) float32 tensor (x, y, z)."""
        normals = self._get_camera_output()["normals"].clone()
        assert normals.shape[0] == 1, "Expected single environment"
        assert normals.shape[-1] == 3, "Expected 3-channel normals"
        return normals.squeeze(0)

    # ------------------------------------------------------------------
    # Optical flow (motion vectors)
    # ------------------------------------------------------------------

    def get_optical_flow(self) -> torch.Tensor:
        """Returns dense optical flow as a (H, W, 2) float32 tensor (dx, dy in pixels)."""
        mv = self._get_camera_output()["motion_vectors"].clone()
        assert mv.shape[0] == 1, "Expected single environment"
        assert mv.shape[-1] == 2, "Expected 2-channel motion vectors"
        return mv.squeeze(0)

    # ------------------------------------------------------------------
    # Semantic segmentation
    # ------------------------------------------------------------------

    def get_semantic_segmentation(self) -> Tuple[torch.Tensor, Dict]:
        """Returns semantic segmentation data.

        Returns:
            Tuple of:
                - (H, W, 4) uint8 tensor with RGBA segmentation colors.
                - dict mapping RGBA id-strings to label dicts (``{"class": ...}``).
        """
        seg = self._get_camera_output()["semantic_segmentation"].clone()
        id_to_labels = self._camera.data.info[0]["semantic_segmentation"]["idToLabels"]
        assert seg.shape[0] == 1, "Expected single environment"
        assert seg.shape[-1] == 4, "Expected 4-channel segmentation (RGBA)"
        return seg.squeeze(0).to(torch.uint8), id_to_labels

    # ------------------------------------------------------------------
    # Instance ID segmentation
    # ------------------------------------------------------------------

    def get_instance_id_segmentation(self) -> Tuple[torch.Tensor, Dict]:
        """Returns per-pixel integer instance IDs and the ID-to-prim-path mapping.

        Requires ``instance_id_segmentation_fast`` in the camera's data types
        with ``colorize_instance_id_segmentation=False``.

        Returns:
            Tuple of:
                - (H, W) int32 tensor of instance IDs.
                - dict mapping ID strings to label dicts (``{"class": <prim_path>}``).
        """
        seg = self._get_camera_output()["instance_id_segmentation_fast"].clone()
        id_to_labels = self._camera.data.info[0]["instance_id_segmentation_fast"]["idToLabels"]
        assert seg.shape[0] == 1, "Expected single environment"
        return seg.squeeze(0).squeeze(-1).to(torch.int32), id_to_labels

    @staticmethod
    def _instance_object_key_from_binding(binding: Tuple[str, ...], prim_path: str) -> Tuple[str, str]:
        """Convert tracking binding to a stable per-object semantic key.

        The returned key is used to allocate temporally consistent object IDs
        and names in :meth:`get_object_instance_segmentation`.
        """
        kind = binding[0]
        if kind == "RIGID":
            return ("RIGID", binding[1])
        if kind == "ARTICULATION":
            return ("ARTICULATION", binding[1])
        if kind == "STATIC":
            if not prim_path or prim_path.upper() in ("BACKGROUND", "UNLABELLED", "UNKNOWN", "INVALID"):
                return ("STATIC", "background")
            return ("STATIC", prim_path)
        if not prim_path:
            return ("UNSUPPORTED", "unknown")
        return ("UNSUPPORTED", prim_path)

    def _get_or_create_instance_identity(
        self, instance_key: Tuple[str, str]
    ) -> Tuple[int, str, Tuple[int, int, int, int]]:
        """Delegate to the shared registry."""
        return self._registry.get_or_create_instance_identity(instance_key)

    def get_object_instance_segmentation(self, env: Any) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Build a semantic-style RGBA image where each tracked object has a unique label.

        Unlike :meth:`get_semantic_segmentation`, this method does not depend on
        authored semantic tags. It uses ``instance_id_segmentation_fast`` plus
        prim-path binding resolution to assign pixels to:
        - rigid objects,
        - articulated objects,
        - static/background geometry,
        - unsupported objects.

        The assigned object IDs, names, and colors are persistent across frames,
        so if an object disappears and later reappears it keeps the same name.

        Args:
            env: The gymnasium-wrapped IsaacLab environment.

        Returns:
            Tuple of:
                - (H, W, 4) uint8 RGBA object-instance segmentation.
                - List of metadata dicts for visible objects in this frame.
        """
        inst_ids, id_to_labels = self.get_instance_id_segmentation()
        H, W = inst_ids.shape
        device = inst_ids.device
        scene = env.unwrapped.scene

        seg_rgba = torch.zeros(H, W, 4, dtype=torch.uint8, device=device)
        object_rows: Dict[int, Dict[str, Any]] = {}

        for uid in inst_ids.unique():
            uid_key = uid.item()
            pixel_mask = inst_ids == uid
            if not pixel_mask.any():
                continue

            label = id_to_labels.get(uid_key, "")
            prim_path = label.get("class", "") if isinstance(label, dict) else str(label)
            if not prim_path or prim_path.upper() in ("BACKGROUND", "UNLABELLED", "UNKNOWN", "INVALID"):
                binding: Tuple[str, ...] = ("STATIC",)
            else:
                binding = self._resolve_tracking_binding(prim_path, scene)

            instance_key = self._instance_object_key_from_binding(binding, prim_path)
            object_id, object_name, rgba = self._get_or_create_instance_identity(instance_key)
            seg_rgba[pixel_mask] = torch.tensor(rgba, dtype=torch.uint8, device=device)

            if object_id not in object_rows:
                kind, source = instance_key
                object_rows[object_id] = {
                    "object_id": object_id,
                    "object_name": object_name,
                    "rgba": rgba,
                    "class_name": kind.lower(),
                    "track_type": kind,
                    "asset_name": source,
                    "pixel_count": 0,
                }
            object_rows[object_id]["pixel_count"] += int(pixel_mask.sum().item())

        semantic_info = sorted(object_rows.values(), key=lambda x: x["pixel_count"], reverse=True)
        return seg_rgba, semantic_info

    # ------------------------------------------------------------------
    # 3D scene flow — exact adjacent-frame ground truth (SE(3) based)
    # ------------------------------------------------------------------

    def cache_scene_flow_frame(self, env: Any) -> None:
        """Cache per-pixel world points and tracking anchors for this frame.

        Must be called once per simulation step *after* :meth:`update`.
        The cached data is consumed by :meth:`compute_exact_scene_flow`
        on the *next* frame to produce one-frame-lag flow.

        Args:
            env: The gymnasium-wrapped IsaacLab environment.
        """
        from isaaclab.utils.math import unproject_depth

        depth = self.get_depth()
        intrinsics = self.get_intrinsics()
        extrinsics = self.get_extrinsics()

        H, W = depth.shape
        device = depth.device

        # Unproject to world (preserving existing column-major → row-major fix)
        points_cam = unproject_depth(depth, intrinsics)  # (H*W, 3)
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        points_world = (points_cam @ R.T) + t  # (H*W, 3)
        points_world = points_world.reshape(W, H, 3).permute(1, 0, 2).contiguous()

        valid_mask = torch.isfinite(depth)

        inst_ids, id_to_labels = self.get_instance_id_segmentation()

        scene = env.unwrapped.scene

        track_type = torch.full((H, W), TrackType.UNSUPPORTED, dtype=torch.uint8, device=device)
        local_points = torch.zeros(H, W, 3, dtype=torch.float32, device=device)

        rigid_keys = torch.full((H, W), -1, dtype=torch.int64, device=device)
        artic_keys = torch.full((H, W), -1, dtype=torch.int64, device=device)
        artic_body_idx = torch.full((H, W), -1, dtype=torch.int64, device=device)

        self._rigid_key_to_name = {}
        self._artic_key_to_name = {}

        rigid_pose_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        artic_pose_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}

        for uid in inst_ids.unique():
            uid_key = uid.item()
            if uid_key not in id_to_labels:
                continue

            label = id_to_labels[uid_key]
            prim_path = label.get("class", "") if isinstance(label, dict) else str(label)
            if not prim_path or prim_path.upper() in ("BACKGROUND", "UNLABELLED", "UNKNOWN", "INVALID"):
                pixel_mask = (inst_ids == uid) & valid_mask
                if pixel_mask.any():
                    track_type[pixel_mask] = TrackType.STATIC
                continue

            binding = self._resolve_tracking_binding(prim_path, scene)

            pixel_mask = (inst_ids == uid) & valid_mask
            if not pixel_mask.any():
                continue

            pts = points_world[pixel_mask]  # (N, 3)

            if binding[0] == "STATIC":
                track_type[pixel_mask] = TrackType.STATIC

            elif binding[0] == "RIGID":
                asset_name = binding[1]
                track_type[pixel_mask] = TrackType.RIGID

                key = hash(asset_name) & 0x7FFFFFFFFFFFFFFF
                self._rigid_key_to_name[key] = asset_name
                rigid_keys[pixel_mask] = key

                if asset_name not in rigid_pose_cache:
                    obj = scene[asset_name]
                    pos = obj.data.root_link_pos_w[0].clone()
                    quat = obj.data.root_link_quat_w[0].clone()  # (w, x, y, z)
                    rigid_pose_cache[asset_name] = (pos, quat)

                pos_t, quat_t = rigid_pose_cache[asset_name]
                q_local = quat_apply_inverse(
                    quat_t.unsqueeze(0).expand(pts.shape[0], -1), pts - pos_t.unsqueeze(0)
                )
                local_points[pixel_mask] = q_local

            elif binding[0] == "ARTICULATION":
                asset_name, body_idx = binding[1], binding[2]
                track_type[pixel_mask] = TrackType.ARTICULATION

                artic_key = hash(asset_name) & 0x7FFFFFFFFFFFFFFF
                self._artic_key_to_name[artic_key] = asset_name
                artic_keys[pixel_mask] = artic_key
                artic_body_idx[pixel_mask] = body_idx

                cache_key = (asset_name, body_idx)
                if cache_key not in artic_pose_cache:
                    artic_obj = scene[asset_name]
                    link_pos = artic_obj.data.body_link_pos_w[0, body_idx].clone()
                    link_quat = artic_obj.data.body_link_quat_w[0, body_idx].clone()
                    artic_pose_cache[cache_key] = (link_pos, link_quat)

                pos_t, quat_t = artic_pose_cache[cache_key]
                q_local = quat_apply_inverse(
                    quat_t.unsqueeze(0).expand(pts.shape[0], -1), pts - pos_t.unsqueeze(0)
                )
                local_points[pixel_mask] = q_local

            else:
                pass  # UNSUPPORTED — already default

        self._prev_points_world = points_world
        self._prev_extrinsics = extrinsics
        self._prev_track_type = track_type
        self._prev_valid_mask = valid_mask
        self._prev_local_points = local_points
        self._prev_rigid_keys = rigid_keys
        self._prev_artic_keys = artic_keys
        self._prev_artic_body_idx = artic_body_idx

    def compute_exact_scene_flow(self, env: Any) -> Optional[SceneFlowResult]:
        """Compute exact adjacent-frame 3D scene flow for the *previous* frame.

        Uses the cached anchors from the previous call to
        :meth:`cache_scene_flow_frame` together with the current frame's
        SE(3) poses to produce ground-truth displacement.

        Returns ``None`` if no previous frame has been cached yet (first frame).

        Args:
            env: The gymnasium-wrapped IsaacLab environment.

        Returns:
            A :class:`SceneFlowResult` for the *previous* frame, or ``None``.
        """
        if self._prev_points_world is None:
            return None

        H, W = self._prev_points_world.shape[:2]
        device = self._prev_points_world.device

        scene = env.unwrapped.scene

        flow = torch.zeros(H, W, 3, dtype=torch.float32, device=device)
        valid = self._prev_valid_mask.clone()

        # --- STATIC pixels: displacement = 0 (already zero in flow) ---

        # --- RIGID pixels ---
        rigid_mask = self._prev_track_type == TrackType.RIGID
        if rigid_mask.any():
            rigid_pose_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            for key, name in self._rigid_key_to_name.items():
                obj = scene[name]
                rigid_pose_cache[name] = (
                    obj.data.root_link_pos_w[0].clone(),
                    obj.data.root_link_quat_w[0].clone(),
                )

            for key, name in self._rigid_key_to_name.items():
                sel = rigid_mask & (self._prev_rigid_keys == key)
                if not sel.any():
                    continue
                pos_tp1, quat_tp1 = rigid_pose_cache[name]
                q_local = self._prev_local_points[sel]
                p_tp1 = quat_apply(
                    quat_tp1.unsqueeze(0).expand(q_local.shape[0], -1), q_local
                ) + pos_tp1.unsqueeze(0)
                flow[sel] = p_tp1 - self._prev_points_world[sel]

        # --- ARTICULATION pixels ---
        artic_mask = self._prev_track_type == TrackType.ARTICULATION
        if artic_mask.any():
            artic_pose_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
            for artic_key, name in self._artic_key_to_name.items():
                artic_obj = scene[name]
                num_bodies = artic_obj.data.body_link_pos_w.shape[1]
                for bi in range(num_bodies):
                    cache_key = (name, bi)
                    if cache_key not in artic_pose_cache:
                        artic_pose_cache[cache_key] = (
                            artic_obj.data.body_link_pos_w[0, bi].clone(),
                            artic_obj.data.body_link_quat_w[0, bi].clone(),
                        )

            for artic_key, name in self._artic_key_to_name.items():
                artic_obj = scene[name]
                num_bodies = artic_obj.data.body_link_pos_w.shape[1]
                for bi in range(num_bodies):
                    sel = artic_mask & (self._prev_artic_keys == artic_key) & (self._prev_artic_body_idx == bi)
                    if not sel.any():
                        continue
                    pos_tp1, quat_tp1 = artic_pose_cache[(name, bi)]
                    q_local = self._prev_local_points[sel]
                    p_tp1 = quat_apply(
                        quat_tp1.unsqueeze(0).expand(q_local.shape[0], -1), q_local
                    ) + pos_tp1.unsqueeze(0)
                    flow[sel] = p_tp1 - self._prev_points_world[sel]

        # --- UNSUPPORTED pixels: mark invalid ---
        unsupported = self._prev_track_type == TrackType.UNSUPPORTED
        valid[unsupported] = False

        return SceneFlowResult(
            scene_flow_3d=flow,
            scene_flow_valid_mask=valid,
            scene_flow_track_type=self._prev_track_type.clone(),
        )

    # ------------------------------------------------------------------
    # World-space → camera-relative 3D flow conversion
    # ------------------------------------------------------------------

    def _world_to_camera_relative_flow(
        self,
        p_world_src: torch.Tensor,
        p_world_dst: torch.Tensor,
        extr_src: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a pair of world-space point clouds to camera-relative 3D flow.

        Computes ``p_cam_dst - p_cam_src`` where each point is expressed in
        its respective camera coordinate frame.

        Args:
            p_world_src: (H, W, 3) world positions at the source frame.
            p_world_dst: (H, W, 3) world positions at the destination frame.
            extr_src: (4, 4) cam-to-world extrinsics of the source frame.

        Returns:
            (H, W, 3) float32 camera-relative 3D displacement.
        """
        H, W = p_world_src.shape[:2]

        R_src = extr_src[:3, :3]
        t_src = extr_src[:3, 3]
        p_cam_src = (p_world_src.reshape(-1, 3) - t_src) @ R_src

        extr_dst = self.get_extrinsics()
        R_dst = extr_dst[:3, :3]
        t_dst = extr_dst[:3, 3]
        p_cam_dst = (p_world_dst.reshape(-1, 3) - t_dst) @ R_dst

        flow = (p_cam_dst - p_cam_src).reshape(H, W, 3)
        flow[~torch.isfinite(flow)] = 0.0
        return flow

    def world_to_camera_scene_flow(
        self, world_flow: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Convert adjacent-frame world-space scene flow to camera-relative.

        For each pixel the result is ``p_cam_{t+1} - p_cam_t`` where
        ``p_cam_t`` lives in the source camera frame and ``p_cam_{t+1}`` in
        the current camera frame.  Static pixels will have non-zero flow
        when the camera moves.

        Must be called after :meth:`compute_exact_scene_flow` and before
        :meth:`cache_scene_flow_frame` on the same step (so that the
        cached previous-frame data is still available).

        Args:
            world_flow: (H, W, 3) world-space 3D displacement (from
                :attr:`SceneFlowResult.scene_flow_3d`).

        Returns:
            (H, W, 3) camera-relative 3D displacement, or ``None`` if
            no previous frame data is cached.
        """
        if self._prev_points_world is None or self._prev_extrinsics is None:
            return None
        p_dst = self._prev_points_world + world_flow
        return self._world_to_camera_relative_flow(
            self._prev_points_world, p_dst, self._prev_extrinsics
        )

    def world_to_camera_anchor_flow(
        self, points_world_k: torch.Tensor, anchor_frame: int
    ) -> Optional[torch.Tensor]:
        """Convert anchor-frame world-space flow to camera-relative.

        For each anchor pixel the result is ``p_cam_k - p_cam_anchor``
        where ``p_cam_anchor`` lives in the anchor camera frame and
        ``p_cam_k`` in the current camera frame.

        Args:
            points_world_k: (H, W, 3) reconstructed world positions at the
                current frame (from :attr:`FirstFrameFlowResult.points_world_k`).
            anchor_frame: Which anchor frame to reference.

        Returns:
            (H, W, 3) camera-relative 3D displacement, or ``None`` if
            the anchor frame was not initialised.
        """
        data = self._ff_anchors.get(anchor_frame)
        if data is None:
            return None
        return self._world_to_camera_relative_flow(
            data.p0_world, points_world_k, data.extrinsics
        )

    # ------------------------------------------------------------------
    # True 2D optical flow (camera ego-motion + object motion)
    # ------------------------------------------------------------------

    def compute_true_optical_flow(
        self,
        scene_flow_3d: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Compute true 2D optical flow including camera ego-motion.

        For each pixel at the *previous* frame, projects its world-space
        position (optionally displaced by *scene_flow_3d*) into the *current*
        camera to obtain the (dx, dy) pixel displacement.  This captures
        both camera motion (parallax on static surfaces) and object motion.

        Must be called *after* :meth:`cache_scene_flow_frame` on the previous
        frame and :meth:`update` on the current frame.

        Args:
            scene_flow_3d: (H, W, 3) world-space displacement from the
                previous frame to the current frame (from
                :meth:`compute_exact_scene_flow`).  If ``None``, all pixels
                are treated as static (only camera ego-motion contributes).

        Returns:
            (H, W, 2) float32 tensor of (dx, dy) pixel displacement, or
            ``None`` if no previous frame has been cached.
        """
        if self._prev_points_world is None:
            return None

        H, W = self._prev_points_world.shape[:2]
        device = self._prev_points_world.device

        if scene_flow_3d is not None:
            pts_world = self._prev_points_world + scene_flow_3d
        else:
            pts_world = self._prev_points_world

        intrinsics = self.get_intrinsics()  # (3, 3)
        extrinsics = self.get_extrinsics()  # (4, 4) cam-to-world
        R = extrinsics[:3, :3]
        t_cam = extrinsics[:3, 3]

        # World → camera (inverse of cam-to-world: p_cam = (p_world - t) @ R)
        pts_cam = (pts_world.reshape(-1, 3) - t_cam.unsqueeze(0)) @ R  # (H*W, 3)

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        Z = pts_cam[:, 2]
        u_proj = (fx * pts_cam[:, 0] / Z + cx).reshape(H, W)
        v_proj = (fy * pts_cam[:, 1] / Z + cy).reshape(H, W)

        grid_v, grid_u = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        flow = torch.stack([u_proj - grid_u, v_proj - grid_v], dim=-1)

        valid = (
            self._prev_valid_mask
            if self._prev_valid_mask is not None
            else torch.ones(H, W, dtype=torch.bool, device=device)
        )
        behind = Z.reshape(H, W) <= 0
        flow[~valid | behind] = 0.0

        return flow

    # ------------------------------------------------------------------
    # First-frame-anchored 3D trajectory flow
    # ------------------------------------------------------------------

    def init_anchor_frame(self, env: Any, anchor_frame: int = 0) -> None:
        """Store per-pixel anchor state from the current frame.

        Can be called at multiple simulation steps to create additional
        anchors.  Each *anchor_frame* value can only be initialised once.

        For each pixel with finite depth the method records:

        * ``p0_world`` — the 3-D world position.
        * ``track_type`` — STATIC / RIGID / ARTICULATION / UNSUPPORTED.
        * ``trackable_mask`` — True for every type except UNSUPPORTED.
        * For RIGID / ARTICULATION pixels: the body-local anchor
          ``q0 = T_anchor^{-1} * p_anchor`` so that later frames can
          reconstruct ``p_k = T_k * q0``.
        * For STATIC pixels: the world point itself (immovable).

        Args:
            env: The gymnasium-wrapped IsaacLab environment.
            anchor_frame: Index of the simulation step being anchored.
        """
        if anchor_frame in self._ff_anchors:
            raise RuntimeError(f"Anchor frame {anchor_frame} already initialised")

        from isaaclab.utils.math import unproject_depth

        depth = self.get_depth()
        intrinsics = self.get_intrinsics()
        extrinsics = self.get_extrinsics()

        H, W = depth.shape
        device = depth.device

        points_cam = unproject_depth(depth, intrinsics)
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        points_world = (points_cam @ R.T) + t
        points_world = points_world.reshape(W, H, 3).permute(1, 0, 2).contiguous()

        valid_depth = torch.isfinite(depth)
        inst_ids, id_to_labels = self.get_instance_id_segmentation()
        scene = env.unwrapped.scene

        track_type = torch.full((H, W), TrackType.UNSUPPORTED, dtype=torch.uint8, device=device)
        local_points = torch.zeros(H, W, 3, dtype=torch.float32, device=device)
        rigid_keys = torch.full((H, W), -1, dtype=torch.int64, device=device)
        artic_keys = torch.full((H, W), -1, dtype=torch.int64, device=device)
        artic_body_idx = torch.full((H, W), -1, dtype=torch.int64, device=device)

        ff_rigid_key_to_name: Dict[int, str] = {}
        ff_artic_key_to_name: Dict[int, str] = {}

        rigid_pose_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        artic_pose_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}

        for uid in inst_ids.unique():
            uid_key = uid.item()
            if uid_key not in id_to_labels:
                continue

            label = id_to_labels[uid_key]
            prim_path = label.get("class", "") if isinstance(label, dict) else str(label)
            if not prim_path or prim_path.upper() in ("BACKGROUND", "UNLABELLED", "UNKNOWN", "INVALID"):
                pixel_mask = (inst_ids == uid) & valid_depth
                if pixel_mask.any():
                    track_type[pixel_mask] = TrackType.STATIC
                continue

            binding = self._resolve_tracking_binding(prim_path, scene)
            pixel_mask = (inst_ids == uid) & valid_depth
            if not pixel_mask.any():
                continue

            pts = points_world[pixel_mask]

            if binding[0] == "STATIC":
                track_type[pixel_mask] = TrackType.STATIC
                local_points[pixel_mask] = pts

            elif binding[0] == "RIGID":
                asset_name = binding[1]
                track_type[pixel_mask] = TrackType.RIGID
                key = hash(asset_name) & 0x7FFFFFFFFFFFFFFF
                ff_rigid_key_to_name[key] = asset_name
                rigid_keys[pixel_mask] = key

                if asset_name not in rigid_pose_cache:
                    obj = scene[asset_name]
                    rigid_pose_cache[asset_name] = (
                        obj.data.root_link_pos_w[0].clone(),
                        obj.data.root_link_quat_w[0].clone(),
                    )
                pos_t, quat_t = rigid_pose_cache[asset_name]
                q_local = quat_apply_inverse(
                    quat_t.unsqueeze(0).expand(pts.shape[0], -1), pts - pos_t.unsqueeze(0)
                )
                local_points[pixel_mask] = q_local

            elif binding[0] == "ARTICULATION":
                asset_name, body_idx = binding[1], binding[2]
                track_type[pixel_mask] = TrackType.ARTICULATION
                artic_key = hash(asset_name) & 0x7FFFFFFFFFFFFFFF
                ff_artic_key_to_name[artic_key] = asset_name
                artic_keys[pixel_mask] = artic_key
                artic_body_idx[pixel_mask] = body_idx

                cache_key = (asset_name, body_idx)
                if cache_key not in artic_pose_cache:
                    artic_obj = scene[asset_name]
                    artic_pose_cache[cache_key] = (
                        artic_obj.data.body_link_pos_w[0, body_idx].clone(),
                        artic_obj.data.body_link_quat_w[0, body_idx].clone(),
                    )
                pos_t, quat_t = artic_pose_cache[cache_key]
                q_local = quat_apply_inverse(
                    quat_t.unsqueeze(0).expand(pts.shape[0], -1), pts - pos_t.unsqueeze(0)
                )
                local_points[pixel_mask] = q_local

        self._ff_anchors[anchor_frame] = _AnchorFrameData(
            p0_world=points_world,
            extrinsics=extrinsics,
            trackable_mask=(track_type != TrackType.UNSUPPORTED) & valid_depth,
            track_type=track_type,
            local_points=local_points,
            rigid_keys=rigid_keys,
            artic_keys=artic_keys,
            artic_body_idx=artic_body_idx,
            rigid_key_to_name=ff_rigid_key_to_name,
            artic_key_to_name=ff_artic_key_to_name,
        )

    def compute_anchor_frame_flow(
        self,
        env: Any,
        anchor_frame: int = 0,
        *,
        occlusion_tol: float = 0.0001,
    ) -> FirstFrameFlowResult:
        """Compute flow from an anchor frame's pixels to the current frame k.

        For every trackable anchor pixel:

        * **STATIC**: ``p_k = p_anchor`` (no displacement).
        * **RIGID**: ``p_k = T_k * q_anchor``.
        * **ARTICULATION**: same as RIGID but using the per-body-link pose.

        Additionally computes visibility masks by projecting each ``p_k``
        into the current camera and comparing against the current depth map.

        Args:
            env: The gymnasium-wrapped IsaacLab environment.
            anchor_frame: Which anchor frame to compute flow from.
            occlusion_tol: Depth tolerance (metres) for the occlusion test.

        Returns:
            A :class:`FirstFrameFlowResult`.

        Raises:
            RuntimeError: If the anchor frame was not initialised via
                :meth:`init_anchor_frame`.
        """
        if anchor_frame not in self._ff_anchors:
            raise RuntimeError(
                f"Anchor frame {anchor_frame} not initialised. "
                "Call init_anchor_frame first."
            )

        data = self._ff_anchors[anchor_frame]
        H, W = data.p0_world.shape[:2]
        device = data.p0_world.device

        scene = env.unwrapped.scene
        trackable = data.trackable_mask.clone()

        p_k = data.p0_world.clone()

        # --- RIGID anchors ---
        rigid_mask = (data.track_type == TrackType.RIGID) & trackable
        if rigid_mask.any():
            for key, name in data.rigid_key_to_name.items():
                sel = rigid_mask & (data.rigid_keys == key)
                if not sel.any():
                    continue
                obj = scene[name]
                pos_k = obj.data.root_link_pos_w[0].clone()
                quat_k = obj.data.root_link_quat_w[0].clone()
                q_local = data.local_points[sel]
                p_k[sel] = quat_apply(
                    quat_k.unsqueeze(0).expand(q_local.shape[0], -1), q_local
                ) + pos_k.unsqueeze(0)

        # --- ARTICULATION anchors ---
        artic_mask = (data.track_type == TrackType.ARTICULATION) & trackable
        if artic_mask.any():
            for artic_key, name in data.artic_key_to_name.items():
                artic_obj = scene[name]
                num_bodies = artic_obj.data.body_link_pos_w.shape[1]
                for bi in range(num_bodies):
                    sel = artic_mask & (data.artic_keys == artic_key) & (data.artic_body_idx == bi)
                    if not sel.any():
                        continue
                    pos_k = artic_obj.data.body_link_pos_w[0, bi].clone()
                    quat_k = artic_obj.data.body_link_quat_w[0, bi].clone()
                    q_local = data.local_points[sel]
                    p_k[sel] = quat_apply(
                        quat_k.unsqueeze(0).expand(q_local.shape[0], -1), q_local
                    ) + pos_k.unsqueeze(0)

        flow_0k = p_k - data.p0_world
        # Pixels with non-finite world positions (e.g. infinite depth for
        # background) produce Inf - Inf = NaN; zero them out.
        flow_0k[~torch.isfinite(flow_0k)] = 0.0

        # --- Visibility / projection masks ---
        intrinsics = self.get_intrinsics()  # (3, 3)
        extrinsics = self.get_extrinsics()  # (4, 4) cam-to-world
        depth_k = self.get_depth()  # (H, W)

        R_inv = extrinsics[:3, :3].T
        t_inv = -(R_inv @ extrinsics[:3, 3])

        p_k_flat = p_k.reshape(-1, 3)
        p_cam = (p_k_flat @ R_inv.T) + t_inv  # (H*W, 3)
        p_cam = p_cam.reshape(H, W, 3)

        z_cam = p_cam[..., 2]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        u = (p_cam[..., 0] / z_cam) * fx + cx
        v = (p_cam[..., 1] / z_cam) * fy + cy

        # Half-pixel tolerance for floating-point precision at image edges.
        _PIX_EPS = 0.5
        in_frame = (
            (z_cam > 0)
            & (u >= -_PIX_EPS) & (u < W - 1 + _PIX_EPS)
            & (v >= -_PIX_EPS) & (v < H - 1 + _PIX_EPS)
        )

        # NaN from non-finite world points (e.g. infinite depth) would
        # produce undefined indices after .long(); replace with 0 so
        # clamp/long are well-defined.  in_frame already excludes them.
        u = torch.nan_to_num(u, nan=0.0)
        v = torch.nan_to_num(v, nan=0.0)

        # Sample depth at all 4 bilinear-cell corners to handle depth
        # discontinuities at object edges.
        u_fl = u.floor().clamp(0, W - 1).long()
        u_ce = (u_fl + 1).clamp(0, W - 1)
        v_fl = v.floor().clamp(0, H - 1).long()
        v_ce = (v_fl + 1).clamp(0, H - 1)

        d00 = depth_k[v_fl, u_fl]
        d01 = depth_k[v_fl, u_ce]
        d10 = depth_k[v_ce, u_fl]
        d11 = depth_k[v_ce, u_ce]

        min_depth_diff = torch.min(
            torch.min(torch.abs(z_cam - d00), torch.abs(z_cam - d01)),
            torch.min(torch.abs(z_cam - d10), torch.abs(z_cam - d11)),
        )

        visible_now = in_frame & (min_depth_diff < occlusion_tol)

        return FirstFrameFlowResult(
            flow3d_from_first=flow_0k,
            trackable_mask=trackable,
            in_frame_mask=in_frame & trackable,
            visible_now_mask=visible_now & trackable,
            points_world_k=p_k,
        )

    # ------------------------------------------------------------------
    # Prim-path → tracking binding resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_tracking_binding(
        prim_path: str, scene: Any
    ) -> Tuple[str, ...]:
        """Classify a USD prim path into a tracking category.

        Returns one of:
            ("STATIC",)
            ("RIGID", asset_name)
            ("ARTICULATION", asset_name, body_idx)
            ("UNSUPPORTED",)
        """
        path_parts = prim_path.strip("/").split("/")

        for name in scene.rigid_objects.keys():
            if name in path_parts:
                return ("RIGID", name)

        for name, artic in scene.articulations.items():
            if name in path_parts:
                body_names = artic.data.body_names
                body_idx = _find_body_index_for_prim(prim_path, body_names)
                if body_idx is not None:
                    return ("ARTICULATION", name, body_idx)
                return ("ARTICULATION", name, 0)

        # Prim exists but doesn't belong to any tracked dynamic asset —
        # treat it as static world geometry (zero displacement).
        return ("STATIC",)

    # ------------------------------------------------------------------
    # Per-object semantic info
    # ------------------------------------------------------------------

    def _get_object_id(self, rgba: tuple) -> int:
        """Delegate to the shared registry."""
        return self._registry.get_object_id(rgba)

    def get_semantic_info(self) -> List[Dict[str, Any]]:
        """Returns per-object metadata for every semantic class visible in this frame.

        For each distinct RGBA colour present in the segmentation image, reports:
        - ``object_id``: temporally consistent integer ID (same object → same ID
          across frames).
        - ``object_name``: stable name, e.g. ``"obj_0"``, ``"obj_1"``, ...
          If the USD prim has a ``semanticLabel`` the label is appended:
          ``"obj_0_cracker_box"``.
        - ``rgba``: the RGBA colour tuple.
        - ``class_name``: raw semantic class from USD (may be ``""`` or
          ``"BACKGROUND"``).
        - ``pixel_count``: number of pixels belonging to this object.

        Returns:
            List of dicts, one per visible object, sorted by descending pixel count.
        """
        seg_data, id_to_labels = self.get_semantic_segmentation()

        rgba_to_class: Dict[tuple, str] = {}
        for key, val in id_to_labels.items():
            rgba_tuple = tuple(eval(key)) if isinstance(key, str) else key
            rgba_to_class[rgba_tuple] = val.get("class", "")

        seg_np = seg_data.cpu().numpy()
        flat_seg = seg_np.reshape(-1, 4)
        unique_colors = np.unique(flat_seg, axis=0)

        results: List[Dict[str, Any]] = []
        for color in unique_colors:
            color_tuple = tuple(int(c) for c in color)
            mask = (seg_np == color.reshape(1, 1, 4)).all(axis=-1)
            pixel_count = int(mask.sum())
            if pixel_count == 0:
                continue

            object_id = self._get_object_id(color_tuple)
            class_name = rgba_to_class.get(color_tuple, "")

            # Build a human-readable, temporally consistent name
            if class_name and class_name.upper() not in ("UNKNOWN", "UNLABELLED", "BACKGROUND"):
                object_name = f"obj_{object_id}_{class_name}"
            else:
                object_name = f"obj_{object_id}"

            results.append({
                "object_id": object_id,
                "object_name": object_name,
                "rgba": color_tuple,
                "class_name": class_name,
                "pixel_count": pixel_count,
            })

        results.sort(key=lambda x: x["pixel_count"], reverse=True)
        return results


# ------------------------------------------------------------------
# Helper: find body index for articulation prim path
# ------------------------------------------------------------------


def _find_body_index_for_prim(prim_path: str, body_names: list[str]) -> int | None:
    """Find which articulation body a prim path belongs to.

    Matches the last body name that appears as a path component.
    """
    path_parts = prim_path.strip("/").split("/")
    best_idx: int | None = None
    best_depth = -1
    for idx, bname in enumerate(body_names):
        for depth, part in enumerate(path_parts):
            if part == bname and depth > best_depth:
                best_idx = idx
                best_depth = depth
    return best_idx


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def create_static_camera(
    position: Tuple[float, float, float],
    target: Tuple[float, float, float],
    width: int = 640,
    height: int = 480,
    focal_length: float = 24.0,
    prim_path: str = "/World/StaticCamera",
    instance_registry: Optional[ObjectInstanceRegistry] = None,
) -> IsaacLabArenaCameraHandler:
    """Create and initialise a static camera sensor looking from *position* at *target*.

    Must be called **after** the simulation app has been launched and the
    environment has been reset so that the USD stage is ready.

    The camera is configured with all sensor data types needed for 3D
    reconstruction data generation: RGB, depth, normals, motion vectors
    (optical flow), and semantic segmentation.

    Args:
        position: Camera position in world frame (x, y, z).
        target: Look-at point in world frame (x, y, z).
        width: Image width in pixels.
        height: Image height in pixels.
        focal_length: Focal length in mm.
        prim_path: USD prim path where the camera will be spawned.
        instance_registry: Optional shared :class:`ObjectInstanceRegistry`.
            Pass the same instance to every camera so that object IDs,
            names, and colors are consistent across viewpoints.

    Returns:
        An initialised :class:`IsaacLabArenaCameraHandler`.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    quat_wxyz = _look_at_quaternion_ros(position, target)

    cfg = CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=height,
        width=width,
        data_types=ALL_DATA_TYPES,
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=position,
            rot=quat_wxyz,
            convention="ros",
        ),
    )

    camera = Camera(cfg)
    camera._initialize_callback(None)
    camera.reset([0])
    return IsaacLabArenaCameraHandler(camera, instance_registry=instance_registry)


def _look_at_quaternion_ros(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Tuple[float, float, float, float]:
    """Compute a quaternion (w, x, y, z) for a ROS-convention camera at *eye* looking at *target*.

    ROS camera convention: x = right, y = down, z = forward (optical axis).
    """
    eye_arr = np.array(eye, dtype=np.float64)
    target_arr = np.array(target, dtype=np.float64)
    up_arr = np.array(up, dtype=np.float64)

    forward = target_arr - eye_arr
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up_arr)
    right /= np.linalg.norm(right)

    down = np.cross(forward, right)

    R_cam_to_world = np.column_stack([right, down, forward])
    quat_tensor = quat_from_matrix(torch.from_numpy(R_cam_to_world).float())
    return tuple(quat_tensor.tolist())
