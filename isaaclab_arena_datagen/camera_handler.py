# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import torch
from typing import Any

import isaaclab.sim as sim_utils
import omni.usd
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix
from pxr import Gf, UsdGeom

from isaaclab_arena_datagen.geometry.rotation import Rotation
from isaaclab_arena_datagen.geometry.transform_se3 import TransformSE3
from isaaclab_arena_datagen.geometry.translation import Translation

from .object_registry import InstanceKey, ObjectInstanceRegistry, ObjectType
from .scene_flow import BACKGROUND_LABELS, SceneFlowComputer, SceneFlowResult

ALL_DATA_TYPES = [
    "rgb",
    "distance_to_image_plane",
    "normals",
    "motion_vectors",
    "semantic_segmentation",
    "instance_id_segmentation_fast",
]


class IsaacLabArenaCameraHandler:
    """Handles a static camera sensor in an IsaacLab Arena environment.

    Wraps an IsaacLab Camera sensor and provides methods to extract
    RGB, depth, normals, optical flow, semantic segmentation,
    intrinsic matrix, and camera-to-world extrinsic transform.

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
        instance_registry: ObjectInstanceRegistry | None = None,
    ):
        """Initialize the camera handler wrapping an Isaac Lab camera sensor."""
        self._camera = camera
        self._camera_name = camera_name
        self._registry = instance_registry or ObjectInstanceRegistry()

        # Override for get_T_W_from_C() when camera is moved via set_world_pose
        # (the sensor's pos_w / quat_w_ros do not update after USD prim changes)
        self._override_T_W_from_C: TransformSE3 | None = None

        # Cached USD transform op for set_world_pose(); created on first use.
        self._xform_op: Any = None

        # Scene flow computation (composition, not inheritance)
        self._scene_flow = SceneFlowComputer()

    @property
    def camera_name(self) -> str:
        """Return the camera identifier string."""
        return self._camera_name

    def _get_camera_output(self) -> dict[str, torch.Tensor]:
        return convert_dict_to_backend(self._camera.data.output, backend="torch")  # type: ignore[no-any-return]

    def update(self, dt: float) -> None:
        """Read the latest render data from the camera sensor."""
        self._camera.update(dt)

    def set_world_pose(
        self,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> None:
        """Move the camera to *position* and orient it to look at *target*.

        Updates the USD prim transform so the next :meth:`update` call
        renders from the new viewpoint.  Call once per step *before*
        :meth:`update` for each step where the camera should move.
        """
        eye_W_3 = np.array(position, dtype=np.float64)
        target_W_3 = np.array(target, dtype=np.float64)
        up_W_3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        forward_W_3 = target_W_3 - eye_W_3
        forward_norm = np.linalg.norm(forward_W_3)
        if forward_norm < 1e-12:
            raise ValueError("Camera position and target are identical; cannot compute look-at direction.")
        forward_W_3 /= forward_norm

        # If the view direction is (anti-)parallel to the world up axis (e.g. a
        # straight-down or straight-up camera), the right axis is undefined; fall
        # back to a different up reference so the look-at basis stays well-conditioned.
        if abs(float(np.dot(forward_W_3, up_W_3))) > 0.999:
            up_W_3 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        right_W_3 = np.cross(forward_W_3, up_W_3)
        right_norm = np.linalg.norm(right_W_3)
        if right_norm < 1e-12:
            raise ValueError("Camera forward direction is parallel to the up vector; cannot compute right axis.")
        right_W_3 /= right_norm

        cam_up_W_3 = np.cross(right_W_3, forward_W_3)

        # USD camera convention: x = right, y = up, z = backward.
        # Row-vector convention: v_world = v_local * M, so the rows of M
        # are the camera basis vectors expressed in the world frame.
        mat = Gf.Matrix4d(
            float(right_W_3[0]),
            float(right_W_3[1]),
            float(right_W_3[2]),
            0.0,
            float(cam_up_W_3[0]),
            float(cam_up_W_3[1]),
            float(cam_up_W_3[2]),
            0.0,
            float(-forward_W_3[0]),
            float(-forward_W_3[1]),
            float(-forward_W_3[2]),
            0.0,
            float(eye_W_3[0]),
            float(eye_W_3[1]),
            float(eye_W_3[2]),
            1.0,
        )

        prim_path = self._camera.cfg.prim_path
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)

        if self._xform_op is None:
            xformable.ClearXformOpOrder()
            self._xform_op = xformable.AddTransformOp()

        self._xform_op.Set(mat)

        # Cache ROS-convention cam-to-world extrinsics so that
        # get_T_W_from_C() returns the correct pose even though
        # the camera sensor's internal pos_w / quat_w_ros do not
        # update after direct USD prim changes.
        # ROS camera axes: x = right, y = down, z = forward.
        rotation_W_from_C = torch.zeros(3, 3, dtype=torch.float32)
        rotation_W_from_C[0, 0], rotation_W_from_C[1, 0], rotation_W_from_C[2, 0] = (
            float(right_W_3[0]),
            float(right_W_3[1]),
            float(right_W_3[2]),
        )
        rotation_W_from_C[0, 1], rotation_W_from_C[1, 1], rotation_W_from_C[2, 1] = (
            float(-cam_up_W_3[0]),
            float(-cam_up_W_3[1]),
            float(-cam_up_W_3[2]),
        )
        rotation_W_from_C[0, 2], rotation_W_from_C[1, 2], rotation_W_from_C[2, 2] = (
            float(forward_W_3[0]),
            float(forward_W_3[1]),
            float(forward_W_3[2]),
        )
        translation_W_from_C = torch.tensor(
            [float(eye_W_3[0]), float(eye_W_3[1]), float(eye_W_3[2])], dtype=torch.float32
        )
        self._override_T_W_from_C = TransformSE3(
            rotation=Rotation(R=rotation_W_from_C.unsqueeze(0)),
            translation=Translation(t=translation_W_from_C.unsqueeze(0)),
        )

    # ------------------------------------------------------------------
    # Core outputs
    # ------------------------------------------------------------------

    def get_rgb(self) -> torch.Tensor:
        """Returns RGB image as a (H, W, 3) uint8 tensor."""
        rgb_hw3 = self._get_camera_output()["rgb"].clone()
        assert rgb_hw3.shape[0] == 1, "Expected single environment"
        assert rgb_hw3.shape[-1] == 3, "Expected 3-channel RGB"
        return rgb_hw3.squeeze(0)

    def get_depth(self) -> torch.Tensor:
        """Returns depth map as a (H, W) float32 tensor (distance to image plane)."""
        depth_hw = self._get_camera_output()["distance_to_image_plane"].clone()
        assert depth_hw.shape[0] == 1, "Expected single environment"
        return depth_hw.squeeze()

    def get_intrinsics(self) -> torch.Tensor:
        """Returns the 3x3 camera intrinsic matrix."""
        assert self._camera.data.intrinsic_matrices.shape[0] == 1, "Expected single environment"
        return self._camera.data.intrinsic_matrices.data[0].clone()

    def get_T_W_from_C(self) -> TransformSE3:
        """Returns the camera-to-world transform as a TransformSE3.

        If the camera was repositioned via :meth:`set_world_pose`, the
        override extrinsics are returned (the sensor's internal ``pos_w``
        / ``quat_w_ros`` do not update after direct USD prim changes).
        """
        if self._override_T_W_from_C is not None:
            device = self._camera.data.pos_w.device
            return self._override_T_W_from_C.to(device)

        assert self._camera.data.pos_w.shape[0] == 1, "Expected single environment"
        translation_W_from_C = self._camera.data.pos_w.data[0].clone()
        quat_W_from_C_4 = self._camera.data.quat_w_ros.data[0].clone()

        quat_4 = quat_W_from_C_4.reshape(4)
        rotation_W_from_C = matrix_from_quat(quat_4)
        return TransformSE3(
            rotation=Rotation(R=rotation_W_from_C.unsqueeze(0)),
            translation=Translation(t=translation_W_from_C.unsqueeze(0)),
        )

    # ------------------------------------------------------------------
    # Normals
    # ------------------------------------------------------------------

    def get_normals(self) -> torch.Tensor:
        """Returns surface normals as a (H, W, 3) float32 tensor (x, y, z)."""
        normals_hw3 = self._get_camera_output()["normals"].clone()
        assert normals_hw3.shape[0] == 1, "Expected single environment"
        assert normals_hw3.shape[-1] == 3, "Expected 3-channel normals"
        return normals_hw3.squeeze(0)

    # ------------------------------------------------------------------
    # Optical flow (motion vectors)
    # ------------------------------------------------------------------

    def get_optical_flow(self) -> torch.Tensor:
        """Returns dense optical flow as a (H, W, 2) float32 tensor (dx, dy in pixels)."""
        motion_vectors_hw2 = self._get_camera_output()["motion_vectors"].clone()
        assert motion_vectors_hw2.shape[0] == 1, "Expected single environment"
        assert motion_vectors_hw2.shape[-1] == 2, "Expected 2-channel motion vectors"
        return motion_vectors_hw2.squeeze(0)

    # ------------------------------------------------------------------
    # Semantic segmentation
    # ------------------------------------------------------------------

    def get_semantic_segmentation(self) -> tuple[torch.Tensor, dict]:
        """Returns semantic segmentation data.

        Returns:
            Tuple of:
                - (H, W, 4) uint8 tensor with RGBA segmentation colors.
                - dict mapping RGBA id-strings to label dicts (``{"class": ...}``).
        """
        seg_rgba_hw4 = self._get_camera_output()["semantic_segmentation"].clone()
        id_to_labels = self._camera.data.info[0]["semantic_segmentation"]["idToLabels"]
        assert seg_rgba_hw4.shape[0] == 1, "Expected single environment"
        assert seg_rgba_hw4.shape[-1] == 4, "Expected 4-channel segmentation (RGBA)"
        return seg_rgba_hw4.squeeze(0).to(torch.uint8), id_to_labels

    # ------------------------------------------------------------------
    # Instance ID segmentation
    # ------------------------------------------------------------------

    def get_instance_id_segmentation(self) -> tuple[torch.Tensor, dict]:
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
    def _instance_object_key_from_binding(binding: tuple[str, ...], prim_path: str) -> InstanceKey:
        """Convert tracking binding to a stable per-object semantic key."""
        kind = binding[0]
        if kind == "RIGID":
            return InstanceKey(ObjectType.RIGID, binding[1])
        if kind == "ARTICULATION":
            return InstanceKey(ObjectType.ARTICULATION, binding[1])
        if kind == "STATIC":
            if not prim_path or prim_path.upper() in BACKGROUND_LABELS:
                return InstanceKey(ObjectType.STATIC, "background")
            return InstanceKey(ObjectType.STATIC, prim_path)
        if not prim_path:
            return InstanceKey(ObjectType.UNSUPPORTED, "unknown")
        return InstanceKey(ObjectType.UNSUPPORTED, prim_path)

    def _get_or_create_instance_identity(self, instance_key: InstanceKey) -> tuple[int, str, tuple[int, int, int, int]]:
        """Delegate to the shared registry."""
        return self._registry.get_or_create_instance_identity(instance_key)

    def get_object_instance_segmentation(self, env: Any) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Build a semantic-style RGBA image where each tracked object has a unique label.

        Args:
            env: The gymnasium-wrapped IsaacLab environment.

        Returns:
            Tuple of:
                - (H, W, 4) uint8 RGBA object-instance segmentation.
                - List of metadata dicts for visible objects in this frame.
        """
        instance_ids_hw, id_to_labels = self.get_instance_id_segmentation()
        H, W = instance_ids_hw.shape
        device = instance_ids_hw.device
        scene = env.unwrapped.scene

        seg_rgba_hw4 = torch.zeros(H, W, 4, dtype=torch.uint8, device=device)
        object_rows: dict[int, dict[str, Any]] = {}

        for instance_id in instance_ids_hw.unique():
            uid_key = instance_id.item()
            mask_hw = instance_ids_hw == instance_id
            if not mask_hw.any():
                continue

            label = id_to_labels.get(uid_key, "")
            prim_path = label.get("class", "") if isinstance(label, dict) else str(label)
            if not prim_path or prim_path.upper() in BACKGROUND_LABELS:
                binding: tuple[str, ...] = ("STATIC",)
            else:
                binding = self._resolve_tracking_binding(prim_path, scene)

            instance_key = self._instance_object_key_from_binding(binding, prim_path)
            object_id, object_name, rgba = self._get_or_create_instance_identity(instance_key)
            seg_rgba_hw4[mask_hw] = torch.tensor(rgba, dtype=torch.uint8, device=device)

            if object_id not in object_rows:
                object_rows[object_id] = {
                    "object_id": object_id,
                    "object_name": object_name,
                    "rgba": rgba,
                    "class_name": instance_key.kind.label,
                    "track_type": instance_key.kind.name,
                    "asset_name": instance_key.asset_name,
                    "pixel_count": 0,
                    "visible_body_indices": set(),
                }
            object_rows[object_id]["pixel_count"] += int(mask_hw.sum().item())
            if binding[0] == "ARTICULATION" and len(binding) > 2:
                object_rows[object_id]["visible_body_indices"].add(binding[2])

        for row in object_rows.values():
            row["visible_body_indices"] = sorted(row["visible_body_indices"])

        semantic_info = sorted(object_rows.values(), key=lambda x: x["pixel_count"], reverse=True)
        return seg_rgba_hw4, semantic_info

    # ------------------------------------------------------------------
    # Prim-path -> tracking binding resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_tracking_binding(prim_path: str, scene: Any) -> tuple[Any, ...]:
        """Classify a USD prim path into a tracking category.

        Returns one of:
            ("STATIC",)
            ("RIGID", asset_name)
            ("ARTICULATION", asset_name, body_idx)
        """
        path_parts_lower = [p.lower() for p in prim_path.strip("/").split("/")]

        for name in scene.rigid_objects.keys():
            if name.lower() in path_parts_lower:
                return ("RIGID", name)

        for name, articulation_obj in scene.articulations.items():
            if name.lower() in path_parts_lower:
                body_names = list(articulation_obj.data.body_names)
                body_idx = _find_body_index_for_prim(prim_path, body_names)
                if body_idx is not None:
                    return ("ARTICULATION", name, body_idx)
                return ("ARTICULATION", name, 0)

        return ("STATIC",)

    # ------------------------------------------------------------------
    # Scene flow delegation (composition)
    # ------------------------------------------------------------------

    def reset_scene_flow(self) -> None:
        """Clear cached scene-flow state (call at an episode boundary).

        Ensures the next frame computes no flow against a frame from a previous
        episode (which would be a spurious jump across the scene reset).
        """
        self._scene_flow.reset()

    def close(self) -> None:
        """Detach the camera's replicator annotators and remove its prim.

        Call when the camera is no longer needed (e.g. between eval-runner jobs).
        Standalone cameras are not owned by the env's sensor manager, so without
        this their ``rgb``/etc. annotators linger in the global replicator
        registry attached to a torn-down render product and clash with cameras
        created for the next job (``Annotator rgb is not attached to any render
        products``). Mirrors :meth:`isaaclab.sensors.Camera.__del__`.
        """
        camera = self._camera
        try:
            for annotators in camera._rep_registry.values():  # noqa: SLF001
                for annotator, render_product_path in zip(annotators, camera._render_product_paths):  # noqa: SLF001
                    annotator.detach([render_product_path])
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"[datagen] Warning: failed to detach camera annotators for {self._camera_name}: {exc}")
        try:
            stage = omni.usd.get_context().get_stage()
            if stage is not None and stage.GetPrimAtPath(camera.cfg.prim_path).IsValid():
                stage.RemovePrim(camera.cfg.prim_path)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"[datagen] Warning: failed to remove camera prim {camera.cfg.prim_path}: {exc}")

    def cache_scene_flow_frame(self, env: Any) -> None:
        """Cache per-pixel world points and tracking anchors for this frame.

        Args:
            env: The gymnasium-wrapped IsaacLab environment.
        """
        instance_ids_hw, id_to_labels = self.get_instance_id_segmentation()
        self._scene_flow.cache_frame(
            depth_hw=self.get_depth(),
            intrinsics_33=self.get_intrinsics(),
            T_W_from_C=self.get_T_W_from_C(),
            instance_ids_hw=instance_ids_hw,
            id_to_labels=id_to_labels,
            scene=env.unwrapped.scene,
            resolve_binding_fn=self._resolve_tracking_binding,
        )

    def compute_exact_scene_flow(self, env: Any) -> SceneFlowResult | None:
        """Compute exact adjacent-frame 3D scene flow for the *previous* frame.

        Args:
            env: The gymnasium-wrapped IsaacLab environment.
        """
        return self._scene_flow.compute_exact_flow(env.unwrapped.scene)

    def convert_scene_flow_W_to_C(self, flow_W_hw3: torch.Tensor) -> torch.Tensor | None:
        """Convert adjacent-frame world-space scene flow to camera-relative.

        Args:
            flow_W_hw3: (H, W, 3) world-space 3D displacement.
        """
        return self._scene_flow.convert_scene_flow_W_to_C(flow_W_hw3, self.get_T_W_from_C())

    def compute_true_optical_flow(self, scene_flow_W_hw3: torch.Tensor | None = None) -> torch.Tensor | None:
        """Compute true 2D optical flow including camera ego-motion.

        Args:
            scene_flow_W_hw3: (H, W, 3) world-space displacement, or ``None``.
        """
        return self._scene_flow.compute_true_optical_flow(
            self.get_intrinsics(), self.get_T_W_from_C(), scene_flow_W_hw3
        )

    # ------------------------------------------------------------------
    # Semantic info (legacy colour-based)
    # ------------------------------------------------------------------

    def _get_object_id(self, rgba: tuple) -> int:
        """Delegate to the shared registry's colour-based ID assignment."""
        return self._registry.get_object_id(rgba)

    def get_semantic_info(self) -> list[dict[str, Any]]:
        """Parse semantic segmentation into a per-object summary.

        Returns:
            List of dicts, sorted by pixel count (descending).
        """
        seg_rgba_hw4, _id_to_labels = self.get_semantic_segmentation()

        seg_flat_n4 = seg_rgba_hw4.reshape(-1, 4)
        unique_colors = seg_flat_n4.unique(dim=0)

        results: list[dict[str, Any]] = []
        for color in unique_colors:
            color_tuple = tuple(color.tolist())
            mask_n = (seg_flat_n4 == color.unsqueeze(0)).all(dim=1)
            pixel_count = int(mask_n.sum().item())

            object_id = self._get_object_id(color_tuple)

            class_name = "unknown"

            object_name = class_name
            if class_name.upper() in BACKGROUND_LABELS or class_name == "unknown":
                object_name = "background"
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
    for idx, body_name in enumerate(body_names):
        for depth, part in enumerate(path_parts):
            if part == body_name and depth > best_depth:
                best_idx = idx
                best_depth = depth
    return best_idx


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def create_static_camera(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
    width: int = 640,
    height: int = 480,
    focal_length: float = 24.0,
    prim_path: str = "/World/StaticCamera",
    instance_registry: ObjectInstanceRegistry | None = None,
) -> IsaacLabArenaCameraHandler:
    """Create and initialise a static camera sensor looking from *position* at *target*.

    Must be called **after** the simulation app has been launched and the
    environment has been reset so that the USD stage is ready.

    Args:
        position: Camera position in world frame (x, y, z).
        target: Look-at point in world frame (x, y, z).
        width: Image width in pixels.
        height: Image height in pixels.
        focal_length: Focal length in mm.
        prim_path: USD prim path where the camera will be spawned.
        instance_registry: Optional shared :class:`ObjectInstanceRegistry`.

    Returns:
        An initialised :class:`IsaacLabArenaCameraHandler`.
    """
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
    camera._initialize_callback(None)  # pylint: disable=protected-access  # no public init API
    camera.reset([0])
    return IsaacLabArenaCameraHandler(camera, instance_registry=instance_registry)


def _look_at_quaternion_ros(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Compute a quaternion (w, x, y, z) for a ROS-convention camera at *eye* looking at *target*.

    ROS camera convention: x = right, y = down, z = forward (optical axis).
    """
    eye_W_3 = np.array(eye, dtype=np.float64)
    target_W_3 = np.array(target, dtype=np.float64)
    up_W_3 = np.array(up, dtype=np.float64)

    forward_W_3 = target_W_3 - eye_W_3
    forward_norm = np.linalg.norm(forward_W_3)
    if forward_norm < 1e-12:
        raise ValueError("Eye and target are identical; cannot compute look-at direction.")
    forward_W_3 /= forward_norm

    # If the view direction is (anti-)parallel to the up axis (e.g. a straight-down
    # or straight-up camera), the right axis is undefined; fall back to a different
    # up reference so the look-at basis stays well-conditioned.
    if abs(float(np.dot(forward_W_3, up_W_3))) > 0.999:
        up_W_3 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right_W_3 = np.cross(forward_W_3, up_W_3)
    right_norm = np.linalg.norm(right_W_3)
    if right_norm < 1e-12:
        raise ValueError("Forward direction is parallel to the up vector; cannot compute right axis.")
    right_W_3 /= right_norm

    down_W_3 = np.cross(forward_W_3, right_W_3)

    rotation_W_from_C = np.column_stack([right_W_3, down_W_3, forward_W_3])
    quat_W_from_C_4 = quat_from_matrix(torch.from_numpy(rotation_W_from_C).float())
    return tuple(quat_W_from_C_4.tolist())
