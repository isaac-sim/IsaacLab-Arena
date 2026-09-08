# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Viewport video recording that preserves Arena's configured viewer frame."""

from __future__ import annotations

import numpy as np
from typing import Literal

from isaaclab.envs.utils.video_recorder import VideoRecorder
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.utils.configclass import configclass

_CAMERA_INTRINSIC_ATTRIBUTES = (
    "projection",
    "horizontalAperture",
    "verticalAperture",
    "horizontalApertureOffset",
    "verticalApertureOffset",
    "focalLength",
    "clippingRange",
    "clippingPlanes",
    "fStop",
    "focusDistance",
    "stereoRole",
    "shutter:open",
    "shutter:close",
    "exposure",
)


def _define_camera_with_matching_intrinsics(stage, source_path: str, destination_path: str) -> None:
    """Define a camera whose optical properties match an existing camera."""
    source_prim = stage.GetPrimAtPath(source_path)
    destination_prim = stage.DefinePrim(destination_path, "Camera")
    if not source_prim.IsValid():
        return

    for attribute_name in _CAMERA_INTRINSIC_ATTRIBUTES:
        source_attribute = source_prim.GetAttribute(attribute_name)
        if not source_attribute.IsValid():
            continue
        value = source_attribute.Get()
        if value is not None:
            destination_prim.GetAttribute(attribute_name).Set(value)


def _as_numpy_xyz(value) -> np.ndarray:
    """Convert an Isaac Lab array-like XYZ value to a CPU NumPy array."""
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=float)


def resolve_viewer_camera_pose(
    scene,
    cfg: ArenaViewportVideoRecorderCfg,
    eye: np.ndarray,
    lookat: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Resolve viewer-relative eye and target coordinates into the simulation world frame.

    Args:
        scene: Interactive scene containing environment origins and tracked assets.
        cfg: Arena viewport video recorder configuration.
        eye: Camera eye coordinates relative to the configured viewer origin.
        lookat: Camera target coordinates relative to the configured viewer origin.

    Returns:
        World-frame camera eye and target coordinates.
    """
    origin_type = cfg.viewer_origin_type
    if origin_type == "world":
        origin = np.zeros(3, dtype=float)
    else:
        assert 0 <= cfg.viewer_env_index < len(scene.env_origins), (
            f"Viewer environment index {cfg.viewer_env_index} is outside the available range "
            f"[0, {len(scene.env_origins) - 1}]."
        )
        if origin_type == "env":
            origin = _as_numpy_xyz(scene.env_origins[cfg.viewer_env_index])
        else:
            assert cfg.viewer_asset_name is not None, f"Viewer origin type '{origin_type}' requires an asset name."
            asset = scene[cfg.viewer_asset_name]
            if origin_type == "asset_root":
                root_pos_w = getattr(asset.data.root_pos_w, "torch", asset.data.root_pos_w)
                origin = _as_numpy_xyz(root_pos_w[cfg.viewer_env_index])
            else:
                assert origin_type == "asset_body", f"Unsupported viewer origin type: '{origin_type}'."
                assert cfg.viewer_body_name is not None, "Viewer origin type 'asset_body' requires a body name."
                body_ids, _ = asset.find_bodies(cfg.viewer_body_name)
                assert (
                    len(body_ids) == 1
                ), f"Viewer body pattern '{cfg.viewer_body_name}' must resolve to exactly one body; got {body_ids}."
                body_pos_w = getattr(asset.data.body_pos_w, "torch", asset.data.body_pos_w)
                origin = _as_numpy_xyz(body_pos_w[cfg.viewer_env_index, body_ids[0]])

    world_eye = origin + eye
    world_lookat = origin + lookat
    return tuple(float(x) for x in world_eye), tuple(float(x) for x in world_lookat)


class ArenaViewportVideoRecorder(VideoRecorder):
    """Record an Arena viewer pose without moving the interactive Kit viewport camera."""

    cfg: ArenaViewportVideoRecorderCfg

    def __init__(self, cfg: ArenaViewportVideoRecorderCfg, scene):
        # Isaac Lab may overwrite cfg.eye/lookat from a visualizer configuration during construction.
        # Preserve the task ViewerCfg values copied into the recorder before that happens.
        self._viewer_eye = np.asarray(cfg.eye, dtype=float)
        self._viewer_lookat = np.asarray(cfg.lookat, dtype=float)
        super().__init__(cfg, scene)
        cfg.eye = tuple(float(x) for x in self._viewer_eye)
        cfg.lookat = tuple(float(x) for x in self._viewer_lookat)

        # The report camera follows ViewerCfg, not a visualizer's stale construction-time camera.
        # In particular, disable VideoRecorder's per-frame Newton visualizer synchronization.
        self._matched_visualizer = None

        if self._backend == "kit":
            _define_camera_with_matching_intrinsics(
                scene.stage,
                source_path=self._capture.cfg.camera_prim_path,
                destination_path=cfg.camera_prim_path,
            )
            self._capture.cfg.camera_prim_path = cfg.camera_prim_path

    def render_rgb_array(self) -> np.ndarray | None:
        """Render the report camera after resolving its configured origin into world coordinates."""
        if self._backend is None or self._capture is None:
            return None

        eye, lookat = resolve_viewer_camera_pose(
            self._scene,
            self.cfg,
            self._viewer_eye,
            self._viewer_lookat,
        )
        self.cfg.eye = eye
        self.cfg.lookat = lookat

        if self._backend == "kit":
            self._capture.cfg.eye = eye
            self._capture.cfg.lookat = lookat
            from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view

            set_kit_renderer_camera_view(
                eye=eye,
                target=lookat,
                camera_prim_path=self.cfg.camera_prim_path,
            )
        else:
            self._capture.update_camera(eye, lookat)

        return super().render_rgb_array()


@configclass
class ArenaViewportVideoRecorderCfg(VideoRecorderCfg):
    """Configure Arena's frame-aware, viewport-isolated report video recorder."""

    class_type: type = ArenaViewportVideoRecorder

    viewer_origin_type: Literal["world", "env", "asset_root", "asset_body"] = "world"
    """Frame in which the configured camera eye and target are expressed."""

    viewer_env_index: int = 0
    """Environment supplying the viewer origin."""

    viewer_asset_name: str | None = None
    """Asset supplying the viewer origin for asset tracking modes."""

    viewer_body_name: str | None = None
    """Body supplying the viewer origin for body tracking mode."""

    camera_prim_path: str = "/World/ArenaReportCamera"
    """Dedicated Kit camera prim used for report recording."""
