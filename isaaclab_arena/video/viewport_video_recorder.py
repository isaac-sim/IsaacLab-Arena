# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Viewport video recording that preserves Arena's configured viewer frame."""

from __future__ import annotations

import contextlib
import numpy as np
import os
from typing import Literal

from isaaclab.envs.utils.video_recorder import VideoRecorder
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.utils.configclass import configclass


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


def _define_report_camera(stage, camera_prim_path: str, env_index: int):
    """Define the report camera and copy the selected environment's RTX scene partition."""
    from pxr import Sdf, UsdGeom

    camera_prim = UsdGeom.Camera.Define(stage, camera_prim_path).GetPrim()
    env_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_index}")
    env_partition_attr = env_prim.GetAttribute("primvars:omni:scenePartition")
    if env_partition_attr.IsValid() and env_partition_attr.Get() is not None:
        camera_prim.CreateAttribute(
            "omni:scenePartition",
            Sdf.ValueTypeNames.Token,
            custom=True,
        ).Set(env_partition_attr.Get())
    return camera_prim


def _set_report_camera_pose(
    stage,
    camera_prim_path: str,
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
) -> None:
    """Author a world-frame eye and target directly on the report camera USD prim."""
    import torch

    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
    from pxr import Gf, UsdGeom

    eye_tensor = torch.tensor([eye], dtype=torch.float32, device="cpu")
    lookat_tensor = torch.tensor([lookat], dtype=torch.float32, device="cpu")
    rotation_matrix = create_rotation_matrix_from_view(
        eye_tensor,
        lookat_tensor,
        up_axis=UsdGeom.GetStageUpAxis(stage),
        device="cpu",
    )
    assert not torch.isnan(rotation_matrix).any(), "Report camera eye and lookat must define a valid view."
    quat_xyzw = quat_from_matrix(rotation_matrix)[0]

    camera_prim = UsdGeom.Camera.Define(stage, camera_prim_path).GetPrim()
    camera_xform = UsdGeom.Xformable(camera_prim)
    camera_xform.ClearXformOpOrder()
    camera_xform.SetResetXformStack(True)

    translate_attr = camera_prim.GetAttribute("xformOp:translate")
    translate_op = UsdGeom.XformOp(translate_attr) if translate_attr else camera_xform.AddTranslateOp()
    orient_attr = camera_prim.GetAttribute("xformOp:orient")
    orient_op = (
        UsdGeom.XformOp(orient_attr) if orient_attr else camera_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    )
    camera_xform.SetXformOpOrder([translate_op, orient_op], camera_xform.GetResetXformStack())

    translate_op.Set(Gf.Vec3d(*eye))
    orient_op.Set(
        Gf.Quatd(
            float(quat_xyzw[3]),
            Gf.Vec3d(float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])),
        )
    )


class ArenaViewportVideoRecorder(VideoRecorder):
    """Record an Arena viewer pose without moving the interactive Kit viewport camera."""

    cfg: ArenaViewportVideoRecorderCfg

    def __init__(self, cfg: ArenaViewportVideoRecorderCfg, env):
        self._viewer_eye = np.asarray(cfg.eye, dtype=float)
        self._viewer_lookat = np.asarray(cfg.lookat, dtype=float)
        self._scene = env.scene
        self._rgb_annotator = None
        self._render_product = None
        super().__init__(cfg, env)
        _define_report_camera(self._scene.stage, cfg.camera_prim_path, cfg.viewer_env_index)

    def _get_frame(self) -> np.ndarray | None:
        """Capture one frame from Arena's dedicated report camera."""
        return self.render_rgb_array()

    def _clip_path(self, index: int) -> str:
        """Return a report-compatible episode video path."""
        filename = f"{self.cfg.output_filename_prefix}-episode-{index}.mp4"
        return os.path.join(self._effective_output_dir(), filename)

    def render_rgb_array(self) -> np.ndarray | None:
        """Render the report camera after resolving its configured origin into world coordinates."""
        eye, lookat = resolve_viewer_camera_pose(
            self._scene,
            self.cfg,
            self._viewer_eye,
            self._viewer_lookat,
        )
        _set_report_camera_pose(self._scene.stage, self.cfg.camera_prim_path, eye, lookat)

        import omni.kit.app
        import omni.replicator.core as rep
        from isaaclab.app.settings_manager import get_settings_manager

        settings = get_settings_manager()
        headless = not bool(settings.get("/isaaclab/has_gui", False))

        if self._rgb_annotator is None:
            resolution = (self.cfg.window_width, self.cfg.window_height)
            self._render_product = rep.create.render_product(self.cfg.camera_prim_path, resolution)
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
        elif headless and self._render_product is not None:
            with contextlib.suppress(Exception):
                self._render_product.resume()

        play_flag = settings.get("/app/player/playSimulations")
        settings.set_bool("/app/player/playSimulations", False)
        try:
            omni.kit.app.get_app().update()
        finally:
            settings.set_bool("/app/player/playSimulations", bool(play_flag))

        try:
            rgb_data = self._rgb_annotator.get_data()
            if isinstance(rgb_data, dict):
                rgb_data = rgb_data.get("data", np.array([], dtype=np.uint8))
            rgb_data = np.asarray(rgb_data, dtype=np.uint8)
            if rgb_data.size == 0:
                return np.zeros((self.cfg.window_height, self.cfg.window_width, 3), dtype=np.uint8)
            if rgb_data.ndim == 1:
                rgb_data = rgb_data.reshape(self.cfg.window_height, self.cfg.window_width, -1)
            return rgb_data[:, :, :3]
        finally:
            if headless and self._render_product is not None:
                with contextlib.suppress(Exception):
                    self._render_product.pause()

    def close(self) -> None:
        """Flush video frames and release Replicator resources."""
        super().close()
        if self._rgb_annotator is not None:
            with contextlib.suppress(Exception):
                self._rgb_annotator.detach()
        if self._render_product is not None:
            with contextlib.suppress(Exception):
                self._render_product.destroy()
        self._rgb_annotator = None
        self._render_product = None


@configclass
class ArenaViewportVideoRecorderCfg(VideoRecorderCfg):
    """Configure Arena's frame-aware, viewport-isolated report video recorder."""

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

    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Report camera position relative to the configured viewer origin."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Report camera target relative to the configured viewer origin."""

    window_width: int = 1280
    """Recorded frame width in pixels."""

    window_height: int = 720
    """Recorded frame height in pixels."""
