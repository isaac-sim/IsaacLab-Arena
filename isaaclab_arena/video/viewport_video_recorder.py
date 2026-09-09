# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Report video recording that reads its viewpoint in the task's configured viewer frame."""

from __future__ import annotations

import numpy as np
from typing import Literal

from isaaclab.envs.utils.video_recorder import VideoRecorder
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.utils.configclass import configclass

SUPPORTED_VIEWER_ORIGIN_TYPES = ("world", "env")
"""Viewer frames the report video recorder can resolve."""


def resolve_viewer_origin(scene, origin_type: str, env_index: int) -> np.ndarray:
    """Return the world-frame origin that a task's viewer eye and target are measured from.

    Args:
        scene: Interactive scene supplying the per-environment origins.
        origin_type: Viewer frame, mirroring the task's ``ViewerCfg.origin_type``.
        env_index: Environment supplying the origin when the frame is environment-relative.

    Returns:
        The world-frame origin, as an XYZ array.
    """
    if origin_type == "world":
        return np.zeros(3, dtype=float)
    # The asset-tracking frames cannot be resolved here: Isaac Lab builds the video recorder before
    # sim.reset(), so no asset has a physics view yet and root/body poses are unreadable.
    assert origin_type == "env", (
        f"Viewer origin type '{origin_type}' is not supported for report video recording; "
        f"expected one of {SUPPORTED_VIEWER_ORIGIN_TYPES}."
    )
    num_envs = len(scene.env_origins)
    assert (
        0 <= env_index < num_envs
    ), f"Viewer environment index {env_index} is outside the available range [0, {num_envs - 1}]."
    return scene.env_origins[env_index].detach().cpu().numpy().astype(float)


class ArenaViewportVideoRecorder(VideoRecorder):
    """Record the report video from the task's viewpoint, resolved into world coordinates.

    Isaac Lab forwards ``ViewerCfg.eye`` and ``ViewerCfg.lookat`` to the recorder but not
    ``ViewerCfg.origin_type``, so an environment-relative task viewpoint is recorded as though it
    were already in world coordinates. That aims the camera at empty space whenever the selected
    environment origin is non-zero, which is every parallel-environment run.
    """

    cfg: ArenaViewportVideoRecorderCfg

    def __init__(self, cfg: ArenaViewportVideoRecorderCfg, scene):
        # Resolve before delegating: the base class copies eye/lookat into the backend capture
        # config, and that copy is what ultimately aims the camera.
        # ManagerBasedRLEnv re-derives cfg.eye/lookat from cfg.viewer on every environment
        # construction, so overwriting them here cannot accumulate across rebuilds.
        origin = resolve_viewer_origin(scene, cfg.viewer_origin_type, cfg.viewer_env_index)
        cfg.eye = tuple(float(x) for x in origin + np.asarray(cfg.eye, dtype=float))
        cfg.lookat = tuple(float(x) for x in origin + np.asarray(cfg.lookat, dtype=float))
        super().__init__(cfg, scene)


@configclass
class ArenaViewportVideoRecorderCfg(VideoRecorderCfg):
    """Configure the report video recorder to read its viewpoint in the task's viewer frame."""

    class_type: type = ArenaViewportVideoRecorder

    viewer_origin_type: Literal["world", "env", "asset_root", "asset_body"] = "world"
    """Frame the configured eye and target are expressed in, mirroring ``ViewerCfg.origin_type``."""

    viewer_env_index: int = 0
    """Environment supplying the viewer origin when the frame is environment-relative."""
