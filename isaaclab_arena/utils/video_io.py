# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared frame-encoding and video/gif writing helpers.

One encoder for every Arena recorder (robot-POV-cam :class:`CameraObsVideoRecorder`,
the :class:`SynchronizedVisualizer`, …) so they stay consistent and there is a
single place to fix encoding bugs. Heavy deps (moviepy, Pillow) are imported
lazily so this module is importable before the Isaac Sim app launches.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def to_uint8(frame: torch.Tensor | np.ndarray, *, drop_alpha: bool = True) -> np.ndarray:
    """Convert a camera frame to an ``HxWx3`` ``uint8`` numpy array.

    Handles the encodings Arena cameras emit: ``uint8`` passthrough, float in
    ``[0, 1]`` (``mdp.image`` with ``normalize=True``) rescaled to ``[0, 255]``,
    and float already in ``[0, 255]`` clamped.

    Args:
        frame: A single frame (torch tensor or ndarray), ``HxWxC``.
        drop_alpha: Keep only the first 3 channels (drop alpha). No-op for RGB.
    """
    arr = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    if arr.dtype == np.uint8:
        out = arr
    elif arr.dtype.kind == "f":
        # normalize=True yields [0, 1]; otherwise assume already [0, 255].
        scale = 255.0 if arr.size and float(arr.max()) <= 1.0 else 1.0
        out = np.clip(arr * scale, 0, 255).astype(np.uint8)
    else:
        out = np.clip(arr, 0, 255).astype(np.uint8)
    return out[..., :3] if drop_alpha else out


def write_video(frames: list[np.ndarray], path: str, fps: int) -> None:
    """Encode ``frames`` to an mp4 at ``path`` via moviepy.

    Closes the clip in a ``finally`` so the ffmpeg subprocess / file handles are
    released deterministically (matters when writing many videos in a loop).
    """
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    clip = ImageSequenceClip(frames, fps=fps)
    try:
        clip.write_videofile(path, logger=None, audio=False)
    finally:
        clip.close()


def write_gif(frames: list[np.ndarray], path: str, fps: int) -> None:
    """Encode ``frames`` to an animated gif at ``path`` via Pillow."""
    from PIL import Image

    pil_frames = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / max(fps, 1))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
