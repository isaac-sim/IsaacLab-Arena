# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared frame-encoding and video/gif writing helpers.

One place to fix frame-encoding bugs. Heavy deps (moviepy, Pillow) are imported
lazily so this module is importable before the Isaac Sim app launches.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def to_uint8(frame: torch.Tensor | np.ndarray, *, drop_alpha: bool = True) -> np.ndarray:
    """Convert a camera frame to an HxWx3 uint8 array that never aliases the source.

    Accepts the encodings Arena cameras emit: uint8 passthrough, normalized float
    in [0, 1] rescaled to [0, 255], and float already in [0, 255] clamped. Copies
    the input so retaining the result is safe even when the source is a reused
    render-annotator buffer.

    Args:
        frame: A single HxWxC frame (torch tensor or ndarray).
        drop_alpha: Keep only the first 3 channels. No-op for RGB.
    """
    # Decouple from buffers the caller or renderer may overwrite next step:
    # - GPU tensor: .cpu() already produces an owned host copy.
    # - CPU tensor: .numpy() shares storage with the tensor, so copy explicitly.
    # - ndarray: np.array(..., copy) detaches from the source.
    if hasattr(frame, "detach"):
        arr = frame.detach().cpu().numpy()
        if frame.device.type == "cpu":
            arr = arr.copy()
    else:
        arr = np.array(frame)
    assert arr.size, "to_uint8 received an empty frame."
    if arr.dtype == np.uint8:
        out = arr
    elif arr.dtype.kind == "f":
        # normalize=True yields [0, 1]; otherwise assume already [0, 255]. Heuristic:
        # a [0, 255] frame whose brightest pixel is <= 1.0 (a near-black scene) is
        # misread as normalized and scaled up. Pass uint8 to avoid the ambiguity.
        scale = 255.0 if float(arr.max()) <= 1.0 else 1.0
        out = np.clip(arr * scale, 0, 255).astype(np.uint8)
    else:
        out = np.clip(arr, 0, 255).astype(np.uint8)
    return out[..., :3] if drop_alpha else out


def write_video(frames: list[np.ndarray], path: str, fps: int) -> None:
    """Encode frames to an mp4 via moviepy, closing the clip to release ffmpeg handles."""
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    assert frames, "write_video called with zero frames."
    clip = ImageSequenceClip(frames, fps=fps)
    try:
        clip.write_videofile(path, logger=None, audio=False)
    finally:
        clip.close()


def write_gif(frames: list[np.ndarray], path: str, fps: int) -> None:
    """Encode frames to an animated gif via Pillow."""
    from PIL import Image

    assert frames, "write_gif called with zero frames."
    pil_frames = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / max(fps, 1))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
