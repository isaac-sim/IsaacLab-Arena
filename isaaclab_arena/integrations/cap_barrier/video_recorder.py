# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded exterior-camera video evidence for the CAP grocery producer."""

from __future__ import annotations

import numpy as np
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg

from isaaclab_arena.video.camera_observation_video_recorder import _validate_encoded_video

from .grocery_scene_spec import CAP_GROCERY_CAMERA_NAME

CAP_GROCERY_VIDEO_FPS = 10
CAP_GROCERY_VIDEO_SAMPLE_EVERY_FRAMES = 20
CAP_GROCERY_VIDEO_FIRST_CAPTURE_GENERATION = 2


@dataclass(frozen=True)
class _ExteriorRgbFrame:
    """One detached RGB frame from the existing grocery exterior camera."""

    camera_name: str
    width: int
    height: int
    rgb: bytes


def _extract_exterior_rgb(environment: Any, *, frame_index: int) -> _ExteriorRgbFrame:
    """Render and detach RGB only; depth and calibration are not video inputs."""
    unwrapped = environment.unwrapped
    if CAP_GROCERY_CAMERA_NAME not in unwrapped.scene.sensors:
        raise RuntimeError(
            f"{CAP_GROCERY_CAMERA_NAME} sensor is not attached; build the environment with cameras enabled"
        )
    if not hasattr(unwrapped, "sim"):
        raise RuntimeError("CAP grocery video environment does not expose its simulation context")
    unwrapped.sim.render(skip_app_pumping=False)
    camera = unwrapped.scene[CAP_GROCERY_CAMERA_NAME]
    rgb_tensor = camera.data.output["rgb"][0, ..., :3]
    height, width = int(rgb_tensor.shape[0]), int(rgb_tensor.shape[1])
    rgb = np.ascontiguousarray(rgb_tensor.detach().cpu().numpy().astype(np.uint8)).tobytes()

    del frame_index
    return _ExteriorRgbFrame(
        camera_name=CAP_GROCERY_CAMERA_NAME,
        width=width,
        height=height,
        rgb=rgb,
    )


class GroceryVideoRecorder:
    """Stream the existing exterior camera to one explicitly finalized MP4."""

    def __init__(
        self,
        adapter: Any,
        marker_sink: Callable[[str], None],
        *,
        output_path: str | os.PathLike[str],
        frame_extractor: Callable[..., Any] | None = None,
        writer_factory: Callable[..., Any] | None = None,
        validator: Callable[[str, int], None] | None = None,
    ) -> None:
        final_path = Path(output_path)
        if not final_path.is_absolute():
            raise ValueError("grocery video output path must be absolute")
        if final_path.suffix.lower() != ".mp4":
            raise ValueError("grocery video output path must end in .mp4")
        if not final_path.parent.is_dir():
            raise FileNotFoundError(f"grocery video output directory does not exist: {final_path.parent}")

        temporary_path = Path(f"{final_path}.part")
        finalize_request_path = Path(f"{final_path}.finalize")
        for path, label in (
            (final_path, "output"),
            (temporary_path, "partial output"),
            (finalize_request_path, "finalize request"),
        ):
            if path.exists():
                raise FileExistsError(f"grocery video {label} already exists: {path}")

        self._adapter = adapter
        self._marker_sink = marker_sink
        self._final_path = final_path
        self._temporary_path = temporary_path
        self._finalize_request_path = finalize_request_path
        self._frame_extractor = frame_extractor or _extract_exterior_rgb
        self._writer_factory = writer_factory or imageio_ffmpeg.write_frames
        self._validator = validator or _validate_encoded_video
        self._writer: Any | None = None
        self._frame_shape: tuple[int, int, int] | None = None
        self._frame_count = 0
        self._capture_index = 0
        self._generation: int | None = None
        self._closed = False
        self._finalized = False

        marker_sink(
            "CAP_GROCERY_VIDEO_ARMED "
            f"path={self._final_path} finalize_request={self._finalize_request_path} "
            f"camera={CAP_GROCERY_CAMERA_NAME} fps={CAP_GROCERY_VIDEO_FPS}"
        )

    @property
    def finalize_request_path(self) -> Path:
        """Return the one-shot request path that atomically closes the video."""
        return self._finalize_request_path

    @property
    def frame_count(self) -> int:
        """Return the number of frames submitted to the encoder."""
        return self._frame_count

    @property
    def finalized(self) -> bool:
        """Return whether the validated MP4 was atomically published."""
        return self._finalized

    def __call__(self, frame: int) -> None:
        self.on_physics_frame(frame)

    def begin_generation(self, generation: int) -> None:
        """Arm recording only for the post-reset task generation and later."""
        if self._closed:
            raise RuntimeError("grocery video recorder is closed")
        if generation <= 0:
            raise ValueError("grocery video generation must be positive")
        if self._generation is not None and generation <= self._generation:
            raise ValueError(f"grocery video generation must advance: current={self._generation}, next={generation}")
        self._generation = generation
        self._marker_sink(
            "CAP_GROCERY_VIDEO_GENERATION_ARMED "
            f"generation={generation} "
            f"capture_enabled={int(generation >= CAP_GROCERY_VIDEO_FIRST_CAPTURE_GENERATION)}"
        )

    def on_physics_frame(self, frame: int) -> None:
        """Sample at 10 Hz or honor an explicit finalize request."""
        if self._finalized:
            return
        if self._closed:
            raise RuntimeError("grocery video recorder is closed")
        if self._generation is None:
            raise RuntimeError("grocery video recorder used before begin_generation")
        if frame < 0:
            raise ValueError("physics frame index must be nonnegative")

        if self._finalize_request_path.exists():
            self._honor_finalize_request()
            return
        if self._generation < CAP_GROCERY_VIDEO_FIRST_CAPTURE_GENERATION:
            return
        if frame % CAP_GROCERY_VIDEO_SAMPLE_EVERY_FRAMES != 0:
            return

        captured = self._frame_extractor(
            self._adapter._environment,
            frame_index=self._capture_index,
        )
        rgb = self._decode_rgb(captured)
        self._capture_index += 1
        self._write_frame(rgb)

    def _decode_rgb(self, captured: Any) -> np.ndarray:
        if getattr(captured, "camera_name", None) != CAP_GROCERY_CAMERA_NAME:
            raise RuntimeError(
                f"grocery video extractor returned the wrong camera: {getattr(captured, 'camera_name', None)!r}"
            )
        width = int(captured.width)
        height = int(captured.height)
        if width <= 0 or height <= 0:
            raise ValueError(f"grocery video frame dimensions must be positive, got {width}x{height}")
        expected_bytes = width * height * 3
        if len(captured.rgb) != expected_bytes:
            raise ValueError(
                f"grocery video RGB payload size mismatch: expected {expected_bytes}, got {len(captured.rgb)}"
            )
        return np.frombuffer(captured.rgb, dtype=np.uint8).reshape(height, width, 3)

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self._open_writer(frame)
        if tuple(frame.shape) != self._frame_shape:
            self._discard_partial()
            raise ValueError(f"grocery video frame shape changed from {self._frame_shape} to {tuple(frame.shape)}")
        try:
            self._writer.send(np.ascontiguousarray(frame))
            self._frame_count += 1
        except BaseException:
            self._discard_partial()
            self._closed = True
            raise

    def _open_writer(self, frame: np.ndarray) -> None:
        height, width, channels = frame.shape
        if channels != 3:
            raise ValueError(f"grocery video frame must be RGB, got shape {frame.shape}")
        writer = self._writer_factory(
            str(self._temporary_path),
            (width, height),
            pix_fmt_in="rgb24",
            fps=CAP_GROCERY_VIDEO_FPS,
            macro_block_size=1,
            output_params=["-f", "mp4"],
        )
        try:
            writer.send(None)
        except BaseException:
            try:
                writer.close()
            finally:
                self._temporary_path.unlink(missing_ok=True)
            raise
        self._writer = writer
        self._frame_shape = tuple(frame.shape)

    def _require_regular_finalize_request(self) -> None:
        if not self._finalize_request_path.is_file():
            raise RuntimeError(f"grocery video finalize request is not a regular file: {self._finalize_request_path}")

    def _honor_finalize_request(self) -> None:
        try:
            self._require_regular_finalize_request()
            self._finalize()
        except BaseException:
            if not self._closed:
                try:
                    self._discard_partial()
                finally:
                    self._closed = True
            raise

    def _finalize(self) -> None:
        if self._writer is None or self._frame_count <= 0:
            self._closed = True
            raise RuntimeError("cannot finalize grocery video before at least one frame is recorded")

        writer = self._writer
        self._writer = None
        published = False
        try:
            writer.close()
            if not self._temporary_path.is_file() or self._temporary_path.stat().st_size <= 0:
                raise RuntimeError(f"ffmpeg produced no grocery video: {self._temporary_path}")
            self._validator(str(self._temporary_path), self._frame_count)
            file_descriptor = os.open(self._temporary_path, os.O_RDONLY)
            try:
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
            # A same-filesystem hard-link is an atomic, no-overwrite publication.
            # Keeping the request file makes finalization externally auditable.
            os.link(self._temporary_path, self._final_path)
            published = True
            self._temporary_path.unlink()
            directory_descriptor = os.open(
                self._final_path.parent,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            self._temporary_path.unlink(missing_ok=True)
            if published:
                self._final_path.unlink(missing_ok=True)
            self._closed = True
            raise

        self._finalized = True
        self._closed = True
        self._marker_sink(
            f"CAP_GROCERY_VIDEO_RECORDED path={self._final_path} frames={self._frame_count} fps={CAP_GROCERY_VIDEO_FPS}"
        )

    def _discard_partial(self) -> None:
        writer = self._writer
        self._writer = None
        try:
            if writer is not None:
                writer.close()
        finally:
            self._temporary_path.unlink(missing_ok=True)

    def close(self) -> None:
        """Finalize on a pending request; otherwise discard the partial stream."""
        if self._closed:
            return
        if self._finalize_request_path.exists():
            self._honor_finalize_request()
            return

        try:
            self._discard_partial()
        finally:
            self._closed = True
        self._marker_sink(
            f"CAP_GROCERY_VIDEO_DISCARDED path={self._final_path} frames={self._frame_count} reason=no-finalize-request"
        )


def make_grocery_video_recorder(
    adapter: Any,
    marker_sink: Callable[[str], None],
    *,
    output_path: str | os.PathLike[str],
) -> GroceryVideoRecorder:
    """Build the main-thread exterior-camera video observer."""
    return GroceryVideoRecorder(
        adapter,
        marker_sink,
        output_path=output_path,
    )
