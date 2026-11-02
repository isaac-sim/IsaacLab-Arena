# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation adapter that lets a VLM measure 3D points by choosing pixels in gridded, zoomable images."""

from __future__ import annotations

import json
import numpy as np
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from isaaclab_arena_vlm_agent.image_utils import image_to_jpeg_data_url, resize_camera_geometry
from isaaclab_arena_vlm_agent.observation_adapter import CAMERA_OBSERVATION_GROUP, VLMObservationAdapter

DEPTH_DATA_TYPE = "distance_to_image_plane"
"""Camera data type holding per-pixel depth along the optical axis, in meters."""

PIXEL_MEASUREMENT_PROMPT = (
    "Measure positions by pointing at pixels, never by computing them from camera_geometry yourself. Every image"
    " carries a yellow pixel grid labelled in that image's own transmitted coordinates, u to the right and v down."
    " Every response also contains zoom_requests and point_queries lists; leave both empty to act. While either is"
    " non-empty the action fields are ignored, and the reply answers the requests in the same decision; after the last"
    " allowed round, act. A zoom request names a camera image, a center pixel u and v, and half_size in pixels; the"
    " reply is a magnified crop from the rendered image with a finer grid labelled in the same full-image coordinates"
    " and a magenta cross at the requested center, so read a target's pixel from the crop's grid labels, to a fraction"
    " of a pixel. A point query names a camera image and a pixel u and v in those full-image coordinates (fractions"
    " allowed); the reply gives point_root_m, the 3D point in the robot root frame where that pixel's ray first hits a"
    " surface, measured from the camera's depth image. Zoom before pointing at small parts: zoom wide around the"
    " object, then close around the exact spot, then point. A surface point is not an object center or axis; point at"
    " several spots and combine them. Check every returned z against the expected height of the surface you meant to"
    " hit; a much higher z means the pixel hit the gripper or arm, and a lower one means it missed the part. Earlier"
    " measurements are listed in recent_point_measurements with how many decisions ago they were taken; measurements of"
    " fixed objects stay valid. "
)
"""Describe the measurement protocol; appended to the system prompt when pixel measurement is enabled."""


class PointQuery(BaseModel):
    """Request the 3D surface point under one transmitted-image pixel."""

    model_config = ConfigDict(extra="forbid", strict=True)

    camera: str
    u: float
    v: float


class ZoomRequest(BaseModel):
    """Request a magnified crop centered on one transmitted-image pixel."""

    model_config = ConfigDict(extra="forbid", strict=True)

    camera: str
    u: float
    v: float
    half_size: float


class MeasurementRequests(BaseModel):
    """Read only the zoom and point lists of a response, so placeholder action fields cannot block them."""

    model_config = ConfigDict(extra="ignore", strict=True)

    zoom_requests: list[ZoomRequest] = []
    point_queries: list[PointQuery] = []


@dataclass
class CameraView:
    """One camera's rendered color and depth with its calibration in the robot root frame R."""

    rgb: np.ndarray
    """Rendered HWC uint8 pixels."""

    depth: np.ndarray
    """Rendered HxW depth along the optical axis, in meters."""

    intrinsic_matrix: np.ndarray
    """3x3 intrinsics in rendered pixels."""

    t_R_C: np.ndarray
    """Optical-frame origin in the robot root frame, in meters."""

    q_R_C: np.ndarray
    """XYZW rotation from the optical frame (+X right, +Y down, +Z forward) into the robot root frame."""

    sent_size: tuple[int, int]
    """Transmitted (width, height) in pixels; requests use these coordinates."""

    def measure_point(self, u: float, v: float) -> tuple[np.ndarray | None, float]:
        """Return the robot-root surface point under a transmitted-image pixel and its depth.

        Args:
            u: Transmitted-image column, in pixels.
            v: Transmitted-image row, in pixels.

        Returns:
            The point in meters, or None when the pixel is outside the image or sees no surface,
            and the depth read at that pixel.
        """
        height, width = self.depth.shape
        u_render = u * width / self.sent_size[0]
        v_render = v * height / self.sent_size[1]
        if not (0.0 <= u_render < width and 0.0 <= v_render < height):
            return None, float("nan")
        # Intrinsics place pixel centers at half-integer coordinates, so pixel i covers [i, i + 1).
        column, row = int(np.floor(u_render)), int(np.floor(v_render))
        z_C = float(self.depth[row, column])
        if not np.isfinite(z_C) or z_C <= 0.0:
            return None, z_C
        p_C = np.linalg.solve(self.intrinsic_matrix, np.array([u_render, v_render, 1.0])) * z_C
        return Rotation.from_quat(self.q_R_C).apply(p_C) + self.t_R_C, z_C


def draw_pixel_grid(frame: Image.Image, origin_px: tuple[float, float], scale: float, step_px: float) -> Image.Image:
    """Return a copy of a frame with labelled grid lines at multiples of step_px image coordinates.

    Image coordinate c appears at frame pixel (c - origin_px) * scale on each axis.

    Args:
        frame: Frame to annotate.
        origin_px: Image coordinates (u, v) at the frame's top-left corner.
        scale: Frame pixels per image pixel.
        step_px: Grid spacing in image pixels.
    """
    frame = frame.convert("RGBA")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default(size=11)
    except TypeError:
        # Pillow before 10.1 has only a fixed-size bitmap font.
        font = ImageFont.load_default()
    for axis, extent in enumerate(frame.size):
        first = int(np.ceil(origin_px[axis] / step_px))
        last = int(np.floor((origin_px[axis] + extent / scale) / step_px))
        for index in range(first, last + 1):
            value = index * step_px
            offset = (value - origin_px[axis]) * scale
            if axis == 0:
                line, label_xy = [(offset, 0), (offset, frame.height)], (offset + 2, 1)
            else:
                line, label_xy = [(0, offset), (frame.width, offset)], (1, offset + 1)
            draw.line(line, fill=(255, 255, 0, 110), width=1)
            label = f"{value:g}"
            draw.rectangle(draw.textbbox(label_xy, label, font=font), fill=(0, 0, 0, 150))
            draw.text(label_xy, label, fill=(255, 255, 0, 255), font=font)
    return Image.alpha_composite(frame, overlay).convert("RGB")


def resize_with_pixel_grid(image: np.ndarray, max_width_or_height: int, step_px: float) -> np.ndarray:
    """Return an image shrunk to fit max_width_or_height, with a pixel grid labelled in its own coordinates.

    Args:
        image: HWC uint8 pixels; the alpha channel, if present, is discarded.
        max_width_or_height: Maximum width or height of the result, in pixels; smaller images are kept.
        step_px: Grid spacing in result pixels.
    """
    assert image.ndim == 3 and image.shape[-1] in (3, 4), f"Unexpected image shape {image.shape}"
    assert image.dtype == np.uint8, f"Expected uint8 pixels, got {image.dtype}"
    frame = Image.fromarray(image[..., :3])
    frame.thumbnail((max_width_or_height, max_width_or_height))
    return np.asarray(draw_pixel_grid(frame, (0.0, 0.0), 1.0, step_px))


class PixelMeasurementObservationAdapter(VLMObservationAdapter):
    """Add pixel grids, zoomed crops, and depth-based point measurement to an image observation adapter.

    Images are returned at their transmitted size with a labelled pixel grid. The model may then request
    magnified crops and the robot-root 3D point under any pixel, measured from rendered depth, before it
    acts. Recent measurements are reported in later requests as recent_point_measurements.
    """

    def __init__(
        self,
        image_adapter: VLMObservationAdapter,
        max_width_or_height: int,
        *,
        max_rounds: int = 5,
        memory_size: int = 12,
        grid_step_px: float = 40.0,
        zoom_output_px: int = 512,
        zoom_half_size_range_px: tuple[float, float] = (12.0, 80.0),
        max_zooms_per_round: int = 4,
        max_points_per_round: int = 8,
    ):
        """Wrap an image adapter and configure request limits and image annotation.

        Args:
            image_adapter: Adapter returning rendered RGB images and their camera_geometry.
            max_width_or_height: Transmitted image width or height limit, in pixels.
            max_rounds: Zoom or point rounds allowed per decision before an action is required.
            memory_size: Recent successful measurements reported in later requests, per environment.
            grid_step_px: Grid spacing on full images, in transmitted pixels.
            zoom_output_px: Longer side of each magnified crop, in pixels.
            zoom_half_size_range_px: Allowed zoom half sizes, in transmitted pixels; requests are clamped.
            max_zooms_per_round: Zoom requests answered per round; later ones are reported as skipped.
            max_points_per_round: Point queries answered per round; later ones are reported as skipped.
        """
        assert isinstance(image_adapter, VLMObservationAdapter)
        assert max_width_or_height > 0, "max_width_or_height must be positive"
        assert type(max_rounds) is int and max_rounds >= 1, "max_rounds must be a positive integer"
        assert type(memory_size) is int and memory_size >= 0, "memory_size must be a non-negative integer"
        assert grid_step_px > 0 and zoom_output_px > 0, "Grid spacing and zoom size must be positive"
        assert 0 < zoom_half_size_range_px[0] <= zoom_half_size_range_px[1], "Expected a valid zoom half-size range"
        assert max_zooms_per_round >= 1 and max_points_per_round >= 1, "Per-round request limits must be positive"
        self.image_adapter = image_adapter
        self.max_width_or_height = max_width_or_height
        self.max_tool_rounds = max_rounds
        self.memory_size = memory_size
        self.grid_step_px = grid_step_px
        self.zoom_output_px = zoom_output_px
        self.zoom_half_size_range_px = zoom_half_size_range_px
        self.max_zooms_per_round = max_zooms_per_round
        self.max_points_per_round = max_points_per_round
        self._views: dict[int, dict[str, CameraView]] = {}
        self._memory: dict[int, deque] = {}
        self._decisions: dict[int, int] = {}

    def extract_vlm_input_from_observation(self, env, observation, env_id: int):
        """Return gridded images at their transmitted size and prompt values with recent measurements.

        The camera_geometry intrinsics and image sizes are rescaled to the returned images.
        """
        images, prompt_values = self.image_adapter.extract_vlm_input_from_observation(env, observation, env_id)
        # Injected adapters may reuse calibration dictionaries across predictions.
        prompt_values = deepcopy(prompt_values)
        geometry = prompt_values.get("camera_geometry", {})
        rendered_geometry = deepcopy(geometry)
        gridded_images, sent_sizes = {}, {}
        for name, image in images.items():
            gridded = resize_with_pixel_grid(image, self.max_width_or_height, self.grid_step_px)
            gridded_images[name] = gridded
            sent_sizes[name] = (gridded.shape[1], gridded.shape[0])
            if name in geometry:
                resize_camera_geometry(geometry[name], (image.shape[1], image.shape[0]), sent_sizes[name])
        self._views[env_id] = self.read_camera_views(env, images, env_id, rendered_geometry, sent_sizes)
        self._decisions[env_id] = self._decisions.get(env_id, 0) + 1
        recent = self.recent_measurements(env_id)
        if recent:
            prompt_values["recent_point_measurements"] = recent
        return gridded_images, prompt_values

    def read_camera_views(
        self,
        env,
        images: dict[str, np.ndarray],
        env_id: int,
        geometry: dict[str, dict[str, Any]],
        sent_sizes: dict[str, tuple[int, int]],
    ) -> dict[str, CameraView]:
        """Read depth for each camera image and pair it with its rendered-size calibration.

        Args:
            env: Arena environment, optionally gym-wrapped, whose cameras render depth.
            images: Rendered HWC RGB images indexed by camera observation name.
            env_id: Environment index whose cameras are read.
            geometry: camera_geometry for the rendered images, indexed by camera observation name.
            sent_sizes: Transmitted (width, height) per camera.

        Returns:
            Camera views indexed by camera observation name.
        """
        base = env.unwrapped
        camera_cfg = getattr(base.cfg.observations, CAMERA_OBSERVATION_GROUP)
        views = {}
        for name, image in images.items():
            assert name in geometry, f"Pixel measurement requires camera_geometry for camera '{name}'"
            sensor = base.scene[getattr(camera_cfg, name).params["sensor_cfg"].name]
            assert (
                DEPTH_DATA_TYPE in sensor.data.output
            ), f"Pixel measurement requires {DEPTH_DATA_TYPE} on camera '{name}'"
            depth = sensor.data.output[DEPTH_DATA_TYPE]
            depth = getattr(depth, "torch", depth)[env_id].detach().cpu().numpy().reshape(image.shape[:2])
            poses = geometry[name]["poses_in_robot_roots"]
            assert len(poses) == 1, "Pixel measurement requires exactly one commanded robot root"
            (pose,) = poses.values()
            views[name] = CameraView(
                rgb=image,
                depth=depth,
                intrinsic_matrix=np.asarray(geometry[name]["intrinsic_matrix"], dtype=np.float64),
                t_R_C=np.asarray(pose["position_root_m"], dtype=np.float64),
                q_R_C=np.asarray(pose["quaternion_root_xyzw"], dtype=np.float64),
                sent_size=sent_sizes[name],
            )
        return views

    def get_system_prompt_addition(self) -> str:
        """Return the zoom and point request protocol."""
        return PIXEL_MEASUREMENT_PROMPT

    def extend_response_model(self, action_model: type[BaseModel]) -> type[BaseModel]:
        """Return the action response model extended with zoom_requests and point_queries lists."""
        return create_model(
            f"{action_model.__name__}WithPixelMeasurement",
            __base__=action_model,
            zoom_requests=(list[ZoomRequest], ...),
            point_queries=(list[PointQuery], ...),
        )

    def read_tool_requests(self, payload: Any) -> MeasurementRequests | None:
        """Return a response's zoom and point requests regardless of its action fields, or None when there are none."""
        try:
            requests = MeasurementRequests.model_validate(payload)
        except ValidationError:
            return None
        if not (requests.zoom_requests or requests.point_queries):
            return None
        return requests

    def recent_measurements(self, env_id: int) -> list[dict[str, Any]]:
        """Return recent successful measurements for one environment, with how many decisions ago each was taken."""
        decision = self._decisions.get(env_id, 0)
        entries = []
        for measured_decision, measurement in self._memory.get(env_id, ()):
            entries.append({**measurement, "decisions_ago": decision - measured_decision})
        return entries

    def answer_tool_requests(
        self, requests: MeasurementRequests, env_id: int, is_last_round: bool
    ) -> list[dict[str, Any]]:
        """Answer zoom requests and point queries as user chat content.

        Args:
            requests: Zoom requests and point queries read from a response.
            env_id: Environment whose current camera views answer the requests.
            is_last_round: Whether to tell the model that its next response must act.

        Returns:
            Text and image content parts, one zoom result per crop followed by all measurements.
        """
        views = self._views.get(env_id, {})
        content = []
        for index, request in enumerate(requests.zoom_requests):
            if index < self.max_zooms_per_round:
                info, url = self.render_zoom(views, request)
            else:
                skipped = f"skipped; at most {self.max_zooms_per_round} zooms per round"
                info, url = {**request.model_dump(), "error": skipped}, None
            content.append({"type": "text", "text": json.dumps({"zoom": info})})
            if url is not None:
                content.append({"type": "image_url", "image_url": {"url": url}})
        if requests.point_queries:
            memory = self._memory.setdefault(env_id, deque(maxlen=self.memory_size))
            measurements = []
            for index, query in enumerate(requests.point_queries):
                if index >= self.max_points_per_round:
                    skipped = f"skipped; at most {self.max_points_per_round} point queries per round"
                    measurements.append({**query.model_dump(), "error": skipped})
                    continue
                measurement = self.measure(views, query)
                if "point_root_m" in measurement:
                    memory.append((self._decisions.get(env_id, 0), measurement))
                measurements.append(measurement)
            content.append({"type": "text", "text": json.dumps({"point_measurements": measurements})})
        if is_last_round:
            note = "That was the last zoom or point round for this decision; return an action now."
            content.append({"type": "text", "text": note})
        return content

    def measure(self, views: dict[str, CameraView], query: PointQuery) -> dict[str, Any]:
        """Return the measured point for one query, or an error the model can correct."""
        result: dict[str, Any] = {"camera": query.camera, "u": query.u, "v": query.v}
        view = views.get(query.camera)
        if view is None:
            result["error"] = f"unknown camera; use one of {sorted(views)}"
            return result
        p_R, z_C = view.measure_point(query.u, query.v)
        if p_R is None:
            result["error"] = "pixel outside the image or no surface at that pixel"
        else:
            result["point_root_m"] = [round(float(value), 4) for value in p_R]
            result["depth_m"] = round(z_C, 4)
        return result

    def render_zoom(self, views: dict[str, CameraView], request: ZoomRequest) -> tuple[dict[str, Any], str | None]:
        """Return the visible crop bounds and a magnified, gridded crop for one zoom request.

        Args:
            views: Camera views indexed by camera name.
            request: Zoom center and half size, in transmitted pixels.

        Returns:
            Crop description, and a JPEG data URL or None when the request cannot be answered.
        """
        info: dict[str, Any] = {"camera": request.camera, "u": request.u, "v": request.v}
        view = views.get(request.camera)
        if view is None:
            info["error"] = f"unknown camera; use one of {sorted(views)}"
            return info, None
        sent_width, sent_height = view.sent_size
        half = float(np.clip(request.half_size, *self.zoom_half_size_range_px))
        u_min, u_max = max(0.0, request.u - half), min(float(sent_width), request.u + half)
        v_min, v_max = max(0.0, request.v - half), min(float(sent_height), request.v + half)
        if u_max - u_min < 1.0 or v_max - v_min < 1.0:
            info["error"] = "zoom center outside the image"
            return info, None
        height, width = view.rgb.shape[:2]
        scale_u, scale_v = width / sent_width, height / sent_height
        span = max(u_max - u_min, v_max - v_min)
        scale = self.zoom_output_px / span
        # Resample from the exact fractional bounds so grid labels line up with the image content.
        crop = Image.fromarray(view.rgb[..., :3]).resize(
            (round((u_max - u_min) * scale), round((v_max - v_min) * scale)),
            Image.LANCZOS,
            box=(u_min * scale_u, v_min * scale_v, u_max * scale_u, v_max * scale_v),
        )
        # Aim for at most about a dozen labelled lines per axis.
        step = next((step for step in (1, 2, 5, 10, 20, 50) if span / step <= 12), 100)
        crop = draw_pixel_grid(crop, (u_min, v_min), scale, step)
        draw = ImageDraw.Draw(crop)
        center_x, center_y = (request.u - u_min) * scale, (request.v - v_min) * scale
        draw.line([(center_x - 6, center_y), (center_x + 6, center_y)], fill=(255, 0, 255), width=1)
        draw.line([(center_x, center_y - 6), (center_x, center_y + 6)], fill=(255, 0, 255), width=1)
        info |= {
            "u_range": [round(u_min, 1), round(u_max, 1)],
            "v_range": [round(v_min, 1), round(v_max, 1)],
            "grid_step_px": step,
        }
        return info, image_to_jpeg_data_url(np.asarray(crop), max(crop.size))[0]

    def reset(self, env_ids: list[int] | None = None) -> None:
        """Discard views, measurements, and decision counts for selected environments, or all."""
        if env_ids is None:
            self._views.clear()
            self._memory.clear()
            self._decisions.clear()
            return
        for env_id in env_ids:
            self._views.pop(env_id, None)
            self._memory.pop(env_id, None)
            self._decisions.pop(env_id, None)
