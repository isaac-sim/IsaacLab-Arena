# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tools that let a VLM zoom into gridded camera images and measure 3D points from camera depth."""

from __future__ import annotations

import json
import numpy as np
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from isaaclab_arena_vlm_agent.camera_view import DEPTH_DATA_TYPE, CameraView
from isaaclab_arena_vlm_agent.image_utils import image_to_jpeg_data_url, resize_camera_geometry
from isaaclab_arena_vlm_agent.tool_set import ToolResult, VLMToolSet, make_function_tool

PIXEL_MEASUREMENT_PROMPT = (Path(__file__).parent / "prompts" / "pixel_measurement.txt").read_text().strip()
"""Explain how to measure with the zoom and measure_point tools; appended to the system prompt."""


ZOOM_TOOL = "zoom"
"""Name of the tool returning a magnified, gridded crop of a camera image."""

MEASURE_POINT_TOOL = "measure_point"
"""Name of the tool returning the 3D point under a camera image pixel."""


class PointQuery(BaseModel):
    """Request the 3D surface point under one transmitted-image pixel."""

    model_config = ConfigDict(extra="forbid", strict=True)

    camera: str = Field(description="Camera image name.")
    u: float = Field(description="Pixel column in the full image's grid coordinates; fractions allowed.")
    v: float = Field(description="Pixel row in the full image's grid coordinates; fractions allowed.")


class ZoomRequest(BaseModel):
    """Request a magnified crop centered on one transmitted-image pixel."""

    model_config = ConfigDict(extra="forbid", strict=True)

    camera: str = Field(description="Camera image name.")
    u: float = Field(description="Crop center column in the full image's grid coordinates.")
    v: float = Field(description="Crop center row in the full image's grid coordinates.")
    half_size: float = Field(description="Half the crop width, in full-image pixels.")


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


class PixelMeasurementToolSet(VLMToolSet):
    """Let the model zoom into gridded camera images and measure 3D points from camera depth before it acts.

    Images are sent with a labelled pixel grid. The zoom tool returns a magnified crop of the rendered image,
    and the measure_point tool returns the robot-root 3D point under a pixel. Recent measurements are reported
    in later requests as recent_point_measurements.
    """

    def __init__(
        self,
        max_width_or_height: int,
        *,
        max_rounds: int = 5,
        memory_size: int = 12,
        grid_step_px: float = 40.0,
        zoom_output_px: int = 512,
        zoom_half_size_range_px: tuple[float, float] = (12.0, 80.0),
        max_zooms_per_decision: int = 12,
        max_points_per_decision: int = 24,
    ):
        """Configure request limits and image annotation.

        Args:
            max_width_or_height: Transmitted image width or height limit, in pixels.
            max_rounds: Tool-call rounds allowed per decision before an action is required.
            memory_size: Recent successful measurements reported in later requests, per environment.
            grid_step_px: Grid spacing on full images, in transmitted pixels.
            zoom_output_px: Longer side of each magnified crop, in pixels.
            zoom_half_size_range_px: Allowed zoom half sizes, in transmitted pixels; requests are clamped.
            max_zooms_per_decision: Zoom calls answered per decision; later ones return an error.
            max_points_per_decision: Point measurements answered per decision; later ones return an error.
        """
        assert max_width_or_height > 0, "max_width_or_height must be positive"
        assert type(max_rounds) is int and max_rounds >= 1, "max_rounds must be a positive integer"
        assert type(memory_size) is int and memory_size >= 0, "memory_size must be a non-negative integer"
        assert grid_step_px > 0 and zoom_output_px > 0, "Grid spacing and zoom size must be positive"
        assert 0 < zoom_half_size_range_px[0] <= zoom_half_size_range_px[1], "Expected a valid zoom half-size range"
        assert max_zooms_per_decision >= 1 and max_points_per_decision >= 1, "Per-decision limits must be positive"
        self.max_width_or_height = max_width_or_height
        self.max_rounds = max_rounds
        self.memory_size = memory_size
        self.grid_step_px = grid_step_px
        self.zoom_output_px = zoom_output_px
        self.zoom_half_size_range_px = zoom_half_size_range_px
        self.max_zooms_per_decision = max_zooms_per_decision
        self.max_points_per_decision = max_points_per_decision
        self._views: dict[int, dict[str, CameraView]] = {}
        self._sent_sizes: dict[int, dict[str, tuple[int, int]]] = {}
        self._call_counts: dict[int, dict[str, int]] = {}
        self._memory: dict[int, deque] = {}
        self._decisions: dict[int, int] = {}

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return the zoom and measure_point function tools."""
        return [
            make_function_tool(
                ZOOM_TOOL,
                "Return a magnified crop of a camera image around a pixel, with a finer grid labelled in the same"
                " full-image coordinates and a magenta cross at the center.",
                ZoomRequest,
            ),
            make_function_tool(
                MEASURE_POINT_TOOL,
                "Return point_root_m, the 3D point in the robot root frame where a pixel's ray first hits a"
                " surface, measured from the camera's depth image.",
                PointQuery,
            ),
        ]

    def get_system_prompt_addition(self) -> str:
        """Return how to measure positions with the tools."""
        return PIXEL_MEASUREMENT_PROMPT

    def prepare_request(self, observation_adapter, env, observation, env_id, images, prompt_values):
        """Return gridded images at their transmitted size and prompt values with recent measurements.

        The camera_geometry intrinsics and image sizes are rescaled to the returned images.
        """
        views = observation_adapter.read_camera_views(env, observation, env_id)
        for name in images:
            assert name in views, f"Pixel measurement requires a camera view for image '{name}'"
            assert views[name].depth is not None, f"Pixel measurement requires {DEPTH_DATA_TYPE} on camera '{name}'"
            assert len(views[name].poses_in_robot_roots) == 1, "Pixel measurement requires one commanded robot root"
        geometry = prompt_values.get("camera_geometry", {})
        gridded_images, sent_sizes = {}, {}
        for name, image in images.items():
            gridded = resize_with_pixel_grid(image, self.max_width_or_height, self.grid_step_px)
            gridded_images[name] = gridded
            sent_sizes[name] = (gridded.shape[1], gridded.shape[0])
            if name in geometry:
                resize_camera_geometry(geometry[name], (image.shape[1], image.shape[0]), sent_sizes[name])
        self._views[env_id] = {name: views[name] for name in images}
        self._sent_sizes[env_id] = sent_sizes
        self._call_counts[env_id] = {ZOOM_TOOL: 0, MEASURE_POINT_TOOL: 0}
        self._decisions[env_id] = self._decisions.get(env_id, 0) + 1
        recent = self.recent_measurements(env_id)
        if recent:
            prompt_values["recent_point_measurements"] = recent
        return gridded_images, prompt_values

    def call_tool(self, name: str, arguments: str, env_id: int) -> ToolResult:
        """Answer one zoom or measure_point call from the environment's current decision."""
        limits = {ZOOM_TOOL: self.max_zooms_per_decision, MEASURE_POINT_TOOL: self.max_points_per_decision}
        if name not in limits:
            return ToolResult(json.dumps({"error": f"unknown tool {name!r}; use {ZOOM_TOOL} or {MEASURE_POINT_TOOL}"}))
        try:
            request = (ZoomRequest if name == ZOOM_TOOL else PointQuery).model_validate_json(arguments)
        except ValidationError as exc:
            return ToolResult(json.dumps({"error": f"invalid arguments: {exc}"}))
        counts = self._call_counts.setdefault(env_id, {ZOOM_TOOL: 0, MEASURE_POINT_TOOL: 0})
        if counts[name] >= limits[name]:
            error = f"skipped; at most {limits[name]} {name} calls per decision, so return an action"
            return ToolResult(json.dumps({**request.model_dump(), "error": error}))
        counts[name] += 1
        views = self._views.get(env_id, {})
        sent_sizes = self._sent_sizes.get(env_id, {})
        if name == ZOOM_TOOL:
            info, url = self.render_zoom(views, sent_sizes, request)
            return ToolResult(json.dumps(info), [url] if url is not None else [])
        measurement = self.measure(views, sent_sizes, request)
        if "point_root_m" in measurement:
            memory = self._memory.setdefault(env_id, deque(maxlen=self.memory_size))
            memory.append((self._decisions.get(env_id, 0), measurement))
        return ToolResult(json.dumps(measurement))

    def recent_measurements(self, env_id: int) -> list[dict[str, Any]]:
        """Return recent successful measurements for one environment, with how many decisions ago each was taken."""
        decision = self._decisions.get(env_id, 0)
        entries = []
        for measured_decision, measurement in self._memory.get(env_id, ()):
            entries.append({**measurement, "decisions_ago": decision - measured_decision})
        return entries

    def measure(
        self, views: dict[str, CameraView], sent_sizes: dict[str, tuple[int, int]], query: PointQuery
    ) -> dict[str, Any]:
        """Return the measured point for one query, or an error the model can correct.

        Args:
            views: Camera views indexed by camera name.
            sent_sizes: Transmitted (width, height) per camera; queries use these pixels.
            query: Camera and transmitted pixel to measure.

        Returns:
            The query with point_root_m and depth_m, or with an error.
        """
        result: dict[str, Any] = {"camera": query.camera, "u": query.u, "v": query.v}
        view = views.get(query.camera)
        if view is None:
            result["error"] = f"unknown camera; use one of {sorted(views)}"
            return result
        (root_name,) = view.poses_in_robot_roots
        sent_width, sent_height = sent_sizes[query.camera]
        height, width = view.rgb.shape[:2]
        p_R, z_C = view.measure_point(query.u * width / sent_width, query.v * height / sent_height, root_name)
        if p_R is None:
            result["error"] = "pixel outside the image or no surface at that pixel"
        else:
            result["point_root_m"] = [round(float(value), 4) for value in p_R]
            result["depth_m"] = round(z_C, 4)
        return result

    def render_zoom(
        self, views: dict[str, CameraView], sent_sizes: dict[str, tuple[int, int]], request: ZoomRequest
    ) -> tuple[dict[str, Any], str | None]:
        """Return the visible crop bounds and a magnified, gridded crop for one zoom request.

        Args:
            views: Camera views indexed by camera name.
            sent_sizes: Transmitted (width, height) per camera; requests use these pixels.
            request: Zoom center and half size, in transmitted pixels.

        Returns:
            Crop description, and a JPEG data URL or None when the request cannot be answered.
        """
        info: dict[str, Any] = {"camera": request.camera, "u": request.u, "v": request.v}
        view = views.get(request.camera)
        if view is None:
            info["error"] = f"unknown camera; use one of {sorted(views)}"
            return info, None
        sent_width, sent_height = sent_sizes[request.camera]
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
        state = (self._views, self._sent_sizes, self._call_counts, self._memory, self._decisions)
        for per_env in state:
            if env_ids is None:
                per_env.clear()
            else:
                for env_id in env_ids:
                    per_env.pop(env_id, None)
