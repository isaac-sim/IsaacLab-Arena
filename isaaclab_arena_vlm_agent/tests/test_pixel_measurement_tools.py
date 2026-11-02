# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check pixel grids, zoomed crops, and depth back-projection without a simulation runtime."""

import base64
import io
import json
import numpy as np

import pytest
from PIL import Image

from isaaclab_arena_vlm_agent.camera_view import CameraView
from isaaclab_arena_vlm_agent.observation_adapter import VLMObservationAdapter
from isaaclab_arena_vlm_agent.pixel_measurement_tools import (
    MEASURE_POINT_TOOL,
    ZOOM_TOOL,
    PixelMeasurementToolSet,
    ZoomRequest,
)

TABLE_Z_R = 0.1
"""Height of the synthetic table in the robot root frame R, in meters."""


def make_view() -> CameraView:
    """Return a 64x48 camera 0.9 m above a flat table, looking straight down."""
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[8:16, 40:48, 0] = 255
    depth = np.full((48, 64), 1.0 - TABLE_Z_R)
    depth[0, 0] = np.inf
    intrinsics = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    # A half turn about X maps optical +Z (forward) to root -Z (down) and optical +Y (down) to root -Y.
    pose = {"position_root_m": [0.5, 0.0, 1.0], "quaternion_root_xyzw": [1.0, 0.0, 0.0, 0.0]}
    return CameraView(
        rgb=rgb,
        intrinsic_matrix=intrinsics,
        t_W_C=np.array(pose["position_root_m"]),
        q_W_C=np.array(pose["quaternion_root_xyzw"]),
        poses_in_robot_roots={"robot": pose},
        depth=depth,
    )


SENT_SIZES = {"top": (32, 24)}
"""The synthetic camera is transmitted at half size."""


def decode(url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_points_are_back_projected_from_depth():
    view = make_view()
    p_R = np.array([0.52, -0.03, TABLE_Z_R])
    p_C = np.array([p_R[0] - 0.5, -p_R[1], 1.0 - p_R[2]])
    u_render, v_render = (view.intrinsic_matrix @ p_C)[:2] / p_C[2]
    measured, z_C = view.measure_point(u_render, v_render, "robot")
    np.testing.assert_allclose(measured, p_R, atol=1e-9)
    assert z_C == pytest.approx(1.0 - TABLE_Z_R)
    # Column 63.8 lies inside the last pixel [63, 64).
    assert view.measure_point(63.8, 47.8, "robot")[0] is not None
    # Pixels outside the image or without a surface have no point.
    for u, v in [(-1.0, 5.0), (5.0, 48.0), (0.0, 0.0)]:
        assert view.measure_point(u, v, "robot")[0] is None


def test_zoom_is_clamped_magnified_and_labelled_in_full_image_coordinates():
    measurement = PixelMeasurementToolSet(32, zoom_output_px=120)
    info, url = measurement.render_zoom(
        {"top": make_view()}, SENT_SIZES, ZoomRequest(camera="top", u=16.0, v=12.0, half_size=4.0)
    )
    # The 4-pixel request is widened to the 12-pixel minimum and clipped to the 32x24 image.
    assert info["u_range"] == [4.0, 28.0] and info["v_range"] == [0.0, 24.0]
    assert info["grid_step_px"] == 2
    crop = decode(url).convert("RGB")
    assert crop.size == (120, 120)
    # The red square covers transmitted u 20-24 and v 4-8; sample between grid lines.
    red = crop.getpixel((round((22.5 - 4.0) * 5), round(6.5 * 5)))
    assert red[0] > 180 and red[1] < 120
    info, url = measurement.render_zoom(
        {"top": make_view()}, SENT_SIZES, ZoomRequest(camera="top", u=100, v=1, half_size=20)
    )
    assert url is None and "outside" in info["error"]


class StaticImageAdapter(VLMObservationAdapter):
    """Return the synthetic camera view at its rendered size."""

    def read_camera_views(self, env, observation, env_id):
        return {"top": make_view()}

    def extract_vlm_input_from_observation(self, env, observation, env_id):
        view = make_view()
        return {"top": view.rgb}, {"camera_geometry": {"top": view.to_camera_geometry()}}


def prepare(tool_set: PixelMeasurementToolSet, env_id: int = 0):
    adapter = StaticImageAdapter()
    images, prompt_values = adapter.extract_vlm_input_from_observation(None, None, env_id)
    return tool_set.prepare_request(adapter, None, None, env_id, images, prompt_values)


def test_images_and_calibration_are_returned_at_the_transmitted_size():
    images, prompt_values = prepare(PixelMeasurementToolSet(32))
    assert images["top"].shape == (24, 32, 3)
    geometry = prompt_values["camera_geometry"]["top"]
    assert geometry["image_size_px"] == [32, 24]
    np.testing.assert_allclose(geometry["intrinsic_matrix"], make_view().intrinsic_matrix * [[0.5], [0.5], [1.0]])
    assert "recent_point_measurements" not in prompt_values


def test_tool_calls_are_answered_within_limits_and_remembered_until_reset():
    tool_set = PixelMeasurementToolSet(32, max_zooms_per_decision=1, max_points_per_decision=2)
    assert [tool["function"]["name"] for tool in tool_set.get_tool_definitions()] == [ZOOM_TOOL, MEASURE_POINT_TOOL]
    prepare(tool_set)
    zoom = json.dumps({"camera": "top", "u": 16.0, "v": 12.0, "half_size": 8.0})
    point = {"camera": "top", "u": 16.0, "v": 12.0}
    result = tool_set.call_tool(ZOOM_TOOL, zoom, 0)
    assert len(result.image_urls) == 1 and json.loads(result.text)["u_range"] == [4.0, 28.0]
    assert "skipped" in json.loads(tool_set.call_tool(ZOOM_TOOL, zoom, 0).text)["error"]
    measurement = json.loads(tool_set.call_tool(MEASURE_POINT_TOOL, json.dumps(point), 0).text)
    np.testing.assert_allclose(measurement["point_root_m"], [0.5, 0.0, TABLE_Z_R], atol=1e-4)
    unknown = tool_set.call_tool(MEASURE_POINT_TOOL, json.dumps({**point, "camera": "side"}), 0)
    assert "unknown camera" in json.loads(unknown.text)["error"]
    assert "skipped" in json.loads(tool_set.call_tool(MEASURE_POINT_TOOL, json.dumps(point), 0).text)["error"]
    assert "invalid arguments" in json.loads(tool_set.call_tool(ZOOM_TOOL, '{"camera": "top"}', 0).text)["error"]
    assert "unknown tool" in json.loads(tool_set.call_tool("grasp", "{}", 0).text)["error"]
    _, prompt_values = prepare(tool_set)
    remembered = prompt_values["recent_point_measurements"]
    assert len(remembered) == 1 and remembered[0]["decisions_ago"] == 1
    assert tool_set.recent_measurements(1) == []
    tool_set.reset([0])
    assert tool_set.recent_measurements(0) == []
