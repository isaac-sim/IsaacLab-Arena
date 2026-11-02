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
from pydantic import BaseModel, ConfigDict

from isaaclab_arena_vlm_agent.observation_adapter import VLMObservationAdapter
from isaaclab_arena_vlm_agent.pixel_measurement import (
    CameraView,
    PixelMeasurementObservationAdapter,
    PointQuery,
    ZoomRequest,
    resize_with_pixel_grid,
)

TABLE_Z_R = 0.1
"""Height of the synthetic table in the robot root frame R, in meters."""


def make_view() -> CameraView:
    """Return a 64x48 camera 0.9 m above a flat table, looking straight down, transmitted at half size."""
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[8:16, 40:48, 0] = 255
    depth = np.full((48, 64), 1.0 - TABLE_Z_R)
    depth[0, 0] = np.inf
    intrinsics = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    # A half turn about X maps optical +Z (forward) to root -Z (down) and optical +Y (down) to root -Y.
    return CameraView(
        rgb=rgb,
        depth=depth,
        intrinsic_matrix=intrinsics,
        t_R_C=np.array([0.5, 0.0, 1.0]),
        q_R_C=np.array([1.0, 0.0, 0.0, 0.0]),
        sent_size=(32, 24),
    )


def decode(url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_measured_points_match_projected_table_points():
    view = make_view()
    p_R = np.array([0.52, -0.03, TABLE_Z_R])
    p_C = np.array([p_R[0] - 0.5, -p_R[1], 1.0 - p_R[2]])
    u_render, v_render = (view.intrinsic_matrix @ p_C)[:2] / p_C[2]
    measured, z_C = view.measure_point(u_render / 2, v_render / 2)
    np.testing.assert_allclose(measured, p_R, atol=1e-9)
    assert z_C == pytest.approx(1.0 - TABLE_Z_R)


@pytest.mark.parametrize("u,v", [(-1.0, 5.0), (5.0, 24.0), (0.0, 0.0)])
def test_pixels_outside_the_image_or_without_a_surface_have_no_point(u, v):
    measured, _ = make_view().measure_point(u, v)
    assert measured is None


def test_pixels_cover_the_full_image_up_to_the_far_edge():
    # Transmitted u 31.9 is rendered column 63.8, inside the last pixel [63, 64).
    measured, _ = make_view().measure_point(31.9, 23.9)
    assert measured is not None


def test_zoom_is_clamped_magnified_and_labelled_in_full_image_coordinates():
    measurement = make_adapter(zoom_output_px=120)
    info, url = measurement.render_zoom({"top": make_view()}, ZoomRequest(camera="top", u=16.0, v=12.0, half_size=4.0))
    # The 4-pixel request is widened to the 12-pixel minimum and clipped to the 32x24 image.
    assert info["u_range"] == [4.0, 28.0] and info["v_range"] == [0.0, 24.0]
    assert info["grid_step_px"] == 2
    crop = decode(url).convert("RGB")
    assert crop.size == (120, 120)
    # The red square covers transmitted u 20-24 and v 4-8; sample between grid lines.
    red = crop.getpixel((round((22.5 - 4.0) * 5), round(6.5 * 5)))
    assert red[0] > 180 and red[1] < 120


@pytest.mark.parametrize(
    "request_,error",
    [
        (ZoomRequest(camera="side", u=1, v=1, half_size=20), "unknown camera"),
        (ZoomRequest(camera="top", u=100, v=1, half_size=20), "outside"),
    ],
)
def test_unanswerable_zoom_requests_return_errors(request_, error):
    info, url = make_adapter().render_zoom({"top": make_view()}, request_)
    assert url is None and error in info["error"]


def test_gridded_images_are_shrunk_to_the_transmitted_size():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    frame = resize_with_pixel_grid(image, 32, step_px=8).astype(int)
    assert frame.shape == (24, 32, 3)
    yellow = (frame[..., 0] > 80) & (frame[..., 1] > 80) & (frame[..., 2] < 60)
    assert yellow.any()


class TargetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    target: list[float]


class StaticImageAdapter(VLMObservationAdapter):
    """Return the synthetic camera image at its rendered size with its rendered-size calibration."""

    def extract_vlm_input_from_observation(self, env, observation, env_id):
        view = make_view()
        geometry = {"image_size_px": [64, 48], "intrinsic_matrix": view.intrinsic_matrix.tolist()}
        return {"top": view.rgb}, {"camera_geometry": {"top": geometry}}


class FixedViewAdapter(PixelMeasurementObservationAdapter):
    def read_camera_views(self, env, images, env_id, geometry, sent_sizes):
        assert geometry["top"]["image_size_px"] == [64, 48] and sent_sizes == {"top": (32, 24)}
        return {"top": make_view()}


def make_adapter(**kwargs) -> FixedViewAdapter:
    return FixedViewAdapter(StaticImageAdapter(), 32, **kwargs)


MEASUREMENT_PAYLOAD = {
    "target": [],
    "zoom_requests": [{"camera": "top", "u": 16.0, "v": 12.0, "half_size": 8.0}],
    "point_queries": [{"camera": "top", "u": 16.0, "v": 12.0}, {"camera": "top", "u": 99.0, "v": 1.0}],
}


def test_images_and_calibration_are_returned_at_the_transmitted_size():
    images, prompt_values = make_adapter().extract_vlm_input_from_observation(None, None, 0)
    assert images["top"].shape == (24, 32, 3)
    geometry = prompt_values["camera_geometry"]["top"]
    assert geometry["image_size_px"] == [32, 24]
    np.testing.assert_allclose(geometry["intrinsic_matrix"], make_view().intrinsic_matrix * [[0.5], [0.5], [1.0]])
    assert "recent_point_measurements" not in prompt_values


def test_measurements_are_reported_and_remembered_until_reset():
    adapter = make_adapter()
    adapter.extract_vlm_input_from_observation(None, None, 0)
    assert issubclass(adapter.extend_response_model(TargetResponse), TargetResponse)
    assert adapter.read_tool_requests({**MEASUREMENT_PAYLOAD, "zoom_requests": [], "point_queries": []}) is None
    content = adapter.answer_tool_requests(adapter.read_tool_requests(MEASUREMENT_PAYLOAD), 0, is_last_round=False)
    assert [part["type"] for part in content] == ["text", "image_url", "text"]
    measurements = json.loads(content[2]["text"])["point_measurements"]
    np.testing.assert_allclose(measurements[0]["point_root_m"], [0.5, 0.0, TABLE_Z_R], atol=1e-4)
    assert "error" in measurements[1]
    _, prompt_values = adapter.extract_vlm_input_from_observation(None, None, 0)
    remembered = prompt_values["recent_point_measurements"]
    assert len(remembered) == 1 and remembered[0]["decisions_ago"] == 1
    assert adapter.recent_measurements(1) == []
    adapter.reset([0])
    assert adapter.recent_measurements(0) == []


def test_measure_reports_unknown_cameras():
    result = make_adapter().measure({"top": make_view()}, PointQuery(camera="side", u=1.0, v=1.0))
    assert "unknown camera" in result["error"]


def test_requests_over_the_round_limit_are_reported_as_skipped():
    adapter = make_adapter(max_zooms_per_round=1, max_points_per_round=1)
    adapter.extract_vlm_input_from_observation(None, None, 0)
    zoom = {"camera": "top", "u": 16.0, "v": 12.0, "half_size": 8.0}
    point = {"camera": "top", "u": 16.0, "v": 12.0}
    requests = adapter.read_tool_requests({"zoom_requests": [zoom, zoom], "point_queries": [point, point]})
    content = adapter.answer_tool_requests(requests, 0, is_last_round=True)
    assert [part["type"] for part in content] == ["text", "image_url", "text", "text", "text"]
    assert "skipped" in json.loads(content[2]["text"])["zoom"]["error"]
    measurements = json.loads(content[3]["text"])["point_measurements"]
    assert "point_root_m" in measurements[0] and "skipped" in measurements[1]["error"]
    assert "return an action now" in content[4]["text"]


def test_adapters_without_tools_ask_for_no_tool_rounds():
    adapter = StaticImageAdapter()
    assert adapter.max_tool_rounds == 0 and adapter.get_system_prompt_addition() == ""
    assert adapter.extend_response_model(TargetResponse) is TargetResponse
    assert adapter.read_tool_requests(MEASUREMENT_PAYLOAD) is None
