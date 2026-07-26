# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import math
import threading
from types import SimpleNamespace

import pytest

import isaaclab_arena.integrations.cap_barrier.grocery_physical_result as result_module
from isaaclab_arena.integrations.cap_barrier.grocery_physical_result import (
    CAP_GROCERY_PHYSICAL_RESULT_SCHEMA,
    GREY_BIN_ANGULAR_VELOCITY_THRESHOLD_RAD_S,
    GREY_BIN_LINEAR_VELOCITY_THRESHOLD_M_S,
    GREY_BIN_MAX_X_SEPARATION_M,
    GREY_BIN_MAX_Y_SEPARATION_M,
    GREY_BIN_MAX_Z_SEPARATION_M,
    GroceryPhysicalResultObserver,
    capture_grocery_physical_result,
)
from isaaclab_arena.integrations.cap_barrier.grocery_scene_spec import (
    CAP_GROCERY_BIN_ASSET,
    CAP_GROCERY_OBJECT_ASSET,
)
from isaaclab_arena_environments.libero_object_packing_environment import (
    _make_libero_packing_task,
)


def _observer(tmp_path, *, success: bool, capture=None):
    request_path = tmp_path / "physical-result.request"
    result_path = tmp_path / "physical-result.jsonl"
    markers: list[str] = []
    capture_calls: list[int] = []
    expected_thread = threading.get_ident()

    def default_capture():
        capture_calls.append(threading.get_ident())
        return {"success": success, "criterion": "synthetic"}

    def marker_sink(marker: str) -> None:
        if marker.startswith("CAP_GROCERY_PHYSICAL_RESULT success="):
            assert result_path.is_file()
        markers.append(marker)

    observer = GroceryPhysicalResultObserver(
        request_path=request_path,
        result_path=result_path,
        capture=capture or default_capture,
        marker_sink=marker_sink,
    )
    return (
        observer,
        request_path,
        result_path,
        markers,
        capture_calls,
        expected_thread,
    )


def test_result_observer_waits_for_request_then_publishes_exactly_once(tmp_path) -> None:
    observer, request, result, markers, calls, expected_thread = _observer(
        tmp_path,
        success=True,
    )

    observer.on_physics_frame(0)
    observer.on_physics_frame(20)
    assert not result.exists()
    assert not calls

    request.write_text("evaluate\n", encoding="utf-8")
    observer.on_physics_frame(21)
    assert not result.exists()
    observer.on_physics_frame(40)
    first_bytes = result.read_bytes()
    observer.on_physics_frame(60)

    assert calls == [expected_thread]
    assert result.read_bytes() == first_bytes
    assert first_bytes.endswith(b"\n") and first_bytes.count(b"\n") == 1
    record = json.loads(first_bytes)
    assert record["schema_version"] == CAP_GROCERY_PHYSICAL_RESULT_SCHEMA
    assert record["success"] is True
    assert record["serve_generation_frame_index"] == 40
    assert len(markers) == 1
    assert markers[0].startswith("CAP_GROCERY_PHYSICAL_RESULT success=1 ")
    assert not list(tmp_path.glob(f".{result.name}.*"))


def test_result_observer_publishes_truthful_false_instead_of_timing_out(tmp_path) -> None:
    observer, request, result, markers, *_ = _observer(tmp_path, success=False)
    request.write_text("evaluate\n", encoding="utf-8")

    observer.on_physics_frame(0)

    record = json.loads(result.read_text(encoding="utf-8"))
    assert record["success"] is False
    assert markers == [
        f"CAP_GROCERY_PHYSICAL_RESULT success=0 path={result}"
    ]


def test_result_observer_propagates_capture_error_without_success_record(tmp_path) -> None:
    def fail():
        raise RuntimeError("synthetic capture failure")

    observer, request, result, markers, *_ = _observer(
        tmp_path,
        success=False,
        capture=fail,
    )
    request.write_text("evaluate\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="synthetic capture failure"):
        observer.on_physics_frame(0)

    assert not result.exists()
    assert len(markers) == 1
    assert markers[0].startswith("CAP_GROCERY_PHYSICAL_RESULT_ERROR ")
    assert "CAP_GROCERY_PHYSICAL_RESULT success=" not in markers[0]
    assert not list(tmp_path.glob(f".{result.name}.*"))


def test_result_observer_rejects_nonfinite_evidence(tmp_path) -> None:
    observer, request, result, markers, *_ = _observer(
        tmp_path,
        success=True,
        capture=lambda: {"success": True, "bad": math.nan},
    )
    request.write_text("evaluate\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Out of range float values"):
        observer.on_physics_frame(0)

    assert not result.exists()
    assert markers[0].startswith("CAP_GROCERY_PHYSICAL_RESULT_ERROR ")


def test_result_observer_rejects_stale_request_or_result(tmp_path) -> None:
    request = tmp_path / "request"
    result = tmp_path / "result.jsonl"
    request.write_text("stale\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="request already exists"):
        GroceryPhysicalResultObserver(
            request_path=request,
            result_path=result,
            capture=lambda: {"success": True},
            marker_sink=lambda _marker: None,
        )

    request.unlink()
    result.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="result already exists"):
        GroceryPhysicalResultObserver(
            request_path=request,
            result_path=result,
            capture=lambda: {"success": True},
            marker_sink=lambda _marker: None,
        )


def test_capture_uses_existing_resting_in_bin_criterion_and_live_inputs(
    monkeypatch,
) -> None:
    import warp as wp
    import isaaclab_arena.tasks.terminations as terminations

    object_data = SimpleNamespace(
        root_lin_vel_w=[[0.01, 0.02, 0.03]],
        root_ang_vel_w=[[0.1, 0.2, 0.3]],
    )
    bin_data = SimpleNamespace(
        root_lin_vel_w=[[0.001, 0.002, 0.003]],
        root_ang_vel_w=[[0.01, 0.02, 0.03]],
    )
    environment = SimpleNamespace(
        scene={
            CAP_GROCERY_OBJECT_ASSET: SimpleNamespace(data=object_data),
            CAP_GROCERY_BIN_ASSET: SimpleNamespace(data=bin_data),
        }
    )
    adapter = SimpleNamespace(
        _unwrapped=environment,
        physics_step_count=1234,
        synchronize=lambda: None,
        gripper_position=lambda: 0.0,
    )
    predicate_call: dict[str, object] = {}

    def resting_in_bin(environment_arg, **kwargs):
        predicate_call["environment"] = environment_arg
        predicate_call.update(kwargs)
        return [SimpleNamespace(item=lambda: True)]

    poses = {
        CAP_GROCERY_OBJECT_ASSET: {
            "pos_w": [0.34, 0.15, 0.06],
            "quat_w_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        CAP_GROCERY_BIN_ASSET: {
            "pos_w": [0.46, -0.15, 0.00],
            "quat_w_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    monkeypatch.setattr(wp, "to_torch", lambda values: values)
    monkeypatch.setattr(terminations, "resting_in_bin", resting_in_bin)
    monkeypatch.setattr(result_module, "_capture_scene_poses", lambda _env: poses)

    record = capture_grocery_physical_result(adapter)

    assert record["success"] is True
    assert record["live_object_poses"] == poses
    assert record["object_linear_velocity_m_s"] == [0.01, 0.02, 0.03]
    assert record["object_angular_velocity_rad_s"] == [0.1, 0.2, 0.3]
    assert record["bin_linear_velocity_m_s"] == [0.001, 0.002, 0.003]
    assert record["bin_angular_velocity_rad_s"] == [0.01, 0.02, 0.03]
    assert record["producer_physics_step_count"] == 1234
    assert predicate_call["environment"] is environment
    assert predicate_call["object_cfg"].name == CAP_GROCERY_OBJECT_ASSET
    assert predicate_call["target_object_cfg"].name == CAP_GROCERY_BIN_ASSET
    assert predicate_call["max_x_separation"] == GREY_BIN_MAX_X_SEPARATION_M
    assert predicate_call["max_y_separation"] == GREY_BIN_MAX_Y_SEPARATION_M
    assert predicate_call["max_z_separation"] == GREY_BIN_MAX_Z_SEPARATION_M
    assert (
        predicate_call["lin_vel_threshold"]
        == GREY_BIN_LINEAR_VELOCITY_THRESHOLD_M_S
    )
    assert (
        predicate_call["ang_vel_threshold"]
        == GREY_BIN_ANGULAR_VELOCITY_THRESHOLD_RAD_S
    )


def test_physical_criterion_constants_track_existing_grey_bin_task_defaults() -> None:
    parameters = inspect.signature(_make_libero_packing_task).parameters

    assert parameters["max_x_separation"].default == GREY_BIN_MAX_X_SEPARATION_M
    assert parameters["max_y_separation"].default == GREY_BIN_MAX_Y_SEPARATION_M
    assert parameters["max_z_separation"].default == GREY_BIN_MAX_Z_SEPARATION_M
    assert (
        parameters["lin_vel_threshold"].default
        == GREY_BIN_LINEAR_VELOCITY_THRESHOLD_M_S
    )
    assert (
        parameters["ang_vel_threshold"].default
        == GREY_BIN_ANGULAR_VELOCITY_THRESHOLD_RAD_S
    )
