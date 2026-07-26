# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One-shot physical outcome evidence for the CAP grocery-to-bin producer."""

from __future__ import annotations

import datetime
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from isaaclab_arena.integrations.cap_barrier.grocery_scene_spec import (
    CAP_GROCERY_BIN_ASSET,
    CAP_GROCERY_OBJECT_ASSET,
)

CAP_GROCERY_PHYSICAL_RESULT_SCHEMA = "cap.grocery-physical-result.v1"

# These are the existing grey-bin packing criterion defaults in
# isaaclab_arena_environments.libero_object_packing_environment.
GREY_BIN_MAX_X_SEPARATION_M = 0.15
GREY_BIN_MAX_Y_SEPARATION_M = 0.10
GREY_BIN_MAX_Z_SEPARATION_M = 0.11
GREY_BIN_LINEAR_VELOCITY_THRESHOLD_M_S = 0.05
GREY_BIN_ANGULAR_VELOCITY_THRESHOLD_RAD_S = 0.5

_CRITERION_NAME = "isaaclab_arena.tasks.terminations.resting_in_bin"
_REQUEST_POLL_EVERY_FRAMES = 20


def _vector(values) -> list[float]:
    return [float(value) for value in values]


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _capture_scene_poses(environment) -> dict[str, dict[str, list[float]]]:
    from isaaclab_arena.recording.common_terms import world_pose_xyz_quat_xyzw

    return {
        name: world_pose_xyz_quat_xyzw(environment, name, 0)
        for name in (CAP_GROCERY_OBJECT_ASSET, CAP_GROCERY_BIN_ASSET)
    }


def capture_grocery_physical_result(adapter) -> dict[str, Any]:
    """Evaluate the existing grey-bin predicate and capture its live inputs."""
    import warp as wp
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.tasks.terminations import resting_in_bin

    adapter.synchronize()
    environment = adapter._unwrapped
    live_object_poses = _capture_scene_poses(environment)
    grocery_object = environment.scene[CAP_GROCERY_OBJECT_ASSET]
    grocery_bin = environment.scene[CAP_GROCERY_BIN_ASSET]
    object_linear_velocity = _vector(
        wp.to_torch(grocery_object.data.root_lin_vel_w)[0]
    )
    object_angular_velocity = _vector(
        wp.to_torch(grocery_object.data.root_ang_vel_w)[0]
    )
    bin_linear_velocity = _vector(wp.to_torch(grocery_bin.data.root_lin_vel_w)[0])
    bin_angular_velocity = _vector(wp.to_torch(grocery_bin.data.root_ang_vel_w)[0])
    success = bool(
        resting_in_bin(
            environment,
            object_cfg=SceneEntityCfg(CAP_GROCERY_OBJECT_ASSET),
            target_object_cfg=SceneEntityCfg(CAP_GROCERY_BIN_ASSET),
            max_x_separation=GREY_BIN_MAX_X_SEPARATION_M,
            max_y_separation=GREY_BIN_MAX_Y_SEPARATION_M,
            max_z_separation=GREY_BIN_MAX_Z_SEPARATION_M,
            lin_vel_threshold=GREY_BIN_LINEAR_VELOCITY_THRESHOLD_M_S,
            ang_vel_threshold=GREY_BIN_ANGULAR_VELOCITY_THRESHOLD_RAD_S,
        )[0].item()
    )
    object_position = live_object_poses[CAP_GROCERY_OBJECT_ASSET]["pos_w"]
    bin_position = live_object_poses[CAP_GROCERY_BIN_ASSET]["pos_w"]

    return {
        "success": success,
        "criterion": _CRITERION_NAME,
        "criterion_parameters": {
            "max_x_separation_m": GREY_BIN_MAX_X_SEPARATION_M,
            "max_y_separation_m": GREY_BIN_MAX_Y_SEPARATION_M,
            "max_z_separation_m": GREY_BIN_MAX_Z_SEPARATION_M,
            "linear_velocity_threshold_m_s": GREY_BIN_LINEAR_VELOCITY_THRESHOLD_M_S,
            "angular_velocity_threshold_rad_s": GREY_BIN_ANGULAR_VELOCITY_THRESHOLD_RAD_S,
        },
        "object_name": CAP_GROCERY_OBJECT_ASSET,
        "bin_name": CAP_GROCERY_BIN_ASSET,
        "live_object_poses": live_object_poses,
        "object_linear_velocity_m_s": object_linear_velocity,
        "object_angular_velocity_rad_s": object_angular_velocity,
        "bin_linear_velocity_m_s": bin_linear_velocity,
        "bin_angular_velocity_rad_s": bin_angular_velocity,
        "observed_separation_m": {
            axis: abs(float(object_position[index]) - float(bin_position[index]))
            for index, axis in enumerate(("x", "y", "z"))
        },
        "observed_object_linear_speed_m_s": _norm(object_linear_velocity),
        "observed_object_angular_speed_rad_s": _norm(object_angular_velocity),
        "observed_bin_linear_speed_m_s": _norm(bin_linear_velocity),
        "observed_bin_angular_speed_rad_s": _norm(bin_angular_velocity),
        "gripper_position_rad": float(adapter.gripper_position()),
        "producer_physics_step_count": int(adapter.physics_step_count),
    }


def _atomic_write_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"physical result already exists: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"physical result directory does not exist: {path.parent}")

    payload = json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard-link publication is atomic and refuses to overwrite a stale or
        # competing result, unlike os.replace().
        os.link(temporary_path, path)
        temporary_path.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


class GroceryPhysicalResultObserver:
    """Write exactly one physical result after the supervisor requests it."""

    def __init__(
        self,
        *,
        request_path: Path,
        result_path: Path,
        capture: Callable[[], Mapping[str, Any]],
        marker_sink: Callable[[str], None],
    ) -> None:
        if not request_path.is_absolute() or not result_path.is_absolute():
            raise ValueError("physical result request and output paths must be absolute")
        if request_path == result_path:
            raise ValueError("physical result request and output paths must differ")
        if request_path.exists():
            raise FileExistsError(f"physical result request already exists: {request_path}")
        if result_path.exists():
            raise FileExistsError(f"physical result already exists: {result_path}")
        self._request_path = request_path
        self._result_path = result_path
        self._capture = capture
        self._marker_sink = marker_sink
        self._attempted = False

    def on_physics_frame(self, frame: int) -> None:
        if self._attempted or frame % _REQUEST_POLL_EVERY_FRAMES != 0:
            return
        if not self._request_path.exists():
            return
        if not self._request_path.is_file():
            raise RuntimeError(
                f"physical result request is not a regular file: {self._request_path}"
            )

        self._attempted = True
        try:
            captured = self._capture()
            if not isinstance(captured, Mapping):
                raise TypeError("physical result capture must return a mapping")
            record = dict(captured)
            if type(record.get("success")) is not bool:
                raise TypeError("physical result success must be a bool")
            record["schema_version"] = CAP_GROCERY_PHYSICAL_RESULT_SCHEMA
            record["serve_generation_frame_index"] = int(frame)
            record["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
            _atomic_write_jsonl(self._result_path, record)
        except BaseException as error:
            detail = str(error).replace("\n", " ")
            self._marker_sink(
                "CAP_GROCERY_PHYSICAL_RESULT_ERROR "
                f"path={self._result_path} detail={type(error).__name__}:{detail}"
            )
            raise

        self._marker_sink(
            "CAP_GROCERY_PHYSICAL_RESULT "
            f"success={int(record['success'])} path={self._result_path}"
        )


def make_grocery_physical_result_observer(
    adapter,
    marker_sink: Callable[[str], None],
    *,
    request_path: str | os.PathLike[str],
    result_path: str | os.PathLike[str],
) -> Callable[[int], None]:
    """Build the one-shot main-thread physical-result callback."""
    request = Path(request_path)
    result = Path(result_path)
    observer = GroceryPhysicalResultObserver(
        request_path=request,
        result_path=result,
        capture=lambda: capture_grocery_physical_result(adapter),
        marker_sink=marker_sink,
    )
    return observer.on_physics_frame
