# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-physics-step retention diagnostics for the CAP grocery producer."""

from __future__ import annotations

import json
import math
import os
import queue
import re
import stat
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .grocery_scene_spec import CAP_GROCERY_OBJECT_ASSET
from .joint_mapping import PANDA_ARM_JOINTS

CAP_GROCERY_RETENTION_TRACE_SCHEMA = "cap.grocery-retention-trace.v1"
CAP_GROCERY_RETENTION_TRACE_FLUSH_EVERY_RECORDS = 20
CAP_GROCERY_RETENTION_CONTACT_MEASUREMENTS_INCLUDED = False
CAP_GROCERY_RETENTION_TRACE_QUEUE_CAPACITY = 64
CAP_GROCERY_RETENTION_TRACE_CALLBACK_TARGET_NS = 500_000
CAP_GROCERY_RETENTION_TRACE_CALLBACK_POISON_NS = 5_000_000
CAP_GROCERY_RETENTION_TRACE_CLOSE_TIMEOUT_S = 30.0

_FINGER_FRAME_NAMES = {
    "left": "tool_leftfinger",
    "right": "tool_rightfinger",
}
_ROBOTIQ_JOINT_NAMES = (
    "finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
    "left_inner_finger_joint",
)
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def _require_boot_id(value: str) -> str:
    if not isinstance(value, str) or _BOOT_ID_RE.fullmatch(value) is None:
        raise RuntimeError(f"invalid kernel boot identity {value!r}")
    return value


def _read_boot_id(path: Path = _BOOT_ID_PATH) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"kernel boot identity is not a regular file: {path}")
        payload = os.read(descriptor, 64)
        if os.read(descriptor, 1):
            raise RuntimeError("kernel boot identity exceeds 64 bytes")
    finally:
        os.close(descriptor)
    try:
        value = payload.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError("kernel boot identity is not ASCII") from error
    return _require_boot_id(value)


def _as_python(value: Any) -> Any:
    value = value.torch if hasattr(value, "torch") else value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def _finite_vector(values: Sequence[Any], *, length: int, label: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != length:
        raise RuntimeError(f"{label} must contain {length} values, got {len(result)}")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} contains a nonfinite value")
    return result


def _first_vector(value: Any, *, length: int, label: str) -> list[float]:
    rows = _as_python(value)
    if len(rows) != 1:
        raise RuntimeError(
            f"{label} must contain exactly one environment row, got {len(rows)}"
        )
    return _finite_vector(rows[0], length=length, label=label)


def _normalized_quaternion_xyzw(
    values: Sequence[float], *, label: str
) -> tuple[float, float, float, float]:
    quaternion = tuple(_finite_vector(values, length=4, label=label))
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 0.0:
        raise ValueError(f"{label} has zero norm")
    return tuple(value / norm for value in quaternion)


def _quaternion_multiply_xyzw(
    lhs: Sequence[float],
    rhs: Sequence[float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _pose_in_reference_frame(
    reference_position_w: Sequence[float],
    reference_quaternion_w_xyzw: Sequence[float],
    target_position_w: Sequence[float],
    target_quaternion_w_xyzw: Sequence[float],
) -> dict[str, list[float]]:
    """Return the target pose expressed in the reference frame."""
    reference_quaternion = _normalized_quaternion_xyzw(
        reference_quaternion_w_xyzw,
        label="reference world quaternion",
    )
    target_quaternion = _normalized_quaternion_xyzw(
        target_quaternion_w_xyzw,
        label="target world quaternion",
    )
    reference_inverse = (
        -reference_quaternion[0],
        -reference_quaternion[1],
        -reference_quaternion[2],
        reference_quaternion[3],
    )
    delta = tuple(
        float(target) - float(reference)
        for reference, target in zip(
            reference_position_w, target_position_w, strict=True
        )
    )
    rotated_delta = _quaternion_multiply_xyzw(
        _quaternion_multiply_xyzw(reference_inverse, (*delta, 0.0)),
        reference_quaternion,
    )[:3]
    relative_quaternion = _normalized_quaternion_xyzw(
        _quaternion_multiply_xyzw(reference_inverse, target_quaternion),
        label="relative quaternion",
    )
    return {
        "position_m": list(rotated_delta),
        "quaternion_xyzw": list(relative_quaternion),
    }


def _pose(
    position: Sequence[Any], quaternion_xyzw: Sequence[Any], *, label: str
) -> dict[str, list[float]]:
    return {
        "position_m": _finite_vector(position, length=3, label=f"{label} position"),
        "quaternion_xyzw": _finite_vector(
            quaternion_xyzw, length=4, label=f"{label} quaternion"
        ),
    }


def _require_retention_roster(joint_names: Sequence[str]) -> tuple[int, ...]:
    robotiq_indices = tuple(
        index for index, name in enumerate(joint_names) if name not in PANDA_ARM_JOINTS
    )
    robotiq_names = tuple(joint_names[index] for index in robotiq_indices)
    if robotiq_names != _ROBOTIQ_JOINT_NAMES:
        raise RuntimeError(
            "CAP grocery retention trace requires the exact physical Robotiq joint roster, "
            f"expected={list(_ROBOTIQ_JOINT_NAMES)}, got={list(robotiq_names)}"
        )
    return robotiq_indices


def _retention_state_from_samples(
    *,
    joint_names: Sequence[str],
    positions: Sequence[Any],
    velocities: Sequence[Any],
    applied_torques: Sequence[Any],
    can_position_w: Sequence[Any],
    can_quaternion_w: Sequence[Any],
    can_linear_velocity_w: Sequence[Any],
    can_angular_velocity_w: Sequence[Any],
    target_frame_names: Sequence[str],
    target_positions_w: Sequence[Sequence[Any]],
    target_quaternions_w: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    """Materialize one validated retention state from host samples."""
    if not (
        len(joint_names) == len(positions) == len(velocities) == len(applied_torques)
    ):
        raise RuntimeError(
            "DROID joint state arrays do not match the articulation joint roster"
        )
    robotiq_indices = _require_retention_roster(joint_names)
    robotiq_joints = []
    for index in robotiq_indices:
        values = _finite_vector(
            (positions[index], velocities[index], applied_torques[index]),
            length=3,
            label=f"Robotiq joint {joint_names[index]} state",
        )
        robotiq_joints.append(
            {
                "name": joint_names[index],
                "position_rad": values[0],
                "velocity_rad_s": values[1],
                "applied_torque_diagnostic_n_m": values[2],
            }
        )

    # Pinned BaseRigidObjectData and BaseFrameTransformerData both expose xyzw.
    # Keep the convention named in every serialized quaternion field as well.
    can_world_pose = _pose(
        can_position_w, can_quaternion_w, label="grocery can world pose"
    )

    if len(target_positions_w) != len(target_frame_names) or len(
        target_quaternions_w
    ) != len(target_frame_names):
        raise RuntimeError(
            "inner-finger frame arrays do not match the target frame roster"
        )

    inner_finger_frames = {}
    for side, frame_name in _FINGER_FRAME_NAMES.items():
        if target_frame_names.count(frame_name) != 1:
            raise RuntimeError(
                f"CAP grocery retention trace requires exactly one {frame_name!r} target frame, "
                f"got {target_frame_names}"
            )
        frame_index = target_frame_names.index(frame_name)
        world_pose = _pose(
            target_positions_w[frame_index],
            target_quaternions_w[frame_index],
            label=f"{side} inner-finger world pose",
        )
        inner_finger_frames[side] = {
            "frame_name": frame_name,
            "world_pose": world_pose,
            "pose_in_can_frame": _pose_in_reference_frame(
                can_world_pose["position_m"],
                can_world_pose["quaternion_xyzw"],
                world_pose["position_m"],
                world_pose["quaternion_xyzw"],
            ),
        }

    return {
        "robotiq_joints": robotiq_joints,
        "can": {
            "asset_name": CAP_GROCERY_OBJECT_ASSET,
            "world_pose": can_world_pose,
            "linear_velocity_world_m_s": _finite_vector(
                can_linear_velocity_w,
                length=3,
                label="grocery can world linear velocity",
            ),
            "angular_velocity_world_rad_s": _finite_vector(
                can_angular_velocity_w,
                length=3,
                label="grocery can world angular velocity",
            ),
        },
        "inner_finger_frames": inner_finger_frames,
    }


def capture_grocery_retention_state(adapter: Any) -> dict[str, Any]:
    """Synchronously capture one state outside the production physics callback."""
    adapter.synchronize()
    environment = adapter._unwrapped
    joint_names = tuple(adapter.joint_names)
    positions, velocities, applied_torques = adapter.read_joint_state()
    grocery_object = environment.scene[CAP_GROCERY_OBJECT_ASSET]
    frame_data = environment.scene["ee_frame"].data
    target_positions_w = _as_python(frame_data.target_pos_w)
    target_quaternions_w = _as_python(frame_data.target_quat_w)
    if len(target_positions_w) != 1 or len(target_quaternions_w) != 1:
        raise RuntimeError(
            "inner-finger frame data must contain exactly one environment row"
        )
    return _retention_state_from_samples(
        joint_names=joint_names,
        positions=positions,
        velocities=velocities,
        applied_torques=applied_torques,
        can_position_w=_first_vector(
            grocery_object.data.root_pos_w,
            length=3,
            label="grocery can world position",
        ),
        can_quaternion_w=_first_vector(
            grocery_object.data.root_quat_w,
            length=4,
            label="grocery can world quaternion",
        ),
        can_linear_velocity_w=_first_vector(
            grocery_object.data.root_lin_vel_w,
            length=3,
            label="grocery can world linear velocity",
        ),
        can_angular_velocity_w=_first_vector(
            grocery_object.data.root_ang_vel_w,
            length=3,
            label="grocery can world angular velocity",
        ),
        target_frame_names=tuple(frame_data.target_frame_names),
        target_positions_w=target_positions_w[0],
        target_quaternions_w=target_quaternions_w[0],
    )


def _canonical_jsonl_line(record: Mapping[str, Any]) -> str:
    return (
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def _open_exclusive_text(path: Path) -> TextIO:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        return os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n", buffering=64 * 1024
        )
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


@dataclass
class _SnapshotSlot:
    index: int
    decision_ready: threading.Event
    persist: bool = False
    payload: Mapping[str, Any] | None = None
    host_values: Any = None
    device_values: Any = None
    copy_complete: Any = None
    copy_recorded: bool = False
    stage_stream: Any = None


@dataclass(frozen=True)
class _PendingRecord:
    slot: _SnapshotSlot
    record_without_state: Mapping[str, Any]


@dataclass(frozen=True)
class _FlushRequest:
    done: threading.Event


_STOP_WORKER = object()


class _MappingSnapshotter:
    """Stage test-only mapping captures without owning production tensor state."""

    def __init__(self, capture: Callable[[], Mapping[str, Any]]) -> None:
        self._capture = capture

    def make_slots(self, count: int) -> tuple[_SnapshotSlot, ...]:
        return tuple(
            _SnapshotSlot(index=index, decision_ready=threading.Event())
            for index in range(count)
        )

    def preflight(self, _slot: _SnapshotSlot) -> None:
        pass

    def stage(self, slot: _SnapshotSlot) -> None:
        captured = self._capture()
        if not isinstance(captured, Mapping):
            raise TypeError("grocery retention trace capture must return a mapping")
        slot.payload = dict(captured)

    def materialize(self, slot: _SnapshotSlot) -> Mapping[str, Any]:
        if slot.payload is None:
            raise RuntimeError(
                "grocery retention trace mapping slot has no staged payload"
            )
        return slot.payload

    def discard(self, _slot: _SnapshotSlot) -> None:
        pass

    def release(self, slot: _SnapshotSlot) -> None:
        slot.payload = None


class _TorchRetentionSnapshotter:
    """Stage live tensors into owned slots without synchronizing the Kit thread."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._torch = adapter._torch
        self._device = self._torch.device(adapter._unwrapped.device)
        self._is_cuda = self._device.type == "cuda"
        self._joint_names = tuple(adapter.joint_names)
        _require_retention_roster(self._joint_names)
        frame_data = adapter._unwrapped.scene["ee_frame"].data
        self._target_frame_names = tuple(frame_data.target_frame_names)
        for frame_name in _FINGER_FRAME_NAMES.values():
            if self._target_frame_names.count(frame_name) != 1:
                raise RuntimeError(
                    f"CAP grocery retention trace requires exactly one {frame_name!r} target frame, "
                    f"got {self._target_frame_names}"
                )

        sources = self._source_views()
        self._source_lengths = tuple(source.numel() for source in sources)
        self._value_count = sum(self._source_lengths)
        if self._value_count <= 0:
            raise RuntimeError("grocery retention trace tensor snapshot has no values")
        source_dtype = sources[0].dtype
        for source in sources:
            if source.device != self._device:
                raise RuntimeError(
                    "grocery retention trace tensor source is on the wrong device: "
                    f"expected={self._device}, got={source.device}"
                )
            if source.dtype != source_dtype:
                raise RuntimeError(
                    "grocery retention trace tensor sources must use one dtype: "
                    f"expected={source_dtype}, got={source.dtype}"
                )
        self._dtype = source_dtype

    @staticmethod
    def _as_tensor(value: Any) -> Any:
        return value.torch if hasattr(value, "torch") else value

    def _source_views(self) -> tuple[Any, ...]:
        grocery_object = self._adapter._unwrapped.scene[CAP_GROCERY_OBJECT_ASSET]
        frame_data = self._adapter._unwrapped.scene["ee_frame"].data
        sources = (
            self._adapter._robot.data.joint_pos,
            self._adapter._robot.data.joint_vel,
            self._adapter._robot.data.applied_torque,
            grocery_object.data.root_pos_w,
            grocery_object.data.root_quat_w,
            grocery_object.data.root_lin_vel_w,
            grocery_object.data.root_ang_vel_w,
            frame_data.target_pos_w,
            frame_data.target_quat_w,
        )
        return tuple(self._as_tensor(source).detach().reshape(-1) for source in sources)

    def make_slots(self, count: int) -> tuple[_SnapshotSlot, ...]:
        slots = []
        for index in range(count):
            host_values = self._torch.empty(
                self._value_count,
                dtype=self._dtype,
                device="cpu",
                pin_memory=self._is_cuda,
            )
            if self._is_cuda and not host_values.is_pinned():
                raise RuntimeError(
                    "grocery retention trace CUDA host slot is not pinned"
                )
            slots.append(
                _SnapshotSlot(
                    index=index,
                    decision_ready=threading.Event(),
                    host_values=host_values,
                    device_values=self._torch.empty(
                        self._value_count,
                        dtype=self._dtype,
                        device=self._device,
                    ),
                    copy_complete=self._torch.cuda.Event(enable_timing=False)
                    if self._is_cuda
                    else None,
                )
            )
        return tuple(slots)

    def preflight(self, slot: _SnapshotSlot) -> None:
        try:
            self.stage(slot)
        finally:
            self.discard(slot)

    def stage(self, slot: _SnapshotSlot) -> None:
        sources = self._source_views()
        if tuple(source.numel() for source in sources) != self._source_lengths:
            raise RuntimeError(
                "grocery retention trace tensor source shape changed after construction"
            )
        slot.copy_recorded = False
        slot.stage_stream = (
            self._torch.cuda.current_stream(self._device) if self._is_cuda else None
        )
        try:
            self._torch.cat(sources, out=slot.device_values)
            slot.host_values.copy_(slot.device_values, non_blocking=self._is_cuda)
        finally:
            if self._is_cuda:
                # Even a failed stage may have queued earlier work. Recording in
                # finally gives close a bounded ownership fence for that slot.
                slot.copy_complete.record(slot.stage_stream)
            slot.copy_recorded = True

    def _wait_for_copy(self, slot: _SnapshotSlot, *, allow_failed_stage: bool) -> None:
        if not slot.copy_recorded:
            if not allow_failed_stage:
                raise RuntimeError(
                    "grocery retention trace tensor slot has no recorded copy completion"
                )
            if self._is_cuda and slot.stage_stream is not None:
                # Cleanup-only fallback if CUDA event recording itself failed.
                # Synchronize the exact stream that owned the staged work; CUDA
                # current-stream state is thread-local and cleanup runs elsewhere.
                slot.stage_stream.synchronize()
                slot.stage_stream = None
            return
        if self._is_cuda:
            # This wait is worker-owned. The Kit callback never calls a CUDA
            # synchronization primitive.
            slot.copy_complete.synchronize()
        slot.copy_recorded = False
        slot.stage_stream = None

    def materialize(self, slot: _SnapshotSlot) -> Mapping[str, Any]:
        self._wait_for_copy(slot, allow_failed_stage=False)
        values = slot.host_values.tolist()
        chunks = []
        offset = 0
        for length in self._source_lengths:
            chunks.append(values[offset : offset + length])
            offset += length
        (
            positions,
            velocities,
            applied_torques,
            can_position_w,
            can_quaternion_w,
            can_linear_velocity_w,
            can_angular_velocity_w,
            target_positions_flat,
            target_quaternions_flat,
        ) = chunks
        frame_count = len(self._target_frame_names)
        target_positions_w = [
            target_positions_flat[index * 3 : (index + 1) * 3]
            for index in range(frame_count)
        ]
        target_quaternions_w = [
            target_quaternions_flat[index * 4 : (index + 1) * 4]
            for index in range(frame_count)
        ]
        return _retention_state_from_samples(
            joint_names=self._joint_names,
            positions=positions,
            velocities=velocities,
            applied_torques=applied_torques,
            can_position_w=can_position_w,
            can_quaternion_w=can_quaternion_w,
            can_linear_velocity_w=can_linear_velocity_w,
            can_angular_velocity_w=can_angular_velocity_w,
            target_frame_names=self._target_frame_names,
            target_positions_w=target_positions_w,
            target_quaternions_w=target_quaternions_w,
        )

    def discard(self, slot: _SnapshotSlot) -> None:
        self._wait_for_copy(slot, allow_failed_stage=True)

    def release(self, _slot: _SnapshotSlot) -> None:
        pass


class GroceryRetentionTraceWriter:
    """Stage bounded physics snapshots and write canonical rows off-thread."""

    def __init__(
        self,
        adapter: Any,
        marker_sink: Callable[[str], None],
        *,
        output_path: str | os.PathLike[str],
        flush_every_records: int = CAP_GROCERY_RETENTION_TRACE_FLUSH_EVERY_RECORDS,
        capture: Callable[[], Mapping[str, Any]] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        callback_clock_ns: Callable[[], int] = time.perf_counter_ns,
        queue_capacity: int = CAP_GROCERY_RETENTION_TRACE_QUEUE_CAPACITY,
        close_timeout_s: float = CAP_GROCERY_RETENTION_TRACE_CLOSE_TIMEOUT_S,
        _snapshotter: Any = None,
        _boot_id: str | None = None,
    ) -> None:
        path = Path(output_path)
        if not path.is_absolute():
            raise ValueError("grocery retention trace output path must be absolute")
        if path.suffix.lower() != ".jsonl":
            raise ValueError("grocery retention trace output path must end in .jsonl")
        if not path.parent.is_dir():
            raise FileNotFoundError(
                f"grocery retention trace output directory does not exist: {path.parent}"
            )
        if type(flush_every_records) is not int or flush_every_records <= 0:
            raise ValueError(
                "grocery retention trace flush interval must be a positive integer"
            )
        if type(queue_capacity) is not int or queue_capacity <= 0:
            raise ValueError(
                "grocery retention trace queue capacity must be a positive integer"
            )
        if not math.isfinite(close_timeout_s) or close_timeout_s <= 0.0:
            raise ValueError(
                "grocery retention trace close timeout must be finite and positive"
            )
        if capture is not None and _snapshotter is not None:
            raise ValueError(
                "grocery retention trace accepts either capture or _snapshotter, not both"
            )
        boot_id = (
            _require_boot_id(_boot_id) if _boot_id is not None else _read_boot_id()
        )

        snapshotter = _snapshotter
        if snapshotter is None:
            snapshotter = (
                _MappingSnapshotter(capture)
                if capture is not None
                else _TorchRetentionSnapshotter(adapter)
            )
        slots = snapshotter.make_slots(queue_capacity)
        if len(slots) != queue_capacity:
            raise RuntimeError(
                f"grocery retention trace snapshotter returned {len(slots)} slots, expected {queue_capacity}"
            )
        snapshotter.preflight(slots[0])

        self._adapter = adapter
        self._marker_sink = marker_sink
        self._path = path
        self._flush_every_records = flush_every_records
        self._snapshotter = snapshotter
        self._monotonic_ns = monotonic_ns
        self._callback_clock_ns = callback_clock_ns
        self._close_timeout_s = float(close_timeout_s)
        self._boot_id = boot_id
        self._work_queue: queue.Queue[Any] = queue.Queue(maxsize=queue_capacity)
        self._free_slots: queue.Queue[_SnapshotSlot] = queue.Queue(
            maxsize=queue_capacity
        )
        for slot in slots:
            self._free_slots.put_nowait(slot)
        self._abandoned_slots: list[_SnapshotSlot] = []
        self._state_lock = threading.Lock()
        self._source_generation: int | None = None
        self._local_generation_index = 0
        self._next_generation_frame_index = 0
        self._last_physics_step_count: int | None = None
        self._last_reset_count: int | None = None
        self._generation_has_observations = False
        self._observed_count = 0
        self._record_count = 0
        self._dropped_over_target = 0
        self._dropped_queue_full = 0
        self._dropped_callback_poison = 0
        self._max_callback_ns = 0
        self._closed = False
        self._failure: BaseException | None = None
        self._failure_reported = False
        self._deferred_reaper: threading.Thread | None = None
        self._worker = threading.Thread(
            target=self._worker_main,
            name="cap-grocery-retention-trace-writer",
            daemon=True,
        )
        stream = _open_exclusive_text(path)
        self._stream = stream
        try:
            marker_sink(
                "CAP_GROCERY_RETENTION_TRACE_ARMED "
                f"path={path} flush_every_records={flush_every_records} "
                f"queue_capacity={queue_capacity} callback_target_ns={CAP_GROCERY_RETENTION_TRACE_CALLBACK_TARGET_NS} "
                f"boot_id={self._boot_id} "
                "contacts=omitted diagnostic_only=true qualification_eligible=false"
            )
            self._worker.start()
        except BaseException:
            try:
                stream.close()
            finally:
                path.unlink(missing_ok=True)
                self._closed = True
            raise

    @property
    def record_count(self) -> int:
        """Return the number of complete records accepted by the writer."""
        with self._state_lock:
            return self._record_count

    def begin_generation(self, generation: int) -> None:
        """Begin the next local generation without recording a wire generation ID."""
        self._require_writable()
        try:
            self._begin_generation(generation)
        except BaseException as error:
            self._mark_reported_failure(error)
            raise

    def _begin_generation(self, generation: int) -> None:
        if type(generation) is not int or generation <= 0:
            raise ValueError(
                "grocery retention trace generation must be a positive integer"
            )
        if (
            self._source_generation is not None
            and generation <= self._source_generation
        ):
            raise ValueError(
                f"grocery retention trace generation must advance: current={self._source_generation}, next={generation}"
            )
        if self._generation_has_observations:
            self._flush_worker()
        self._source_generation = generation
        self._local_generation_index += 1
        self._next_generation_frame_index = 0
        self._generation_has_observations = False
        self._marker_sink(
            f"CAP_GROCERY_RETENTION_TRACE_GENERATION local_generation_index={self._local_generation_index}"
        )

    def __call__(self, frame: int) -> None:
        self.on_physics_frame(frame)

    def on_physics_frame(self, frame: int) -> None:
        """Capture and append the successful physics step identified by a local frame counter."""
        self._require_writable()
        try:
            self._record_physics_frame(frame)
        except BaseException as error:
            self._mark_reported_failure(error)
            raise

    def _record_physics_frame(self, frame: int) -> None:
        callback_started_ns = self._callback_clock_ns()
        if type(callback_started_ns) is not int or callback_started_ns < 0:
            raise RuntimeError(
                f"grocery retention trace callback clock must be a nonnegative integer, got {callback_started_ns!r}"
            )
        if self._source_generation is None:
            raise RuntimeError("grocery retention trace used before begin_generation")
        if type(frame) is not int or frame != self._next_generation_frame_index:
            raise ValueError(
                "grocery retention trace frame must advance without gaps: "
                f"expected={self._next_generation_frame_index}, got={frame!r}"
            )

        physics_step_count = int(self._adapter.physics_step_count)
        reset_count = int(self._adapter.reset_count)
        if physics_step_count <= 0:
            raise RuntimeError(
                "grocery retention trace observed no completed physics step"
            )
        if (
            self._last_physics_step_count is not None
            and physics_step_count != self._last_physics_step_count + 1
        ):
            raise RuntimeError(
                "grocery retention trace physics-step counter did not advance exactly once: "
                f"previous={self._last_physics_step_count}, current={physics_step_count}"
            )
        if frame == 0:
            if (
                self._last_reset_count is not None
                and reset_count <= self._last_reset_count
            ):
                raise RuntimeError(
                    "grocery retention trace generation changed without a distinct reset: "
                    f"previous={self._last_reset_count}, current={reset_count}"
                )
        elif reset_count != self._last_reset_count:
            raise RuntimeError(
                "grocery retention trace reset counter changed inside a generation: "
                f"previous={self._last_reset_count}, current={reset_count}"
            )

        admitted_action = float(self._adapter.last_admitted_binary_gripper_action)
        if admitted_action not in (0.0, 1.0):
            raise RuntimeError(
                f"admitted DROID gripper action must be binary, got {admitted_action!r}"
            )
        timestamp_ns = self._monotonic_ns()
        if type(timestamp_ns) is not int or timestamp_ns < 0:
            raise RuntimeError(
                f"monotonic timestamp must be a nonnegative integer, got {timestamp_ns!r}"
            )
        record_without_state = {
            "schema_version": CAP_GROCERY_RETENTION_TRACE_SCHEMA,
            "record_type": "physics_step",
            "boot_id": self._boot_id,
            "monotonic_timestamp_ns": timestamp_ns,
            "local_counters": {
                "generation_index": self._local_generation_index,
                "generation_frame_index": frame,
                "generation_first_step": frame == 0,
                "physics_step_count": physics_step_count,
                "reset_count": reset_count,
            },
            "admitted_binary_gripper_action": admitted_action,
        }
        with self._state_lock:
            self._observed_count += 1
        try:
            slot = self._free_slots.get_nowait()
        except queue.Empty:
            self._advance_observation_counters(physics_step_count, reset_count)
            callback_elapsed_ns = self._finish_callback_timing(callback_started_ns)
            if callback_elapsed_ns >= CAP_GROCERY_RETENTION_TRACE_CALLBACK_POISON_NS:
                self._poison_on_callback_overrun(callback_elapsed_ns)
            with self._state_lock:
                self._dropped_queue_full += 1
            self._raise_if_failed()
            return

        slot.persist = False
        slot.decision_ready.clear()
        queued = False
        try:
            self._snapshotter.stage(slot)
            self._work_queue.put_nowait(_PendingRecord(slot, record_without_state))
            queued = True
        except BaseException as error:
            if queued:
                slot.persist = False
                slot.decision_ready.set()
            else:
                # Keep the owned buffers alive until close. A failed CUDA stage
                # may still have work ordered before its completion event.
                self._abandoned_slots.append(slot)
            self._fail_synchronously(error)

        self._advance_observation_counters(physics_step_count, reset_count)
        try:
            callback_elapsed_ns = self._finish_callback_timing(callback_started_ns)
        except BaseException as error:
            slot.persist = False
            slot.decision_ready.set()
            self._fail_synchronously(error)
        if callback_elapsed_ns >= CAP_GROCERY_RETENTION_TRACE_CALLBACK_POISON_NS:
            with self._state_lock:
                self._dropped_callback_poison += 1
            slot.persist = False
            slot.decision_ready.set()
            error = RuntimeError(
                "grocery retention trace callback exceeded the fail-closed bound: "
                f"elapsed_ns={callback_elapsed_ns}, bound_ns={CAP_GROCERY_RETENTION_TRACE_CALLBACK_POISON_NS}"
            )
            self._fail_synchronously(error)
        if callback_elapsed_ns > CAP_GROCERY_RETENTION_TRACE_CALLBACK_TARGET_NS:
            with self._state_lock:
                self._dropped_over_target += 1
            slot.persist = False
        else:
            slot.persist = True
        slot.decision_ready.set()
        self._raise_if_failed()

    def _advance_observation_counters(
        self, physics_step_count: int, reset_count: int
    ) -> None:
        self._next_generation_frame_index += 1
        self._last_physics_step_count = physics_step_count
        self._last_reset_count = reset_count
        self._generation_has_observations = True

    def _finish_callback_timing(self, callback_started_ns: int) -> int:
        callback_finished_ns = self._callback_clock_ns()
        if (
            type(callback_finished_ns) is not int
            or callback_finished_ns < callback_started_ns
        ):
            raise RuntimeError(
                "grocery retention trace callback clock moved backwards: "
                f"started={callback_started_ns}, finished={callback_finished_ns!r}"
            )
        elapsed_ns = callback_finished_ns - callback_started_ns
        with self._state_lock:
            self._max_callback_ns = max(self._max_callback_ns, elapsed_ns)
        return elapsed_ns

    def _poison_on_callback_overrun(self, elapsed_ns: int) -> None:
        if elapsed_ns < CAP_GROCERY_RETENTION_TRACE_CALLBACK_POISON_NS:
            self._raise_if_failed()
            return
        with self._state_lock:
            self._dropped_callback_poison += 1
        self._fail_synchronously(
            RuntimeError(
                "grocery retention trace callback exceeded the fail-closed bound: "
                f"elapsed_ns={elapsed_ns}, bound_ns={CAP_GROCERY_RETENTION_TRACE_CALLBACK_POISON_NS}"
            )
        )

    def _worker_main(self) -> None:
        records_since_flush = 0
        first_error: BaseException | None = None
        try:
            while True:
                item = self._work_queue.get()
                if item is _STOP_WORKER:
                    break
                if isinstance(item, _FlushRequest):
                    try:
                        self._stream.flush()
                        records_since_flush = 0
                    except BaseException as error:
                        self._set_failure(error)
                        raise
                    finally:
                        item.done.set()
                    continue
                if not isinstance(item, _PendingRecord):
                    raise RuntimeError(
                        f"unexpected grocery retention trace work item {type(item)!r}"
                    )
                slot = item.slot
                slot_fenced = False
                try:
                    slot.decision_ready.wait()
                    if slot.persist:
                        state = self._snapshotter.materialize(slot)
                        slot_fenced = True
                        record = dict(item.record_without_state)
                        record["state"] = dict(state)
                        payload = _canonical_jsonl_line(record)
                        written = self._stream.write(payload)
                        if written != len(payload):
                            raise OSError(
                                f"short grocery retention trace write: expected {len(payload)}, wrote {written}"
                            )
                        with self._state_lock:
                            self._record_count += 1
                        records_since_flush += 1
                        if records_since_flush >= self._flush_every_records:
                            self._stream.flush()
                            records_since_flush = 0
                    else:
                        self._snapshotter.discard(slot)
                        slot_fenced = True
                except BaseException as error:
                    # Publish the sticky failure before making any slot
                    # available. A callback racing worker cleanup must fail
                    # rather than restage buffers from the failed record.
                    self._set_failure(error)
                    if not slot_fenced:
                        try:
                            self._snapshotter.discard(slot)
                        except BaseException as cleanup_error:
                            self._set_failure(cleanup_error)
                            self._abandoned_slots.append(slot)
                        else:
                            try:
                                self._snapshotter.release(slot)
                            except BaseException as cleanup_error:
                                self._set_failure(cleanup_error)
                                self._abandoned_slots.append(slot)
                    else:
                        try:
                            self._snapshotter.release(slot)
                        except BaseException as cleanup_error:
                            self._set_failure(cleanup_error)
                            self._abandoned_slots.append(slot)
                    raise
                else:
                    self._snapshotter.release(slot)
                    self._free_slots.put_nowait(slot)
        except BaseException as error:
            first_error = error
        finally:
            try:
                self._stream.flush()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            try:
                self._stream.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            if first_error is not None:
                self._set_failure(first_error)

    def _flush_worker(self) -> None:
        done = threading.Event()
        self._put_control(_FlushRequest(done))
        if not done.wait(self._close_timeout_s):
            self._fail_synchronously(
                RuntimeError("grocery retention trace worker flush timed out")
            )
        self._raise_if_failed()

    def _put_control(self, item: Any) -> None:
        deadline = time.monotonic() + self._close_timeout_s
        while self._worker.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._fail_synchronously(
                    RuntimeError("grocery retention trace worker queue timed out")
                )
            try:
                self._work_queue.put(item, timeout=min(remaining, 0.1))
                return
            except queue.Full:
                self._raise_if_failed()
        self._raise_if_failed()
        self._fail_synchronously(
            RuntimeError(
                "grocery retention trace worker stopped before accepting control work"
            )
        )

    def _set_failure(self, error: BaseException) -> None:
        with self._state_lock:
            if self._failure is None:
                self._failure = error

    def _fail_synchronously(self, error: BaseException) -> None:
        self._mark_reported_failure(error)
        raise error

    def _mark_reported_failure(self, error: BaseException) -> None:
        with self._state_lock:
            if self._failure is None:
                self._failure = error
            self._failure_reported = True

    def _raise_if_failed(self) -> None:
        with self._state_lock:
            error = self._failure
            already_reported = self._failure_reported
            if error is not None and not already_reported:
                self._failure_reported = True
        if error is None:
            return
        if already_reported:
            raise RuntimeError(
                "grocery retention trace writer previously failed"
            ) from error
        raise error

    def _require_writable(self) -> None:
        if self._closed:
            raise RuntimeError("grocery retention trace writer is closed")
        self._raise_if_failed()

    def _reclaim_queued_slots(self) -> None:
        """Fence and release staged records left behind by a failed worker."""
        deadline = time.monotonic() + self._close_timeout_s
        while True:
            try:
                item = self._work_queue.get_nowait()
            except queue.Empty:
                return
            if item is _STOP_WORKER:
                continue
            if isinstance(item, _FlushRequest):
                item.done.set()
                continue
            if not isinstance(item, _PendingRecord):
                self._set_failure(
                    RuntimeError(
                        f"unexpected grocery retention trace work item {type(item)!r}"
                    )
                )
                continue

            slot = item.slot
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not slot.decision_ready.wait(remaining):
                self._set_failure(
                    RuntimeError(
                        "grocery retention trace queued-slot decision timed out"
                    )
                )
                self._abandoned_slots.append(slot)
                continue
            try:
                self._snapshotter.discard(slot)
            except BaseException as error:
                self._set_failure(error)
                self._abandoned_slots.append(slot)
                continue
            try:
                self._snapshotter.release(slot)
            except BaseException as error:
                self._set_failure(error)

    def _reclaim_abandoned_slots(self) -> None:
        unreleased_slots = []
        for slot in self._abandoned_slots:
            try:
                self._snapshotter.discard(slot)
            except BaseException as error:
                self._set_failure(error)
                unreleased_slots.append(slot)
            else:
                try:
                    self._snapshotter.release(slot)
                except BaseException as error:
                    self._set_failure(error)
        self._abandoned_slots = unreleased_slots

    def _reclaim_owned_slots(self) -> None:
        self._reclaim_queued_slots()
        self._reclaim_abandoned_slots()

    def _reap_after_worker_exit(self) -> None:
        """Retain ownership after a timed-out close until every slot is fenced."""
        self._worker.join()
        try:
            self._reclaim_owned_slots()
        except BaseException as error:
            self._set_failure(error)

    def _start_deferred_reaper(self) -> None:
        if self._deferred_reaper is not None:
            return
        reaper = threading.Thread(
            target=self._reap_after_worker_exit,
            name="cap-grocery-retention-trace-deferred-reaper",
            daemon=True,
        )
        self._deferred_reaper = reaper
        reaper.start()

    def close(self) -> None:
        """Drain and close the worker, propagating any unreported output failure."""
        if self._closed:
            return
        close_deadline = time.monotonic() + self._close_timeout_s
        close_control_error: BaseException | None = None
        if self._worker.is_alive():
            try:
                self._put_control(_STOP_WORKER)
            except BaseException as error:
                close_control_error = error
            remaining = max(0.0, close_deadline - time.monotonic())
            self._worker.join(remaining)
            if self._worker.is_alive():
                timeout_error = RuntimeError(
                    "grocery retention trace worker did not stop within the close bound"
                )
                self._set_failure(timeout_error)
                self._start_deferred_reaper()
                self._closed = True
                with self._state_lock:
                    prior_failure = self._failure
                    self._failure_reported = True
                if prior_failure is not None and prior_failure is not timeout_error:
                    raise timeout_error from prior_failure
                raise timeout_error
        if not self._worker.is_alive():
            reclaimer = threading.Thread(
                target=self._reclaim_owned_slots,
                name="cap-grocery-retention-trace-reclaimer",
                daemon=True,
            )
            reclaimer.start()
            remaining = max(0.0, close_deadline - time.monotonic())
            reclaimer.join(remaining)
            if reclaimer.is_alive():
                timeout_error = RuntimeError(
                    "grocery retention trace slot reclamation did not stop within the close bound"
                )
                self._set_failure(timeout_error)
                self._closed = True
                with self._state_lock:
                    prior_failure = self._failure
                    self._failure_reported = True
                if prior_failure is not None and prior_failure is not timeout_error:
                    raise timeout_error from prior_failure
                raise timeout_error
        self._closed = True
        with self._state_lock:
            failure = self._failure
            failure_reported = self._failure_reported
            observed_count = self._observed_count
            record_count = self._record_count
            dropped_over_target = self._dropped_over_target
            dropped_queue_full = self._dropped_queue_full
            dropped_callback_poison = self._dropped_callback_poison
            max_callback_ns = self._max_callback_ns
        dropped_count = (
            dropped_over_target + dropped_queue_full + dropped_callback_poison
        )
        summary = (
            f"path={self._path} boot_id={self._boot_id} "
            f"observed={observed_count} persisted={record_count} dropped={dropped_count} "
            f"dropped_over_target={dropped_over_target} dropped_queue_full={dropped_queue_full} "
            f"dropped_callback_poison={dropped_callback_poison} max_callback_ns={max_callback_ns} "
            f"generations={self._local_generation_index} qualification_eligible=false"
        )
        if failure is None and observed_count != record_count + dropped_count:
            failure = RuntimeError(
                "grocery retention trace accounting mismatch: "
                f"observed={observed_count}, persisted={record_count}, dropped={dropped_count}"
            )
            self._set_failure(failure)
        if failure is None and dropped_count:
            failure = RuntimeError(
                "grocery retention trace is incomplete because physics frames were dropped: "
                f"observed={observed_count}, persisted={record_count}, dropped={dropped_count}"
            )
            self._set_failure(failure)
        if failure is None:
            self._marker_sink(
                f"CAP_GROCERY_RETENTION_TRACE_CLOSED {summary} accounting_complete=true"
            )
        else:
            marker_kind = (
                "CAP_GROCERY_RETENTION_TRACE_INCOMPLETE"
                if dropped_count and observed_count == record_count + dropped_count
                else "CAP_GROCERY_RETENTION_TRACE_FAILED"
            )
            self._marker_sink(f"{marker_kind} {summary} accounting_complete=false")
        if failure is not None and not failure_reported:
            with self._state_lock:
                self._failure_reported = True
            raise failure
        if close_control_error is not None:
            raise close_control_error


def make_grocery_retention_trace_writer(
    adapter: Any,
    marker_sink: Callable[[str], None],
    *,
    output_path: str | os.PathLike[str],
) -> GroceryRetentionTraceWriter:
    """Build the opt-in bounded asynchronous grocery retention observer."""
    return GroceryRetentionTraceWriter(
        adapter,
        marker_sink,
        output_path=output_path,
    )
