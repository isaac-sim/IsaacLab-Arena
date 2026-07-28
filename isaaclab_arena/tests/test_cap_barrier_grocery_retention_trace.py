# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the opt-in CAP grocery retention trace."""

from __future__ import annotations

import json
import math
import queue
import stat
import threading
from types import SimpleNamespace

import pytest

import isaaclab_arena.integrations.cap_barrier.grocery_retention_trace as trace_module
from isaaclab_arena.integrations.cap_barrier.grocery_retention_trace import (
    CAP_GROCERY_RETENTION_CONTACT_MEASUREMENTS_INCLUDED,
    CAP_GROCERY_RETENTION_TRACE_SCHEMA,
    GroceryRetentionTraceWriter,
    _canonical_jsonl_line,
    capture_grocery_retention_state,
    _pose_in_reference_frame,
)
from isaaclab_arena.integrations.cap_barrier.grocery_scene_spec import (
    CAP_GROCERY_OBJECT_ASSET,
)
from isaaclab_arena.integrations.cap_barrier.joint_mapping import PANDA_ARM_JOINTS

_ROBOTIQ_JOINTS = (
    "finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
    "left_inner_finger_joint",
)
_QUARTER_TURN_Z_XYZW = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
_BOOT_ID = "01234567-89ab-cdef-0123-456789abcdef"


class _FakeCaptureAdapter:
    def __init__(self) -> None:
        self.joint_names = (*PANDA_ARM_JOINTS, *_ROBOTIQ_JOINTS)
        count = len(self.joint_names)
        self._positions = [0.1 * index for index in range(count)]
        self._velocities = [0.01 * index for index in range(count)]
        self._torques = [0.001 * index for index in range(count)]
        self.synchronize_calls = 0
        grocery_object = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=[[1.0, 2.0, 3.0]],
                root_quat_w=[list(_QUARTER_TURN_Z_XYZW)],
                root_lin_vel_w=[[0.1, 0.2, 0.3]],
                root_ang_vel_w=[[0.4, 0.5, 0.6]],
            )
        )
        ee_frame = SimpleNamespace(
            data=SimpleNamespace(
                target_frame_names=[
                    "end_effector",
                    "tool_rightfinger",
                    "tool_leftfinger",
                ],
                target_pos_w=[
                    [
                        [9.0, 9.0, 9.0],
                        [0.0, 2.0, 3.0],
                        [1.0, 3.0, 3.0],
                    ]
                ],
                target_quat_w=[
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        list(_QUARTER_TURN_Z_XYZW),
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ],
            )
        )
        self._unwrapped = SimpleNamespace(
            scene={
                CAP_GROCERY_OBJECT_ASSET: grocery_object,
                "ee_frame": ee_frame,
            }
        )
        self._robot = object()

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def read_joint_state(self):
        return self._positions, self._velocities, self._torques


class _WriterAdapter:
    def __init__(self) -> None:
        self.physics_step_count = 0
        self.reset_count = 1
        self.last_admitted_binary_gripper_action = 0.0


class _FakeTorchAdapter(_WriterAdapter):
    def __init__(self, torch) -> None:
        super().__init__()
        self._torch = torch
        self.joint_names = (*PANDA_ARM_JOINTS, *_ROBOTIQ_JOINTS)
        count = len(self.joint_names)
        self._robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=torch.arange(count, dtype=torch.float32).reshape(1, count)
                * 0.1,
                joint_vel=torch.arange(count, dtype=torch.float32).reshape(1, count)
                * 0.01,
                applied_torque=torch.arange(count, dtype=torch.float32).reshape(
                    1, count
                )
                * 0.001,
            )
        )
        grocery_object = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[1.0, 2.0, 3.0]]),
                root_quat_w=torch.tensor([list(_QUARTER_TURN_Z_XYZW)]),
                root_lin_vel_w=torch.tensor([[0.1, 0.2, 0.3]]),
                root_ang_vel_w=torch.tensor([[0.4, 0.5, 0.6]]),
            )
        )
        ee_frame = SimpleNamespace(
            data=SimpleNamespace(
                target_frame_names=[
                    "end_effector",
                    "tool_rightfinger",
                    "tool_leftfinger",
                ],
                target_pos_w=torch.tensor(
                    [
                        [
                            [9.0, 9.0, 9.0],
                            [0.0, 2.0, 3.0],
                            [1.0, 3.0, 3.0],
                        ]
                    ]
                ),
                target_quat_w=torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 1.0],
                            list(_QUARTER_TURN_Z_XYZW),
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ]
                ),
            )
        )
        self._unwrapped = SimpleNamespace(
            device="cpu",
            scene={
                CAP_GROCERY_OBJECT_ASSET: grocery_object,
                "ee_frame": ee_frame,
            },
        )

    def synchronize(self) -> None:
        raise AssertionError(
            "the production trace callback must not call adapter.synchronize"
        )


def _step(
    writer: GroceryRetentionTraceWriter,
    adapter: _WriterAdapter,
    frame: int,
    *,
    action: float,
) -> None:
    adapter.physics_step_count += 1
    adapter.last_admitted_binary_gripper_action = action
    writer(frame)


def test_capture_records_six_named_joints_can_and_existing_inner_finger_frames() -> (
    None
):
    adapter = _FakeCaptureAdapter()

    captured = capture_grocery_retention_state(adapter)

    assert adapter.synchronize_calls == 1
    joints = captured["robotiq_joints"]
    assert [joint["name"] for joint in joints] == list(_ROBOTIQ_JOINTS)
    assert joints[0] == {
        "name": "finger_joint",
        "position_rad": 0.7000000000000001,
        "velocity_rad_s": 0.07,
        "applied_torque_diagnostic_n_m": 0.007,
    }
    assert captured["can"] == {
        "asset_name": CAP_GROCERY_OBJECT_ASSET,
        "world_pose": {
            "position_m": [1.0, 2.0, 3.0],
            "quaternion_xyzw": list(_QUARTER_TURN_Z_XYZW),
        },
        "linear_velocity_world_m_s": [0.1, 0.2, 0.3],
        "angular_velocity_world_rad_s": [0.4, 0.5, 0.6],
    }
    fingers = captured["inner_finger_frames"]
    assert fingers["left"]["frame_name"] == "tool_leftfinger"
    assert fingers["left"]["world_pose"]["position_m"] == [1.0, 3.0, 3.0]
    assert fingers["left"]["pose_in_can_frame"]["position_m"] == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    assert fingers["left"]["pose_in_can_frame"]["quaternion_xyzw"] == pytest.approx(
        [
            0.0,
            0.0,
            -math.sqrt(0.5),
            math.sqrt(0.5),
        ]
    )
    assert fingers["right"]["frame_name"] == "tool_rightfinger"
    assert fingers["right"]["pose_in_can_frame"]["position_m"] == pytest.approx(
        [0.0, 1.0, 0.0]
    )


def test_pose_in_reference_frame_preserves_noncommuting_quaternion_order() -> None:
    root_half = math.sqrt(0.5)

    relative = _pose_in_reference_frame(
        [1.0, 2.0, 3.0],
        [0.0, 0.0, root_half, root_half],
        [2.0, 2.0, 3.0],
        [root_half, 0.0, 0.0, root_half],
    )

    assert relative["position_m"] == pytest.approx([0.0, -1.0, 0.0])
    assert relative["quaternion_xyzw"] == pytest.approx([0.5, -0.5, -0.5, 0.5])


@pytest.mark.parametrize(
    "joint_names",
    [
        (*PANDA_ARM_JOINTS, *_ROBOTIQ_JOINTS[:-1]),
        (*PANDA_ARM_JOINTS, *_ROBOTIQ_JOINTS, "unexpected_joint"),
        (*PANDA_ARM_JOINTS, "renamed_finger_joint", *_ROBOTIQ_JOINTS[1:]),
    ],
)
def test_capture_requires_exact_physical_robotiq_joint_roster(joint_names) -> None:
    adapter = _FakeCaptureAdapter()
    adapter.joint_names = joint_names
    count = len(joint_names)
    adapter._positions = [0.0] * count
    adapter._velocities = [0.0] * count
    adapter._torques = [0.0] * count

    with pytest.raises(RuntimeError, match="exact physical Robotiq joint roster"):
        capture_grocery_retention_state(adapter)


def test_v1_explicitly_omits_unvalidated_contact_measurements() -> None:
    captured = capture_grocery_retention_state(_FakeCaptureAdapter())
    serialized = json.dumps(captured, sort_keys=True)

    assert CAP_GROCERY_RETENTION_CONTACT_MEASUREMENTS_INCLUDED is False
    assert "contact" not in serialized
    assert "force" not in serialized
    assert "applied_torque_diagnostic_n_m" in serialized


def test_production_snapshotter_stages_tensor_state_without_adapter_synchronize(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    adapter = _FakeTorchAdapter(torch)
    callback_clock = iter((0, 100))
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "tensor-snapshot.jsonl",
        callback_clock_ns=lambda: next(callback_clock),
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=1.0)
    writer.close()

    record = json.loads(
        (tmp_path / "tensor-snapshot.jsonl").read_text(encoding="utf-8")
    )
    assert [joint["name"] for joint in record["state"]["robotiq_joints"]] == list(
        _ROBOTIQ_JOINTS
    )
    assert record["state"]["can"]["world_pose"]["position_m"] == [1.0, 2.0, 3.0]


def test_writer_emits_canonical_step_rows_with_only_local_generation_identity(
    tmp_path,
) -> None:
    output = tmp_path / "retention.jsonl"
    markers: list[str] = []
    adapter = _WriterAdapter()
    timestamps = iter((100, 200, 300))
    writer = GroceryRetentionTraceWriter(
        adapter,
        markers.append,
        output_path=output,
        flush_every_records=2,
        capture=lambda: {"sample": "state"},
        monotonic_ns=lambda: next(timestamps),
        _boot_id=_BOOT_ID,
    )

    writer.begin_generation(41)
    _step(writer, adapter, 0, action=0.0)
    _step(writer, adapter, 1, action=1.0)
    writer.begin_generation(99)
    adapter.reset_count = 2
    _step(writer, adapter, 0, action=1.0)
    writer.close()
    writer.close()

    raw_lines = output.read_text(encoding="utf-8").splitlines(keepends=True)
    records = [json.loads(line) for line in raw_lines]
    assert len(records) == 3
    assert raw_lines == [_canonical_jsonl_line(record) for record in records]
    assert all(
        line.startswith('{"admitted_binary_gripper_action":') for line in raw_lines
    )
    assert all(
        record["schema_version"] == CAP_GROCERY_RETENTION_TRACE_SCHEMA
        for record in records
    )
    assert all(record["record_type"] == "physics_step" for record in records)
    assert all(record["boot_id"] == _BOOT_ID for record in records)
    assert [record["monotonic_timestamp_ns"] for record in records] == [100, 200, 300]
    assert [record["admitted_binary_gripper_action"] for record in records] == [
        0.0,
        1.0,
        1.0,
    ]
    assert [record["local_counters"] for record in records] == [
        {
            "generation_index": 1,
            "generation_frame_index": 0,
            "generation_first_step": True,
            "physics_step_count": 1,
            "reset_count": 1,
        },
        {
            "generation_index": 1,
            "generation_frame_index": 1,
            "generation_first_step": False,
            "physics_step_count": 2,
            "reset_count": 1,
        },
        {
            "generation_index": 2,
            "generation_frame_index": 0,
            "generation_first_step": True,
            "physics_step_count": 3,
            "reset_count": 2,
        },
    ]
    assert all(
        set(record)
        == {
            "admitted_binary_gripper_action",
            "boot_id",
            "local_counters",
            "monotonic_timestamp_ns",
            "record_type",
            "schema_version",
            "state",
        }
        for record in records
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert "flush_every_records=2" in markers[0]
    assert markers[0].endswith(
        f"boot_id={_BOOT_ID} contacts=omitted diagnostic_only=true qualification_eligible=false"
    )
    assert (
        f"boot_id={_BOOT_ID} observed=3 persisted=3 dropped=0 "
        "dropped_over_target=0 dropped_queue_full=0 dropped_callback_poison=0"
        in markers[-1]
    )
    assert markers[-1].endswith(
        "generations=2 qualification_eligible=false accounting_complete=true"
    )


@pytest.mark.parametrize(
    ("output_path", "message"),
    [
        ("relative.jsonl", "must be absolute"),
        ("/tmp/retention.json", "must end in .jsonl"),
    ],
)
def test_writer_rejects_invalid_output_paths(output_path: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GroceryRetentionTraceWriter(
            _WriterAdapter(),
            lambda _marker: None,
            output_path=output_path,
            capture=lambda: {},
        )


def test_writer_opens_output_exclusively_and_requires_existing_directory(
    tmp_path,
) -> None:
    existing = tmp_path / "existing.jsonl"
    existing.write_text("stale\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        GroceryRetentionTraceWriter(
            _WriterAdapter(),
            lambda _marker: None,
            output_path=existing,
            capture=lambda: {},
        )
    assert existing.read_text(encoding="utf-8") == "stale\n"

    with pytest.raises(FileNotFoundError, match="directory does not exist"):
        GroceryRetentionTraceWriter(
            _WriterAdapter(),
            lambda _marker: None,
            output_path=tmp_path / "missing" / "trace.jsonl",
            capture=lambda: {},
        )


def test_writer_removes_exclusive_output_when_armed_marker_fails(tmp_path) -> None:
    output = tmp_path / "marker-failure.jsonl"

    def fail_marker(_marker: str) -> None:
        raise RuntimeError("synthetic marker failure")

    with pytest.raises(RuntimeError, match="marker failure"):
        GroceryRetentionTraceWriter(
            _WriterAdapter(),
            fail_marker,
            output_path=output,
            capture=lambda: {},
        )

    assert not output.exists()
    retry = GroceryRetentionTraceWriter(
        _WriterAdapter(),
        lambda _marker: None,
        output_path=output,
        capture=lambda: {},
    )
    retry.close()


def test_writer_fails_closed_on_nonfinite_capture_and_generation_or_counter_gaps(
    tmp_path,
) -> None:
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "nonfinite.jsonl",
        capture=lambda: {"bad": math.nan},
    )
    writer.begin_generation(1)
    adapter.physics_step_count = 1
    writer(0)
    with pytest.raises(ValueError, match="Out of range float values"):
        writer.close()
    assert (tmp_path / "nonfinite.jsonl").read_bytes() == b""

    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "counter-gap.jsonl",
        capture=lambda: {},
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    adapter.physics_step_count += 2
    with pytest.raises(RuntimeError, match="physics-step counter"):
        writer(1)
    writer.close()


def test_capture_exception_is_sticky_and_cannot_recover(tmp_path) -> None:
    calls = 0

    def capture():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic capture failure")
        return {}

    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "capture-failure.jsonl",
        capture=capture,
    )
    writer.begin_generation(1)
    adapter.physics_step_count = 1

    with pytest.raises(RuntimeError, match="capture failure"):
        writer(0)
    with pytest.raises(RuntimeError, match="previously failed"):
        writer(0)
    writer.close()

    assert calls == 1
    assert (tmp_path / "capture-failure.jsonl").read_bytes() == b""


def test_writer_requires_a_distinct_reset_for_each_local_generation(tmp_path) -> None:
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "resets.jsonl",
        capture=lambda: {},
    )
    writer.begin_generation(4)
    _step(writer, adapter, 0, action=0.0)
    writer.begin_generation(5)
    adapter.physics_step_count += 1

    with pytest.raises(RuntimeError, match="without a distinct reset"):
        writer(0)

    writer.close()


def test_writer_rejects_reset_change_inside_generation(tmp_path) -> None:
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "in-generation-reset.jsonl",
        capture=lambda: {},
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    adapter.physics_step_count += 1
    adapter.reset_count += 1

    with pytest.raises(RuntimeError, match="reset counter changed"):
        writer(1)

    writer.close()


class _FailingStream:
    def __init__(self, *, write_error: bool = False, flush_failures: int = 0) -> None:
        self.write_error = write_error
        self.flush_failures = flush_failures
        self.flush_calls = 0
        self.closed = False
        self.payloads: list[str] = []

    def write(self, payload: str) -> int:
        if self.write_error:
            raise OSError("synthetic write failure")
        self.payloads.append(payload)
        return len(payload)

    def flush(self) -> None:
        self.flush_calls += 1
        if self.flush_failures:
            self.flush_failures -= 1
            raise OSError("synthetic flush failure")

    def close(self) -> None:
        self.closed = True


class _GenerationFlushFailureWithBlockedCleanup(_FailingStream):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_flush_started = threading.Event()
        self.release_cleanup_flush = threading.Event()

    def flush(self) -> None:
        self.flush_calls += 1
        if self.flush_calls == 1:
            raise OSError("synthetic generation flush failure")
        self.cleanup_flush_started.set()
        assert self.release_cleanup_flush.wait(5.0)


@pytest.mark.parametrize(
    ("stream", "message"),
    [
        (_FailingStream(write_error=True), "write failure"),
        (_FailingStream(flush_failures=1), "flush failure"),
    ],
)
def test_enabled_writer_propagates_write_and_bounded_flush_failures(
    tmp_path,
    monkeypatch,
    stream: _FailingStream,
    message: str,
) -> None:
    monkeypatch.setattr(trace_module, "_open_exclusive_text", lambda _path: stream)
    adapter = _WriterAdapter()
    markers: list[str] = []
    writer = GroceryRetentionTraceWriter(
        adapter,
        markers.append,
        output_path=tmp_path / "failure.jsonl",
        flush_every_records=1,
        capture=lambda: {},
    )
    writer.begin_generation(1)
    adapter.physics_step_count = 1

    writer(0)
    with pytest.raises(OSError, match=message):
        writer.close()
    assert stream.closed is True
    assert not any("CAP_GROCERY_RETENTION_TRACE_CLOSED" in marker for marker in markers)
    assert writer.record_count == int(not stream.write_error)


def test_generation_boundary_flush_is_load_bearing(tmp_path, monkeypatch) -> None:
    stream = _FailingStream()
    monkeypatch.setattr(trace_module, "_open_exclusive_text", lambda _path: stream)
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "generation-flush.jsonl",
        flush_every_records=20,
        capture=lambda: {},
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    assert stream.flush_calls == 0

    writer.begin_generation(2)

    assert stream.flush_calls == 1
    writer.close()


def test_generation_flush_failure_is_sticky_before_waiter_release(
    tmp_path,
    monkeypatch,
) -> None:
    stream = _GenerationFlushFailureWithBlockedCleanup()
    monkeypatch.setattr(trace_module, "_open_exclusive_text", lambda _path: stream)
    snapshotter = _QueuedSlotTrackingSnapshotter()
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "generation-flush-failure.jsonl",
        flush_every_records=20,
        _snapshotter=snapshotter,
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)

    with pytest.raises(OSError, match="generation flush failure"):
        writer.begin_generation(2)
    assert stream.cleanup_flush_started.wait(5.0)

    adapter.physics_step_count += 1
    with pytest.raises(RuntimeError, match="previously failed"):
        writer(1)
    assert snapshotter.staged == [0]

    stream.release_cleanup_flush.set()
    writer._worker.join(5.0)
    assert not writer._worker.is_alive()
    writer.close()


def test_close_propagates_final_flush_failure_but_still_closes(
    tmp_path, monkeypatch
) -> None:
    stream = _FailingStream(flush_failures=1)
    monkeypatch.setattr(trace_module, "_open_exclusive_text", lambda _path: stream)
    writer = GroceryRetentionTraceWriter(
        _WriterAdapter(),
        lambda _marker: None,
        output_path=tmp_path / "close-failure.jsonl",
        capture=lambda: {},
    )

    with pytest.raises(OSError, match="flush failure"):
        writer.close()

    assert stream.closed is True


def test_callback_only_stages_while_worker_owns_json_and_output(
    tmp_path, monkeypatch
) -> None:
    callback_thread = threading.current_thread().name
    canonical_threads: list[str] = []
    original_canonical = trace_module._canonical_jsonl_line

    def canonical(record):
        canonical_threads.append(threading.current_thread().name)
        return original_canonical(record)

    monkeypatch.setattr(trace_module, "_canonical_jsonl_line", canonical)
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "thread-ownership.jsonl",
        capture=lambda: {"sample": "state"},
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    writer.close()

    assert canonical_threads == ["cap-grocery-retention-trace-writer"]
    assert callback_thread not in canonical_threads


def test_over_target_callback_is_dropped_and_reported_without_poisoning(
    tmp_path,
) -> None:
    markers: list[str] = []
    adapter = _WriterAdapter()
    callback_clock = iter(
        (0, trace_module.CAP_GROCERY_RETENTION_TRACE_CALLBACK_TARGET_NS + 1)
    )
    writer = GroceryRetentionTraceWriter(
        adapter,
        markers.append,
        output_path=tmp_path / "over-target.jsonl",
        capture=lambda: {"sample": "state"},
        callback_clock_ns=lambda: next(callback_clock),
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    with pytest.raises(
        RuntimeError, match="incomplete because physics frames were dropped"
    ):
        writer.close()

    assert (tmp_path / "over-target.jsonl").read_bytes() == b""
    assert (
        "observed=1 persisted=0 dropped=1 dropped_over_target=1 "
        "dropped_queue_full=0 dropped_callback_poison=0 "
        f"max_callback_ns={trace_module.CAP_GROCERY_RETENTION_TRACE_CALLBACK_TARGET_NS + 1}"
        in markers[-1]
    )
    assert markers[-1].startswith("CAP_GROCERY_RETENTION_TRACE_INCOMPLETE ")
    assert markers[-1].endswith("accounting_complete=false")


def test_callback_at_poison_bound_fails_run_and_never_persists_frame(tmp_path) -> None:
    markers: list[str] = []
    adapter = _WriterAdapter()
    callback_clock = iter(
        (0, trace_module.CAP_GROCERY_RETENTION_TRACE_CALLBACK_POISON_NS)
    )
    writer = GroceryRetentionTraceWriter(
        adapter,
        markers.append,
        output_path=tmp_path / "poison.jsonl",
        capture=lambda: {"sample": "state"},
        callback_clock_ns=lambda: next(callback_clock),
    )
    writer.begin_generation(1)
    adapter.physics_step_count = 1

    with pytest.raises(RuntimeError, match="exceeded the fail-closed bound"):
        writer(0)
    with pytest.raises(RuntimeError, match="previously failed"):
        writer(1)
    writer.close()

    assert (tmp_path / "poison.jsonl").read_bytes() == b""
    assert not any("CAP_GROCERY_RETENTION_TRACE_CLOSED" in marker for marker in markers)
    assert markers[-1].startswith("CAP_GROCERY_RETENTION_TRACE_INCOMPLETE ")


class _BlockingStream(_FailingStream):
    def __init__(self) -> None:
        super().__init__()
        self.write_started = threading.Event()
        self.release_write = threading.Event()

    def write(self, payload: str) -> int:
        self.write_started.set()
        assert self.release_write.wait(5.0)
        return super().write(payload)


class _BlockingWriteFailure(_FailingStream):
    def __init__(self) -> None:
        super().__init__()
        self.write_started = threading.Event()
        self.release_write = threading.Event()

    def write(self, _payload: str) -> int:
        self.write_started.set()
        assert self.release_write.wait(5.0)
        raise OSError("synthetic blocked write failure")


def test_full_bounded_queue_drops_without_blocking_or_fabricating_a_row(
    tmp_path, monkeypatch
) -> None:
    stream = _BlockingStream()
    monkeypatch.setattr(trace_module, "_open_exclusive_text", lambda _path: stream)
    markers: list[str] = []
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        markers.append,
        output_path=tmp_path / "queue-full.jsonl",
        queue_capacity=1,
        capture=lambda: {"sample": "state"},
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    assert stream.write_started.wait(5.0)

    _step(writer, adapter, 1, action=1.0)
    stream.release_write.set()
    with pytest.raises(
        RuntimeError, match="incomplete because physics frames were dropped"
    ):
        writer.close()

    assert len(stream.payloads) == 1
    assert "observed=2 persisted=1 dropped=1" in markers[-1]
    assert "dropped_queue_full=1" in markers[-1]
    assert markers[-1].startswith("CAP_GROCERY_RETENTION_TRACE_INCOMPLETE ")


def test_queue_full_at_poison_bound_counts_one_disjoint_drop(
    tmp_path,
    monkeypatch,
) -> None:
    stream = _BlockingStream()
    monkeypatch.setattr(trace_module, "_open_exclusive_text", lambda _path: stream)
    markers: list[str] = []
    adapter = _WriterAdapter()
    callback_clock = iter(
        (
            0,
            100,
            200,
            200 + trace_module.CAP_GROCERY_RETENTION_TRACE_CALLBACK_POISON_NS,
        )
    )
    writer = GroceryRetentionTraceWriter(
        adapter,
        markers.append,
        output_path=tmp_path / "queue-full-poison.jsonl",
        queue_capacity=1,
        callback_clock_ns=lambda: next(callback_clock),
        capture=lambda: {"sample": "state"},
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    assert stream.write_started.wait(5.0)

    with pytest.raises(RuntimeError, match="exceeded the fail-closed bound"):
        _step(writer, adapter, 1, action=1.0)
    stream.release_write.set()
    writer._worker.join(5.0)
    writer.close()

    assert "observed=2 persisted=1 dropped=1" in markers[-1]
    assert "dropped_queue_full=0" in markers[-1]
    assert "dropped_callback_poison=1" in markers[-1]
    assert markers[-1].startswith("CAP_GROCERY_RETENTION_TRACE_INCOMPLETE ")
    assert not any(
        marker.startswith("CAP_GROCERY_RETENTION_TRACE_CLOSED ") for marker in markers
    )


def test_worker_error_is_sticky_and_is_reported_by_next_callback(
    tmp_path, monkeypatch
) -> None:
    stream = _FailingStream(write_error=True)
    monkeypatch.setattr(trace_module, "_open_exclusive_text", lambda _path: stream)
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "worker-error.jsonl",
        capture=lambda: {},
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    writer._worker.join(5.0)
    assert not writer._worker.is_alive()

    adapter.physics_step_count += 1
    with pytest.raises(OSError, match="synthetic write failure"):
        writer(1)
    writer.close()


class _QueuedSlotTrackingSnapshotter:
    def __init__(self) -> None:
        self.staged: list[int] = []
        self.discarded: list[int] = []
        self.released: list[int] = []

    def make_slots(self, count: int):
        return tuple(
            trace_module._SnapshotSlot(
                index=index,
                decision_ready=threading.Event(),
            )
            for index in range(count)
        )

    def preflight(self, _slot) -> None:
        pass

    def stage(self, slot) -> None:
        self.staged.append(slot.index)

    def materialize(self, _slot):
        return {}

    def discard(self, slot) -> None:
        self.discarded.append(slot.index)

    def release(self, slot) -> None:
        self.released.append(slot.index)


def test_close_fences_and_releases_records_queued_behind_worker_failure(
    tmp_path,
    monkeypatch,
) -> None:
    stream = _BlockingWriteFailure()
    monkeypatch.setattr(trace_module, "_open_exclusive_text", lambda _path: stream)
    snapshotter = _QueuedSlotTrackingSnapshotter()
    callback_clock = iter((0, 100, 200, 300, 400, 500))
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "queued-worker-failure.jsonl",
        queue_capacity=3,
        callback_clock_ns=lambda: next(callback_clock),
        _snapshotter=snapshotter,
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    assert stream.write_started.wait(5.0)
    _step(writer, adapter, 1, action=1.0)
    _step(writer, adapter, 2, action=1.0)

    stream.release_write.set()
    writer._worker.join(5.0)
    assert not writer._worker.is_alive()
    with pytest.raises(OSError, match="blocked write failure"):
        writer.close()

    assert snapshotter.staged == [0, 1, 2]
    assert snapshotter.discarded == [1, 2]
    assert snapshotter.released == [0, 1, 2]
    assert writer._work_queue.empty()


def test_close_timeout_never_publishes_terminal_marker_while_worker_is_live(
    tmp_path,
    monkeypatch,
) -> None:
    stream = _BlockingStream()
    monkeypatch.setattr(trace_module, "_open_exclusive_text", lambda _path: stream)
    markers: list[str] = []
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        markers.append,
        output_path=tmp_path / "close-timeout.jsonl",
        queue_capacity=1,
        close_timeout_s=0.05,
        capture=lambda: {"sample": "state"},
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    assert stream.write_started.wait(5.0)

    with pytest.raises(RuntimeError, match="did not stop within the close bound"):
        writer.close()

    assert writer._worker.is_alive()
    assert not any(
        marker.startswith(
            (
                "CAP_GROCERY_RETENTION_TRACE_CLOSED ",
                "CAP_GROCERY_RETENTION_TRACE_FAILED ",
                "CAP_GROCERY_RETENTION_TRACE_INCOMPLETE ",
            )
        )
        for marker in markers
    )
    stream.release_write.set()
    writer._worker.join(5.0)
    assert not writer._worker.is_alive()
    assert writer._deferred_reaper is not None
    writer._deferred_reaper.join(5.0)
    assert not writer._deferred_reaper.is_alive()


def test_close_timeout_deferred_reaper_fences_queued_slots_after_worker_exit(
    tmp_path,
    monkeypatch,
) -> None:
    stream = _BlockingWriteFailure()
    monkeypatch.setattr(trace_module, "_open_exclusive_text", lambda _path: stream)
    snapshotter = _QueuedSlotTrackingSnapshotter()
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "close-timeout-queued.jsonl",
        queue_capacity=3,
        close_timeout_s=0.05,
        _snapshotter=snapshotter,
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    assert stream.write_started.wait(5.0)
    _step(writer, adapter, 1, action=1.0)
    _step(writer, adapter, 2, action=1.0)

    with pytest.raises(RuntimeError, match="did not stop within the close bound"):
        writer.close()
    assert writer._worker.is_alive()
    assert writer._deferred_reaper is not None

    stream.release_write.set()
    writer._deferred_reaper.join(5.0)
    assert not writer._deferred_reaper.is_alive()
    assert snapshotter.discarded == [1, 2]
    assert snapshotter.released == [0, 1, 2]
    assert writer._work_queue.empty()


def test_boot_id_failure_precedes_exclusive_output_creation(tmp_path) -> None:
    output = tmp_path / "invalid-boot-id.jsonl"

    with pytest.raises(RuntimeError, match="invalid kernel boot identity"):
        GroceryRetentionTraceWriter(
            _WriterAdapter(),
            lambda _marker: None,
            output_path=output,
            capture=lambda: {},
            _boot_id="not-a-boot-id",
        )

    assert not output.exists()


class _StageFailureSnapshotter:
    def __init__(self) -> None:
        self.discarded: list[int] = []
        self.released: list[int] = []
        self.stage_threads: list[str] = []
        self.materialize_threads: list[str] = []

    def make_slots(self, count: int):
        return tuple(
            trace_module._SnapshotSlot(index=index, decision_ready=threading.Event())
            for index in range(count)
        )

    def preflight(self, _slot) -> None:
        pass

    def stage(self, _slot) -> None:
        raise RuntimeError("synthetic stage failure")

    def materialize(self, _slot):
        raise AssertionError("failed stage must never be materialized")

    def discard(self, slot) -> None:
        self.discarded.append(slot.index)

    def release(self, slot) -> None:
        self.released.append(slot.index)


class _BlockingDiscardSnapshotter(_StageFailureSnapshotter):
    def __init__(self) -> None:
        super().__init__()
        self.discard_started = threading.Event()
        self.release_discard = threading.Event()
        self.discard_done = threading.Event()

    def discard(self, slot) -> None:
        self.discard_started.set()
        assert self.release_discard.wait(5.0)
        super().discard(slot)
        self.discard_done.set()


def test_failed_stage_is_retained_then_reclaimed_at_close(tmp_path) -> None:
    snapshotter = _StageFailureSnapshotter()
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "stage-failure.jsonl",
        queue_capacity=1,
        _snapshotter=snapshotter,
    )
    writer.begin_generation(1)
    adapter.physics_step_count = 1

    with pytest.raises(RuntimeError, match="synthetic stage failure"):
        writer(0)
    writer.close()

    assert snapshotter.discarded == [0]
    assert snapshotter.released == [0]


def test_slot_reclamation_timeout_never_publishes_a_terminal_marker(
    tmp_path,
) -> None:
    snapshotter = _BlockingDiscardSnapshotter()
    markers: list[str] = []
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        markers.append,
        output_path=tmp_path / "reclaim-timeout.jsonl",
        queue_capacity=1,
        close_timeout_s=0.05,
        _snapshotter=snapshotter,
    )
    writer.begin_generation(1)
    adapter.physics_step_count = 1
    with pytest.raises(RuntimeError, match="synthetic stage failure"):
        writer(0)

    with pytest.raises(
        RuntimeError,
        match="slot reclamation did not stop within the close bound",
    ):
        writer.close()

    assert snapshotter.discard_started.is_set()
    assert not any(
        marker.startswith(
            (
                "CAP_GROCERY_RETENTION_TRACE_CLOSED ",
                "CAP_GROCERY_RETENTION_TRACE_FAILED ",
                "CAP_GROCERY_RETENTION_TRACE_INCOMPLETE ",
            )
        )
        for marker in markers
    )
    snapshotter.release_discard.set()
    assert snapshotter.discard_done.wait(5.0)


class _ReleaseTrackingSnapshotter(_StageFailureSnapshotter):
    def stage(self, _slot) -> None:
        self.stage_threads.append(threading.current_thread().name)

    def materialize(self, _slot):
        self.materialize_threads.append(threading.current_thread().name)
        return {}


class _WorkerFenceFailureSnapshotter(_ReleaseTrackingSnapshotter):
    def __init__(self) -> None:
        super().__init__()
        self.discard_started = threading.Event()
        self.release_discard = threading.Event()

    def materialize(self, _slot):
        raise RuntimeError("synthetic worker fence failure")

    def discard(self, slot) -> None:
        self.discard_started.set()
        assert self.release_discard.wait(5.0)
        super().discard(slot)


def test_worker_fence_failure_is_sticky_before_slot_release(tmp_path) -> None:
    snapshotter = _WorkerFenceFailureSnapshotter()
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "worker-fence-failure.jsonl",
        queue_capacity=1,
        _snapshotter=snapshotter,
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    assert snapshotter.discard_started.wait(5.0)

    assert writer._free_slots.empty()
    with pytest.raises(RuntimeError, match="synthetic worker fence failure"):
        _step(writer, adapter, 1, action=0.0)
    assert snapshotter.stage_threads == [threading.current_thread().name]

    snapshotter.release_discard.set()
    writer._worker.join(5.0)
    writer.close()
    assert snapshotter.discarded == [0]
    assert snapshotter.released == [0]


class _FakeCudaStream:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _FailingCudaEvent:
    def __init__(self) -> None:
        self.recorded_stream = None

    def record(self, stream) -> None:
        self.recorded_stream = stream
        raise RuntimeError("synthetic CUDA event record failure")


class _FakeCudaHostValues:
    def copy_(self, _source, *, non_blocking: bool) -> None:
        assert non_blocking is True


def test_torch_preflight_fences_exact_stage_stream_when_event_record_fails() -> None:
    stage_stream = _FakeCudaStream()
    cleanup_stream = _FakeCudaStream()
    current_streams = iter((stage_stream, cleanup_stream))
    cuda = SimpleNamespace(current_stream=lambda _device: next(current_streams))
    torch = SimpleNamespace(
        cuda=cuda,
        cat=lambda _sources, *, out: out,
    )
    source = SimpleNamespace(numel=lambda: 1)
    snapshotter = object.__new__(trace_module._TorchRetentionSnapshotter)
    snapshotter._torch = torch
    snapshotter._device = object()
    snapshotter._is_cuda = True
    snapshotter._source_lengths = (1,)
    snapshotter._source_views = lambda: (source,)
    event = _FailingCudaEvent()
    slot = trace_module._SnapshotSlot(
        index=0,
        decision_ready=threading.Event(),
        host_values=_FakeCudaHostValues(),
        device_values=object(),
        copy_complete=event,
    )

    with pytest.raises(RuntimeError, match="event record failure"):
        snapshotter.preflight(slot)

    assert event.recorded_stream is stage_stream
    assert stage_stream.synchronize_calls == 1
    assert cleanup_stream.synchronize_calls == 0
    assert slot.copy_recorded is False
    assert slot.stage_stream is None


def test_failure_after_enqueue_makes_worker_discard_and_release_slot(tmp_path) -> None:
    snapshotter = _ReleaseTrackingSnapshotter()
    adapter = _WriterAdapter()
    callback_clock = iter((10, 9))
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "post-enqueue-failure.jsonl",
        queue_capacity=1,
        callback_clock_ns=lambda: next(callback_clock),
        _snapshotter=snapshotter,
    )
    writer.begin_generation(1)
    adapter.physics_step_count = 1

    with pytest.raises(RuntimeError, match="clock moved backwards"):
        writer(0)
    writer.close()

    assert snapshotter.discarded == [0]
    assert snapshotter.released == [0]


def test_snapshot_materialization_is_worker_owned(tmp_path) -> None:
    snapshotter = _ReleaseTrackingSnapshotter()
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "worker-materialize.jsonl",
        queue_capacity=1,
        _snapshotter=snapshotter,
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    writer.close()

    assert snapshotter.stage_threads == [threading.current_thread().name]
    assert snapshotter.materialize_threads == ["cap-grocery-retention-trace-writer"]
    assert snapshotter.released == [0]


def test_enqueue_failure_retains_then_reclaims_staged_slot(
    tmp_path, monkeypatch
) -> None:
    snapshotter = _ReleaseTrackingSnapshotter()
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "enqueue-failure.jsonl",
        queue_capacity=1,
        _snapshotter=snapshotter,
    )
    writer.begin_generation(1)
    adapter.physics_step_count = 1

    def fail_enqueue(_item) -> None:
        raise queue.Full("synthetic enqueue failure")

    monkeypatch.setattr(writer._work_queue, "put_nowait", fail_enqueue)
    with pytest.raises(queue.Full, match="synthetic enqueue failure"):
        writer(0)
    writer.close()

    assert snapshotter.discarded == [0]
    assert snapshotter.released == [0]


def test_worker_write_failure_still_releases_materialized_slot(
    tmp_path, monkeypatch
) -> None:
    stream = _FailingStream(write_error=True)
    monkeypatch.setattr(trace_module, "_open_exclusive_text", lambda _path: stream)
    snapshotter = _ReleaseTrackingSnapshotter()
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        lambda _marker: None,
        output_path=tmp_path / "worker-release.jsonl",
        queue_capacity=1,
        _snapshotter=snapshotter,
    )
    writer.begin_generation(1)
    adapter.physics_step_count = 1
    writer(0)

    with pytest.raises(OSError, match="synthetic write failure"):
        writer.close()

    assert snapshotter.discarded == []
    assert snapshotter.released == [0]


def test_clean_close_rejects_internal_accounting_drift(tmp_path) -> None:
    markers: list[str] = []
    adapter = _WriterAdapter()
    writer = GroceryRetentionTraceWriter(
        adapter,
        markers.append,
        output_path=tmp_path / "accounting-drift.jsonl",
        capture=lambda: {},
    )
    writer.begin_generation(1)
    _step(writer, adapter, 0, action=0.0)
    with writer._state_lock:
        writer._observed_count += 1

    with pytest.raises(RuntimeError, match="accounting mismatch"):
        writer.close()

    assert markers[-1].startswith("CAP_GROCERY_RETENTION_TRACE_FAILED ")
    assert "observed=2 persisted=1 dropped=0" in markers[-1]
    assert markers[-1].endswith("accounting_complete=false")
