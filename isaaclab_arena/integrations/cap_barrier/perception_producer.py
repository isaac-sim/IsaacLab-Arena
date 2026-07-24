# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Nonblocking CAP perception producer for the Arena barrier half.

The barrier drives Kit at 200 Hz in lockstep with the ROS control plane. This
producer must never stall that loop, so it splits into two halves:

* ``extract_camera_frame`` reads the ``exterior_cam`` RGB-D output and camera
  pose from an already-stepped Arena environment. RTX rendering is triggered by
  the data access, so the caller decides how often to pay the render cost (the
  barrier itself never reads the camera per physics tick).

* ``PerceptionFrameProducer`` owns a background gRPC client-streaming thread with
  a single-slot latest-frame mailbox. ``offer`` is nonblocking: it replaces any
  unsent frame and returns immediately, so a slow or disconnected bridge drops
  frames instead of back-pressuring the physics loop. The thread reconnects and
  restarts the stream on failure, and never fabricates frames.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import grpc

from ._generated.cap_perception_proto import cap_perception_pb2 as _pb2
from ._generated.cap_perception_proto import cap_perception_pb2_grpc as _pb2_grpc

_CAM = "exterior_cam"
_DEPTH_DT = "distance_to_image_plane"
_DEFAULT_ENDPOINT = "127.0.0.1:50061"
_RECONNECT_BACKOFF_S = 0.5
_PUBLISH_ACK_TIMEOUT_S = 1.0
_CLOSE_TIMEOUT_S = 2.0
_MAX_GRPC_MESSAGE_BYTES = 8 * 1024 * 1024
_PERMANENT_RPC_CODES = frozenset(
    {
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.FAILED_PRECONDITION,
        grpc.StatusCode.OUT_OF_RANGE,
        grpc.StatusCode.PERMISSION_DENIED,
        grpc.StatusCode.UNAUTHENTICATED,
        grpc.StatusCode.UNIMPLEMENTED,
    }
)


@dataclass(frozen=True)
class PerceptionFrame:
    """One rendered RGB-D frame plus source evidence, detached from Kit tensors."""

    camera_name: str
    width: int
    height: int
    rgb: bytes
    depth: bytes
    intrinsic_matrix: tuple[float, ...]
    camera_pose: tuple[float, ...]
    frame_index: int
    capture_monotonic_ns: int

    def to_proto(self) -> _pb2.CameraFrame:
        return _pb2.CameraFrame(
            camera_name=self.camera_name,
            width=self.width,
            height=self.height,
            rgb=self.rgb,
            depth=self.depth,
            intrinsic_matrix=list(self.intrinsic_matrix),
            camera_pose=list(self.camera_pose),
            frame_index=self.frame_index,
            capture_monotonic_ns=self.capture_monotonic_ns,
        )


def extract_camera_frame(
    environment: Any,
    *,
    frame_index: int,
    camera_name: str = _CAM,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> PerceptionFrame:
    """Render and detach one RGB-D frame + world pose from an already-stepped env.

    Ordinary CAP control steps deliberately skip Kit/RTX. A non-skipped render
    prepares Kit visualizers when present; Camera.data then performs the camera
    render itself when no visualizer pumps the Kit app. Call this only when a
    frame is actually wanted, after a physics step.
    """
    import numpy as np

    unwrapped = environment.unwrapped
    if camera_name not in unwrapped.scene.sensors:
        raise RuntimeError(f"{camera_name} sensor is not attached; build the env with enable_cameras=True")
    if not hasattr(unwrapped, "sim"):
        raise RuntimeError("CAP camera environment does not expose its simulation context")
    unwrapped.sim.render(skip_app_pumping=False)
    camera = unwrapped.scene[camera_name]

    rgb_tensor = camera.data.output["rgb"][0, ..., :3]
    height, width = int(rgb_tensor.shape[0]), int(rgb_tensor.shape[1])
    rgb = np.ascontiguousarray(rgb_tensor.detach().cpu().numpy().astype(np.uint8)).tobytes()

    depth_bytes = b""
    if _DEPTH_DT in camera.data.output:
        depth_raw = camera.data.output[_DEPTH_DT][0].squeeze(-1).detach().cpu().numpy().astype(np.float32)
        depth = np.ascontiguousarray(np.nan_to_num(depth_raw, nan=0.0, posinf=0.0, neginf=0.0))
        depth_bytes = depth.tobytes()

    intrinsics = camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
    pos = camera.data.pos_w[0].detach().cpu().numpy().astype(np.float64)
    # Isaac Lab exposes camera quaternions in xyzw order. The CAP perception
    # transport is frozen as xyz + wxyz, so convert exactly once at this boundary.
    quat_xyzw = camera.data.quat_w_ros[0].detach().cpu().numpy().astype(np.float64)

    return PerceptionFrame(
        camera_name=camera_name,
        width=width,
        height=height,
        rgb=rgb,
        depth=depth_bytes,
        intrinsic_matrix=tuple(float(v) for v in intrinsics.reshape(-1)),
        camera_pose=(
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(quat_xyzw[3]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
        ),
        frame_index=frame_index,
        capture_monotonic_ns=monotonic_ns(),
    )


class PerceptionFrameProducer:
    """Background gRPC client that streams only the latest offered frame."""

    def __init__(
        self,
        *,
        endpoint: str = _DEFAULT_ENDPOINT,
        channel_factory: Callable[[str], grpc.Channel] | None = None,
        reconnect_backoff_s: float = _RECONNECT_BACKOFF_S,
        publish_ack_timeout_s: float = _PUBLISH_ACK_TIMEOUT_S,
    ) -> None:
        if not math.isfinite(reconnect_backoff_s) or reconnect_backoff_s < 0.0:
            raise ValueError("reconnect_backoff_s must be finite and nonnegative")
        if not math.isfinite(publish_ack_timeout_s) or publish_ack_timeout_s <= 0.0:
            raise ValueError("publish_ack_timeout_s must be finite and positive")
        self._endpoint = endpoint
        self._channel_factory = channel_factory or self._default_channel
        self._reconnect_backoff_s = reconnect_backoff_s
        self._publish_ack_timeout_s = publish_ack_timeout_s
        self._lock = threading.Lock()
        self._pending: PerceptionFrame | None = None
        self._inflight: PerceptionFrame | None = None
        self._new_frame = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_call: Any | None = None
        self._poisoned = False
        self._offered = 0
        self._dropped = 0
        self._sent = 0
        self._superseded = 0
        self._failed = 0
        self._stream_starts = 0

    @staticmethod
    def _default_channel(endpoint: str) -> grpc.Channel:
        return grpc.insecure_channel(
            endpoint,
            options=(
                ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
            ),
        )

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("producer already started")
        if self._stop.is_set():
            raise RuntimeError("producer is closed")
        self._thread = threading.Thread(target=self._run, name="cap-perception-producer", daemon=True)
        self._thread.start()

    def offer(self, frame: PerceptionFrame) -> bool:
        """Install ``frame`` as the latest to send; never blocks the caller."""
        with self._new_frame:
            if self._stop.is_set():
                raise RuntimeError("perception producer is closed")
            if self._poisoned:
                raise RuntimeError("perception producer is poisoned by a permanent transport failure")
            replaced = self._pending is not None
            self._pending = frame
            self._offered += 1
            if replaced:
                self._dropped += 1
            self._new_frame.notify()
        return not replaced

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "offered": self._offered,
                "sent": self._sent,
                "superseded": self._superseded,
                "failed": self._failed,
                "dropped": self._dropped,
                "stream_starts": self._stream_starts,
            }

    def _take_pending(self, timeout_s: float) -> PerceptionFrame | None:
        with self._new_frame:
            if self._pending is None:
                self._new_frame.wait(timeout=timeout_s)
            frame = self._pending
            self._pending = None
            if frame is not None:
                self._inflight = frame
            return frame

    def _requeue_inflight(self) -> None:
        # A stream can die after a frame is taken but before it is delivered.
        # Re-offer that in-flight frame (unless a newer one already arrived) so the
        # next stream still carries the latest real frame and none is silently lost.
        with self._new_frame:
            if self._inflight is not None and self._pending is None:
                self._pending = self._inflight
                self._new_frame.notify()
            self._inflight = None

    def _wait_for_ready(self, ready: threading.Event) -> bool:
        deadline = time.monotonic() + self._reconnect_backoff_s + 1.0
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            if ready.wait(timeout=min(remaining, 0.05)):
                return True
        return False

    def _settle_inflight(self, *, superseded: bool) -> None:
        with self._lock:
            self._sent += 1
            self._superseded += int(superseded)
            self._inflight = None

    def _poison_inflight(self) -> None:
        with self._lock:
            self._failed += 1
            self._inflight = None
            self._poisoned = True

    def _run(self) -> None:
        while not self._stop.is_set():
            ready = threading.Event()
            channel: grpc.Channel | None = None

            def _on_state(state: grpc.ChannelConnectivity, _ready: threading.Event = ready) -> None:
                if state is grpc.ChannelConnectivity.READY:
                    _ready.set()
                else:
                    _ready.clear()

            try:
                channel = self._channel_factory(self._endpoint)
                channel.subscribe(_on_state, try_to_connect=True)
                if not self._wait_for_ready(ready):
                    if self._stop.wait(self._reconnect_backoff_s):
                        break
                    continue
                stub = _pb2_grpc.CapPerceptionStub(channel)
                while ready.is_set() and not self._stop.is_set():
                    frame = self._take_pending(timeout_s=0.1)
                    if frame is None:
                        continue
                    if not ready.is_set():
                        self._requeue_inflight()
                        break
                    # Use a finite one-frame client stream so PublishAck is an
                    # actual remote-consumption boundary. Clearing _inflight
                    # merely because gRPC pulled from a long-lived iterator can
                    # lose the only snapshot on a post-yield disconnect.
                    with self._lock:
                        self._stream_starts += 1
                    call = stub.PublishFrames.future(
                        iter((frame.to_proto(),)),
                        timeout=self._publish_ack_timeout_s,
                    )
                    with self._lock:
                        self._active_call = call
                    try:
                        acknowledgement = call.result()
                    finally:
                        with self._lock:
                            if self._active_call is call:
                                self._active_call = None
                    received = int(acknowledgement.frames_received)
                    if received not in (0, 1):
                        self._poison_inflight()
                        return
                    # The bridge validates every received frame, but its frozen
                    # count is the number installed in the latest-frame store.
                    # Zero therefore means this retry was already present or a
                    # newer frame won; either value is terminal remote settlement.
                    self._settle_inflight(superseded=received == 0)
            except grpc.RpcError as error:
                if self._stop.is_set():
                    break
                if error.code() in _PERMANENT_RPC_CODES:
                    self._poison_inflight()
                    return
                self._stop.wait(self._reconnect_backoff_s)
            except Exception:
                if self._stop.is_set():
                    break
                self._stop.wait(self._reconnect_backoff_s)
            finally:
                if channel is not None:
                    with suppress(Exception):
                        channel.unsubscribe(_on_state)
                    with suppress(Exception):
                        channel.close()
                self._requeue_inflight()

    def close(self) -> None:
        self._stop.set()
        with self._new_frame:
            self._new_frame.notify_all()
        with self._lock:
            active_call = self._active_call
        if active_call is not None:
            with suppress(Exception):
                active_call.cancel()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(_CLOSE_TIMEOUT_S, self._publish_ack_timeout_s + 0.5))
            if thread.is_alive():
                raise RuntimeError("perception producer worker did not stop within its bound")
            self._thread = None
