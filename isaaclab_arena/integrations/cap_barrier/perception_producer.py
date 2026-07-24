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
_MAX_GRPC_MESSAGE_BYTES = 8 * 1024 * 1024


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
    """Read one RGB-D frame + world pose from an already-stepped Arena env.

    Accessing the camera output triggers the RTX render for the current sim
    state; call this only when a frame is actually wanted, after a physics step.
    """
    import numpy as np

    unwrapped = environment.unwrapped
    if camera_name not in unwrapped.scene.sensors:
        raise RuntimeError(f"{camera_name} sensor is not attached; build the env with enable_cameras=True")
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
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._endpoint = endpoint
        self._channel_factory = channel_factory or self._default_channel
        self._reconnect_backoff_s = reconnect_backoff_s
        self._sleep = sleep
        self._lock = threading.Lock()
        self._pending: PerceptionFrame | None = None
        self._inflight: PerceptionFrame | None = None
        self._new_frame = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offered = 0
        self._dropped = 0
        self._sent = 0
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
        self._thread = threading.Thread(target=self._run, name="cap-perception-producer", daemon=True)
        self._thread.start()

    def offer(self, frame: PerceptionFrame) -> bool:
        """Install ``frame`` as the latest to send; never blocks the caller."""
        with self._new_frame:
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

    def _frame_iterator(self, ready: threading.Event):
        while not self._stop.is_set():
            frame = self._take_pending(timeout_s=0.1)
            if frame is None:
                # Idle client streams do not observe server death, so end the RPC
                # and let the run loop reconnect once the channel breaks.
                if not ready.is_set():
                    return
                continue
            if not ready.is_set():
                # Do not spend the latest frame on a broken channel; requeue it so
                # the next stream carries it, then end this RPC to force reconnect.
                self._requeue_inflight()
                return
            yield frame.to_proto()
            with self._lock:
                self._sent += 1
                self._inflight = None

    def _run(self) -> None:
        while not self._stop.is_set():
            ready = threading.Event()

            def _on_state(state: grpc.ChannelConnectivity, _ready: threading.Event = ready) -> None:
                if state is grpc.ChannelConnectivity.READY:
                    _ready.set()
                else:
                    _ready.clear()

            channel = self._channel_factory(self._endpoint)
            channel.subscribe(_on_state, try_to_connect=True)
            try:
                if not ready.wait(timeout=self._reconnect_backoff_s + 1.0):
                    if self._stop.wait(self._reconnect_backoff_s):
                        break
                    continue
                stub = _pb2_grpc.CapPerceptionStub(channel)
                with self._lock:
                    self._stream_starts += 1
                stub.PublishFrames(self._frame_iterator(ready))
            except grpc.RpcError:
                if self._stop.is_set():
                    break
                self._sleep(self._reconnect_backoff_s)
            except Exception:
                if self._stop.is_set():
                    break
                self._sleep(self._reconnect_backoff_s)
            finally:
                with suppress(Exception):
                    channel.unsubscribe(_on_state)
                channel.close()
                self._requeue_inflight()

    def close(self) -> None:
        self._stop.set()
        with self._new_frame:
            self._new_frame.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
