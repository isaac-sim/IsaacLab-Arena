# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kit-free tests for the CAP perception producer.

The camera frame extraction is exercised against a fake Arena env (no Kit), and
the nonblocking streaming path runs against a real in-process gRPC server that
mirrors the ROS bridge servicer. Requires the generated stubs; run
generate_perception_stubs.sh first (skipped otherwise).
"""

from __future__ import annotations

import numpy as np
import socket
import struct
import threading
import time
from concurrent import futures
from types import SimpleNamespace

import pytest

grpc = pytest.importorskip("grpc")

try:
    from isaaclab_arena.integrations.cap_barrier._generated.cap_perception_proto import cap_perception_pb2 as pb2
    from isaaclab_arena.integrations.cap_barrier._generated.cap_perception_proto import (
        cap_perception_pb2_grpc as pb2_grpc,
    )
except ImportError:  # pragma: no cover - stubs not generated in this checkout
    pytest.skip(
        "run generate_perception_stubs.sh to build the CAP perception gRPC stubs",
        allow_module_level=True,
    )

from isaaclab_arena.integrations.cap_barrier.perception_producer import (
    PerceptionFrame,
    PerceptionFrameProducer,
    extract_camera_frame,
)


def _fake_environment(
    width: int = 4,
    height: int = 3,
    *,
    quat_xyzw: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.9),
):
    import torch

    rgb = torch.arange(height * width * 4, dtype=torch.uint8).reshape(1, height, width, 4)
    depth = torch.full((1, height, width, 1), 0.75, dtype=torch.float32)
    intrinsics = torch.tensor([[[200.0, 0.0, 2.0], [0.0, 201.0, 1.5], [0.0, 0.0, 1.0]]])
    camera = SimpleNamespace(
        data=SimpleNamespace(
            output={"rgb": rgb, "distance_to_image_plane": depth},
            intrinsic_matrices=intrinsics,
            pos_w=torch.tensor([[1.0, 2.0, 3.0]]),
            # CameraData.quat_w_ros is xyzw. Use four distinct values so the
            # producer's frozen wxyz transport conversion cannot pass vacuously.
            quat_w_ros=torch.tensor([quat_xyzw]),
        )
    )

    class _Scene:
        def __init__(self, sensors):
            self.sensors = sensors

        def __getitem__(self, key):
            return self.sensors[key]

    unwrapped = SimpleNamespace(scene=_Scene({"exterior_cam": camera}))
    return SimpleNamespace(unwrapped=unwrapped), width, height


def test_extract_camera_frame_detaches_rgbd_pose_and_intrinsics() -> None:
    pytest.importorskip("torch")
    env, width, height = _fake_environment()

    frame = extract_camera_frame(env, frame_index=5, monotonic_ns=lambda: 4242)

    assert frame.camera_name == "exterior_cam"
    assert (frame.width, frame.height) == (width, height)
    assert len(frame.rgb) == width * height * 3
    assert frame.rgb == bytes(
        (np.arange(height * width * 4).reshape(height, width, 4)[..., :3]).astype(np.uint8).reshape(-1)
    )
    depth = struct.unpack(f"<{width * height}f", frame.depth)
    assert all(value == pytest.approx(0.75) for value in depth)
    assert frame.intrinsic_matrix == pytest.approx((200.0, 0.0, 2.0, 0.0, 201.0, 1.5, 0.0, 0.0, 1.0))
    assert frame.camera_pose == pytest.approx((1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3))
    assert frame.frame_index == 5
    assert frame.capture_monotonic_ns == 4242


def test_maple_oblique_pose_survives_cap_wxyz_transport() -> None:
    import torch

    from isaaclab.utils.math import matrix_from_quat

    from isaaclab_arena_environments.maple_cameras import _CAM_EYE, _CAM_TARGET, _agentview_ros_quat_xyzw

    camera_xyzw = _agentview_ros_quat_xyzw(_CAM_EYE, _CAM_TARGET)
    env, _, _ = _fake_environment(quat_xyzw=camera_xyzw)

    frame = extract_camera_frame(env, frame_index=0)

    # CAP transports xyz + wxyz. Convert back to Isaac Lab's xyzw at the
    # consumer edge and prove the actual oblique optical axis is unchanged.
    transport_wxyz = frame.camera_pose[3:]
    assert transport_wxyz == pytest.approx((camera_xyzw[3], camera_xyzw[0], camera_xyzw[1], camera_xyzw[2]))
    consumer_xyzw = torch.tensor(
        [[transport_wxyz[1], transport_wxyz[2], transport_wxyz[3], transport_wxyz[0]]],
        dtype=torch.float32,
    )
    camera_forward = matrix_from_quat(consumer_xyzw)[0, :, 2]
    expected_forward = torch.tensor(_CAM_TARGET, dtype=torch.float32) - torch.tensor(_CAM_EYE, dtype=torch.float32)
    expected_forward /= torch.linalg.norm(expected_forward)
    assert float(camera_forward @ expected_forward) > 0.999
    # arena_droid_b1 pins T_world_base to identity, so this world translation is
    # also the planner-base camera translation.
    assert frame.camera_pose[:3] == pytest.approx((1.0, 2.0, 3.0))


def _sample_frame(frame_index: int, capture_ns: int) -> PerceptionFrame:
    return PerceptionFrame(
        camera_name="exterior_cam",
        width=2,
        height=1,
        rgb=bytes(range(6)),
        depth=struct.pack("<2f", 0.5, 0.6),
        intrinsic_matrix=(100.0, 0.0, 1.0, 0.0, 100.0, 0.5, 0.0, 0.0, 1.0),
        camera_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        frame_index=frame_index,
        capture_monotonic_ns=capture_ns,
    )


def test_offer_drops_unsent_frames_without_blocking() -> None:
    producer = PerceptionFrameProducer(endpoint="127.0.0.1:0")

    assert producer.offer(_sample_frame(0, 10)) is True
    assert producer.offer(_sample_frame(1, 11)) is False
    assert producer.offer(_sample_frame(2, 12)) is False

    frame = producer._take_pending(timeout_s=0.0)
    assert frame is not None and frame.frame_index == 2
    stats = producer.stats
    assert stats["offered"] == 3
    assert stats["dropped"] == 2


class _RecordingServicer(pb2_grpc.CapPerceptionServicer):
    def __init__(self) -> None:
        self.frames: list[pb2.CameraFrame] = []
        self.streams = 0
        self.event = threading.Event()

    def PublishFrames(self, request_iterator, context):
        self.streams += 1
        count = 0
        for count, frame in enumerate(request_iterator, start=1):
            self.frames.append(frame)
            self.event.set()
        return pb2.PublishAck(frames_received=count)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_producer_streams_latest_frame_and_restarts_after_disconnect() -> None:
    servicer = _RecordingServicer()

    def _serve(port: int):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        pb2_grpc.add_CapPerceptionServicer_to_server(servicer, server)
        server.add_insecure_port(f"127.0.0.1:{port}")
        server.start()
        return server

    port = _free_port()
    server = _serve(port)
    producer = PerceptionFrameProducer(endpoint=f"127.0.0.1:{port}", reconnect_backoff_s=0.05)
    producer.start()
    try:
        producer.offer(_sample_frame(7, 100))
        assert servicer.event.wait(timeout=5.0)
        deadline = time.monotonic() + 5.0
        while not servicer.frames and time.monotonic() < deadline:
            time.sleep(0.01)
        assert servicer.frames[0].frame_index == 7
        assert servicer.frames[0].camera_name == "exterior_cam"

        # Drop the server to force an RpcError, then bring it back on the same port.
        server.stop(grace=None).wait(timeout=2.0)
        time.sleep(0.1)
        servicer.event.clear()
        server = _serve(port)
        producer.offer(_sample_frame(8, 200))
        assert servicer.event.wait(timeout=5.0)
        indexes = {frame.frame_index for frame in servicer.frames}
        assert 8 in indexes
        assert servicer.streams >= 2
        assert producer.stats["stream_starts"] >= 2
    finally:
        producer.close()
        server.stop(grace=None).wait(timeout=2.0)
