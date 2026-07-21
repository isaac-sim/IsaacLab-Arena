# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stream the CAP barrier exterior_cam RGB-D to the ROS perception bridge.

Kit + GPU smoke (not run in CI). Boots the fixed arena_droid_b1 profile with
cameras enabled, then renders and streams the exterior_cam over the frozen
client-streaming gRPC contract to a running cap_perception_bridge node. The
producer is nonblocking: the sampling loop paces itself and drops frames rather
than stalling, and it never fabricates a frame.

Generate the gRPC stubs once before running (does not touch the pinned venv/lock):

    isaaclab_arena/integrations/cap_barrier/generate_perception_stubs.sh

Then, with the ROS bridge already listening on 127.0.0.1:50061:

    ./dev_run.sh isaaclab_arena/scripts/run_cap_barrier_perception_stream.py \
        --headless --enable_cameras --device cuda:0
"""

from __future__ import annotations

import time

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

_STREAM_HZ = 10.0
_STREAM_SECONDS = 20.0
_ENDPOINT = "127.0.0.1:50061"


def _stream(device: str) -> None:
    import torch

    from isaaclab_arena.integrations.cap_barrier.franka_env import make_cap_franka_environment
    from isaaclab_arena.integrations.cap_barrier.perception_producer import (
        PerceptionFrameProducer,
        extract_camera_frame,
    )

    adapter = make_cap_franka_environment(device=device, enable_cameras=True)
    producer = PerceptionFrameProducer(endpoint=_ENDPOINT)
    try:
        environment = adapter._environment
        unwrapped = adapter._unwrapped
        environment.reset()
        producer.start()
        period_s = 1.0 / _STREAM_HZ
        deadline = time.monotonic() + _STREAM_SECONDS
        next_tick = time.monotonic()
        frame_index = 0
        hold = torch.zeros((1, unwrapped.action_manager.total_action_dim), device=unwrapped.device)
        while time.monotonic() < deadline:
            environment.step(hold)
            frame = extract_camera_frame(environment, frame_index=frame_index)
            producer.offer(frame)
            frame_index += 1
            next_tick += period_s
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        stats = producer.stats
        print(
            "CAP_PERCEPTION_STREAM_TRACE "
            f"frames={frame_index} offered={stats['offered']} sent={stats['sent']} "
            f"dropped={stats['dropped']} stream_starts={stats['stream_starts']}",
            flush=True,
        )
        print(
            "CAP_PERCEPTION_STREAM_OK"
            if stats["sent"] > 0 and stats["stream_starts"] >= 1
            else "CAP_PERCEPTION_STREAM_FAILED",
            flush=True,
        )
    finally:
        producer.close()
        adapter.close()


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    args_cli = parser.parse_args()
    with SimulationAppContext(args_cli):
        _stream(args_cli.device)


if __name__ == "__main__":
    main()
