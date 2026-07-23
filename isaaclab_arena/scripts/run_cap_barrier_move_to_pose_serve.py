# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serve the B=1 DROID barrier half for the move_to_pose (arm-motion) smoke.

Cold-start Kit first, exactly like the open_gripper serve and the production
smoke: this producer prints ``CAP_PRODUCTION_KIT_ENV_READY``, waits for the CAP
control plane to become serviceable, attaches generation 1, and runs the
generation-1 FENCE bootstrap so the control plane can activate the hold and
joint-streaming controllers.

It then serves the barrier like a real episode for an EXTERNAL ARM policy: it
keeps ticking PHYSICS at the 200 Hz base and applies whatever arm command the
control plane streams each frame (the guarded joint_trajectory command the
server-side robot.move_to_pose skill executes under its arm lease), publishing
joint_states continuously. Unlike the open_gripper serve it does NOT watch for a
gripper transition, and unlike the production smoke it does NOT choreograph its
own arm+reset sequence -- it just serves frames so the arm can be commanded and
the hold->joint_trajectory controller switch can complete inside update().

This is the producer the move_to_pose smoke needs: without continuous physics
ticks the arm lease's controller switch cannot complete (Jazzy CM switches only
inside update()) and the arbiter denies the arm lease. It follows a ResetEpisode
(if the connector issues one) into the next generation and re-bootstraps, so the
motion can land on either the pre- or post-reset generation.
"""

from __future__ import annotations

import math
import os
import time

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

_SHM_NAME = "/cap_arena_b1"
_BOOTSTRAP_FENCE_FRAMES = 200
_BASE_PERIOD_S = 0.005
# Bounded window for an external policy to attach, (optionally) ResetEpisode,
# acquire the arm lease, plan+execute move_to_pose, and release. Must comfortably
# exceed the connector's own exchange timeouts so this producer never stops
# serving mid-exchange.
_SERVE_TIMEOUT_S = 600.0
# A ResetEpisode (gen 1 -> 2) may precede the motion; allow a small margin.
_MAX_GENERATIONS_FOLLOWED = 4
# 200 Hz base / 20 = ~10 Hz camera sampling, matching the standalone stream default.
_PERCEPTION_SAMPLE_EVERY_FRAMES = 20


class _PerceptionStreamSink:
    """Co-resident, nonblocking perception producer driven from the serve loop.

    Identical contract to the open_gripper serve: samples the exterior_cam right
    after a physics step on the main/Kit thread at a decimated rate and offers it
    to the latest-frame gRPC producer. The offer never blocks the 200 Hz loop and
    drops rather than stalls; a sampling error is logged once and never propagates
    into the barrier serve loop.
    """

    def __init__(self, adapter, endpoint: str, marker_sink) -> None:
        from isaaclab_arena.integrations.cap_barrier.perception_producer import (
            PerceptionFrameProducer,
        )

        self._adapter = adapter
        self._marker_sink = marker_sink
        self._producer = PerceptionFrameProducer(endpoint=endpoint)
        self._frame_index = 0
        self._sample_calls = 0
        self._failed = False
        self._producer.start()
        marker_sink(f"CAP_SERVE_KIT_PERCEPTION_STREAM_STARTED endpoint={endpoint}")

    def on_frame(self, _result) -> None:
        self._sample_calls += 1
        if self._sample_calls % _PERCEPTION_SAMPLE_EVERY_FRAMES != 0:
            return
        try:
            from isaaclab_arena.integrations.cap_barrier.perception_producer import (
                extract_camera_frame,
            )

            frame = extract_camera_frame(
                self._adapter._environment, frame_index=self._frame_index
            )
            self._producer.offer(frame)
            self._frame_index += 1
        except Exception as error:  # noqa: BLE001 - never break the serve loop
            if not self._failed:
                self._failed = True
                self._marker_sink(f"CAP_SERVE_KIT_PERCEPTION_SAMPLE_FAILED detail={error!r}")

    def close(self) -> None:
        stats = self._producer.stats
        self._producer.close()
        self._marker_sink(
            "CAP_SERVE_KIT_PERCEPTION_STREAM_TRACE "
            f"offered={stats['offered']} sent={stats['sent']} "
            f"dropped={stats['dropped']} stream_starts={stats['stream_starts']}"
        )


def _max_delta(lhs, rhs) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(lhs, rhs, strict=True))


def _paced_bootstrap_fences(manager, adapter, count: int):
    frozen_state = adapter.abi_positions()
    results = []
    deadline = time.monotonic()
    for _ in range(count):
        results.append(manager.fence())
        deadline += _BASE_PERIOD_S
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    frozen_delta = _max_delta(frozen_state, adapter.abi_positions())
    if frozen_delta > 1e-12:
        raise RuntimeError(f"FENCE advanced DROID state: max_delta={frozen_delta}")
    return results


def _run_serve(
    device: str,
    *,
    perception_stream: str | None = None,
    serve_seconds: float = _SERVE_TIMEOUT_S,
) -> None:
    from isaaclab_arena.integrations.cap_barrier.franka_env import make_cap_franka_environment
    from isaaclab_arena.integrations.cap_barrier.lockstep_manager import ArenaLockstepManager
    from isaaclab_arena.integrations.cap_barrier.production_smoke import (
        CAP_PRODUCTION_STARTUP_RENDEZVOUS_TIMEOUT_S,
        open_production_startup_rendezvous,
        run_physics_until_generation_transition,
    )
    from isaaclab_arena.integrations.cap_barrier.protocol import ControllerTimingSpec
    from isaaclab_arena.integrations.cap_barrier.shared_memory import ArenaBarrierClient

    # The arm motion is applied through joint_streaming; hold keeps the arm still
    # between commands; the gripper controller stays available so a mixed skill
    # sequence still works, but this producer never chooses a gripper action.
    timing = (
        ControllerTimingSpec("hold_controller", 1),
        ControllerTimingSpec("joint_streaming_controller", 1),
        ControllerTimingSpec("robotiq_gripper_controller", 1),
    )
    adapter, client, startup_deadline = open_production_startup_rendezvous(
        lambda: make_cap_franka_environment(
            device=device,
            initial_gripper_closed=True,
            enable_cameras=perception_stream is not None,
        ),
        lambda deadline: ArenaBarrierClient(
            _SHM_NAME,
            open_timeout_s=CAP_PRODUCTION_STARTUP_RENDEZVOUS_TIMEOUT_S,
            startup_deadline_monotonic_s=deadline,
        ),
        marker_sink=lambda marker: print(marker, flush=True),
    )
    perception: _PerceptionStreamSink | None = None
    on_frame = None
    if perception_stream is not None:
        perception = _PerceptionStreamSink(
            adapter, perception_stream, lambda marker: print(marker, flush=True)
        )
        on_frame = perception.on_frame
    try:
        manager = ArenaLockstepManager(
            client=client,
            simulation=adapter,
            joint_mapping=adapter.joint_mapping,
            controller_specs=timing,
            command_timeout_s=30.0,
        )
        generation_1 = manager.attach_initial_generation(
            timeout_s=CAP_PRODUCTION_STARTUP_RENDEZVOUS_TIMEOUT_S,
            startup_deadline_monotonic_s=startup_deadline,
        )
        if generation_1 != 1:
            raise RuntimeError(
                f"serve producer must bootstrap at generation 1, got {generation_1}"
            )
        print("CAP_SERVE_KIT_GENERATION_1_ATTACHED", flush=True)

        bootstrap_fences = _paced_bootstrap_fences(manager, adapter, _BOOTSTRAP_FENCE_FRAMES)
        if bootstrap_fences[0].sequence != 0 or bootstrap_fences[0].physics_tick != 0:
            raise RuntimeError("generation 1 did not begin at sequence/tick zero")
        print(
            f"CAP_SERVE_KIT_BOOTSTRAP_FENCE_OK frames={len(bootstrap_fences)} "
            f"arm0_rad={adapter.arm_positions()[0]:.6f}",
            flush=True,
        )
        print("CAP_SERVE_KIT_ARM_READY_FOR_MOVE_TO_POSE", flush=True)

        # Serve like a real episode: keep ticking PHYSICS and apply whatever arm
        # command the control plane streams, following any ResetEpisode into the
        # next generation and re-bootstrapping. No gripper/arm choreography here --
        # the external move_to_pose skill drives the arm.
        deadline_wall = time.monotonic() + serve_seconds
        total_frames = 0
        generations_served = 0
        while time.monotonic() < deadline_wall:
            remaining_s = deadline_wall - time.monotonic()
            if remaining_s <= 0:
                break
            observation = run_physics_until_generation_transition(
                manager,
                timeout_s=remaining_s,
                on_frame=on_frame,
            )
            total_frames += observation.physics_frames
            generations_served += 1
            print(
                f"CAP_SERVE_KIT_GENERATION_SERVED generation={observation.previous_generation} "
                f"frames={observation.physics_frames} arm0_rad={adapter.arm_positions()[0]:.6f}",
                flush=True,
            )
            if generations_served >= _MAX_GENERATIONS_FOLLOWED:
                break
            try:
                followed_generation = manager.attach_next_generation(timeout_s=30.0)
            except Exception as error:  # noqa: BLE001 - report and stop cleanly
                print(
                    f"CAP_SERVE_KIT_FOLLOW_RESET_FAILED detail={error!r}",
                    flush=True,
                )
                break
            bootstrap = _paced_bootstrap_fences(manager, adapter, _BOOTSTRAP_FENCE_FRAMES)
            print(
                f"CAP_SERVE_KIT_FOLLOWED_RESET generation={followed_generation} "
                f"fences={len(bootstrap)} arm0_rad={adapter.arm_positions()[0]:.6f}",
                flush=True,
            )
        print(
            f"CAP_SERVE_KIT_TRACE generations={generations_served} frames={total_frames}",
            flush=True,
        )
        print("CAP_SERVE_KIT_DONE", flush=True)
    finally:
        if perception is not None:
            perception.close()
        client.close()
        adapter.close()


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument(
        "--perception-stream",
        default=None,
        metavar="HOST:PORT",
        help=(
            "Also stream exterior_cam RGB-D to a cap_perception_bridge at this "
            "endpoint from THIS serving process (e.g. 127.0.0.1:50061). Enables "
            "cameras on the barrier env; the barrier is served by one Kit process."
        ),
    )
    parser.add_argument(
        "--serve-seconds",
        type=float,
        default=float(os.environ.get("CAP_SERVE_SECONDS", _SERVE_TIMEOUT_S)),
        help=(
            "Bounded serve window in seconds (default 600, env CAP_SERVE_SECONDS). "
            "Raise it when a cold external policy load would otherwise eat most of "
            "the window."
        ),
    )
    args_cli = parser.parse_args()
    if not math.isfinite(args_cli.serve_seconds) or args_cli.serve_seconds <= 0.0:
        parser.error("--serve-seconds must be finite and positive")
    if args_cli.perception_stream is not None:
        # The exterior_cam RTX spawn requires AppLauncher cameras; imply the flag
        # so --perception-stream can never be run without it.
        args_cli.enable_cameras = True
    with SimulationAppContext(args_cli):
        _run_serve(
            args_cli.device,
            perception_stream=args_cli.perception_stream,
            serve_seconds=args_cli.serve_seconds,
        )


if __name__ == "__main__":
    main()
