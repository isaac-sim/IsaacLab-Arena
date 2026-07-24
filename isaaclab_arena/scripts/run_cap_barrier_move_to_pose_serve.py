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
joint_states continuously. It reuses the SAME proven serve+reset-follow primitive
as the open_gripper serve (serve_generation_watching_gripper) so the ResetEpisode
fence is followed identically, but ignores the gripper verdict
(declare_open_success=False) -- the gripper watch is purely observational here and
the arm is driven externally; unlike the production smoke it does NOT choreograph
its own arm+reset sequence. It just serves frames so the arm can be commanded and
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
from collections.abc import Callable
from typing import Any

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


def _close_resources(*resources: Any | None) -> None:
    """Close every owned resource in order, then re-raise the first failure."""
    first_error: Exception | None = None
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.close()
        except Exception as error:  # noqa: BLE001 - cleanup must continue
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


class _PerceptionStreamSink:
    """Co-resident perception snapshot producer driven from the serve loop.

    Camera extraction must run on the main/Kit thread and is synchronous. The
    gRPC offer is nonblocking, but rendering is not, so callers may cap captures
    per generation. The grocery graph uses one fresh snapshot per generation:
    after that snapshot, perception inference cannot re-enter RTX from the 200 Hz
    control loop.
    """

    def __init__(
        self,
        adapter,
        endpoint: str,
        marker_sink,
        *,
        frames_per_generation: int | None,
        first_capture_generation: int = 1,
    ) -> None:
        from isaaclab_arena.integrations.cap_barrier.perception_producer import PerceptionFrameProducer

        if frames_per_generation is not None and frames_per_generation <= 0:
            raise ValueError("frames_per_generation must be positive when set")
        if first_capture_generation <= 0:
            raise ValueError("first_capture_generation must be positive")
        self._adapter = adapter
        self._marker_sink = marker_sink
        self._producer = PerceptionFrameProducer(endpoint=endpoint)
        self._frames_per_generation = frames_per_generation
        self._first_capture_generation = first_capture_generation
        self._frame_index = 0
        self._sample_calls = 0
        self._generation: int | None = None
        self._attempted_in_generation = 0
        self._captured_in_generation = 0
        self._failed = False
        self._producer.start()
        marker_sink(f"CAP_SERVE_KIT_PERCEPTION_STREAM_STARTED endpoint={endpoint}")

    def begin_generation(self, generation: int) -> None:
        if generation <= 0:
            raise ValueError("generation must be positive")
        if self._generation is not None and generation <= self._generation:
            raise ValueError(
                f"perception generation must advance: current={self._generation}, next={generation}"
            )
        self._generation = generation
        self._sample_calls = 0
        self._attempted_in_generation = 0
        self._captured_in_generation = 0
        self._failed = False
        self._marker_sink(
            "CAP_SERVE_KIT_PERCEPTION_GENERATION_ARMED "
            f"generation={generation} frames_per_generation={self._frames_per_generation} "
            f"capture_enabled={int(generation >= self._first_capture_generation)}"
        )

    def on_physics_frame(self, _frame: int) -> None:
        if self._generation is None:
            raise RuntimeError("perception capture used before begin_generation")
        if self._generation < self._first_capture_generation:
            return
        if (
            self._frames_per_generation is not None
            and self._attempted_in_generation >= self._frames_per_generation
        ):
            return
        self._sample_calls += 1
        # Capture immediately on the first PHYSICS frame of each generation so
        # reset-follow cannot expose only the preceding generation's snapshot.
        if (self._sample_calls - 1) % _PERCEPTION_SAMPLE_EVERY_FRAMES != 0:
            return
        # Consume the bounded attempt before entering RTX. An extraction error
        # must not turn one requested snapshot into unbounded render retries.
        self._attempted_in_generation += 1
        try:
            from isaaclab_arena.integrations.cap_barrier.perception_producer import extract_camera_frame

            frame = extract_camera_frame(self._adapter._environment, frame_index=self._frame_index)
            self._producer.offer(frame)
            self._frame_index += 1
            self._captured_in_generation += 1
            if (
                self._frames_per_generation is not None
                and self._captured_in_generation == self._frames_per_generation
            ):
                self._marker_sink(
                    "CAP_SERVE_KIT_PERCEPTION_GENERATION_CAPTURED "
                    f"generation={self._generation} frames={self._captured_in_generation} "
                    f"attempts={self._attempted_in_generation}"
                )
        except Exception as error:  # noqa: BLE001 - never break the serve loop
            if not self._failed:
                self._failed = True
                self._marker_sink(
                    "CAP_SERVE_KIT_PERCEPTION_SAMPLE_FAILED "
                    f"generation={self._generation} attempt={self._attempted_in_generation} "
                    f"detail={error!r}"
                )

    def close(self) -> None:
        self._producer.close()
        stats = self._producer.stats
        self._marker_sink(
            "CAP_SERVE_KIT_PERCEPTION_STREAM_TRACE "
            f"offered={stats['offered']} sent={stats['sent']} "
            f"superseded={stats.get('superseded', 0)} failed={stats.get('failed', 0)} "
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
    perception_frames_per_generation: int | None = None,
    perception_first_capture_generation: int = 1,
    serve_seconds: float = _SERVE_TIMEOUT_S,
    environment_factory: Callable[..., Any] | None = None,
    initial_gripper_closed: bool = True,
    ready_marker: str = "CAP_SERVE_KIT_ARM_READY_FOR_MOVE_TO_POSE",
) -> None:
    from isaaclab_arena.integrations.cap_barrier.franka_env import make_cap_franka_environment
    from isaaclab_arena.integrations.cap_barrier.lockstep_manager import ArenaLockstepManager
    from isaaclab_arena.integrations.cap_barrier.production_smoke import (
        CAP_PRODUCTION_STARTUP_RENDEZVOUS_TIMEOUT_S,
        open_production_startup_rendezvous,
    )
    from isaaclab_arena.integrations.cap_barrier.protocol import ControllerTimingSpec
    from isaaclab_arena.integrations.cap_barrier.serve import ServeExit, serve_generation_watching_gripper
    from isaaclab_arena.integrations.cap_barrier.shared_memory import ArenaBarrierClient

    if environment_factory is None:
        environment_factory = make_cap_franka_environment

    # The arm motion is applied through joint_streaming; hold keeps the arm still
    # between commands; the gripper controller stays available so a mixed skill
    # sequence still works, but this producer never chooses a gripper action.
    timing = (
        ControllerTimingSpec("hold_controller", 1),
        ControllerTimingSpec("joint_streaming_controller", 1),
        ControllerTimingSpec("robotiq_gripper_controller", 1),
    )
    adapter, client, startup_deadline = open_production_startup_rendezvous(
        lambda: environment_factory(
            device=device,
            initial_gripper_closed=initial_gripper_closed,
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
    on_physics_frame = None
    if perception_stream is not None:
        perception = _PerceptionStreamSink(
            adapter,
            perception_stream,
            lambda marker: print(marker, flush=True),
            frames_per_generation=perception_frames_per_generation,
            first_capture_generation=perception_first_capture_generation,
        )
        on_physics_frame = perception.on_physics_frame
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
            raise RuntimeError(f"serve producer must bootstrap at generation 1, got {generation_1}")
        print("CAP_SERVE_KIT_GENERATION_1_ATTACHED", flush=True)

        bootstrap_fences = _paced_bootstrap_fences(manager, adapter, _BOOTSTRAP_FENCE_FRAMES)
        if bootstrap_fences[0].sequence != 0 or bootstrap_fences[0].physics_tick != 0:
            raise RuntimeError("generation 1 did not begin at sequence/tick zero")
        print(
            f"CAP_SERVE_KIT_BOOTSTRAP_FENCE_OK frames={len(bootstrap_fences)} "
            f"arm0_rad={adapter.arm_positions()[0]:.6f}",
            flush=True,
        )
        if perception is not None:
            perception.begin_generation(generation_1)
        print(ready_marker, flush=True)

        # Serve like a real episode for an external ARM policy, reusing the SAME
        # proven serve+reset-follow primitive as the open_gripper serve
        # (serve_generation_watching_gripper): it ticks PHYSICS at the 200 Hz base
        # every frame -- applying whatever arm command the control plane streams
        # (the guarded joint_trajectory the move_to_pose skill executes under its
        # lease) and publishing joint_states -- and returns on the reset fence
        # (GENERATION_ADVANCED / BARRIER_INTERRUPTED). We ignore the gripper verdict
        # (declare_open_success=False): the gripper watch is purely observational
        # here; the arm is driven externally. Following the ResetEpisode this way is
        # what advances to the accepting-goals generation AND keeps physics ticking
        # so the arm lease's controller switch completes and joint_states flows.
        deadline = time.monotonic() + serve_seconds
        generations_served = 0
        for generations_served in range(1, _MAX_GENERATIONS_FOLLOWED + 1):
            observation = serve_generation_watching_gripper(
                manager,
                adapter.gripper_position,
                deadline_monotonic_s=deadline,
                settle_frames=0,
                declare_open_success=False,
                marker_sink=lambda marker: print(marker, flush=True),
                on_physics_frame=on_physics_frame,
            )
            print(
                "CAP_SERVE_KIT_GENERATION_SERVED "
                f"exit={observation.exit_reason} frames={observation.physics_frames} "
                f"arm0_rad={adapter.arm_positions()[0]:.6f}",
                flush=True,
            )
            # A ResetEpisode surfaces as a generation advance or a BarrierInterrupted
            # (sidecar reset-phase deactivation race); follow both. Anything else
            # (timeout / open-transition) means the window is done -- stop serving.
            if observation.exit_reason not in (
                ServeExit.GENERATION_ADVANCED,
                ServeExit.BARRIER_INTERRUPTED,
            ):
                break
            try:
                followed_generation = manager.attach_next_generation(timeout_s=30.0)
            except Exception as error:  # noqa: BLE001 - report and stop cleanly
                print(
                    f"CAP_SERVE_KIT_FOLLOW_RESET_FAILED detail={error!r} after={observation.exit_reason}",
                    flush=True,
                )
                break
            bootstrap = _paced_bootstrap_fences(manager, adapter, _BOOTSTRAP_FENCE_FRAMES)
            if perception is not None:
                perception.begin_generation(followed_generation)
            print(
                f"CAP_SERVE_KIT_FOLLOWED_RESET generation={followed_generation} "
                f"fences={len(bootstrap)} arm0_rad={adapter.arm_positions()[0]:.6f}",
                flush=True,
            )
        print(
            f"CAP_SERVE_KIT_TRACE generations={generations_served}",
            flush=True,
        )
        print("CAP_SERVE_KIT_DONE", flush=True)
    finally:
        _close_resources(perception, client, adapter)


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
