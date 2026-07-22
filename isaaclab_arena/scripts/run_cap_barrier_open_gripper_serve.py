# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serve the B=1 DROID barrier half for the GaP-through-ROS open_gripper skeleton.

Cold-start Kit first, exactly like the production smoke: this producer prints
``CAP_PRODUCTION_KIT_ENV_READY``, waits for the CAP control plane to become
serviceable, attaches generation 1, and runs the generation-1 FENCE bootstrap so
the control plane can activate hold + gripper controllers. It then hands control
to an external policy -- GaP issuing a single ``open_gripper`` through ROS -- by
serving gen-1 PHYSICS frames and observing the gripper actuate, without choosing
the action or asserting the production smoke's arm+reset sequence.

The scene starts with the gripper CLOSED so that one open_gripper yields an
unambiguous closed->open transition.
"""

from __future__ import annotations

import time

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

_SHM_NAME = "/cap_arena_b1"
_BOOTSTRAP_FENCE_FRAMES = 200
_BASE_PERIOD_S = 0.005
# Bounded window for GaP to attach, ResetEpisode, acquire the gripper lease, run
# open_gripper, and release -- spanning both the pre-reset and post-reset serve.
# Must comfortably exceed the connector's own reset/open exchange timeouts (~285 s
# each) so this producer never stops serving mid-exchange.
_SERVE_TIMEOUT_S = 600.0
# PHYSICS frames to confirm the gripper holds open after the transition is first seen.
_SERVE_SETTLE_FRAMES = 40
# One ResetEpisode (gen 1 -> 2) is expected before open_gripper; allow a small margin.
_MAX_GENERATIONS_FOLLOWED = 4
# 200 Hz base / 20 = ~10 Hz camera sampling, matching the standalone stream default.
_PERCEPTION_SAMPLE_EVERY_FRAMES = 20


class _PerceptionStreamSink:
    """Co-resident, nonblocking perception producer driven from the serve loop.

    The barrier is served by exactly ONE Kit process; when a perception endpoint is
    configured, this samples the exterior_cam right after a physics step (on the
    main/Kit thread, so the render is thread-safe) at a decimated rate and offers it
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

    def on_physics_frame(self, _frame: int) -> None:
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


def _run_serve(device: str, *, perception_stream: str | None = None) -> None:
    from isaaclab_arena.integrations.cap_barrier.franka_env import make_cap_franka_environment
    from isaaclab_arena.integrations.cap_barrier.lockstep_manager import ArenaLockstepManager
    from isaaclab_arena.integrations.cap_barrier.production_smoke import (
        CAP_PRODUCTION_STARTUP_RENDEZVOUS_TIMEOUT_S,
        open_production_startup_rendezvous,
    )
    from isaaclab_arena.integrations.cap_barrier.protocol import ControllerTimingSpec
    from isaaclab_arena.integrations.cap_barrier.serve import (
        ServeExit,
        serve_generation_watching_gripper,
    )
    from isaaclab_arena.integrations.cap_barrier.shared_memory import ArenaBarrierClient

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
    on_physics_frame = None
    if perception_stream is not None:
        perception = _PerceptionStreamSink(
            adapter, perception_stream, lambda marker: print(marker, flush=True)
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
            f"gripper_rad={adapter.gripper_position():.6f}",
            flush=True,
        )

        # GaP calls ResetEpisode (gen 1 -> 2) before open_gripper, so follow the
        # reset into the post-reset generation and serve again. The gripper resets
        # to its closed init pose on each reset, so the closed->open actuation lands
        # on the generation GaP opens on. Bound the number of resets we follow.
        deadline = time.monotonic() + _SERVE_TIMEOUT_S
        observation = None
        followed_any_reset = False
        for _ in range(_MAX_GENERATIONS_FOLLOWED):
            observation = serve_generation_watching_gripper(
                manager,
                adapter.gripper_position,
                deadline_monotonic_s=deadline,
                settle_frames=_SERVE_SETTLE_FRAMES,
                declare_open_success=followed_any_reset,
                marker_sink=lambda marker: print(marker, flush=True),
                on_physics_frame=on_physics_frame,
            )
            # A ResetEpisode surfaces to the producer as either a generation advance
            # or a BarrierInterrupted from the sidecar's reset-phase deactivation
            # (a race, same as the production smoke). Treat both as a transition to
            # follow; a genuine fault instead makes attach_next_generation fail and
            # we stop cleanly.
            if observation.exit_reason not in (
                ServeExit.GENERATION_ADVANCED,
                ServeExit.BARRIER_INTERRUPTED,
            ):
                break
            try:
                followed_generation = manager.attach_next_generation(timeout_s=30.0)
            except Exception as error:  # noqa: BLE001 - report and stop cleanly
                print(
                    f"CAP_SERVE_KIT_FOLLOW_RESET_FAILED detail={error!r} "
                    f"after={observation.exit_reason}",
                    flush=True,
                )
                break
            bootstrap = _paced_bootstrap_fences(manager, adapter, _BOOTSTRAP_FENCE_FRAMES)
            followed_any_reset = True
            print(
                f"CAP_SERVE_KIT_FOLLOWED_RESET generation={followed_generation} "
                f"fences={len(bootstrap)} gripper_rad={adapter.gripper_position():.6f}",
                flush=True,
            )
        print(
            "CAP_SERVE_KIT_TRACE "
            f"exit={observation.exit_reason} frames={observation.physics_frames} "
            f"init_cmd_rad={observation.initial_commanded_rad:.6f} "
            f"init_phys_rad={observation.initial_physical_rad:.6f} "
            f"final_cmd_rad={observation.final_commanded_rad:.6f} "
            f"final_phys_rad={observation.final_physical_rad:.6f} "
            f"min_phys_rad={observation.min_physical_rad:.6f} "
            f"max_phys_rad={observation.max_physical_rad:.6f} "
            f"observed_closed={int(observation.observed_closed)} "
            f"open_transition={int(observation.open_transition)}",
            flush=True,
        )
        if observation.exit_reason == ServeExit.OPEN_TRANSITION and observation.open_transition:
            print("CAP_SERVE_KIT_OPEN_GRIPPER_OK", flush=True)
        else:
            print(
                "CAP_SERVE_KIT_OPEN_GRIPPER_NOT_OBSERVED "
                f"reason={observation.exit_reason}",
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
    args_cli = parser.parse_args()
    if args_cli.perception_stream is not None:
        # The exterior_cam RTX spawn requires AppLauncher cameras; imply the flag
        # so --perception-stream can never be run without it.
        args_cli.enable_cameras = True
    with SimulationAppContext(args_cli):
        _run_serve(args_cli.device, perception_stream=args_cli.perception_stream)


if __name__ == "__main__":
    main()
