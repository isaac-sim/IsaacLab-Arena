# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the fixed B=1 Franka half of the CAP ROS barrier smoke.

Start the test-only ROS ``cap_smoke_orchestrator`` first and wait for its
``CAP_SMOKE_READY_FOR_KIT`` marker. This runner intentionally mirrors that
orchestrator's one-shot frame counts; it is not a production episode API.
"""

from __future__ import annotations

import time

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

_SHM_NAME = "/cap_arena_b1"
_FENCE_FRAMES = 200
_GENERATION_1_PHYSICS = 128
_GENERATION_2_PHYSICS = 32
_BASE_PERIOD_S = 0.005


def _max_delta(lhs, rhs) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(lhs, rhs, strict=True))


def _paced_fences(manager, adapter, count: int):
    frozen_state = adapter.arm_positions()
    results = []
    deadline = time.monotonic()
    for _ in range(count):
        results.append(manager.fence())
        deadline += _BASE_PERIOD_S
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    frozen_delta = _max_delta(frozen_state, adapter.arm_positions())
    if frozen_delta > 1e-12:
        raise RuntimeError(f"FENCE advanced Franka state: max_delta={frozen_delta}")
    return results


def _run_smoke(device: str) -> None:
    from isaaclab_arena.integrations.cap_barrier.franka_env import make_cap_franka_environment
    from isaaclab_arena.integrations.cap_barrier.lockstep_manager import ArenaLockstepManager
    from isaaclab_arena.integrations.cap_barrier.protocol import ControllerTimingSpec
    from isaaclab_arena.integrations.cap_barrier.shared_memory import ArenaBarrierClient

    # TODO(CR-21): this fixed demo roster is not a production roster/discovery decision.
    timing = (
        ControllerTimingSpec("hold_controller", 1),
        ControllerTimingSpec("joint_streaming_controller", 1),
    )
    adapter = make_cap_franka_environment(device=device)
    client = ArenaBarrierClient(_SHM_NAME, open_timeout_s=30.0)
    try:
        manager = ArenaLockstepManager(
            client=client,
            simulation=adapter,
            joint_mapping=adapter.joint_mapping,
            controller_specs=timing,
            command_timeout_s=30.0,
        )
        generation_1 = manager.attach_initial_generation(timeout_s=30.0)
        if generation_1 != 1:
            raise RuntimeError(f"smoke must bootstrap at generation 1, got {generation_1}")
        print("CAP_SMOKE_KIT_GENERATION_1_ATTACHED", flush=True)

        generation_1_fences = _paced_fences(manager, adapter, _FENCE_FRAMES)
        if generation_1_fences[0].sequence != 0 or generation_1_fences[0].physics_tick != 0:
            raise RuntimeError("generation 1 did not begin at sequence/tick zero")
        print(
            f"CAP_SMOKE_KIT_BOOTSTRAP_FENCE_OK frames={len(generation_1_fences)} "
            f"physics_steps={adapter.physics_step_count}",
            flush=True,
        )

        bootstrap_pose = adapter.arm_positions()
        generation_1_commands = [manager.physics_step() for _ in range(_GENERATION_1_PHYSICS)]
        changed_target = max(_max_delta(result.commanded_positions, bootstrap_pose) for result in generation_1_commands)
        moved = _max_delta(adapter.arm_positions(), bootstrap_pose)
        if changed_target <= 1e-4:
            raise RuntimeError("joint_streaming never produced a target distinct from the bootstrap pose")
        if moved <= 1e-5:
            raise RuntimeError("Franka did not respond to the streamed trajectory")
        if generation_1_commands[0].physics_tick != 0:
            raise RuntimeError("first generation 1 PHYSICS frame was not phase zero")
        print(
            f"CAP_SMOKE_KIT_STREAMING_OK physics={len(generation_1_commands)} "
            f"max_target_delta={changed_target:.6f} max_motion={moved:.6f}",
            flush=True,
        )

        generation_2 = manager.attach_next_generation(timeout_s=30.0)
        if generation_2 != 2 or adapter.reset_count != 2:
            raise RuntimeError(f"reset fence mismatch: generation={generation_2} reset_count={adapter.reset_count}")
        print("CAP_SMOKE_KIT_GENERATION_2_RESET_STAGED", flush=True)

        generation_2_fences = _paced_fences(manager, adapter, _FENCE_FRAMES)
        if generation_2_fences[0].sequence != 0 or generation_2_fences[0].physics_tick != 0:
            raise RuntimeError("generation 2 did not restart sequence/tick at zero")
        generation_2_commands = [manager.physics_step() for _ in range(_GENERATION_2_PHYSICS)]
        if generation_2_commands[0].physics_tick != 0:
            raise RuntimeError("first generation 2 PHYSICS frame was not phase zero")
        expected_steps = _GENERATION_1_PHYSICS + _GENERATION_2_PHYSICS
        if adapter.physics_step_count != expected_steps:
            raise RuntimeError(
                f"physics step count mismatch: expected {expected_steps}, got {adapter.physics_step_count}"
            )
        print(
            f"CAP_SMOKE_KIT_HOLD_RESUME_OK physics={len(generation_2_commands)} "
            f"total_physics_steps={adapter.physics_step_count}",
            flush=True,
        )
        print("CAP_SMOKE_KIT_DONE", flush=True)
    finally:
        client.close()
        adapter.close()


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    args_cli = parser.parse_args()
    with SimulationAppContext(args_cli):
        _run_smoke(args_cli.device)


if __name__ == "__main__":
    main()
