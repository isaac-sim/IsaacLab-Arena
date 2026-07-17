# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the fixed B=1 DROID half of the CAP ROS barrier smoke.

Start the test-only ROS ``cap_smoke_orchestrator`` first and wait for its
``CAP_SMOKE_READY_FOR_KIT`` marker. This runner follows the orchestrator-owned
reset transition; it is not a production episode API.
"""

from __future__ import annotations

import time

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

_SHM_NAME = "/cap_arena_b1"
_FENCE_FRAMES = 200
_MINIMUM_GENERATION_1_PHYSICS = 128
_GENERATION_2_PHYSICS = 32
_BASE_PERIOD_S = 0.005
_RESET_TIMEOUT_S = 300.0


def _max_delta(lhs, rhs) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(lhs, rhs, strict=True))


def _paced_fences(manager, adapter, count: int):
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


def _run_smoke(device: str) -> None:
    from isaaclab_arena.integrations.cap_barrier.franka_env import make_cap_franka_environment
    from isaaclab_arena.integrations.cap_barrier.lockstep_manager import ArenaLockstepManager
    from isaaclab_arena.integrations.cap_barrier.production_smoke import (
        DroidGripperTransitionProof,
        run_physics_until_generation_transition,
    )
    from isaaclab_arena.integrations.cap_barrier.protocol import ControllerTimingSpec
    from isaaclab_arena.integrations.cap_barrier.shared_memory import ArenaBarrierClient

    # TODO(CR-21): this fixed demo roster is not a production roster/discovery decision.
    timing = (
        ControllerTimingSpec("hold_controller", 1),
        ControllerTimingSpec("joint_streaming_controller", 1),
        ControllerTimingSpec("robotiq_gripper_controller", 1),
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

        initial_positions = adapter.abi_positions()
        bootstrap_pose = initial_positions[:7]
        gripper_proof = DroidGripperTransitionProof(initial_positions)
        maximum_target_delta = 0.0

        def observe_generation_1(result) -> None:
            nonlocal maximum_target_delta
            physical_positions = adapter.abi_positions()
            observed_at_s = time.monotonic()
            step_started_at_s = adapter.last_physics_step_started_at_s
            if step_started_at_s is None:
                raise RuntimeError("DROID adapter did not bracket the PHYSICS step")
            gripper_proof.observe(
                result,
                physical_positions,
                step_started_at_s=step_started_at_s,
                observed_at_s=observed_at_s,
            )
            maximum_target_delta = max(
                maximum_target_delta,
                _max_delta(result.commanded_positions[:7], bootstrap_pose),
            )

        transition = run_physics_until_generation_transition(
            manager,
            timeout_s=_RESET_TIMEOUT_S,
            on_frame=observe_generation_1,
        )
        gripper = gripper_proof.observation()
        moved = _max_delta(adapter.arm_positions(), bootstrap_pose)
        if transition.physics_frames < _MINIMUM_GENERATION_1_PHYSICS:
            raise RuntimeError(
                "reset arrived before the smoke proofs completed: "
                f"expected at least {_MINIMUM_GENERATION_1_PHYSICS}, got {transition.physics_frames}"
            )
        if maximum_target_delta <= 1e-4:
            raise RuntimeError("joint_streaming never produced a target distinct from the bootstrap pose")
        if moved <= 1e-5:
            raise RuntimeError("DROID arm did not respond to the streamed trajectory")
        print(
            "CAP_SMOKE_KIT_GRIPPER_TRANSITION_OK direction=open-close-open "
            f"closed_rad={gripper.closed_position_rad:.6f} final_rad={gripper.final_position_rad:.6f} "
            f"close_s={gripper.close_elapsed_s:.6f} open_s={gripper.open_elapsed_s:.6f} "
            f"arm_command_delta={gripper.maximum_arm_command_delta_rad:.9f} "
            f"arm_physical_delta={gripper.maximum_arm_physical_delta_rad:.9f}",
            flush=True,
        )
        print(
            f"CAP_SMOKE_KIT_STREAMING_OK physics={transition.physics_frames} "
            f"max_target_delta={maximum_target_delta:.6f} max_motion={moved:.6f}",
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
        expected_steps = transition.physics_frames + _GENERATION_2_PHYSICS
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
