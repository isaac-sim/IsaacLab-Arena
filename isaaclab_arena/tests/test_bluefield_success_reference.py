# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bluefield lift history survives placement and clears independently on reset."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_bluefield_success_reference(_simulation_app) -> bool:
    import math
    import torch
    from dataclasses import replace
    from types import SimpleNamespace

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena_environments.bluefield_g1_fixed_base_pick_and_place_environment import (
        BluefieldG1FixedBasePickAndPlaceEnvironment,
        BluefieldG1FixedBasePickAndPlaceEnvironmentCfg,
    )

    task = (
        BluefieldG1FixedBasePickAndPlaceEnvironment()
        .build(BluefieldG1FixedBasePickAndPlaceEnvironmentCfg(embodiment="g1_fixed_base_upper_body_joint"))
        .task
    )
    termination = task.get_termination_cfg()
    criteria = termination.success[0]
    assert len(next(iter(criteria.canonical_predicate_sequences.values()))) == 2
    lifted, placement = criteria.predicate_sequence
    assert placement.required_steps == 1
    assert placement.predicate.keywords["force_threshold"] == 0.5
    assert placement.predicate.keywords["velocity_threshold"] == 0.1
    assert placement.predicate.keywords["support_cone_half_angle_rad"] == math.pi / 4
    assert len(PickAndPlaceTask.get_termination_cfg(task).success[0].predicate_sequence) == 3
    assert termination.timeout_s == 6.0 and "object_dropped" in termination.failures

    reference = lifted.keywords["reference_height"]
    origins = torch.tensor([[30.0, -30.0, 2.0], [-30.0, 30.0, -2.0]], dtype=torch.float64)
    positions = origins.clone()
    positions[:, 2] += reference
    env = SimpleNamespace(
        scene=SimpleNamespace(env_origins=origins),
        arena_world=SimpleNamespace(get_position_w=lambda _name: positions),
        placed=torch.ones(2, dtype=torch.bool),
    )
    # Isolate lift ordering from the separately tested upstream placement predicate.
    tracker = ProgressTracker(
        [replace(criteria, predicate_sequence=[lifted, lambda state: state.placed])],
        num_envs=2,
        device="cpu",
    )
    tracker.step(env)
    assert tracker.is_complete().tolist() == [False, False]  # placed without lifting
    positions[0, 2] += 0.011
    tracker.step(env)
    assert tracker.is_complete().tolist() == [False, False]  # lift is the first milestone
    positions[0, 2] = origins[0, 2] + reference
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True, False]  # retain earlier lift on placement
    positions[1, 2] += 0.011
    tracker.step(env)
    positions[1, 2] = origins[1, 2] + reference
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True, True]
    tracker.reset([0])
    tracker.step(env)
    assert tracker.is_complete().tolist() == [False, True]  # no lift inherited across reset
    return True


def test_bluefield_success_reference() -> None:
    assert run_function_with_persistent_simulation_app(_test_bluefield_success_reference)
