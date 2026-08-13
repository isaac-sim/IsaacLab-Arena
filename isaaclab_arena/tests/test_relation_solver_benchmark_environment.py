# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

_ROBO_LAB_SPEC = (
    Path(__file__).resolve().parents[2] / "isaaclab_arena_environments" / "robolab" / "tasks" / "banana_in_bowl.yaml"
)
_KITCHEN_SPEC = (
    Path(__file__).resolve().parents[2]
    / "isaaclab_arena_environments"
    / "kitchen_bench"
    / "replicator_kitchen_l_shape_banana_bowl.yaml"
)


def _test_environment_benchmark(simulation_app):
    import gymnasium as gym

    from isaaclab_arena.relations.benchmark import BenchmarkScenario, run_environment_benchmark

    registered_before = set(gym.registry)
    results = []
    for include_robot in (True, False):
        scenario = BenchmarkScenario(
            name=f"banana-in-bowl-{'robot' if include_robot else 'no-robot'}",
            num_objects=0,
            num_envs=2,
            max_iters=10,
            warmup_runs=0,
            timed_runs=1,
            final_loss_threshold=1e9,
            graph_spec_path=str(_ROBO_LAB_SPEC),
            include_robot=include_robot,
        )
        results.append(run_environment_benchmark(scenario))

    assert all(result.status == "ok" for result in results), [result.error for result in results]
    assert all(result.build_ms is not None and result.build_ms > 0.0 for result in results)
    assert all(result.reset_ms is not None and result.reset_ms > 0.0 for result in results)
    assert all(result.bring_up_ms is not None and result.bring_up_ms > 0.0 for result in results)
    assert all(result.valid_layout_rate is None for result in results)
    assert [result.include_robot for result in results] == [True, False]

    underconverged_mesh_result = run_environment_benchmark(
        BenchmarkScenario(
            name="kitchen-mesh-no-robot",
            num_objects=0,
            num_envs=1,
            max_iters=10,
            max_placement_attempts=1,
            warmup_runs=0,
            timed_runs=1,
            graph_spec_path=str(_KITCHEN_SPEC),
            include_robot=False,
            collision_mode="mesh",
        )
    )
    assert underconverged_mesh_result.status == "failed"
    assert underconverged_mesh_result.error is not None
    assert "Placement pool could not fill" in underconverged_mesh_result.error
    assert set(gym.registry) == registered_before
    return True


def test_environment_benchmark_builds_robot_variants_sequentially():
    assert run_function_with_persistent_simulation_app(_test_environment_benchmark, headless=True)
