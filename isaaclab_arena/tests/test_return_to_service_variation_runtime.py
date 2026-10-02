# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify replayed faults affect physical state and preserve unselected episodes."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from isaaclab_arena.tests.utils.return_to_service import _require_assets


def _test_replayed_scenarios_preserve_unselected_workcells(_simulation_app):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tests.test_return_to_service_runtime import _step_physics
    from isaaclab_arena.utils.physics_backend import PhysicsBackend
    from isaaclab_arena.variations.sampling_context import VariationReplay
    from isaaclab_arena_environments.return_to_service.scenarios import SCENARIOS
    from isaaclab_arena_environments.return_to_service_environment import (
        ReturnToServiceEnvironment,
        ReturnToServiceEnvironmentCfg,
    )

    conditions = {(0, 0): "combined", (1, 0): "healthy", (1, 1): "obstruction", (0, 1): "battery_filter"}
    with TemporaryDirectory() as directory:
        replay_path = Path(directory) / "faults.jsonl"
        rows = []
        for (env_id, episode), condition in conditions.items():
            rows.append({
                "env_id": env_id,
                "episode_in_env": episode,
                "variations": {"body.scenario": condition},
            })
        replay_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        specification = ReturnToServiceEnvironment().build(
            ReturnToServiceEnvironmentCfg(scenarios=["healthy"], episode_length_s=60.0)
        )
        env = ArenaEnvBuilder(
            specification,
            ArenaEnvBuilderCfg(num_envs=2, presets=PhysicsBackend.PHYSX, variation_replay_path=str(replay_path)),
            hydra_overrides=["body.scenario.enabled=true"],
        ).make_registered()
        try:
            env.reset()
            _step_physics(env, 15)
            base = env.unwrapped
            runtime = base.task_runtime
            assert runtime.scenario_names == ["combined", "healthy"]
            assert [model.scenario for model in runtime.models] == [SCENARIOS["combined"], SCENARIOS["healthy"]]
            assert runtime.snapshots[0].inlet_obstructed
            assert not runtime.snapshots[1].inlet_obstructed

            for selected, expected, obstructed in ((1, "obstruction", True), (0, "battery_filter", False)):
                untouched = 1 - selected
                other_model = runtime.models[untouched]
                other_snapshot = runtime.snapshots[untouched]
                other_status = runtime.statuses[untouched]
                other_poses = {
                    name: base.arena_world.get_pose_w(name)[untouched].clone()
                    for name in ("body", "battery_original", "obstruction")
                }
                base._reset_idx(torch.tensor([selected], device=base.device))
                runtime.update()
                assert runtime.scenario_names[selected] == expected
                assert runtime.models[selected].scenario == SCENARIOS[expected]
                assert runtime.snapshots[selected].inlet_obstructed is obstructed
                assert runtime.models[untouched] is other_model
                assert runtime.snapshots[untouched] is other_snapshot
                assert runtime.statuses[untouched] is other_status
                for name, pose in other_poses.items():
                    torch.testing.assert_close(base.arena_world.get_pose_w(name)[untouched], pose, atol=1e-6, rtol=0)
                assert base.variation_recorder["body.scenario"].sample_for_episode(selected, 1) == expected
                _step_physics(env, 2)
                assert runtime.snapshots[selected].inlet_obstructed is obstructed

            trace_path = Path(directory) / "variation_samples.jsonl"
            base.variation_recorder.write_samples_jsonl(trace_path)
            recorded = VariationReplay.from_jsonl(trace_path)
            for key, condition in conditions.items():
                assert recorded.value("body.scenario", key) == condition
        finally:
            env.close()
    return True


def test_replayed_scenarios_preserve_unselected_workcells():
    _require_assets()
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_replayed_scenarios_preserve_unselected_workcells)
