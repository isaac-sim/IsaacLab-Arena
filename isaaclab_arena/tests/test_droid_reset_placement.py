# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify Droid robot and stand stay aligned after relation placement reset."""

from __future__ import annotations

import traceback
from pathlib import Path

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

_KITCHEN_YAML = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments" / "kitchen_task.yaml"
_STAND_HEIGHT_M = 0.8
_Z_MATCH_EPS = 1e-3


def _test_droid_reset_placement(simulation_app) -> bool:
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    try:
        import warp as wp

        spec = ArenaEnvGraphSpec.from_yaml(_KITCHEN_YAML)
        params = dict(spec.embodiment.params)
        params["stand_height_m"] = _STAND_HEIGHT_M
        spec.embodiment.params = params

        builder = ArenaEnvBuilder(spec.to_arena_env(), ArenaEnvBuilderCfg(num_envs=1))
        env = builder.make_registered()
        env.reset()

        robot_z = wp.to_torch(env.unwrapped.scene["robot"].data.root_link_pose_w)[0, 2].item()
        stand_pos = env.unwrapped.scene["stand"].get_world_poses()[0].torch
        stand_z = stand_pos[0, 2].item()
        assert abs(robot_z - stand_z) < _Z_MATCH_EPS, f"robot z {robot_z} != stand z {stand_z} after reset"

        env.close()
    except Exception as exc:
        print(f"Error: {exc}")
        traceback.print_exc()
        return False

    return True


def test_droid_reset_placement():
    result = run_simulation_app_function(_test_droid_reset_placement, headless=True)
    assert result, f"Test {test_droid_reset_placement.__name__} failed"


if __name__ == "__main__":
    test_droid_reset_placement()
