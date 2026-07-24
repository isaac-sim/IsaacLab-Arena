# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Integration test: build an env from a spec YAML and check the solved layout honors its relations."""

import traceback
from pathlib import Path

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

_GRAPH = Path(__file__).parent / "test_data" / "pick_and_place_maple_table_env_graph.yaml"

# The spec pins the two On-table objects inside this box and the mug to a fixed point.
_POSITION_LIMITS_X = (0.55, 0.7)
_POSITION_LIMITS_Y = (-0.4, -0.1)
_MUG_TARGET_XYZ = (0.65, 0.25, 0.85)
_TOLERANCE_M = 0.05


def _local_position(env, scene_name: str) -> list[float]:
    """Return an object's (x, y, z) position in the env-local frame."""
    import warp as wp

    asset = env.unwrapped.scene[scene_name]
    pos_world = wp.to_torch(asset.data.root_pos_w)[0]
    env_origin = env.unwrapped.scene.env_origins[0].to(pos_world.device)
    return (pos_world - env_origin).tolist()


def _test_yaml_spec_placement_satisfies_relations(simulation_app) -> bool:
    """Solved placement must satisfy the spec's position_limits and at_position relations."""
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    env = None
    try:
        spec = ArenaEnvGraphSpec.from_yaml(_GRAPH)
        arena_env = spec.to_arena_env()
        args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
        env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
        env.reset()

        # Both On-table objects share one position_limits box; solving must land them inside it.
        for name in ("rubiks_cube_hot3d_robolab", "bowl_ycb_robolab"):
            x, y, _ = _local_position(env, name)
            assert (
                _POSITION_LIMITS_X[0] - _TOLERANCE_M <= x <= _POSITION_LIMITS_X[1] + _TOLERANCE_M
            ), f"{name} x={x:.3f} escaped position_limits {_POSITION_LIMITS_X}"
            assert (
                _POSITION_LIMITS_Y[0] - _TOLERANCE_M <= y <= _POSITION_LIMITS_Y[1] + _TOLERANCE_M
            ), f"{name} y={y:.3f} escaped position_limits {_POSITION_LIMITS_Y}"

        # The mug is pinned by at_position, so it lands on the requested point.
        mug = _local_position(env, "mug_ycb_robolab")
        for got, want, axis in zip(mug, _MUG_TARGET_XYZ, "xyz"):
            assert abs(got - want) < _TOLERANCE_M, f"mug {axis}={got:.3f} missed at_position target {want}"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        if env is not None:
            env.close()

    return True


def test_yaml_spec_placement_satisfies_relations():
    """Pytest entry point: YAML spec -> built env -> solved placement honors relations."""
    result = run_simulation_app_function(_test_yaml_spec_placement_satisfies_relations, headless=True)
    assert result, f"Test {test_yaml_spec_placement_satisfies_relations.__name__} failed"


if __name__ == "__main__":
    test_yaml_spec_placement_satisfies_relations()
