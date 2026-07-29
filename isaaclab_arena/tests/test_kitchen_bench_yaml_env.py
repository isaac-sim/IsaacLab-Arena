# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: kitchen_bench graph-spec YAMLs build and reset in sim."""

from __future__ import annotations

import traceback
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

_KITCHEN_BENCH_DIR = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments" / "kitchen_bench"
_KITCHEN_BENCH_YAMLS = sorted(_KITCHEN_BENCH_DIR.glob("*.yaml"))


def _test_kitchen_bench_yaml_env_bringup(simulation_app, *, yaml_path: Path) -> bool:
    """Build an env from a kitchen_bench YAML and reset once."""
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    env = None
    try:
        spec = ArenaEnvGraphSpec.from_yaml(yaml_path)
        arena_env = spec.to_arena_env()
        args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
        env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
        env.reset()

        assert env.unwrapped.cfg is not None
        assert arena_env.name == spec.env_name
        assert "robot" in env.unwrapped.scene.keys(), f"robot missing from scene for {yaml_path.name}"
    except Exception as exc:
        print(f"Error bringing up {yaml_path}: {exc}")
        traceback.print_exc()
        return False
    finally:
        if env is not None:
            env.close()

    return True


def _test_droid_stand_survives_pool_refill(simulation_app, *, yaml_path: Path) -> bool:
    """Keep the instanceable Droid stand loaded across placement-pool refill."""
    from pxr import Usd, UsdGeom

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.relations.placement_events import get_placement_pool

    spec = ArenaEnvGraphSpec.from_yaml(yaml_path)
    arena_env = spec.to_arena_env()
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()

    def visible_stand_meshes() -> tuple[str, ...]:
        stand = env.unwrapped.scene.stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link0/stand_instanceable")
        assert stand.IsValid() and stand.IsActive() and stand.IsLoaded()
        meshes = tuple(prim for prim in Usd.PrimRange(stand, Usd.TraverseInstanceProxies()) if prim.IsA(UsdGeom.Mesh))
        assert meshes
        assert all(
            mesh.IsLoaded() and UsdGeom.Imageable(mesh).ComputeVisibility() != UsdGeom.Tokens.invisible
            for mesh in meshes
        )
        return tuple(str(mesh.GetPath()) for mesh in meshes)

    try:
        env.reset()
        mesh_paths = visible_stand_meshes()
        placement_pool = get_placement_pool(env)
        assert placement_pool is not None
        placement_pool.sample_without_replacement(placement_pool.total_remaining)
        env.reset()
        assert visible_stand_meshes() == mesh_paths
    finally:
        env.close()

    return True


@pytest.mark.parametrize(
    "yaml_path",
    _KITCHEN_BENCH_YAMLS,
    ids=[path.stem for path in _KITCHEN_BENCH_YAMLS],
)
def test_kitchen_bench_yaml_env_bringup(yaml_path: Path):
    """Each kitchen_bench YAML must parse and produce a resettable Arena env."""
    assert yaml_path.is_file(), f"missing kitchen_bench YAML: {yaml_path}"
    result = run_simulation_app_function(
        _test_kitchen_bench_yaml_env_bringup,
        headless=True,
        yaml_path=yaml_path,
    )
    assert result, f"kitchen_bench bring-up failed for {yaml_path.name}"


def test_droid_stand_survives_pool_refill():
    """The Lightwheel kitchen keeps the instanceable Droid stand visible."""
    yaml_path = _KITCHEN_BENCH_DIR / "droid_pick_and_place_lightwheel_kitchen.yaml"
    assert run_simulation_app_function(
        _test_droid_stand_survives_pool_refill,
        headless=True,
        yaml_path=yaml_path,
    )


if __name__ == "__main__":
    for path in _KITCHEN_BENCH_YAMLS:
        test_kitchen_bench_yaml_env_bringup(path)
