# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


def test_h2_debug_registered():
    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    assert registry.is_registered("h2_debug")


def test_h2_debug_robot_menagerie_root_env_override(monkeypatch, tmp_path):
    from isaaclab_arena.embodiments.h2.h2 import ROBOT_MENAGERIE_ROOT_ENV_VAR, resolve_h2_usd_path

    monkeypatch.setenv(ROBOT_MENAGERIE_ROOT_ENV_VAR, str(tmp_path))

    assert resolve_h2_usd_path() == str(tmp_path / "unitree/h2/usd/H2_simple_colliders.usd")


def _test_h2_debug_spawn(simulation_app) -> bool:
    import torch

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()
    ground_plane = asset_registry.get_asset_by_name("ground_plane")()
    light = asset_registry.get_asset_by_name("light")()
    embodiment = asset_registry.get_asset_by_name("h2_debug")()
    scene = Scene(assets=[ground_plane, light])

    arena_env = IsaacLabArenaEnvironment(
        name="h2_debug_spawn_test",
        embodiment=embodiment,
        scene=scene,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(arena_env, args_cli)
    env = env_builder.make_registered()
    try:
        env.reset()
        assert "robot" in env.unwrapped.scene.keys()
        robot = env.unwrapped.scene["robot"]
        assert len(robot.data.joint_names) > 0
        assert len(robot.data.body_names) > 0
        with torch.inference_mode():
            env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
    finally:
        env.close()
    return True


def test_h2_debug_spawn_when_asset_available():
    from isaaclab_arena.embodiments.h2.h2 import h2_debug_usd_exists, resolve_h2_usd_path

    if not h2_debug_usd_exists():
        pytest.skip(
            "H2 USD is not available in this container. Mount robot_menagerie/unitree/h2. "
            f"Current resolved path: {resolve_h2_usd_path()}"
        )

    result = run_simulation_app_function(_test_h2_debug_spawn, headless=True, enable_cameras=False)
    assert result
