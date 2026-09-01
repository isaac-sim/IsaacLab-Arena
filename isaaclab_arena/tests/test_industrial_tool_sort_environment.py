# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the industrial tool-sort environment."""

import re
from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPO_ROOT / "isaaclab_arena" / "assets" / "industrial_tool_sort"
ENVIRONMENT_NAME = "vabar_tool_sort__sort_all_newton"
EXPECTED_ASSETS = {
    "industrial__fr3_workcell_table",
    "industrial__hdr_shadow_receiver",
    "industrial__tool_sort_bin",
    "vabar_tool_sort__hammer",
    "vabar_tool_sort__drill",
    "vabar_tool_sort__round_nut",
    "vabar_tool_sort__clamp",
    "industrial_fr3_robotiq_2f85",
    "industrial_fr3_robotiq_2f85_differential_ik",
}
ASSET_ENTRYPOINTS = [
    "industrial__fr3_workcell_table/industrial__fr3_workcell_table.usda",
    "industrial__hdr_shadow_receiver/industrial__hdr_shadow_receiver.usda",
    "industrial__tool_sort_bin/bin1.usda",
    "industrial__tool_sort_bin/bin2_default.usda",
    "vabar_tool_sort__hammer/vabar_tool_sort__hammer.usda",
    "vabar_tool_sort__drill/vabar_tool_sort__drill.usda",
    "vabar_tool_sort__round_nut/vabar_tool_sort__round_nut.usda",
    "vabar_tool_sort__clamp/vabar_tool_sort__clamp.usda",
    "industrial__fr3_robotiq_2f85/franka_fr3_robotiq_2f85.usda",
]


def test_vendored_usd_dependency_closure_is_complete():
    """Every retained file is reachable from a registered USD entry point."""
    reachable: set[Path] = set()
    pending = [ASSET_ROOT / relative_path for relative_path in ASSET_ENTRYPOINTS]
    while pending:
        usd_file = pending.pop().resolve()
        assert usd_file.is_relative_to(ASSET_ROOT.resolve())
        assert usd_file.exists(), f"Missing USD dependency: {usd_file.relative_to(ASSET_ROOT)}"
        if usd_file in reachable:
            continue
        reachable.add(usd_file)
        if usd_file.suffix not in {".usd", ".usda"}:
            continue
        try:
            contents = usd_file.read_text()
        except UnicodeDecodeError:
            continue
        for reference in re.findall(r"@([^@]+)@", contents):
            if "://" in reference:
                continue
            pending.append(usd_file.parent / reference)

    retained_files = {path.resolve() for path in ASSET_ROOT.rglob("*") if path.is_file()}
    assert retained_files == reachable
    assert {path.suffix for path in retained_files} <= {".usd", ".usda", ".usdc", ".png"}


def _test_tool_sort_registration_and_factory(_simulation_app) -> bool:
    import argparse

    from isaaclab_arena.assets.registries import AssetRegistry, EnvironmentRegistry, RetargeterRegistry, TaskRegistry
    from isaaclab_arena_environments.cli import add_environment_cli_args, build_environment_from_cli
    from isaaclab_arena_environments.industrial_tool_sort_environment import (
        IndustrialToolSortEnvironment,
        IndustrialToolSortEnvironmentCfg,
    )

    asset_registry = AssetRegistry()
    assert EXPECTED_ASSETS <= set(asset_registry.get_all_keys())
    assert TaskRegistry().is_registered("ObjectsInRegionsTask")
    assert EnvironmentRegistry().is_registered(ENVIRONMENT_NAME, ensure_loaded=False)

    parser = argparse.ArgumentParser(exit_on_error=False)
    add_environment_cli_args(parser, IndustrialToolSortEnvironment)
    args = parser.parse_args([
        "--embodiment",
        "industrial_fr3_robotiq_2f85_differential_ik",
        "--teleop_device",
        "keyboard",
    ])
    args.enable_cameras = False
    arena_env = build_environment_from_cli(IndustrialToolSortEnvironment, args)
    assert arena_env.name == ENVIRONMENT_NAME
    assert arena_env.embodiment.name == "industrial_fr3_robotiq_2f85_differential_ik"
    assert arena_env.teleop_device.name == "keyboard"
    assert arena_env.task.episode_length_s == 192.0
    assert [asset.name for asset in arena_env.task.objects] == ["hammer_0", "drill_0", "round_nut_0", "clamp_0"]
    assert RetargeterRegistry().is_registered(
        "keyboard__industrial_fr3_robotiq_2f85_differential_ik",
        ensure_loaded=False,
    )

    camera_env = IndustrialToolSortEnvironment().build(
        IndustrialToolSortEnvironmentCfg(
            enable_cameras=True,
            top_camera_position=[1.0, 2.0, 3.0],
            top_camera_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        )
    )
    top_camera = camera_env.embodiment.camera_config.top_camera
    assert top_camera.width == 1280
    assert top_camera.height == 720
    assert top_camera.offset.pos == (1.0, 2.0, 3.0)
    assert top_camera.offset.rot == (1.0, 0.0, 0.0, 0.0)
    return True


def test_tool_sort_registration_and_factory():
    assert run_function_with_persistent_simulation_app(_test_tool_sort_registration_and_factory)


def _test_shared_tool_sort_assets_spawn_with_physx(_simulation_app) -> bool:
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    background = registry.get_asset_by_name("industrial__fr3_workcell_table")()
    background.set_initial_pose(Pose(position_xyz=(-0.5, -0.1, 0.912)))
    assets = [background, registry.get_asset_by_name("industrial__hdr_shadow_receiver")()]
    for index, name in enumerate([
        "industrial__tool_sort_bin",
        "vabar_tool_sort__hammer",
        "vabar_tool_sort__drill",
        "vabar_tool_sort__round_nut",
        "vabar_tool_sort__clamp",
    ]):
        asset = registry.get_asset_by_name(name)(instance_name=f"industrial_asset_{index}")
        asset.set_initial_pose(Pose(position_xyz=(float(index), 0.0, 2.0)))
        assets.append(asset)

    arena_env = IsaacLabArenaEnvironment(
        name="industrial_tool_sort_assets_physx",
        embodiment=NoEmbodiment(),
        scene=Scene(assets=assets),
        task=NoTask(),
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    try:
        env.reset()
        assert {asset.name for asset in assets} <= set(env.unwrapped.scene.keys())
    finally:
        env.close()
    return True


def test_shared_tool_sort_assets_spawn_with_physx():
    assert run_function_with_persistent_simulation_app(_test_shared_tool_sort_assets_spawn_with_physx)


def _test_tool_sort_newton_smoke(_simulation_app) -> bool:
    import torch

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    registry = EnvironmentRegistry()
    factory_type = registry.get_component_by_name(ENVIRONMENT_NAME)
    cfg_type = registry.get_environment_cfg_type(factory_type)
    arena_env = factory_type().build(cfg_type())
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    try:
        assert env.unwrapped.cfg.sim.dt == 1.0 / 50.0
        assert env.unwrapped.cfg.sim.physics.num_substeps == 10
        observation, _ = env.reset()
        assert all(torch.isfinite(value).all() for value in observation["policy"].values())
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        observation, _, _, _, _ = env.step(action)
        assert all(torch.isfinite(value).all() for value in observation["policy"].values())
    finally:
        env.close()
    return True


def test_tool_sort_newton_smoke():
    assert run_function_with_persistent_simulation_app(_test_tool_sort_newton_smoke)
