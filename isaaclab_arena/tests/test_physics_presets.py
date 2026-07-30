# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ArenaPhysicsCfg preset system and ArenaEnvBuilder integration."""

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_arena_physics_cfg_presets(simulation_app) -> bool:
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg

    cfg = ArenaPhysicsCfg()
    # default resolves to PhysX
    assert isinstance(cfg.default, PhysxCfg)
    assert isinstance(cfg.physx, PhysxCfg)
    assert isinstance(cfg.newton, NewtonCfg)
    assert cfg.physx == cfg.default
    # getattr access
    assert isinstance(getattr(cfg, "physx"), PhysxCfg)
    assert isinstance(getattr(cfg, "newton"), NewtonCfg)
    with pytest.raises(AttributeError):
        getattr(cfg, "unknown_backend")
    # Newton solver tuning
    assert cfg.newton.solver_cfg.solver == "newton"
    return True


def _build_env_cfg(presets: str | None):
    """Build a real env cfg through ArenaEnvBuilder.compose_manager_cfg with the given preset."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    cli_args = ["--num_envs", "1"]
    if presets is not None:
        cli_args += ["--presets", presets]

    args_cli = get_isaaclab_arena_cli_parser().parse_args(cli_args)

    asset_registry = AssetRegistry()
    table = asset_registry.get_asset_by_name("packing_table")()
    scene = Scene(assets=[table])

    arena_env = IsaacLabArenaEnvironment(
        name="test_physics_preset",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
    )

    builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli))
    env_cfg, _ = builder.compose_manager_cfg()
    return env_cfg


def _test_builder_no_presets_defaults_to_physx(simulation_app) -> bool:
    env_cfg = _build_env_cfg(presets=None)
    assert env_cfg.sim.physics is None, f"Expected None (PhysX default), got {type(env_cfg.sim.physics)}"
    assert env_cfg.scene.replicate_physics is False
    return True


def _test_builder_physx_preset(simulation_app) -> bool:
    from isaaclab_physx.physics import PhysxCfg

    env_cfg = _build_env_cfg(presets="physx")
    assert isinstance(env_cfg.sim.physics, PhysxCfg), f"Expected PhysxCfg, got {type(env_cfg.sim.physics)}"
    assert env_cfg.scene.replicate_physics is False
    return True


def _test_builder_newton_preset(simulation_app) -> bool:
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    env_cfg = _build_env_cfg(presets="newton")
    assert isinstance(env_cfg.sim.physics, NewtonCfg), f"Expected NewtonCfg, got {type(env_cfg.sim.physics)}"
    assert env_cfg.scene.replicate_physics is True
    return True


def _test_builder_unknown_preset_raises(simulation_app) -> bool:
    try:
        _build_env_cfg(presets="unknown_backend")
    except (AttributeError, SystemExit):
        return True
    raise AssertionError("Expected AttributeError or SystemExit for unknown preset")


# --- pytest-visible outer functions ---


def test_arena_physics_cfg_presets():
    assert run_simulation_app_function(_test_arena_physics_cfg_presets, headless=HEADLESS)


def test_builder_no_presets_defaults_to_physx():
    assert run_simulation_app_function(_test_builder_no_presets_defaults_to_physx, headless=HEADLESS)


def test_builder_physx_preset():
    assert run_simulation_app_function(_test_builder_physx_preset, headless=HEADLESS)


def test_builder_newton_preset():
    assert run_simulation_app_function(_test_builder_newton_preset, headless=HEADLESS)


def test_builder_unknown_preset_raises():
    assert run_simulation_app_function(_test_builder_unknown_preset_raises, headless=HEADLESS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
