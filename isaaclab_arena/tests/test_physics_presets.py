# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ArenaPhysicsCfg preset system and ArenaEnvBuilder integration."""

import pytest
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_arena.environments.isaaclab_arena_manager_based_env import ArenaPhysicsCfg
from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


# ---------------------------------------------------------------------------
# ArenaPhysicsCfg preset lookup (no simulation required)
# ---------------------------------------------------------------------------


class TestArenaPhysicsCfgPresets:
    """Verify that ArenaPhysicsCfg resolves the correct config for each preset name."""

    def test_default_is_physx(self):
        assert isinstance(ArenaPhysicsCfg().default, PhysxCfg)

    def test_physx_preset(self):
        assert isinstance(ArenaPhysicsCfg().physx, PhysxCfg)

    def test_newton_preset(self):
        assert isinstance(ArenaPhysicsCfg().newton, NewtonCfg)

    def test_physx_and_default_are_equal(self):
        cfg = ArenaPhysicsCfg()
        assert cfg.physx == cfg.default

    def test_getattr_physx(self):
        assert isinstance(getattr(ArenaPhysicsCfg(), "physx"), PhysxCfg)

    def test_getattr_newton(self):
        assert isinstance(getattr(ArenaPhysicsCfg(), "newton"), NewtonCfg)

    def test_getattr_unknown_raises(self):
        with pytest.raises(AttributeError):
            getattr(ArenaPhysicsCfg(), "unknown_backend")


# ---------------------------------------------------------------------------
# Newton solver parameter smoke checks (no simulation required)
# ---------------------------------------------------------------------------


class TestNewtonPresetParameters:
    """Verify the Newton preset has the expected solver tuning."""

    def test_solver_type(self):
        assert ArenaPhysicsCfg().newton.solver_cfg.solver == "newton"


# ---------------------------------------------------------------------------
# ArenaEnvBuilder end-to-end preset tests (requires SimulationApp)
# ---------------------------------------------------------------------------


def _build_env_cfg(presets: str | None):
    """Build a real env cfg through ArenaEnvBuilder.compose_manager_cfg with the given preset."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
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

    builder = ArenaEnvBuilder(arena_env, args_cli)
    return builder.compose_manager_cfg()


def _test_builder_no_presets_defaults_to_physx(simulation_app) -> bool:
    env_cfg = _build_env_cfg(presets=None)
    assert env_cfg.sim.physics is None, f"Expected None (PhysX default), got {type(env_cfg.sim.physics)}"
    assert env_cfg.scene.replicate_physics is False
    return True


def _test_builder_physx_preset(simulation_app) -> bool:
    env_cfg = _build_env_cfg(presets="physx")
    assert isinstance(env_cfg.sim.physics, PhysxCfg), f"Expected PhysxCfg, got {type(env_cfg.sim.physics)}"
    assert env_cfg.scene.replicate_physics is False
    return True


def _test_builder_newton_preset(simulation_app) -> bool:
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
