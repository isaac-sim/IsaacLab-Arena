# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ArenaPhysicsCfg preset system and ArenaEnvBuilder integration."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

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


def _build_env_cfg(presets: str | None, env_cfg_override=None):
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
    ground = asset_registry.get_asset_by_name("ground_plane")()
    scene = Scene(assets=[ground])

    arena_env = IsaacLabArenaEnvironment(
        name="test_physics_preset",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
        env_cfg_override=env_cfg_override,
    )

    builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli))
    env_cfg, _ = builder.compose_manager_cfg()
    return env_cfg


def _test_builder_no_presets_defaults_to_physx(simulation_app) -> bool:
    from isaaclab_physx.physics import PhysxCfg

    env_cfg = _build_env_cfg(presets=None)
    assert isinstance(env_cfg.sim.physics, PhysxCfg)
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


def _test_builder_applies_nested_env_cfg_override(simulation_app) -> bool:
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    env_cfg = _build_env_cfg(
        presets=None,
        env_cfg_override={
            "sim": {
                "dt": 0.02,
                "physics": {
                    "_target_": "isaaclab_newton.physics.NewtonCfg",
                    "num_substeps": 7,
                    "solver_cfg": {
                        "_target_": "isaaclab_newton.physics.MJWarpSolverCfg",
                        "iterations": 23,
                        "enable_multiccd": True,
                    },
                },
            },
            "decimation": 3,
        },
    )

    assert env_cfg.sim.dt == 0.02
    assert env_cfg.decimation == 3
    assert isinstance(env_cfg.sim.physics, NewtonCfg)
    assert env_cfg.sim.physics.num_substeps == 7
    assert isinstance(env_cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
    assert env_cfg.sim.physics.solver_cfg.iterations == 23
    assert env_cfg.sim.physics.solver_cfg.enable_multiccd
    return True


def _test_builder_rejects_unsafe_or_incompatible_targets(simulation_app) -> bool:
    unsafe = {"sim": {"physics": {"_target_": "builtins.dict"}}}
    with pytest.raises(AssertionError, match="outside the approved"):
        _build_env_cfg(presets=None, env_cfg_override=unsafe)

    incompatible = {"sim": {"physics": {"_target_": "isaaclab_newton.physics.MJWarpSolverCfg"}}}
    with pytest.raises(AssertionError, match="incompatible"):
        _build_env_cfg(presets=None, env_cfg_override=incompatible)

    with pytest.raises(AssertionError, match="cannot be overridden"):
        _build_env_cfg(presets=None, env_cfg_override={"sim": {"physics": {"class_type": "malicious"}}})

    with pytest.raises(AssertionError, match="interpolation is not allowed"):
        _build_env_cfg(presets=None, env_cfg_override={"sim": {"dt": "${oc.env:SIM_DT}"}})

    with pytest.raises(ValueError, match="Invalid env_cfg_override"):
        _build_env_cfg(presets=None, env_cfg_override={"sim": {"unknown_field": 1}})
    return True


def _test_cli_preset_rejects_conflicting_yaml_backend(simulation_app) -> bool:
    override = {"sim": {"physics": {"_target_": "isaaclab_newton.physics.NewtonCfg"}}}
    with pytest.raises(AssertionError, match="conflicts with the explicit CLI preset"):
        _build_env_cfg(presets="physx", env_cfg_override=override)
    return True


# --- pytest-visible outer functions ---


def test_arena_physics_cfg_presets():
    assert run_function_with_persistent_simulation_app(_test_arena_physics_cfg_presets, headless=HEADLESS)


def test_builder_no_presets_defaults_to_physx():
    assert run_function_with_persistent_simulation_app(_test_builder_no_presets_defaults_to_physx, headless=HEADLESS)


def test_builder_physx_preset():
    assert run_function_with_persistent_simulation_app(_test_builder_physx_preset, headless=HEADLESS)


def test_builder_newton_preset():
    assert run_function_with_persistent_simulation_app(_test_builder_newton_preset, headless=HEADLESS)


def test_builder_unknown_preset_raises():
    assert run_function_with_persistent_simulation_app(_test_builder_unknown_preset_raises, headless=HEADLESS)


def test_builder_applies_nested_env_cfg_override():
    assert run_function_with_persistent_simulation_app(_test_builder_applies_nested_env_cfg_override, headless=HEADLESS)


def test_builder_rejects_unsafe_or_incompatible_targets():
    assert run_function_with_persistent_simulation_app(
        _test_builder_rejects_unsafe_or_incompatible_targets, headless=HEADLESS
    )


def test_cli_preset_rejects_conflicting_yaml_backend():
    assert run_function_with_persistent_simulation_app(
        _test_cli_preset_rejects_conflicting_yaml_backend, headless=HEADLESS
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
