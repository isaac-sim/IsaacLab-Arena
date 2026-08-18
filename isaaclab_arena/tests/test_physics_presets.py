# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ArenaPhysicsCfg preset system and ArenaEnvBuilder integration."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _test_arena_physics_cfg_presets(simulation_app) -> bool:
    from isaaclab_contrib.deformable.newton_manager_cfg import CoupledMJWarpVBDSolverCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg
    from isaaclab_arena.environments.physics_presets import ARENA_PHYSICS_PRESETS, SimulationBackend

    assert set(ARENA_PHYSICS_PRESETS) == {"default", "physx", "newton", "newton_mjwarp_vbd"}
    assert ARENA_PHYSICS_PRESETS["default"].backend is SimulationBackend.PHYSX
    assert ARENA_PHYSICS_PRESETS["physx"].supported_deformable_kinds == frozenset()
    assert ARENA_PHYSICS_PRESETS["newton"].supported_deformable_kinds == frozenset()
    assert ARENA_PHYSICS_PRESETS["newton_mjwarp_vbd"].supported_deformable_kinds == frozenset({"volume"})
    assert ARENA_PHYSICS_PRESETS["newton_mjwarp_vbd"].replicate_physics is True

    # ArenaPhysicsCfg remains a compatible name-keyed configuration API.
    cfg = ArenaPhysicsCfg()
    assert isinstance(cfg.default, PhysxCfg)
    assert isinstance(cfg.physx, PhysxCfg)
    assert isinstance(cfg.newton, NewtonCfg)
    assert isinstance(cfg.newton_mjwarp_vbd, NewtonCfg)
    assert cfg.physx == cfg.default
    assert isinstance(getattr(cfg, "physx"), PhysxCfg)
    assert isinstance(getattr(cfg, "newton"), NewtonCfg)
    with pytest.raises(AttributeError):
        getattr(cfg, "unknown_backend")
    assert cfg.newton.solver_cfg.solver == "newton"
    assert isinstance(cfg.newton_mjwarp_vbd.solver_cfg, CoupledMJWarpVBDSolverCfg)
    assert cfg.newton_mjwarp_vbd.solver_cfg.soft_solver_cfg.iterations == 10
    assert cfg.newton_mjwarp_vbd.num_substeps == 10
    assert cfg.newton_mjwarp_vbd.model_cfg.shape_material_kd == 100.0
    return True


def _build_env_cfg(
    presets: str | None,
    default_physics_preset: str | None = None,
    env_cfg_callback=None,
):
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
        env_cfg_callback=env_cfg_callback,
        default_physics_preset=default_physics_preset,
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


def _test_builder_vbd_preset(simulation_app) -> bool:
    from isaaclab_contrib.deformable.newton_manager_cfg import CoupledMJWarpVBDSolverCfg

    env_cfg = _build_env_cfg(presets="newton_mjwarp_vbd")
    assert isinstance(env_cfg.sim.physics.solver_cfg, CoupledMJWarpVBDSolverCfg)
    assert env_cfg.sim.physics.solver_cfg.soft_solver_cfg.iterations == 10
    assert env_cfg.scene.replicate_physics is True
    return True


def _test_builder_preset_precedence_and_final_authority(simulation_app) -> bool:
    from isaaclab_physx.physics import PhysxCfg

    def callback(env_cfg):
        env_cfg.sim.physics = None
        env_cfg.scene.replicate_physics = True
        return env_cfg

    # CLI PhysX wins over the environment's Newton default and both selected-preset values win
    # over conflicting callback values.
    env_cfg = _build_env_cfg(
        presets="physx",
        default_physics_preset="newton_mjwarp_vbd",
        env_cfg_callback=callback,
    )
    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert env_cfg.scene.replicate_physics is False

    # The environment default is used when the CLI does not select a preset.
    env_cfg = _build_env_cfg(presets=None, default_physics_preset="newton")
    assert env_cfg.sim.physics.solver_cfg.solver == "newton"
    assert env_cfg.scene.replicate_physics is True
    return True


def _test_builder_unknown_preset_raises(simulation_app) -> bool:
    with pytest.raises(ValueError, match="Unknown physics preset 'unknown_backend'"):
        _build_env_cfg(presets="unknown_backend")
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


def test_builder_vbd_preset():
    assert run_function_with_persistent_simulation_app(_test_builder_vbd_preset, headless=HEADLESS)


def test_builder_preset_precedence_and_final_authority():
    assert run_function_with_persistent_simulation_app(
        _test_builder_preset_precedence_and_final_authority, headless=HEADLESS
    )


def test_builder_unknown_preset_raises():
    assert run_function_with_persistent_simulation_app(_test_builder_unknown_preset_raises, headless=HEADLESS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
