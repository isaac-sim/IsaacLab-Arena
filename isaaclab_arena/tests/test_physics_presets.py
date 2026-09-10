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
    assert isinstance(cfg.default, PhysxCfg)
    assert isinstance(cfg.physx, PhysxCfg)
    assert isinstance(cfg.newton, NewtonCfg)
    assert cfg.physx == cfg.default
    with pytest.raises(AttributeError):
        getattr(cfg, "unknown_backend")
    assert cfg.newton.solver_cfg.solver == "newton"
    return True


def _build_env_cfg(presets: str | None, embodiment=None, env_cfg_callback=None):
    """Build a real env cfg through ArenaEnvBuilder.compose_manager_cfg with the given preset."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()
    table = asset_registry.get_asset_by_name("packing_table")()
    scene = Scene(assets=[table])

    if embodiment is None:
        embodiment = FrankaIKEmbodiment()

    arena_env = IsaacLabArenaEnvironment(
        name="test_physics_preset",
        embodiment=embodiment,
        scene=scene,
        env_cfg_callback=env_cfg_callback,
    )

    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, presets=presets))
    env_cfg, _ = builder.compose_manager_cfg()
    return env_cfg


def _test_builder_preset(simulation_app, presets: str | None, expected_backend: str | None, replicate_physics: bool):
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
    from isaaclab_physx.physics import PhysxCfg

    env_cfg = _build_env_cfg(presets=presets)
    expected_type = {"physx": PhysxCfg, "newton": NewtonCfg}.get(expected_backend)
    if expected_type is None:
        assert env_cfg.sim.physics is None
    else:
        assert isinstance(env_cfg.sim.physics, expected_type)
    assert env_cfg.scene.replicate_physics is replicate_physics
    return True


def _test_assembly_callback_rejects_newton_preset(simulation_app) -> bool:
    from isaaclab_arena_environments.mdp.env_callbacks import assembly_env_cfg_callback

    with pytest.raises(AssertionError, match="Assembly environments require PhysX"):
        _build_env_cfg(presets="newton", env_cfg_callback=assembly_env_cfg_callback)
    return True


def _test_droid_diff_ik_physx_preset_keeps_default_spawn(simulation_app) -> bool:
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab_arena.embodiments.droid.droid import DroidDifferentialIKEmbodiment, spawn_newton_droid

    env_cfg = _build_env_cfg(presets="physx", embodiment=DroidDifferentialIKEmbodiment())
    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert env_cfg.scene.robot.spawn.func is not spawn_newton_droid
    return True


def _test_droid_diff_ik_newton_preset_applies_newton_spawn(simulation_app) -> bool:
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    from isaaclab_arena.embodiments.droid.droid import DroidDifferentialIKEmbodiment, spawn_newton_droid

    env_cfg = _build_env_cfg(presets="newton", embodiment=DroidDifferentialIKEmbodiment())
    assert isinstance(env_cfg.sim.physics, NewtonCfg)
    assert env_cfg.scene.robot.spawn.func is spawn_newton_droid
    return True


def _test_droid_abs_joint_pos_newton_preset_applies_newton_spawn(simulation_app) -> bool:
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    from isaaclab_arena.embodiments.droid.droid import (
        _DROID_NEWTON_GRIPPER_MIMIC_SIGNS,
        DroidAbsoluteJointPositionEmbodiment,
        spawn_newton_droid,
    )
    from isaaclab_arena.embodiments.droid.observations import newton_gripper_pos

    env_cfg = _build_env_cfg(presets="newton", embodiment=DroidAbsoluteJointPositionEmbodiment())
    assert isinstance(env_cfg.sim.physics, NewtonCfg)
    assert env_cfg.scene.robot.spawn.func is spawn_newton_droid
    gripper_joint_names = list(_DROID_NEWTON_GRIPPER_MIMIC_SIGNS)
    assert env_cfg.scene.robot.actuators["gripper"].joint_names_expr == gripper_joint_names
    assert env_cfg.actions.gripper_action.joint_names == gripper_joint_names
    assert env_cfg.observations.policy.gripper_pos.func is newton_gripper_pos
    return True


def _test_droid_rel_joint_pos_newton_preset_applies_newton_gripper(simulation_app) -> bool:
    from isaaclab_arena.embodiments.droid.droid import (
        _DROID_NEWTON_GRIPPER_MIMIC_SIGNS,
        DroidRelativeJointPositionEmbodiment,
    )

    env_cfg = _build_env_cfg(presets="newton", embodiment=DroidRelativeJointPositionEmbodiment())
    gripper_joint_names = list(_DROID_NEWTON_GRIPPER_MIMIC_SIGNS)
    assert env_cfg.scene.robot.actuators["gripper"].joint_names_expr == gripper_joint_names
    assert env_cfg.actions.gripper_action.joint_names == gripper_joint_names
    return True


def test_arena_physics_cfg_presets():
    assert run_function_with_persistent_simulation_app(_test_arena_physics_cfg_presets, headless=HEADLESS)


@pytest.mark.parametrize(
    ("presets", "expected_backend", "replicate_physics"),
    [
        (None, None, False),
        ("physx", "physx", False),
        ("newton", "newton", True),
    ],
)
def test_builder_preset(presets, expected_backend, replicate_physics):
    assert run_function_with_persistent_simulation_app(
        _test_builder_preset,
        headless=HEADLESS,
        presets=presets,
        expected_backend=expected_backend,
        replicate_physics=replicate_physics,
    )


def test_assembly_callback_rejects_newton_preset():
    assert run_function_with_persistent_simulation_app(_test_assembly_callback_rejects_newton_preset, headless=HEADLESS)


def test_droid_diff_ik_physx_preset_keeps_default_spawn():
    assert run_function_with_persistent_simulation_app(
        _test_droid_diff_ik_physx_preset_keeps_default_spawn, headless=HEADLESS
    )


def test_droid_diff_ik_newton_preset_applies_newton_spawn():
    assert run_function_with_persistent_simulation_app(
        _test_droid_diff_ik_newton_preset_applies_newton_spawn, headless=HEADLESS
    )


def test_droid_abs_joint_pos_newton_preset_applies_newton_spawn():
    assert run_function_with_persistent_simulation_app(
        _test_droid_abs_joint_pos_newton_preset_applies_newton_spawn, headless=HEADLESS
    )


def test_droid_rel_joint_pos_newton_preset_applies_newton_gripper():
    assert run_function_with_persistent_simulation_app(
        _test_droid_rel_joint_pos_newton_preset_applies_newton_gripper, headless=HEADLESS
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
