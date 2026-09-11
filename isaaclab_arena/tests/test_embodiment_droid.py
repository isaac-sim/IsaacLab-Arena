# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simulation tests for the Newton DROID differential IK embodiment."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

if TYPE_CHECKING:
    import torch

SETTLE_STEPS = 10
HOLD_STEPS = 30
# Equivalent to 20 control steps at the default 30 Hz; the actual step count is derived at runtime.
LIFT_COMMAND_DURATION_S = 20.0 / 30.0
HOLD_TOLERANCE_M = 0.005
MIN_LIFT_M = 0.025
# This test checks directional response, allowing transient Newton IK overshoot rather than precise pose tracking.
MAX_LIFT_M = 0.105


@contextmanager
def _newton_droid_env(env_name: str):
    """Yield a minimal Newton scene with keyboard-teleoperable DROID differential IK."""
    import gymnasium as gym

    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry
    from isaaclab_arena.embodiments.droid.droid import DroidDifferentialIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.physics_backend import PhysicsBackend
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()

    background = asset_registry.get_asset_by_name("packing_table")()
    embodiment = DroidDifferentialIKEmbodiment()
    embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 1.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    teleop_device = device_registry.get_device_by_name("keyboard")()
    arena_env = IsaacLabArenaEnvironment(
        name=env_name,
        embodiment=embodiment,
        scene=Scene(assets=[background]),
        teleop_device=teleop_device,
    )

    if env_name in gym.registry:
        del gym.registry[env_name]

    builder_cfg = ArenaEnvBuilderCfg(num_envs=1, presets=PhysicsBackend.NEWTON)
    env = ArenaEnvBuilder(arena_env, builder_cfg).make_registered()
    try:
        env.reset()
        yield env, teleop_device.pos_sensitivity
    finally:
        env.close()
        if env_name in gym.registry:
            del gym.registry[env_name]


def _get_ee_pos_w(env) -> torch.Tensor:
    """Return the Robotiq base link position in the env-local world frame."""
    import warp as wp

    robot = env.unwrapped.scene["robot"]
    body_ids, _ = robot.find_bodies("base_link")
    return wp.to_torch(robot.data.body_pos_w)[0, body_ids[0], :] - env.unwrapped.scene.env_origins[0]


def _idle_teleop_action(env) -> torch.Tensor:
    """Return the action produced by a keyboard with no motion keys pressed."""
    import torch

    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    action[:, 6] = 1.0
    return action


def _test_newton_droid_ik_holds_without_teleop_command(simulation_app) -> bool:
    """The arm should stay put when the keyboard emits no motion command."""
    import torch

    with _newton_droid_env("newton_droid_ik_hold_test") as (env, _):
        with torch.inference_mode():
            idle_action = _idle_teleop_action(env)
            for _ in range(SETTLE_STEPS):
                env.step(idle_action)

            initial_ee_pos = _get_ee_pos_w(env)

            for _ in range(HOLD_STEPS):
                env.step(idle_action)

            final_ee_pos = _get_ee_pos_w(env)
            displacement = torch.norm(final_ee_pos - initial_ee_pos).item()
            assert displacement < HOLD_TOLERANCE_M, (
                f"End effector moved {displacement:.4f} m without a teleop command; "
                f"tolerance is {HOLD_TOLERANCE_M:.4f} m."
            )

    return True


def _test_newton_droid_ik_lifts_on_teleop_command(simulation_app) -> bool:
    """A sustained keyboard lift command should produce a bounded upward response."""
    import torch

    with _newton_droid_env("newton_droid_ik_lift_test") as (env, pos_sensitivity):
        with torch.inference_mode():
            idle_action = _idle_teleop_action(env)
            for _ in range(SETTLE_STEPS):
                env.step(idle_action)

            initial_ee_pos = _get_ee_pos_w(env)
            lift_action = idle_action.clone()
            lift_action[:, 2] = pos_sensitivity
            lift_steps = max(1, int(round(LIFT_COMMAND_DURATION_S / env.unwrapped.step_dt)))
            for _ in range(lift_steps):
                env.step(lift_action)

            final_ee_pos = _get_ee_pos_w(env)
            displacement = final_ee_pos - initial_ee_pos
            lift_z = displacement[2].item()
            horizontal = torch.norm(displacement[:2]).item()

            assert lift_z > MIN_LIFT_M, f"Expected at least {MIN_LIFT_M:.3f} m upward motion, got {lift_z:.4f} m."
            assert lift_z < MAX_LIFT_M, f"Expected less than {MAX_LIFT_M:.3f} m upward motion, got {lift_z:.4f} m."
            assert (
                lift_z > horizontal
            ), f"Lift should be primarily vertical; dz={lift_z:.4f} m, horizontal={horizontal:.4f} m."

    return True


@pytest.mark.with_newton
def test_newton_droid_ik_holds_without_teleop_command():
    assert run_function_with_persistent_simulation_app(_test_newton_droid_ik_holds_without_teleop_command)


@pytest.mark.with_newton
def test_newton_droid_ik_lifts_on_teleop_command():
    assert run_function_with_persistent_simulation_app(_test_newton_droid_ik_lifts_on_teleop_command)


def _test_newton_droid_embodiment_config_contract(simulation_app) -> bool:
    """Pin Newton DROID gripper action, close target, and IK body."""
    from isaaclab_arena.embodiments.droid.droid import (
        BinaryJointPositionZeroToOneActionCfg,
        DroidDifferentialIKEmbodiment,
    )
    from isaaclab_arena.embodiments.droid.observations import _DROID_NEWTON_GRIPPER_CLOSE_RAD
    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    embodiment = DroidDifferentialIKEmbodiment()
    assert embodiment.action_config.arm_action.body_name == "base_link"
    assert embodiment.action_config.arm_action.controller.ik_method == "dls"

    embodiment.configure_physics_backend(PhysicsBackend.NEWTON)
    gripper_action = embodiment.action_config.gripper_action
    assert isinstance(gripper_action, BinaryJointPositionZeroToOneActionCfg)
    assert embodiment.action_config.arm_action.body_name == "base_link"
    assert embodiment.action_config.arm_action.controller.ik_method == "adaptive_dls"
    assert gripper_action.close_command_expr["finger_joint"] == _DROID_NEWTON_GRIPPER_CLOSE_RAD
    assert embodiment.observation_config.policy.gripper_pos.func.__name__ == "newton_gripper_pos"

    embodiment.configure_physics_backend(PhysicsBackend.NEWTON)
    with pytest.raises(AssertionError, match="already configured for physics backend"):
        embodiment.configure_physics_backend(PhysicsBackend.PHYSX)
    with pytest.raises(AssertionError, match="already configured for physics backend"):
        embodiment.configure_physics_backend(None)

    physx_embodiment = DroidDifferentialIKEmbodiment()
    physx_embodiment.configure_physics_backend(PhysicsBackend.PHYSX)
    physx_embodiment.configure_physics_backend(PhysicsBackend.PHYSX)
    with pytest.raises(AssertionError, match="already configured for physics backend"):
        physx_embodiment.configure_physics_backend(PhysicsBackend.NEWTON)
    return True


@pytest.mark.with_newton
def test_newton_droid_embodiment_config_contract():
    assert run_function_with_persistent_simulation_app(_test_newton_droid_embodiment_config_contract)
