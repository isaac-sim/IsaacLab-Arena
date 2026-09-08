# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simulation tests for the Newton DROID differential IK embodiment."""

from __future__ import annotations

import gymnasium as gym
import torch

import warp as wp

from isaaclab_arena.assets.device_library import KeyboardCfg
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

SETTLE_STEPS = 10
HOLD_STEPS = 30
# Sustained lift duration; step count is derived from env step_dt at runtime.
LIFT_COMMAND_DURATION_S = 20.0 * (8.0 / 240.0)
HOLD_TOLERANCE_M = 0.005
TARGET_LIFT_M = 0.05
LIFT_TOLERANCE_M = 0.025
KEYBOARD_POS_SENSITIVITY = KeyboardCfg().pos_sensitivity


def _build_newton_droid_env(env_name: str):
    """Build a minimal Newton scene with keyboard-teleoperable DROID differential IK."""
    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.droid.droid import DroidDifferentialIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--presets", "newton"])
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

    env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    env.reset()
    return env, arena_env.name


def _get_ee_pos_w(env) -> torch.Tensor:
    """Return the Robotiq base link position in the env-local world frame."""
    robot = env.unwrapped.scene["robot"]
    body_idx = robot.data.body_names.index("base_link")
    return wp.to_torch(robot.data.body_pos_w)[0, body_idx, :] - env.unwrapped.scene.env_origins[0]


def _idle_teleop_action(device: torch.device) -> torch.Tensor:
    """Return the action produced by a keyboard with no motion keys pressed."""
    return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device)


def _lift_teleop_action(device: torch.device, pos_sensitivity: float = KEYBOARD_POS_SENSITIVITY) -> torch.Tensor:
    """Return the action produced by holding the keyboard lift key (Q)."""
    return torch.tensor([[0.0, 0.0, pos_sensitivity, 0.0, 0.0, 0.0, 1.0]], device=device)


def _test_newton_droid_ik_holds_without_teleop_command(simulation_app) -> bool:
    """The arm should stay put when the keyboard emits no motion command."""
    env, env_name = _build_newton_droid_env("newton_droid_ik_hold_test")

    try:
        with torch.inference_mode():
            device = env.unwrapped.device
            for _ in range(SETTLE_STEPS):
                env.step(_idle_teleop_action(device))

            initial_ee_pos = _get_ee_pos_w(env)

            for _ in range(HOLD_STEPS):
                env.step(_idle_teleop_action(device))

            final_ee_pos = _get_ee_pos_w(env)
            displacement = torch.norm(final_ee_pos - initial_ee_pos).item()
            assert displacement < HOLD_TOLERANCE_M, (
                f"End effector moved {displacement:.4f} m without a teleop command; "
                f"tolerance is {HOLD_TOLERANCE_M:.4f} m."
            )
    finally:
        env.close()
        if env_name in gym.registry:
            del gym.registry[env_name]

    return True


def _test_newton_droid_ik_lifts_on_teleop_command(simulation_app) -> bool:
    """A sustained keyboard lift command should raise the end effector by roughly 5 cm."""
    env, env_name = _build_newton_droid_env("newton_droid_ik_lift_test")

    try:
        with torch.inference_mode():
            device = env.unwrapped.device
            for _ in range(SETTLE_STEPS):
                env.step(_idle_teleop_action(device))

            initial_ee_pos = _get_ee_pos_w(env)
            lift_action = _lift_teleop_action(device)
            lift_steps = max(1, int(round(LIFT_COMMAND_DURATION_S / env.unwrapped.step_dt)))
            for _ in range(lift_steps):
                env.step(lift_action)

            final_ee_pos = _get_ee_pos_w(env)
            displacement = final_ee_pos - initial_ee_pos
            lift_z = displacement[2].item()
            horizontal = torch.norm(displacement[:2]).item()

            assert (
                lift_z > TARGET_LIFT_M - LIFT_TOLERANCE_M
            ), f"Expected at least {TARGET_LIFT_M - LIFT_TOLERANCE_M:.3f} m upward motion, got {lift_z:.4f} m."
            assert (
                lift_z < TARGET_LIFT_M + LIFT_TOLERANCE_M + 0.03
            ), f"Expected roughly {TARGET_LIFT_M:.2f} m upward motion, got {lift_z:.4f} m."
            assert (
                lift_z > horizontal
            ), f"Lift should be primarily vertical; dz={lift_z:.4f} m, horizontal={horizontal:.4f} m."
    finally:
        env.close()
        if env_name in gym.registry:
            del gym.registry[env_name]

    return True


def test_newton_droid_ik_holds_without_teleop_command():
    """Pytest entry point for the no-command hold check."""
    assert run_function_with_persistent_simulation_app(_test_newton_droid_ik_holds_without_teleop_command)


def test_newton_droid_ik_lifts_on_teleop_command():
    """Pytest entry point for the keyboard lift check."""
    assert run_function_with_persistent_simulation_app(_test_newton_droid_ik_lifts_on_teleop_command)


def test_z_newton_droid_embodiment_config_contract():
    """Pin Newton DROID gripper action, close target, and IK body after sim tests."""
    from isaaclab_arena.embodiments.droid.droid import (
        BinaryJointPositionZeroToOneActionCfg,
        DroidDifferentialIKEmbodiment,
    )
    from isaaclab_arena.embodiments.droid.observations import _DROID_NEWTON_GRIPPER_CLOSE_RAD

    embodiment = DroidDifferentialIKEmbodiment()
    assert embodiment.action_config.arm_action.body_name == "base_link"
    assert embodiment.action_config.arm_action.controller.ik_method == "dls"

    embodiment.configure_for_physics("newton")
    gripper_action = embodiment.action_config.gripper_action
    assert isinstance(gripper_action, BinaryJointPositionZeroToOneActionCfg)
    assert embodiment.action_config.arm_action.body_name == "base_link"
    assert embodiment.action_config.arm_action.controller.ik_method == "adaptive_dls"
    assert gripper_action.close_command_expr["finger_joint"] == _DROID_NEWTON_GRIPPER_CLOSE_RAD
    assert embodiment.observation_config.policy.gripper_pos.func.__name__ == "newton_gripper_pos"
