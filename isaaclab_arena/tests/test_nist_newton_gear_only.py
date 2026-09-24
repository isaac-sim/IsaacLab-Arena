# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton stability tests for the NIST gear insertion scene objects.

The DROID robot embodiment is excluded because its Robotiq 2F-85 passive
linkages need a separate Newton mass-floor workaround. These tests focus on
the NIST gear, gear base, and Newton-compatible landing slab.
"""

from __future__ import annotations

import numpy as np

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

NUM_STEPS = 300
SETTLE_LIN_THRESH = 0.01
FINAL_LIN_THRESH = 0.05
BASE_LIN_THRESH = 0.001
MAX_GEAR_LIN_SPEED = 10.0

GEAR_DROP_Z_NEAR = 0.00
GEAR_DROP_Z_HIGH = 0.10


def _run_newton_scene_test(simulation_app, gear_initial_z: float) -> bool:
    """Build a no-robot NIST Newton scene and verify gear stability."""
    import torch

    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.register import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_environments.cli import ensure_environments_registered
    from isaaclab_arena_environments.mdp import nist_gear_insertion_env_cfg_callback

    ensure_environments_registered()

    asset_registry = AssetRegistry()
    gears_and_base = asset_registry.get_asset_by_name("gears_and_base")()
    medium_gear = asset_registry.get_asset_by_name("medium_nist_gear")()
    light = asset_registry.get_asset_by_name("light")(
        spawner_cfg=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
    )
    table = asset_registry.get_asset_by_name("table")()

    table.set_initial_pose(Pose(position_xyz=(0.55, 0.0, -0.009), rotation_xyzw=(0.0, 0.0, 0.707, 0.707)))
    gears_and_base.set_initial_pose(Pose(position_xyz=(0.585, -0.074, 0.0), rotation_xyzw=(0.0, 0.0, 0.9239, 0.3827)))
    medium_gear.set_initial_pose(Pose(position_xyz=(0.50, -0.24, gear_initial_z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    scene = Scene(assets=[table, medium_gear, gears_and_base, light])
    arena_env = IsaacLabArenaEnvironment(
        name=f"nist_newton_scene_test_{gear_initial_z:.2f}",
        scene=scene,
        env_cfg_callback=nist_gear_insertion_env_cfg_callback,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = 1
    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    builder_cfg.presets = "newton"

    builder = ArenaEnvBuilder(arena_env, builder_cfg)
    env = builder.make_registered()

    try:
        env.reset()
        gear_obj = env.unwrapped.scene["medium_nist_gear"]
        base_obj = env.unwrapped.scene["gears_and_base"]
        zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

        gear_settled = False
        max_gear_speed = 0.0

        for _ in range(NUM_STEPS):
            env.step(zero_actions)

            gear_pos = gear_obj.data.root_pos_w.torch[0].cpu().numpy()
            gear_vel = gear_obj.data.root_lin_vel_w.torch[0].cpu().numpy()
            base_pos = base_obj.data.root_pos_w.torch[0].cpu().numpy()
            base_vel = base_obj.data.root_lin_vel_w.torch[0].cpu().numpy()

            if not (
                np.all(np.isfinite(gear_pos))
                and np.all(np.isfinite(gear_vel))
                and np.all(np.isfinite(base_pos))
                and np.all(np.isfinite(base_vel))
            ):
                return False

            gear_speed = float(np.linalg.norm(gear_vel))
            max_gear_speed = max(max_gear_speed, gear_speed)
            gear_settled = gear_settled or gear_speed < SETTLE_LIN_THRESH

        final_gear_speed = float(np.linalg.norm(gear_vel))
        final_base_speed = float(np.linalg.norm(base_vel))
        return (
            gear_settled
            and max_gear_speed < MAX_GEAR_LIN_SPEED
            and final_gear_speed < FINAL_LIN_THRESH
            and final_base_speed < BASE_LIN_THRESH
        )

    finally:
        env.close()


def _test_near_drop(simulation_app) -> bool:
    return _run_newton_scene_test(simulation_app, GEAR_DROP_Z_NEAR)


def _test_high_drop(simulation_app) -> bool:
    return _run_newton_scene_test(simulation_app, GEAR_DROP_Z_HIGH)


def test_nist_scene_objects_stable_newton():
    """Gear drops 17 mm onto the landing slab."""
    result = run_function_with_persistent_simulation_app(_test_near_drop, headless=True)
    assert result, "Newton near-drop stability test failed"


def test_nist_scene_objects_stable_newton_high_drop():
    """Gear drops 117 mm onto the landing slab."""
    result = run_function_with_persistent_simulation_app(_test_high_drop, headless=True)
    assert result, "Newton high-drop stability test failed"
