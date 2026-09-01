# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Production contract tests for the YAM cable-routing environment."""

from __future__ import annotations

import os
import subprocess

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants


def _make_route_geometry(torch):
    """Create correctly directed peg loops in the requested cable order."""
    peg_0 = torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32)
    peg_1 = torch.tensor((0.30, 0.0, 0.0), dtype=torch.float32)
    angles = torch.linspace(0.0, 2.0 * torch.pi, 17)
    peg_0_loop = torch.stack(
        (0.025 * torch.cos(angles), 0.025 * torch.sin(angles), torch.zeros_like(angles)),
        dim=-1,
    )
    peg_1_loop = torch.stack(
        (
            peg_1[0] + 0.025 * torch.cos(-angles),
            peg_1[1] + 0.025 * torch.sin(-angles),
            torch.zeros_like(angles),
        ),
        dim=-1,
    )
    bridge = torch.tensor(((0.10, 0.10, 0.0), (0.20, 0.10, 0.0)), dtype=torch.float32)
    cable = torch.cat((peg_0_loop, bridge, peg_1_loop), dim=0)
    pegs = torch.stack((peg_0, peg_1), dim=0)
    return cable, pegs, peg_0_loop, bridge, peg_1_loop


def _test_yam_cable_routing_contracts(simulation_app) -> bool:
    import torch

    from isaaclab_contrib.coupling import CouplerProxyCfg

    from isaaclab_arena.assets.registries import AssetRegistry, EnvironmentRegistry, TaskRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tasks.cable_routing_task import cable_route_success_from_geometry
    from isaaclab_arena_environments.cli import ensure_environments_registered
    from isaaclab_arena_environments.yam_cable_routing_environment import (
        YamCableRoutingEnvironment,
        YamCableRoutingEnvironmentCfg,
    )

    ensure_environments_registered()
    environment_registry = EnvironmentRegistry()
    assert environment_registry.get_component_by_name("yam_cable_routing") is YamCableRoutingEnvironment
    assert "cable_routing_yam_newton" not in environment_registry.get_all_keys()
    assert AssetRegistry().get_asset_by_name("yam_bimanual").__name__ == "BimanualYamEmbodiment"
    assert TaskRegistry().get_component_by_name("CableRoutingTask").__name__ == "CableRoutingTask"

    cable, pegs, peg_0_loop, bridge, peg_1_loop = _make_route_geometry(torch)
    route_directions = (-1.0, 1.0)
    success = cable_route_success_from_geometry(
        cable[None, ...],
        pegs[None, ...],
        route_directions=route_directions,
    )
    assert success.tolist() == [True]

    wrong_order = torch.cat((peg_1_loop, bridge.flip(0), peg_0_loop), dim=0)
    assert not cable_route_success_from_geometry(
        wrong_order[None, ...],
        pegs[None, ...],
        route_directions=route_directions,
    ).item()
    nonfinite = cable.clone()
    nonfinite[4, 0] = torch.nan
    assert not cable_route_success_from_geometry(
        nonfinite[None, ...],
        pegs[None, ...],
        route_directions=route_directions,
    ).item()

    factory = YamCableRoutingEnvironment()
    arena_env = factory.build(YamCableRoutingEnvironmentCfg())
    assert set(arena_env.scene.assets) == {"table", "board", "peg_0", "peg_1", "cable", "ground", "sky_light"}
    assert arena_env.embodiment.name == "yam_bimanual"
    assert arena_env.task.__class__.__name__ == "CableRoutingTask"
    assert arena_env.supported_physics_presets == ()

    builder_cfg = ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device="cuda:0")
    builder = ArenaEnvBuilder(arena_env, builder_cfg)
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    assert isinstance(env_cfg.sim.physics.solver_cfg, CouplerProxyCfg)
    assert env_cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert env_cfg.decimation == 4
    assert env_cfg.scene.replicate_physics is True
    assert list(vars(env_cfg.actions)) == [
        "left_arm_action",
        "left_gripper_action",
        "right_arm_action",
        "right_gripper_action",
    ]

    preset_builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device="cuda:0", presets="newton"),
    )
    with pytest.raises(AssertionError, match="required custom physics configuration"):
        preset_builder.compose_manager_cfg()

    camera_arena_env = factory.build(YamCableRoutingEnvironmentCfg(enable_cameras=True))
    camera_rig = camera_arena_env.embodiment.camera_config
    assert camera_rig.camera_names() == ["left_wrist_camera", "right_wrist_camera", "top_camera"]
    assert camera_rig.use_tiled_camera is False

    env = builder.make_registered(env_cfg, env_kwargs)
    try:
        observations, _ = env.reset()
        assert env.action_space.shape[-1] == 14
        assert observations["policy"].shape[-1] == 46
        assert {"left_robot", "right_robot", "board", "peg_0", "peg_1", "cable"}.issubset(env.unwrapped.scene.keys())
        for _ in range(2):
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            observations, _, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(observations["policy"]).all()
            assert not terminated.any()
            assert not truncated.any()
        cable_positions = env.unwrapped.scene["cable"].data.segment_pose_w.torch[..., :3]
        assert cable_positions.shape[-2] == 100
        assert torch.isfinite(cable_positions).all()
    finally:
        env.close()

    return True


@pytest.mark.with_newton
def test_yam_cable_routing_contracts() -> None:
    """Validate registration, geometry, configuration, cameras, and runtime stepping."""
    result = subprocess.run(
        [TestConstants.python_path, __file__],
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("ISAACLAB_ARENA_SUBPROCESS_TIMEOUT", "900")),
        start_new_session=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    result = run_function_with_persistent_simulation_app(_test_yam_cable_routing_contracts, headless=True)
    raise SystemExit(0 if result else 1)
