# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime smoke tests for the registered Franka soft-lift scene."""

from __future__ import annotations

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_franka_soft_lift_backends_step(simulation_app) -> bool:
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.franka_soft_lift_environment import (
        FrankaSoftLiftEnvironment,
        FrankaSoftLiftEnvironmentCfg,
    )

    backend_expectations = {
        "physx": False,
        "newton_mjwarp_vbd_proxy": True,
        "newton_mjwarp_vbd": True,
    }

    for preset, expected_replicate_physics in backend_expectations.items():
        arena_env = FrankaSoftLiftEnvironment().build(FrankaSoftLiftEnvironmentCfg(enable_cameras=False))
        builder = ArenaEnvBuilder(
            arena_env,
            ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, presets=preset),
        )
        env = builder.make_registered()
        base_env = env.unwrapped
        try:
            obs, _ = env.reset()
            assert base_env.cfg.scene.replicate_physics is expected_replicate_physics
            assert base_env.cfg.sim.gravity == (0.0, 0.0, -9.81)
            assert env.action_space.shape[-1] == 8
            assert obs["policy"].shape == (1, 93)
            assert torch.isfinite(obs["policy"]).all()

            deformable = base_env.scene["deformable"]
            nodal_before = deformable.data.nodal_pos_w.torch.clone()
            assert nodal_before.shape[1] > 0, f"{preset}: deformable has no simulation nodes"
            assert torch.isfinite(nodal_before).all(), f"{preset}: nodal positions are not finite after reset"

            actions = torch.zeros((base_env.num_envs, env.action_space.shape[-1]), device=base_env.device)
            finite_rewards = True
            finite_observations = True
            for _ in range(15):
                obs, rewards, _terminated, _truncated, _info = env.step(actions)
                finite_rewards &= bool(torch.isfinite(rewards).all())
                finite_observations &= bool(torch.isfinite(obs["policy"]).all())

            nodal_after = deformable.data.nodal_pos_w.torch
            assert finite_rewards, f"{preset}: rewards became non-finite"
            assert finite_observations, f"{preset}: observations became non-finite"
            assert torch.isfinite(nodal_after).all(), f"{preset}: nodal positions became non-finite"

            max_delta = (nodal_after - nodal_before).abs().max().item()
            assert max_delta > 1.0e-7, f"{preset}: deformable did not advance under simulation"
        finally:
            env.close()

    return True


@pytest.mark.with_subprocess
@pytest.mark.with_newton
def test_franka_soft_lift_backends_step() -> None:
    assert run_simulation_app_function(_test_franka_soft_lift_backends_step, headless=HEADLESS)
