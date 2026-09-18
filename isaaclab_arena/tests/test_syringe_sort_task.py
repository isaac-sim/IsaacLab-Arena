# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check syringe containment success in the single-syringe environment.

Teleport the syringe above the sharps receiver and let it fall under gravity.
Verify that it settles inside, triggers success, and resets the episode.
"""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_syringe_drop(_simulation_app, minimum_contained_fraction: float | None = None):
    import torch

    from isaaclab.utils.math import quat_apply

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg
    from isaaclab_arena_environments.isaac_cap.registration import register_components
    from isaaclab_arena_environments.isaac_cap.syringe_sort.environments.environment import (
        SyringeSingleEnvironment,
        SyringeSortEnvironmentCfg,
    )

    register_components()
    arena_env = SyringeSingleEnvironment().build(SyringeSortEnvironmentCfg())
    if minimum_contained_fraction is not None:
        arena_env.task.minimum_contained_fraction = minimum_contained_fraction
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(solve_relations=False)).make_registered()
    try:
        obs, _ = env.reset()
        base = env.unwrapped
        syringe = base.scene["syringe_0"]
        # W is world, R is the receiver, and S is the syringe root frame.
        T_W_R = base.arena_world.get_pose_w("sharps_container")
        # Place S above the aperture: t_W_S = t_W_R + R_W_R * t_R_S.
        t_R_S = T_W_R.new_tensor([[0.0975, -0.1225, 0.30]])
        T_W_S = T_W_R.clone()
        T_W_S[:, :3] = T_W_R[:, :3] + quat_apply(T_W_R[:, 3:], t_R_S)
        # q_W_S rotates +90 degrees about world X, making the syringe's Y axis vertical.
        T_W_S[:, 3:] = T_W_S.new_tensor([[2**-0.5, 0, 0, 2**-0.5]])
        syringe.write_root_pose_to_sim(T_W_S)
        syringe.write_root_velocity_to_sim(torch.zeros((1, 6), device=base.device))
        policy = ZeroActionPolicy(ZeroActionPolicyCfg())
        contained_fractions = []
        with torch.inference_mode():
            for step in range(500):
                obs, _, terminated, truncated, _ = env.step(policy.get_action(env, obs))
                success = base.termination_manager.get_term("success")
                if not terminated.any():
                    syringe_bounds = base.arena_world.get_aabb_w("syringe_0")
                    target_bounds = base.arena_world.get_aabb_w("sharps_container")
                    contained_fractions.append(float(syringe_bounds.volume_fraction_within(target_bounds)[0]))
                if step == 0:
                    assert not success.any(), "Syringe above the container must not count as contained"
                assert not truncated.any(), "Drop test timed out"
                if terminated.any():
                    assert success.all(), "Episode ended without syringe containment success"
                    assert base.episode_length_buf[0] == 0, "Success did not reset the environment"
                    return True
        final_fractions = contained_fractions[-50:]
        assert False, (
            "Syringe did not fall into the container and settle; "
            f"AABB volume contained over the final 50 steps: {100 * min(final_fractions):.4f}%–"
            f"{100 * max(final_fractions):.4f}% (final: {100 * final_fractions[-1]:.4f}%)"
        )
    finally:
        env.close()


@pytest.mark.with_newton
def test_syringe_drop_partial_containment():
    assert run_function_with_persistent_simulation_app(_test_syringe_drop, minimum_contained_fraction=0.95)
