# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""RTX regression coverage for composed Replicator background physics."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_replicator_online_visual_physics_with_rtx(_) -> bool:
    import torch

    import warp as wp

    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    env_spec = "isaaclab_arena_environments/kitchen_bench/replicator_kitchen_peninsula_mustard_bowl.yaml"
    args = get_isaaclab_arena_environments_cli_parser().parse_args(
        ["--env_spec", env_spec, "--headless", "--num_envs", "1", "--enable_cameras"]
    )

    builder = get_arena_builder_from_cli(args)
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env = builder.make_registered(env_cfg, env_kwargs, render_mode="rgb_array")
    try:
        env.reset()
        physics_paths = builder.arena_env.scene.get_background_physics_paths()["replicator_kitchen_peninsula"]
        assert len(physics_paths) == 56
        banana_path = next(
            path for path, object_type in physics_paths.items() if "Banana" in path and object_type == ObjectType.RIGID
        )
        runtime_path = banana_path.replace("{ENV_REGEX_NS}", env.unwrapped.scene.env_prim_paths[0])
        banana_view = env.unwrapped.sim.physics_manager.get_physics_sim_view().create_rigid_body_view(runtime_path)
        assert banana_view.count == 1
        initial_transform = wp.to_torch(banana_view.get_transforms()).clone()
        moved_transform = initial_transform.clone()
        moved_transform[:, 0] += 1.0
        indices = wp.from_torch(torch.tensor([0], device=env.unwrapped.device, dtype=torch.int32))
        banana_view.set_transforms(wp.from_torch(moved_transform), indices=indices)
        banana_view.set_velocities(
            wp.from_torch(torch.ones((1, 6), device=env.unwrapped.device)),
            indices=indices,
        )

        env.reset()
        assert torch.allclose(wp.to_torch(banana_view.get_transforms()), initial_transform, atol=1.0e-5)
        assert torch.count_nonzero(wp.to_torch(banana_view.get_velocities())) == 0
    finally:
        env.close()
    return True


@pytest.mark.with_cameras
def test_replicator_online_visual_physics_with_rtx():
    assert run_function_with_persistent_simulation_app(
        _test_replicator_online_visual_physics_with_rtx,
        enable_cameras=True,
    )
