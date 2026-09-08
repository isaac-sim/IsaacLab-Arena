# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_runtime_state_consumers_use_arena_world(_simulation_app) -> bool:
    import torch
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.tasks import terminations
    from isaaclab_arena.tasks.predicates import spatial

    class ArenaWorldDouble:
        def __init__(self):
            self.poses_w_by_scene_key = {}
            self.poses_e_by_scene_key = {}
            self.root_linear_velocities_w_by_scene_key = {}
            self.pose_w_queries = []
            self.pose_e_queries = []
            self.root_linear_velocity_queries = []

        def get_pose_w(self, scene_key):
            self.pose_w_queries.append(scene_key)
            return self.poses_w_by_scene_key[scene_key]

        def get_pose_e(self, scene_key):
            self.pose_e_queries.append(scene_key)
            return self.poses_e_by_scene_key[scene_key]

        def get_root_linear_velocity_w(self, scene_key):
            self.root_linear_velocity_queries.append(scene_key)
            return self.root_linear_velocities_w_by_scene_key[scene_key]

    identity_quaternion = (0.0, 0.0, 0.0, 1.0)
    arena_world = ArenaWorldDouble()
    arena_world.poses_e_by_scene_key = {
        "object": torch.tensor([
            [0.00, 0.00, 0.00, *identity_quaternion],
            [0.00, 0.00, 0.00, *identity_quaternion],
            [0.00, 0.00, 0.00, *identity_quaternion],
        ]),
        "target": torch.tensor([
            [0.05, 0.10, 0.10, *identity_quaternion],
            [0.11, 0.10, 0.10, *identity_quaternion],
            [0.05, 0.10, 0.21, *identity_quaternion],
        ]),
    }
    arena_world.root_linear_velocities_w_by_scene_key = {
        "object": torch.tensor([
            [0.00, 0.00, 0.00],
            [0.20, 0.00, 0.00],
            [0.30, 0.40, 0.00],
        ])
    }
    arena_world.poses_w_by_scene_key = {
        "object": torch.tensor([
            [0.00, 0.00, 0.00, *identity_quaternion],
            [0.20, 0.00, 0.00, *identity_quaternion],
            [0.00, 0.00, 0.00, 0.0, 0.0, 1.0, 0.0],
        ])
    }
    live_env = SimpleNamespace(arena_world=arena_world, device=torch.device("cpu"), num_envs=3)
    wrapped_env = SimpleNamespace(unwrapped=live_env)

    proximity_result = spatial.objects_in_proximity(
        wrapped_env,
        object_cfg=SceneEntityCfg("object"),
        target_object_cfg=SceneEntityCfg("target"),
        max_x_separation=0.10,
        max_y_separation=0.20,
        max_z_separation=0.20,
    )
    torch.testing.assert_close(proximity_result, torch.tensor([True, False, False]))
    assert arena_world.pose_e_queries == ["object", "target"]

    moving_result = spatial.object_moving(wrapped_env, object_name="object", velocity_threshold=0.10)
    torch.testing.assert_close(moving_result, torch.tensor([False, True, True]))
    assert arena_world.root_linear_velocity_queries == ["object"]

    goal_pose_result = terminations.goal_pose_task_termination(
        wrapped_env,
        object_cfg=SceneEntityCfg("object"),
        target_x_range=(-0.10, 0.10),
        target_orientation_xyzw=identity_quaternion,
        target_orientation_tolerance_rad=0.10,
    )
    torch.testing.assert_close(goal_pose_result, torch.tensor([True, False, False]))
    assert arena_world.pose_w_queries == ["object"]

    return True


def test_runtime_state_consumers_use_arena_world():
    assert run_function_with_persistent_simulation_app(_test_runtime_state_consumers_use_arena_world)


if __name__ == "__main__":
    test_runtime_state_consumers_use_arena_world()
