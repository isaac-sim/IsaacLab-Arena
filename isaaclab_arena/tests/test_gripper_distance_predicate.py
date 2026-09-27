# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check object distance against the gripper's frame-transformer output."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_gripper_distance_from_object_exceeds_threshold(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    import pytest

    from isaaclab_arena.embodiments.gripper import PandaGripper
    from isaaclab_arena.environments.arena_world import ArenaWorld
    from isaaclab_arena.tasks.predicates.spatial import gripper_distance_from_object_exceeds_threshold

    for device in ("cpu", "cuda:0"):
        for dtype in (torch.float32, torch.float64):
            poses = torch.tensor(
                [[0.125, 0, 0, 0, 0, 0, 1], [0.25, 0, 0, 0, 0, 0, 1], [0.5, 0, 0, 0, 0, 0, 1]],
                device=device,
                dtype=dtype,
            )
            frame_positions = torch.zeros((3, 2, 3), device=device, dtype=dtype)
            frame_positions[:, 0, 1] = 1.0
            scene = SimpleNamespace(
                num_envs=3,
                rigid_objects={
                    "object": SimpleNamespace(data=SimpleNamespace(root_pose_w=SimpleNamespace(torch=poses)))
                },
                sensors={
                    "ee_frame": SimpleNamespace(
                        data=SimpleNamespace(
                            target_frame_names=["finger", "tool"],
                            target_pos_w=SimpleNamespace(torch=frame_positions),
                        )
                    )
                },
            )
            env = SimpleNamespace(arena_world=ArenaWorld(scene))
            gripper = PandaGripper(target_frame_name="tool")
            params = dict(subject_name="object", gripper=gripper, distance_threshold_m=0.25)
            result = gripper_distance_from_object_exceeds_threshold(env, **params)
            assert result.tolist() == [False, False, True]
            assert result.device == poses.device and result.dtype == torch.bool

            # Consume the updated sensor output, including any embodiment-owned tool offset.
            frame_positions[:, 1, 0] = 0.25
            assert not gripper_distance_from_object_exceeds_threshold(env, **params).any()
            with pytest.raises(AssertionError, match="target frame"):
                missing = PandaGripper(target_frame_name="missing")
                gripper_distance_from_object_exceeds_threshold(env, **{**params, "gripper": missing})
            with pytest.raises(AssertionError, match="non-negative"):
                gripper_distance_from_object_exceeds_threshold(env, **{**params, "distance_threshold_m": -0.1})
    return True


def test_gripper_distance_from_object_exceeds_threshold() -> None:
    assert run_function_with_persistent_simulation_app(_test_gripper_distance_from_object_exceeds_threshold)
