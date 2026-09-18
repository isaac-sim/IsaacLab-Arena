# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check physical parallel-jaw release independently of commanded motion."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_parallel_jaw_gripper_released(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    import pytest

    from isaaclab_arena.environments.arena_world import ArenaWorld
    from isaaclab_arena.tasks.predicates.gripper import parallel_jaw_gripper_released

    for device in ("cpu", "cuda:0"):
        for dtype in (torch.float32, torch.float64):
            # Closed, grasping, opening but still touching, below/exactly at clearance, and physically open.
            joint_positions = torch.tensor(
                [[0.0], [0.015625], [0.015625], [0.0166015625], [0.017578125], [0.01953125], [0.0625]],
                device=device,
                dtype=dtype,
            )
            data = SimpleNamespace(joint_names=["finger"], joint_pos=SimpleNamespace(torch=joint_positions))
            world = ArenaWorld(SimpleNamespace(num_envs=7, articulations={"robot": SimpleNamespace(data=data)}))
            action = SimpleNamespace(processed_actions=torch.zeros_like(joint_positions))
            env = SimpleNamespace(
                arena_world=world,
                action_manager=SimpleNamespace(get_term={"gripper": action}.__getitem__),
            )
            params = dict(
                robot_name="robot",
                gripper_joint_name="finger",
                jaw_gap_at_zero_joint_m=0.0,
                grasp_width_m=0.03125,
                release_clearance_m=0.00390625,
            )
            expected = [False, False, False, False, False, True, True]
            result = parallel_jaw_gripper_released(env, **params)
            assert result.tolist() == expected
            assert result.device == joint_positions.device and result.dtype == torch.bool

            # An opening command cannot release an object before the jaws actually move.
            action.processed_actions[:] = 0.0625
            assert parallel_jaw_gripper_released(env, **params).tolist() == expected
            data.joint_pos.torch = joint_positions.clone()
            data.joint_pos.torch[2, 0] = 0.0625
            assert parallel_jaw_gripper_released(env, **params)[2]

            # A nonzero calibration offset with the same physical gaps gives the same results.
            params["jaw_gap_at_zero_joint_m"] = 0.0078125
            data.joint_pos.torch = joint_positions - 0.00390625
            assert parallel_jaw_gripper_released(env, **params).tolist() == expected
            with pytest.raises(AssertionError, match="clearance"):
                parallel_jaw_gripper_released(env, **{**params, "release_clearance_m": -0.001})
    return True


def test_parallel_jaw_gripper_released() -> None:
    assert run_function_with_persistent_simulation_app(_test_parallel_jaw_gripper_released)
