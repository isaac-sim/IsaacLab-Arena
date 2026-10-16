# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Frame and TCP-offset regression checks for YAM pose actions."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_world_pose_conversion(_app):
    import torch
    from types import SimpleNamespace

    from isaaclab.utils.math import combine_frame_transforms

    from isaaclab_arena.embodiments.yam.control import absolute_pose_actions

    roots = {
        "left": torch.tensor([[0.2, 0.6, 0.75, 0.0, 0.0, 0.70710678, 0.70710678]]),
        "right": torch.tensor([[-0.1, -0.6, 0.75, 0.0, 0.0, -0.70710678, 0.70710678]]),
    }
    env = SimpleNamespace(
        device="cpu",
        num_envs=1,
        scene={
            f"{side}_robot": SimpleNamespace(data=SimpleNamespace(root_pose_w=SimpleNamespace(torch=root)))
            for side, root in roots.items()
        },
    )
    # The same world pose must result in different commands in the opposed bases.
    world = torch.tensor([[0.1, 0.05, 1.05, 1.0, 0.0, 0.0, 0.0]])
    actions = absolute_pose_actions(env, {"left": world, "right": world}, {"left": 0.2, "right": 0.8})
    assert actions.shape == (1, 16)
    assert not torch.allclose(actions[:, :3], actions[:, 8:11])
    for side, start in (("left", 0), ("right", 8)):
        root = roots[side]
        p, q = combine_frame_transforms(
            root[:, :3], root[:, 3:], actions[:, start : start + 3], actions[:, start + 3 : start + 7]
        )
        assert torch.allclose(p, world[:, :3], atol=1e-6)
        assert ((q * world[:, 3:]).sum(-1).abs() > 0.99999).all()
    assert torch.allclose(actions[:, [7, 15]], torch.tensor([[0.2, 0.8]]))
    return True


def _test_rotated_tcp_jacobian(_app):
    import torch
    from types import SimpleNamespace

    from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

    from isaaclab_arena.embodiments.yam.actions import YamTcpIKAction

    # A revolute joint about base Z and a tool rotated 90 degrees about Y.
    # Finite differences expose the upstream error of using an unrotated lever arm.
    q = quat_from_angle_axis(torch.tensor([1.5707963268]), torch.tensor([[0.0, 1.0, 0.0]]))
    offset = torch.tensor([[0.0, -0.044, 0.13]])
    offset_rotation = quat_from_angle_axis(torch.tensor([0.4]), torch.tensor([[0.0, 0.0, 1.0]]))
    jacobian = torch.zeros(1, 6, 1)
    jacobian[:, 5, 0] = 1.0
    fake = SimpleNamespace(
        jacobian_b=jacobian,
        _jacobian_b=torch.zeros_like(jacobian),
        _offset_pos=offset,
        _offset_rot=offset_rotation,
        _compute_frame_pose=lambda: (torch.zeros(1, 3), quat_mul(q, offset_rotation)),
    )
    actual = YamTcpIKAction._compute_frame_jacobian(fake)[0, :3, 0]
    p = quat_apply(q, offset)
    delta = quat_from_angle_axis(torch.tensor([0.0001]), torch.tensor([[0.0, 0.0, 1.0]]))
    expected = ((quat_apply(delta, p) - p) / 0.0001)[0]
    assert torch.allclose(actual, expected, atol=0.0002)
    return True


def test_world_pose_conversion():
    assert run_function_with_persistent_simulation_app(_test_world_pose_conversion)


def test_rotated_tcp_jacobian():
    assert run_function_with_persistent_simulation_app(_test_rotated_tcp_jacobian)


def _test_yam_grasp_predicate(_app):
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.embodiments.yam.gripper import YamGripper
    from isaaclab_arena.tasks.predicates.gripper import gripper_not_grasping, gripper_released

    positions = {"left_robot": torch.tensor([0.009]), "right_robot": torch.tensor([0.0])}
    world = SimpleNamespace(
        get_joint_position=lambda robot, joint: positions[robot],
        get_joint_position_target=lambda robot, joint: torch.tensor([0.0]),
    )
    env = SimpleNamespace(arena_world=world)
    left = YamGripper(articulation_name="left_robot")
    right = YamGripper(articulation_name="right_robot")
    assert not bool(gripper_not_grasping(env, left, 0.018, 0.004, 0.002)[0])
    assert bool(gripper_not_grasping(env, right, 0.018, 0.004, 0.002)[0])
    assert not bool(gripper_released(env, left, 0.018)[0])
    positions["left_robot"][:] = 0.037
    assert bool(gripper_released(env, left, 0.018)[0])
    return True


def test_yam_grasp_predicate():
    assert run_function_with_persistent_simulation_app(_test_yam_grasp_predicate)
