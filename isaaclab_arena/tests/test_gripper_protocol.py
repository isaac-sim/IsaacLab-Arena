# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for embodiment-owned behavioral grippers."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_world():
    import math
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.environments.arena_world import ArenaWorld

    joint_positions = torch.tensor([[0.01, 0.02, 0.0], [0.04, 0.04, 0.8]])
    half_sqrt_two = math.sqrt(0.5)
    articulation_data = SimpleNamespace(
        joint_names=["panda_finger_joint1", "panda_finger_joint2", "left_driver_joint"],
        joint_pos=SimpleNamespace(torch=joint_positions),
        body_names=["robotiq_base"],
        body_link_pose_w=SimpleNamespace(
            torch=torch.tensor([
                [[0.2, 0.1, 0.3, 0.0, 0.0, half_sqrt_two, half_sqrt_two]],
                [[0.4, 0.2, 0.5, 0.0, 0.0, half_sqrt_two, half_sqrt_two]],
            ])
        ),
    )
    target_positions = torch.tensor([
        [[0.1, 0.0, 0.0], [0.1, 0.02, 0.0], [0.1, -0.02, 0.0]],
        [[0.3, 0.0, 0.0], [0.3, 0.05, 0.0], [0.3, -0.05, 0.0]],
    ])
    sensor_data = SimpleNamespace(
        target_frame_names=["end_effector", "tool_leftfinger", "tool_rightfinger"],
        target_pos_w=SimpleNamespace(torch=target_positions),
    )
    object_poses = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    scene = SimpleNamespace(
        num_envs=2,
        articulations={"robot": SimpleNamespace(data=articulation_data)},
        rigid_objects={
            "object": SimpleNamespace(data=SimpleNamespace(root_pose_w=SimpleNamespace(torch=object_poses)))
        },
        sensors={"ee_frame": SimpleNamespace(data=sensor_data)},
        extras={},
    )
    return ArenaWorld(scene)


def _test_panda_gripper_implements_parallel_jaw_interface(_simulation_app) -> bool:
    import torch

    from isaaclab_arena.embodiments.gripper import PandaGripper

    world = _make_world()
    gripper = PandaGripper()

    torch.testing.assert_close(gripper.get_jaw_gap_m(world), torch.tensor([0.03, 0.08]))
    torch.testing.assert_close(gripper.get_opening_width_m(world), torch.tensor([0.03, 0.08]))
    torch.testing.assert_close(gripper.get_position_w(world), torch.tensor([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]))
    return True


def test_panda_gripper_implements_parallel_jaw_interface() -> None:
    assert run_function_with_persistent_simulation_app(_test_panda_gripper_implements_parallel_jaw_interface)


def _test_robotiq_gripper_measures_tracked_finger_pad_gap(_simulation_app) -> bool:
    import torch

    from isaaclab_arena.embodiments.gripper import RobotiqGripper

    world = _make_world()
    gripper = RobotiqGripper()

    torch.testing.assert_close(gripper.get_jaw_gap_m(world), torch.tensor([0.04, 0.10]))
    torch.testing.assert_close(gripper.get_position_w(world), torch.tensor([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]))
    return True


def test_robotiq_gripper_measures_tracked_finger_pad_gap() -> None:
    assert run_function_with_persistent_simulation_app(_test_robotiq_gripper_measures_tracked_finger_pad_gap)


def _test_robotiq_gripper_measurement_sources_are_independent(_simulation_app) -> bool:
    import torch

    import pytest

    from isaaclab_arena.embodiments.gripper import RobotiqGripper

    world = _make_world()
    joint_measured = RobotiqGripper(driver_joint_name="left_driver_joint")
    body_measured = RobotiqGripper(body_name="robotiq_base")

    assert joint_measured.get_jaw_gap_m(world)[0] > 0.084
    torch.testing.assert_close(
        joint_measured.get_position_w(world),
        torch.tensor([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]),
    )
    torch.testing.assert_close(body_measured.get_jaw_gap_m(world), torch.tensor([0.04, 0.10]))
    torch.testing.assert_close(
        body_measured.get_position_w(world),
        torch.tensor([[0.2, 0.1, 0.3], [0.4, 0.2, 0.5]]),
    )

    with pytest.raises(AssertionError, match="requires body_name"):
        RobotiqGripper(body_point_offset_xyz=(0.1, 0.0, 0.0))
    return True


def test_robotiq_gripper_measurement_sources_are_independent() -> None:
    assert run_function_with_persistent_simulation_app(_test_robotiq_gripper_measurement_sources_are_independent)


def _test_robotiq_gripper_encapsulates_driver_joint_and_body_point_details(_simulation_app) -> bool:
    import torch

    from isaaclab_arena.embodiments.gripper import RobotiqGripper

    world = _make_world()
    gripper = RobotiqGripper(
        driver_joint_name="left_driver_joint",
        body_name="robotiq_base",
        body_point_offset_xyz=(0.1, 0.0, 0.157),
    )

    jaw_gap_m = gripper.get_jaw_gap_m(world)
    assert jaw_gap_m[0] > 0.084
    assert jaw_gap_m[1] < 0.001
    torch.testing.assert_close(
        gripper.get_position_w(world),
        torch.tensor([[0.2, 0.2, 0.457], [0.4, 0.3, 0.657]]),
    )
    return True


def test_robotiq_gripper_encapsulates_driver_joint_and_body_point_details() -> None:
    assert run_function_with_persistent_simulation_app(
        _test_robotiq_gripper_encapsulates_driver_joint_and_body_point_details
    )


def _test_embodiment_requires_a_supported_gripper(_simulation_app) -> bool:
    import pytest

    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.embodiments.gripper import PandaGripper

    embodiment = object.__new__(EmbodimentBase)
    embodiment.name = "test"
    embodiment.gripper = None
    with pytest.raises(AssertionError, match="has no supported gripper"):
        embodiment.get_gripper()

    embodiment.gripper = PandaGripper()
    assert embodiment.get_gripper() is embodiment.gripper
    return True


def test_embodiment_requires_a_supported_gripper() -> None:
    assert run_function_with_persistent_simulation_app(_test_embodiment_requires_a_supported_gripper)


def _test_gripper_predicates_are_implementation_agnostic(_simulation_app) -> bool:
    from types import SimpleNamespace

    from isaaclab_arena.embodiments.gripper import PandaGripper, RobotiqGripper
    from isaaclab_arena.tasks.predicates.gripper import gripper_released
    from isaaclab_arena.tasks.predicates.spatial import gripper_distance_from_object_exceeds_threshold

    env = SimpleNamespace(arena_world=_make_world())

    clears = gripper_released(env, PandaGripper(), grasp_width_m=0.035, release_clearance_m=0.004)
    away = gripper_distance_from_object_exceeds_threshold(
        env, subject_name="object", gripper=RobotiqGripper(), distance_threshold_m=0.2
    )
    assert clears.tolist() == [False, True]
    assert away.tolist() == [False, True]
    return True


def test_gripper_predicates_are_implementation_agnostic() -> None:
    assert run_function_with_persistent_simulation_app(_test_gripper_predicates_are_implementation_agnostic)


def _test_release_predicate_supports_multi_finger_hands(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.tasks.predicates.gripper import gripper_released

    class ThreeFingerHand:
        def get_position_w(self, world):
            return world.get_frame_position_w("ee_frame", "end_effector")

        def get_opening_width_m(self, _world):
            return torch.tensor([0.03, 0.08])

    env = SimpleNamespace(arena_world=_make_world())
    hand = ThreeFingerHand()

    released = gripper_released(env, hand, grasp_width_m=0.035, release_clearance_m=0.004)
    assert released.tolist() == [False, True]
    return True


def test_release_predicate_supports_multi_finger_hands() -> None:
    assert run_function_with_persistent_simulation_app(_test_release_predicate_supports_multi_finger_hands)


def _test_gripper_predicates_validate_distances(_simulation_app) -> bool:
    from types import SimpleNamespace

    import pytest

    from isaaclab_arena.embodiments.gripper import PandaGripper
    from isaaclab_arena.tasks.predicates.gripper import gripper_released
    from isaaclab_arena.tasks.predicates.spatial import gripper_distance_from_object_exceeds_threshold

    env = SimpleNamespace(arena_world=_make_world())
    with pytest.raises(AssertionError, match="Grasp width"):
        gripper_released(env, PandaGripper(), grasp_width_m=0.0, release_clearance_m=0.001)
    with pytest.raises(AssertionError, match="clearance"):
        gripper_released(env, PandaGripper(), grasp_width_m=0.01, release_clearance_m=-0.001)
    with pytest.raises(AssertionError, match="Distance"):
        gripper_distance_from_object_exceeds_threshold(
            env, subject_name="object", gripper=PandaGripper(), distance_threshold_m=-0.1
        )
    return True


def test_gripper_predicates_validate_distances() -> None:
    assert run_function_with_persistent_simulation_app(_test_gripper_predicates_validate_distances)
