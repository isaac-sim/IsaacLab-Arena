# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


def _test_object_on_destination_predicate(_simulation_app) -> bool:
    import warp as wp
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.tasks.predicates.spatial import (
        contact_force_is_upward_support,
        object_centroid_in_destination_footprint,
        object_on_destination,
    )
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    class DummyScene(dict):
        def __init__(self, num_envs: int):
            super().__init__()
            self.env_origins = torch.zeros((num_envs, 3))

    class DummyEnv:
        def __init__(self, num_envs: int):
            self.num_envs = num_envs
            self.device = "cpu"
            self.scene = DummyScene(num_envs)

    class DummyAsset:
        def __init__(self, name: str, bounding_box: AxisAlignedBoundingBox, pose_w: torch.Tensor):
            self.name = name
            self._bounding_box = bounding_box
            self._pose_w = pose_w

        def get_scene_key(self) -> str:
            return self.name

        def get_bounding_box(self) -> AxisAlignedBoundingBox:
            return self._bounding_box

        def get_object_pose(self, _env, is_relative: bool = True) -> torch.Tensor:
            assert not is_relative
            return self._pose_w

    identity_quaternion = (0.0, 0.0, 0.0, 1.0)

    force_matrix_w = torch.tensor([
        [[[0.0, 0.0, 0.05], [0.0, 0.0, 0.0]]],
        [[[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]],
        [[[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        [[[0.0, 0.0, -0.2], [0.0, 0.0, 0.0]]],
        [[[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]],
        [[[1.01, 0.0, 1.0], [0.0, 0.0, 0.0]]],
        [[[1.0, 0.0, 0.5], [-1.0, 0.0, 0.5]]],
    ])
    support_result = contact_force_is_upward_support(
        force_matrix_w,
        force_threshold=0.1,
        support_cone_half_angle_deg=45.0,
    )
    torch.testing.assert_close(
        support_result,
        torch.tensor([False, True, False, False, True, False, True]),
    )
    torch.testing.assert_close(
        contact_force_is_upward_support(
            torch.tensor([[0.0, 0.0, 0.2], [0.2, 0.0, 0.0]]),
            force_threshold=0.1,
            support_cone_half_angle_deg=45.0,
        ),
        torch.tensor([True, False]),
    )

    num_envs = 5
    env = DummyEnv(num_envs)
    yaw_90_quaternion = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    object_pose_w = torch.tensor([
        [-0.2, 0.0, 0.0, *identity_quaternion],
        [0.8, 0.5, 0.0, *identity_quaternion],
        # The object overlaps the destination from outside, but its centroid
        # remains beyond the destination footprint.
        [0.81, 0.0, 0.0, *identity_quaternion],
        [0.2, 0.9, 100.0, *identity_quaternion],
        [0.4, 0.0, 0.0, *identity_quaternion],
    ])
    destination_pose_w = torch.tensor([
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *yaw_90_quaternion],
        [0.0, 0.0, 0.0, *yaw_90_quaternion],
    ])
    object_asset = DummyAsset(
        "object",
        AxisAlignedBoundingBox(min_point=(0.1, -0.1, -0.1), max_point=(0.3, 0.1, 0.1)),
        object_pose_w,
    )
    destination_asset = DummyAsset(
        "destination",
        AxisAlignedBoundingBox(min_point=(-1.0, -0.5, -0.2), max_point=(1.0, 0.5, 0.2)),
        destination_pose_w,
    )
    env.scene[object_asset.name] = object()
    env.scene[destination_asset.name] = object()

    footprint_result = object_centroid_in_destination_footprint(
        env,
        object_asset=object_asset,
        destination_asset=destination_asset,
        footprint_tolerance=0.0,
    )
    torch.testing.assert_close(
        footprint_result,
        torch.tensor([True, True, False, True, False]),
    )

    parent_pose_w = torch.tensor([[1.0, 2.0, 0.0, *yaw_90_quaternion]])
    reference_env = DummyEnv(num_envs=1)
    parent_asset = DummyAsset(
        "parent",
        AxisAlignedBoundingBox(min_point=(-2.0, -2.0, -0.2), max_point=(2.0, 2.0, 0.2)),
        parent_pose_w,
    )
    reference_env.scene[parent_asset.name] = object()
    reference_object_asset = DummyAsset(
        "reference_object",
        AxisAlignedBoundingBox(min_point=(-0.1, -0.1, -0.1), max_point=(0.1, 0.1, 0.1)),
        torch.tensor([[1.25, 1.75, 0.0, *identity_quaternion]]),
    )
    reference_env.scene[reference_object_asset.name] = object()

    reference_destination_asset = DummyAsset(
        "reference_destination",
        AxisAlignedBoundingBox(min_point=(-1.0, -0.5, -0.2), max_point=(1.0, 0.5, 0.2)),
        torch.empty((1, 7)),
    )
    reference_destination_asset.parent_asset = parent_asset
    reference_destination_asset.initial_pose_relative_to_parent = Pose(
        position_xyz=(0.5, 0.0, 0.0),
        rotation_xyzw=yaw_90_quaternion,
    )
    reference_footprint_result = object_centroid_in_destination_footprint(
        reference_env,
        object_asset=reference_object_asset,
        destination_asset=reference_destination_asset,
        footprint_tolerance=0.0,
    )
    assert reference_footprint_result.item()

    combined_env = DummyEnv(num_envs=3)
    combined_object_asset = DummyAsset(
        "object",
        AxisAlignedBoundingBox(min_point=(0.1, -0.1, -0.1), max_point=(0.3, 0.1, 0.1)),
        torch.tensor([
            [-0.2, 0.0, 0.0, *identity_quaternion],
            [-0.2, 0.0, 0.0, *identity_quaternion],
            [0.81, 0.0, 0.0, *identity_quaternion],
        ]),
    )
    combined_destination_asset = DummyAsset(
        "destination",
        AxisAlignedBoundingBox(min_point=(-1.0, -0.5, -0.2), max_point=(1.0, 0.5, 0.2)),
        torch.tensor([[0.0, 0.0, 0.0, *identity_quaternion]]).expand(3, 7),
    )

    class DummyRigidObject:
        def __init__(self):
            self.data = type(
                "DummyRigidObjectData",
                (),
                {"root_lin_vel_w": wp.from_torch(torch.zeros((3, 3)), dtype=wp.vec3)},
            )()

    class DummyContactSensor:
        def __init__(self):
            force_matrix_w = torch.tensor([
                [[[0.0, 0.0, 0.2]]],
                [[[0.2, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.2]]],
            ])
            self.data = type(
                "DummyContactSensorData",
                (),
                {"force_matrix_w": wp.from_torch(force_matrix_w, dtype=wp.vec3)},
            )()

    combined_env.scene[combined_object_asset.name] = DummyRigidObject()
    combined_env.scene[combined_destination_asset.name] = object()
    combined_env.scene["contact_sensor"] = DummyContactSensor()
    combined_result = object_on_destination(
        combined_env,
        object_cfg=SceneEntityCfg(combined_object_asset.name),
        contact_sensor_cfg=SceneEntityCfg("contact_sensor"),
        object_asset=combined_object_asset,
        destination_asset=combined_destination_asset,
        force_threshold=0.1,
        velocity_threshold=0.1,
        support_cone_half_angle_deg=45.0,
        footprint_tolerance=0.0,
    )
    torch.testing.assert_close(combined_result, torch.tensor([True, False, False]))

    return True


def test_object_on_destination_predicate():
    assert run_simulation_app_function(_test_object_on_destination_predicate)
