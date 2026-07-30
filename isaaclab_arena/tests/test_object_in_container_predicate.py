# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import math
import torch

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


def _test_object_in_container_predicate(_simulation_app) -> bool:
    import warp as wp
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.tasks.predicates.predicate_utils import ArenaAssetHandle
    from isaaclab_arena.tasks.predicates.spatial import (
        contact_force_is_upward_support,
        object_in_container,
        object_on_destination,
    )
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose, PosePerEnv

    class DummyScene(dict):
        def __init__(self, num_envs: int):
            super().__init__()
            self.env_origins = torch.zeros((num_envs, 3))

    class DummyEnv:
        def __init__(self, num_envs: int):
            self.num_envs = num_envs
            self.device = "cpu"
            self.scene = DummyScene(num_envs)
            self.unwrapped = self

    class DummyAsset:
        def __init__(
            self,
            name: str,
            bounding_box: AxisAlignedBoundingBox,
            pose_w: torch.Tensor,
        ):
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

        def get_bounding_box_pose(self, _env, is_relative: bool = True) -> torch.Tensor:
            assert not is_relative
            return self._pose_w

    identity_quaternion = (0.0, 0.0, 0.0, 1.0)
    roll_90_quaternion = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    yaw_90_quaternion = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))

    geometry_env = DummyEnv(num_envs=8)
    object_asset = DummyAsset(
        "object",
        AxisAlignedBoundingBox(min_point=(0.1, -0.1, -0.1), max_point=(0.3, 0.1, 0.1)),
        torch.tensor([
            [-0.2, 0.0, 0.2, *identity_quaternion],
            [0.81, 0.0, 0.2, *identity_quaternion],
            [-0.2, 0.0, -0.01, *identity_quaternion],
            [-0.2, 0.0, 2.0, *identity_quaternion],
            [-0.2, 0.8, 0.2, *identity_quaternion],
            [0.4, 0.0, 0.2, *identity_quaternion],
            [0.8, 0.0, 0.2, *identity_quaternion],
            [0.0, 0.31, 0.2, *yaw_90_quaternion],
        ]),
    )
    container_asset = DummyAsset(
        "container",
        AxisAlignedBoundingBox(min_point=(-1.0, -0.5, 0.0), max_point=(1.0, 0.5, 0.4)),
        torch.tensor([
            [0.0, 0.0, 0.0, *identity_quaternion],
            [0.0, 0.0, 0.0, *identity_quaternion],
            [0.0, 0.0, 0.0, *identity_quaternion],
            [0.0, 0.0, 0.0, *identity_quaternion],
            [0.0, 0.0, 0.0, *yaw_90_quaternion],
            [0.0, 0.0, 0.0, *yaw_90_quaternion],
            [0.0, 0.0, 0.0, *identity_quaternion],
            [0.0, 0.0, 0.0, *identity_quaternion],
        ]),
    )
    geometry_env.scene[object_asset.name] = object()
    geometry_env.scene[container_asset.name] = object()

    object_asset_handle = ArenaAssetHandle(object_asset)
    container_asset_handle = ArenaAssetHandle(container_asset)
    containment_term_cfg = TerminationTermCfg(
        func=object_in_container,
        params={
            "object_asset_handle": object_asset_handle,
            "container_asset_handle": container_asset_handle,
        },
    )
    containment_term_cfg.validate()
    copied_containment_term_cfg = copy.deepcopy(containment_term_cfg)
    assert copied_containment_term_cfg.params["object_asset_handle"] is object_asset_handle
    assert copied_containment_term_cfg.params["container_asset_handle"] is container_asset_handle
    assert not hasattr(object_asset_handle, "__dict__")

    torch.testing.assert_close(
        object_in_container(geometry_env, object_asset_handle, container_asset_handle),
        torch.tensor([True, False, False, True, True, False, True, False]),
    )

    reference_env = DummyEnv(num_envs=2)
    reference_env.scene.env_origins = torch.tensor([
        [0.0, 0.0, 0.0],
        [4.0, -3.0, 1.0],
    ])

    base_object = Object.__new__(Object)
    base_object.name = "base_object"
    base_object.object_type = ObjectType.BASE
    base_object.initial_pose = Pose(
        position_xyz=(1.0, 2.0, 3.0),
        rotation_xyzw=roll_90_quaternion,
    )
    reference_env.scene[base_object.name] = object()
    expected_base_pose_relative = torch.tensor([
        [1.0, 2.0, 3.0, *roll_90_quaternion],
        [1.0, 2.0, 3.0, *roll_90_quaternion],
    ])
    expected_base_pose_w = expected_base_pose_relative.clone()
    expected_base_pose_w[:, :3] += reference_env.scene.env_origins
    torch.testing.assert_close(base_object.get_object_pose(reference_env), expected_base_pose_relative)
    torch.testing.assert_close(
        base_object.get_object_pose(reference_env, is_relative=False),
        expected_base_pose_w,
    )

    class DummyFrameView:
        def __init__(self, pose_w: torch.Tensor):
            self._pose_w = pose_w

        def get_world_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
            return self._pose_w[:, :3], self._pose_w[:, 3:]

    per_env_base_object = Object.__new__(Object)
    per_env_base_object.name = "per_env_base_object"
    per_env_base_object.object_type = ObjectType.BASE
    per_env_base_object.initial_pose = PosePerEnv([
        Pose(rotation_xyzw=roll_90_quaternion),
        Pose(rotation_xyzw=yaw_90_quaternion),
    ])
    per_env_base_pose_w = torch.tensor([
        [0.5, 0.25, 0.0, *roll_90_quaternion],
        [4.5, -2.75, 1.0, *yaw_90_quaternion],
    ])
    reference_env.scene[per_env_base_object.name] = DummyFrameView(per_env_base_pose_w)
    torch.testing.assert_close(
        per_env_base_object.get_object_pose(reference_env, is_relative=False),
        per_env_base_pose_w,
    )
    expected_per_env_base_pose_relative = per_env_base_pose_w.clone()
    expected_per_env_base_pose_relative[:, :3] -= reference_env.scene.env_origins
    torch.testing.assert_close(
        per_env_base_object.get_object_pose(reference_env),
        expected_per_env_base_pose_relative,
    )

    rotated_reference = ObjectReference.__new__(ObjectReference)
    rotated_reference.name = "rotated_reference"
    rotated_reference.object_type = ObjectType.BASE
    rotated_reference.parent_asset = base_object
    rotated_reference.initial_pose_relative_to_parent = Pose(
        position_xyz=(0.0, 1.0, 0.0),
        rotation_xyzw=yaw_90_quaternion,
    )
    expected_rotated_reference_pose = torch.tensor([
        [1.0, 2.0, 4.0, 0.5, -0.5, 0.5, 0.5],
        [1.0, 2.0, 4.0, 0.5, -0.5, 0.5, 0.5],
    ])
    torch.testing.assert_close(rotated_reference.get_object_pose(reference_env), expected_rotated_reference_pose)
    expected_rotated_reference_bounding_box_pose = expected_rotated_reference_pose.clone()
    expected_rotated_reference_bounding_box_pose[:, 3:] = torch.tensor(roll_90_quaternion)
    torch.testing.assert_close(
        rotated_reference.get_bounding_box_pose(reference_env),
        expected_rotated_reference_bounding_box_pose,
    )

    reference_object = DummyAsset(
        "reference_object",
        AxisAlignedBoundingBox(min_point=(-0.1, -0.1, -0.1), max_point=(0.1, 0.1, 0.1)),
        torch.tensor([
            [0.8, 0.0, 0.2, *identity_quaternion],
            [4.8, -3.0, 1.2, *identity_quaternion],
        ]),
    )
    reference_parent = Object.__new__(Object)
    reference_parent.name = "reference_parent"
    reference_parent.object_type = ObjectType.BASE
    reference_parent.initial_pose = Pose.identity()
    reference_env.scene[reference_parent.name] = object()

    reference_container = ObjectReference.__new__(ObjectReference)
    reference_container.name = "reference_container"
    reference_container.object_type = ObjectType.BASE
    reference_container.parent_asset = reference_parent
    reference_container.initial_pose_relative_to_parent = Pose(rotation_xyzw=yaw_90_quaternion)
    reference_container._bounding_box = AxisAlignedBoundingBox(
        min_point=(-1.0, -0.5, 0.0),
        max_point=(1.0, 0.5, 0.4),
    )
    reference_env.scene[reference_object.name] = object()
    torch.testing.assert_close(
        reference_container.get_object_pose(reference_env, is_relative=False),
        torch.tensor([
            [0.0, 0.0, 0.0, *yaw_90_quaternion],
            [4.0, -3.0, 1.0, *yaw_90_quaternion],
        ]),
    )
    torch.testing.assert_close(
        reference_container.get_bounding_box_pose(reference_env, is_relative=False),
        torch.tensor([
            [0.0, 0.0, 0.0, *identity_quaternion],
            [4.0, -3.0, 1.0, *identity_quaternion],
        ]),
    )
    torch.testing.assert_close(
        object_in_container(
            reference_env,
            ArenaAssetHandle(reference_object),
            ArenaAssetHandle(reference_container),
        ),
        torch.tensor([True, True]),
    )

    registered_reference = ObjectReference.__new__(ObjectReference)
    registered_reference.name = "registered_reference"
    registered_reference.object_type = ObjectType.RIGID
    registered_reference.parent_asset = reference_parent
    registered_reference.initial_pose_relative_to_parent = Pose(rotation_xyzw=yaw_90_quaternion)
    registered_reference_pose_w = torch.tensor([
        [2.0, 3.0, 4.0, *identity_quaternion],
        [6.0, 0.0, 5.0, *identity_quaternion],
    ])
    registered_reference_data = type(
        "RegisteredReferenceData",
        (),
        {"root_pose_w": wp.from_torch(registered_reference_pose_w)},
    )()
    reference_env.scene[registered_reference.name] = type(
        "RegisteredReferenceEntity",
        (),
        {"data": registered_reference_data},
    )()
    torch.testing.assert_close(
        registered_reference.get_object_pose(reference_env, is_relative=False),
        registered_reference_pose_w,
    )
    registered_reference_pose_relative = registered_reference_pose_w.clone()
    registered_reference_pose_relative[:, :3] -= reference_env.scene.env_origins
    torch.testing.assert_close(
        registered_reference.get_object_pose(reference_env),
        registered_reference_pose_relative,
    )

    force_matrix_w = torch.tensor([
        [[[0.0, 0.0, 0.05], [0.0, 0.0, 0.0]]],
        [[[0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]],
        [[[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]],
        [[[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        [[[0.0, 0.0, -0.2], [0.0, 0.0, 0.0]]],
        [[[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]],
        [[[1.01, 0.0, 1.0], [0.0, 0.0, 0.0]]],
        [[[1.0, 0.0, 0.5], [-1.0, 0.0, 0.5]]],
    ])
    torch.testing.assert_close(
        contact_force_is_upward_support(
            force_matrix_w,
            force_threshold=0.1,
            support_cone_half_angle_deg=45.0,
        ),
        torch.tensor([False, True, True, False, False, True, False, True]),
    )

    combined_env = DummyEnv(num_envs=4)
    combined_object_asset = DummyAsset(
        "combined_object",
        AxisAlignedBoundingBox(min_point=(-0.1, -0.1, -0.1), max_point=(0.1, 0.1, 0.1)),
        torch.tensor([
            [0.0, 0.0, 0.2, *identity_quaternion],
            [1.1, 0.0, 0.2, *identity_quaternion],
            [0.0, 0.0, 0.2, *identity_quaternion],
            [0.0, 0.0, 0.2, *identity_quaternion],
        ]),
    )
    combined_container_asset = DummyAsset(
        "combined_container",
        AxisAlignedBoundingBox(min_point=(-1.0, -0.5, 0.0), max_point=(1.0, 0.5, 0.4)),
        torch.tensor([[0.0, 0.0, 0.0, *identity_quaternion]]).expand(4, 7),
    )

    class DummyRigidObject:
        def __init__(self):
            velocity_w = torch.tensor([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ])
            self.data = type(
                "DummyRigidObjectData",
                (),
                {"root_lin_vel_w": wp.from_torch(velocity_w, dtype=wp.vec3)},
            )()

    class DummyContactSensor:
        def __init__(self):
            filtered_force_matrix_w = torch.tensor([
                [[[0.0, 0.0, 0.2]]],
                [[[0.0, 0.0, 0.2]]],
                [[[0.2, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.2]]],
            ])
            self.data = type(
                "DummyContactSensorData",
                (),
                {"force_matrix_w": wp.from_torch(filtered_force_matrix_w, dtype=wp.vec3)},
            )()

    combined_env.scene[combined_object_asset.name] = DummyRigidObject()
    combined_env.scene[combined_container_asset.name] = object()
    combined_env.scene["contact_sensor"] = DummyContactSensor()
    torch.testing.assert_close(
        object_on_destination(
            combined_env,
            object_cfg=SceneEntityCfg(combined_object_asset.name),
            contact_sensor_cfg=SceneEntityCfg("contact_sensor"),
            force_threshold=0.1,
            velocity_threshold=0.1,
            object_asset_handle=ArenaAssetHandle(combined_object_asset),
            destination_asset_handle=ArenaAssetHandle(combined_container_asset),
            support_cone_half_angle_deg=45.0,
        ),
        torch.tensor([True, False, False, False]),
    )

    return True


def test_object_in_container_predicate():
    assert run_simulation_app_function(_test_object_in_container_predicate)
