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

    from isaaclab_arena.tasks.predicates.predicate_utils import ArenaAssetHandle
    from isaaclab_arena.tasks.predicates.spatial import (
        contact_force_is_upward_support,
        object_in_container,
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
            self.unwrapped = self

    class DummyAsset:
        def __init__(
            self,
            name: str,
            bounding_box: AxisAlignedBoundingBox,
            pose_w: torch.Tensor,
            pose_relative_to_parent: Pose | None = None,
        ):
            self.name = name
            self._bounding_box = bounding_box
            self._pose_w = pose_w
            if pose_relative_to_parent is not None:
                self.initial_pose_relative_to_parent = pose_relative_to_parent

        def get_scene_key(self) -> str:
            return self.name

        def get_bounding_box(self) -> AxisAlignedBoundingBox:
            return self._bounding_box

        def get_object_pose(self, _env, is_relative: bool = True) -> torch.Tensor:
            assert not is_relative
            return self._pose_w

    identity_quaternion = (0.0, 0.0, 0.0, 1.0)
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

    reference_env = DummyEnv(num_envs=1)
    reference_object = DummyAsset(
        "reference_object",
        AxisAlignedBoundingBox(min_point=(-0.1, -0.1, -0.1), max_point=(0.1, 0.1, 0.1)),
        torch.tensor([[0.8, 0.0, 0.2, *identity_quaternion]]),
    )
    reference_container = DummyAsset(
        "reference_container",
        AxisAlignedBoundingBox(min_point=(-1.0, -0.5, 0.0), max_point=(1.0, 0.5, 0.4)),
        torch.tensor([[0.0, 0.0, 0.0, *yaw_90_quaternion]]),
        pose_relative_to_parent=Pose(rotation_xyzw=yaw_90_quaternion),
    )
    reference_env.scene[reference_object.name] = object()
    reference_env.scene[reference_container.name] = object()
    torch.testing.assert_close(
        object_in_container(
            reference_env,
            ArenaAssetHandle(reference_object),
            ArenaAssetHandle(reference_container),
        ),
        torch.tensor([True]),
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
