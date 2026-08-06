# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import math
import torch
from functools import partial

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_object_in_container_predicate(_simulation_app) -> bool:
    from types import SimpleNamespace

    import isaaclab.sim as sim_utils
    import warp as wp
    from isaaclab.cloner import ClonePlan
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import (
        make_progress_tracking_events_cfg,
        make_progress_tracking_recorder_cfg,
    )
    from isaaclab_arena.tasks.predicates.spatial import (
        contact_force_is_upward_support,
        object_in_container,
        object_on_destination,
    )
    from isaaclab_arena.tasks.predicates.spatial_manager_terms import ArenaAssetHandle, object_on_destination_term
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    class DummyScene(dict):
        def __init__(self, num_envs: int):
            super().__init__()
            self.env_origins = torch.zeros((num_envs, 3))
            self.env_regex_ns = "/World/envs/env_.*"
            self.env_fmt = "/World/envs/env_{}"

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
    destination_asset_handle = ArenaAssetHandle(container_asset)
    manager_term_cfg = TerminationTermCfg(
        func=object_on_destination_term,
        params={
            "object_asset_handle": object_asset_handle,
            "destination_asset_handle": destination_asset_handle,
        },
    )
    manager_term_cfg.validate()
    copied_manager_term_cfg = copy.deepcopy(manager_term_cfg)
    assert copied_manager_term_cfg.params["object_asset_handle"] is object_asset_handle
    assert copied_manager_term_cfg.params["destination_asset_handle"] is destination_asset_handle
    assert not hasattr(object_asset_handle, "__dict__")

    progress_objective = ProgressObjective(
        name="asset_handle_copy",
        predicate_groups=[
            partial(
                object_on_destination_term,
                object_asset_handle=object_asset_handle,
                destination_asset_handle=destination_asset_handle,
            )
        ],
    )
    progress_events_cfg = make_progress_tracking_events_cfg([progress_objective])
    progress_recorder_cfg = make_progress_tracking_recorder_cfg([progress_objective])
    copied_event_objective = progress_events_cfg.reset_progress_objectives.params["progress_objectives"][0]
    copied_recorder_objective = progress_recorder_cfg.progress_tracking.progress_objectives[0]
    for copied_progress_objective in (copied_event_objective, copied_recorder_objective):
        copied_predicate = next(iter(copied_progress_objective.canonical_predicate_groups.values()))[0][0]
        assert copied_predicate.keywords["object_asset_handle"] is object_asset_handle
        assert copied_predicate.keywords["destination_asset_handle"] is destination_asset_handle

    torch.testing.assert_close(
        object_in_container(geometry_env, object_asset, container_asset),
        torch.tensor([True, False, False, True, True, False, True, False]),
    )

    object_set_env = DummyEnv(num_envs=4)
    object_set_members = [
        DummyAsset(
            f"member_{member_index}",
            AxisAlignedBoundingBox(
                min_point=(float(member_index), 0.0, 0.0),
                max_point=(float(member_index + 1), 1.0, 1.0),
            ),
            torch.zeros((4, 7)),
        )
        for member_index in range(3)
    ]
    object_set = RigidObjectSet.__new__(RigidObjectSet)
    object_set.name = "object_set"
    object_set.objects = object_set_members
    object_set.member_usd_paths = [f"/member_{member_index}.usd" for member_index in range(3)]

    object_set_spawn_cfg = sim_utils.MultiUsdFileCfg(
        usd_path=[object_set.member_usd_paths[2], object_set.member_usd_paths[2], object_set.member_usd_paths[0]],
    )
    object_set_env.scene[object_set.name] = SimpleNamespace(
        cfg=SimpleNamespace(
            prim_path="/World/envs/env_.*/object_set",
            spawn=object_set_spawn_cfg,
        )
    )
    object_set_env.scene.clone_plan = ClonePlan(
        sources=(
            "/World/envs/env_0/unrelated",
            "/World/envs/env_1/object_set",
            "/World/envs/env_2/object_set",
            "/World/envs/env_0/object_set",
        ),
        destinations=(
            "/World/envs/env_{}/unrelated",
            "/World/envs/env_{}/object_set",
            "/World/envs/env_{}/object_set",
            "/World/envs/env_{}/object_set",
        ),
        clone_mask=torch.tensor([
            [True, True, True, True],
            [False, True, False, True],
            [False, False, False, False],
            [True, False, True, False],
        ]),
    )
    spawned_bounding_boxes = object_set.get_spawned_bounding_box_per_env(object_set_env)
    torch.testing.assert_close(
        spawned_bounding_boxes.min_point,
        torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]),
    )
    torch.testing.assert_close(
        spawned_bounding_boxes.max_point,
        torch.tensor([
            [1.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
        ]),
    )
    single_variant_env = DummyEnv(num_envs=4)
    single_variant_env.scene[object_set.name] = SimpleNamespace(
        cfg=SimpleNamespace(
            prim_path="/World/envs/env_.*/object_set",
            spawn=sim_utils.MultiUsdFileCfg(usd_path=[object_set.member_usd_paths[2]]),
        )
    )
    single_variant_env.scene.clone_plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 4), dtype=torch.bool),
    )
    single_variant_bounding_box = object_set.get_spawned_bounding_box_per_env(single_variant_env)
    torch.testing.assert_close(
        single_variant_bounding_box.min_point,
        object_set_members[2].get_bounding_box().min_point.expand(4, 3),
    )
    torch.testing.assert_close(
        single_variant_bounding_box.max_point,
        object_set_members[2].get_bounding_box().max_point.expand(4, 3),
    )
    single_member_object_set = RigidObjectSet.__new__(RigidObjectSet)
    single_member_object_set.objects = [object_set_members[1]]
    single_member_bounding_boxes = single_member_object_set.get_spawned_bounding_box_per_env(object_set_env)
    torch.testing.assert_close(
        single_member_bounding_boxes.min_point,
        object_set_members[1].get_bounding_box().min_point.expand(4, 3),
    )
    torch.testing.assert_close(
        single_member_bounding_boxes.max_point,
        object_set_members[1].get_bounding_box().max_point.expand(4, 3),
    )

    reference_env = DummyEnv(num_envs=2)
    reference_env.scene.env_origins = torch.tensor([
        [10.0, 20.0, 30.0],
        [14.0, 17.0, 31.0],
    ])

    class DummyFrameView:
        def __init__(self, pose_w: torch.Tensor):
            self._pose_w = pose_w

        def get_world_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
            return self._pose_w[:, :3], self._pose_w[:, 3:]

    rotated_reference = ObjectReference.__new__(ObjectReference)
    rotated_reference.name = "rotated_reference"
    rotated_reference.object_type = ObjectType.BASE
    rotated_reference.initial_pose_relative_to_parent = Pose(
        position_xyz=(0.0, 1.0, 0.0),
        rotation_xyzw=yaw_90_quaternion,
    )
    expected_rotated_reference_pose = torch.tensor([
        [1.0, 2.0, 4.0, 0.5, -0.5, 0.5, 0.5],
        [1.0, 2.0, 4.0, 0.5, -0.5, 0.5, 0.5],
    ])
    rotated_reference_pose_w = expected_rotated_reference_pose[:1].clone()
    rotated_reference_pose_w[:, :3] += reference_env.scene.env_origins[:1]
    reference_env.scene[rotated_reference.name] = DummyFrameView(rotated_reference_pose_w)
    expected_rotated_reference_bounding_box_pose = expected_rotated_reference_pose.clone()
    expected_rotated_reference_bounding_box_pose[:, 3:] = torch.tensor(roll_90_quaternion)
    torch.testing.assert_close(
        rotated_reference.get_bounding_box_pose(reference_env),
        expected_rotated_reference_bounding_box_pose,
    )
    expected_rotated_reference_bounding_box_pose_w = expected_rotated_reference_bounding_box_pose.clone()
    expected_rotated_reference_bounding_box_pose_w[:, :3] += reference_env.scene.env_origins
    torch.testing.assert_close(
        rotated_reference.get_bounding_box_pose(reference_env, is_relative=False),
        expected_rotated_reference_bounding_box_pose_w,
    )

    reference_object = DummyAsset(
        "reference_object",
        AxisAlignedBoundingBox(min_point=(-0.1, -0.1, -0.1), max_point=(0.1, 0.1, 0.1)),
        torch.tensor([
            [10.8, 20.0, 30.2, *identity_quaternion],
            [14.8, 17.0, 31.2, *identity_quaternion],
        ]),
    )
    reference_container = ObjectReference.__new__(ObjectReference)
    reference_container.name = "reference_container"
    reference_container.object_type = ObjectType.BASE
    reference_container.initial_pose_relative_to_parent = Pose(rotation_xyzw=yaw_90_quaternion)
    reference_container._bounding_box = AxisAlignedBoundingBox(
        min_point=(-1.0, -0.5, 0.0),
        max_point=(1.0, 0.5, 0.4),
    )
    reference_container_pose_w = torch.tensor([
        [10.0, 20.0, 30.0, *yaw_90_quaternion],
        [14.0, 17.0, 31.0, *yaw_90_quaternion],
    ])
    reference_env.scene[reference_container.name] = DummyFrameView(reference_container_pose_w)
    reference_env.scene[reference_object.name] = object()
    torch.testing.assert_close(
        reference_container.get_bounding_box_pose(reference_env, is_relative=False),
        torch.tensor([
            [10.0, 20.0, 30.0, *identity_quaternion],
            [14.0, 17.0, 31.0, *identity_quaternion],
        ]),
    )
    torch.testing.assert_close(
        object_in_container(
            reference_env,
            reference_object,
            reference_container,
        ),
        torch.tensor([True, True]),
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
    expected_on_destination = torch.tensor([True, False, False, False])
    torch.testing.assert_close(
        object_on_destination(
            combined_env,
            object_cfg=SceneEntityCfg(combined_object_asset.name),
            contact_sensor_cfg=SceneEntityCfg("contact_sensor"),
            force_threshold=0.1,
            velocity_threshold=0.1,
            object_asset=combined_object_asset,
            destination_asset=combined_container_asset,
            support_cone_half_angle_deg=45.0,
        ),
        expected_on_destination,
    )
    torch.testing.assert_close(
        object_on_destination_term(
            combined_env,
            object_cfg=SceneEntityCfg(combined_object_asset.name),
            contact_sensor_cfg=SceneEntityCfg("contact_sensor"),
            force_threshold=0.1,
            velocity_threshold=0.1,
            object_asset_handle=ArenaAssetHandle(combined_object_asset),
            destination_asset_handle=ArenaAssetHandle(combined_container_asset),
            support_cone_half_angle_deg=45.0,
        ),
        expected_on_destination,
    )

    return True


def test_object_in_container_predicate():
    assert run_function_with_persistent_simulation_app(_test_object_in_container_predicate)
