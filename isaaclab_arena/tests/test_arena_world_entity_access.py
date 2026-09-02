# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify ArenaWorld entity reads and derived entity geometry."""

import torch
from types import SimpleNamespace
from unittest.mock import patch

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _check_geometry_bounds_in_prim_frame(entity_access_module) -> None:
    """Check that runtime bounds remove pose but retain spawned scale."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    wrapper = UsdGeom.Xform.Define(stage, "/World/Wrapper")
    wrapper.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(100.0, -50.0, 3.0))
    wrapper.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(37.0)
    wrapper.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(2.0, 3.0, 4.0))

    reference = UsdGeom.Xform.Define(stage, "/World/Wrapper/Reference")
    cube = UsdGeom.Cube.Define(stage, "/World/Wrapper/Reference/Cube")
    cube.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 0.0, 0.0))

    ignored_cube = UsdGeom.Cube.Define(stage, "/World/Wrapper/Reference/IgnoredCube")
    ignored_cube.GetSizeAttr().Set(100.0)
    ignored_cube.GetPurposeAttr().Set(UsdGeom.Tokens.render)

    bounds = entity_access_module._compute_geometry_bounds_in_prim_frame(reference.GetPrim())
    torch.testing.assert_close(bounds.min_point[0], torch.tensor([1.0, -1.5, -2.0]))
    torch.testing.assert_close(bounds.max_point[0], torch.tensor([3.0, 1.5, 2.0]))


def _check_rigid_object_reads_and_local_aabb_cache(
    arena_world_module,
    entity_access_module,
    axis_aligned_bounding_box_type,
) -> None:
    """Check live rigid-object reads and one cached AABB per entity name."""

    class RuntimeBufferDouble:
        def __init__(self, tensor: torch.Tensor):
            self.torch = tensor

    class RigidObjectDouble:
        def __init__(self, pose_w: torch.Tensor, linear_velocity_w: torch.Tensor):
            self.data = SimpleNamespace(
                root_pose_w=RuntimeBufferDouble(pose_w),
                root_lin_vel_w=RuntimeBufferDouble(linear_velocity_w),
            )

    class SceneDouble:
        def __init__(self):
            self.num_envs = 2
            self.device = "cpu"
            self.rigid_objects = {
                "object": RigidObjectDouble(
                    pose_w=torch.tensor([
                        [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
                    ]),
                    linear_velocity_w=torch.tensor([
                        [0.0, 0.0, 0.0],
                        [0.1, 0.2, 0.3],
                    ]),
                ),
                "destination": RigidObjectDouble(
                    pose_w=torch.tensor([
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ]),
                    linear_velocity_w=torch.zeros((2, 3)),
                ),
            }
            self.extras = {}

    scene = SceneDouble()
    arena_world = arena_world_module.ArenaWorld(scene)
    geometry_build_calls: list[str] = []

    def compute_geometry_bounds(_scene, entity_name: str):
        geometry_build_calls.append(entity_name)
        if entity_name == "object":
            return axis_aligned_bounding_box_type(
                min_point=torch.tensor([-0.1, -0.1, -0.1]).expand(2, 3),
                max_point=torch.tensor([0.1, 0.1, 0.1]).expand(2, 3),
            )
        return axis_aligned_bounding_box_type(
            min_point=torch.tensor([-1.0, -0.5, 0.0]).expand(2, 3),
            max_point=torch.tensor([1.0, 0.5, 0.4]).expand(2, 3),
        )

    initial_pose_w = scene.rigid_objects["object"].data.root_pose_w.torch.clone()
    initial_linear_velocity_w = scene.rigid_objects["object"].data.root_lin_vel_w.torch.clone()
    torch.testing.assert_close(arena_world.get_pose_w("object"), initial_pose_w)
    torch.testing.assert_close(arena_world.get_linear_velocity_w("object"), initial_linear_velocity_w)

    moved_pose_w = initial_pose_w.clone()
    moved_pose_w[:, 0] += 0.25
    changed_linear_velocity_w = initial_linear_velocity_w + 0.5
    scene.rigid_objects["object"].data.root_pose_w.torch = moved_pose_w
    scene.rigid_objects["object"].data.root_lin_vel_w.torch = changed_linear_velocity_w
    torch.testing.assert_close(arena_world.get_pose_w("object"), moved_pose_w)
    torch.testing.assert_close(arena_world.get_linear_velocity_w("object"), changed_linear_velocity_w)

    with patch.object(
        entity_access_module,
        "compute_spawned_geometry_bounds_in_entity_frame",
        side_effect=compute_geometry_bounds,
    ):
        object_bounds = arena_world.get_local_aabb("object")
        destination_bounds = arena_world.get_local_aabb("destination")
        assert arena_world.get_local_aabb("object") is object_bounds
        assert arena_world.get_local_aabb("destination") is destination_bounds

    assert object_bounds is not destination_bounds
    assert geometry_build_calls == ["object", "destination"]


def _check_arena_world_reuses_asset_base_cfg_pose_reader(
    arena_world_module,
    entity_access_module,
) -> None:
    """Check that ArenaWorld caches the reader while returning its latest pose."""

    class SceneDouble:
        def __init__(self):
            self.rigid_objects = {}
            self.extras = {"reference": object()}

    class PoseReaderDouble:
        def __init__(self):
            self.read_count = 0
            self.pose_values = [
                torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            ]

        def get_pose_w(self):
            pose_w = self.pose_values[self.read_count]
            self.read_count += 1
            return pose_w

    scene = SceneDouble()
    pose_reader = PoseReaderDouble()
    with patch.object(entity_access_module, "AssetBaseCfgPoseReader", return_value=pose_reader) as make_pose_reader:
        arena_world = arena_world_module.ArenaWorld(scene)
        first_pose_w = arena_world.get_pose_w("reference")
        second_pose_w = arena_world.get_pose_w("reference")

    make_pose_reader.assert_called_once_with(scene, "reference")
    assert pose_reader.read_count == 2
    torch.testing.assert_close(first_pose_w, pose_reader.pose_values[0])
    torch.testing.assert_close(second_pose_w, pose_reader.pose_values[1])


def _check_asset_base_cfg_pose_reader_uses_current_frame_view_poses(entity_access_module) -> None:
    """Check FrameView construction and live pose reads in environment row order."""

    class FrameViewDouble:
        def __init__(self):
            self.prim_paths = [
                "/World/envs/env_0/reference",
                "/World/envs/env_1/reference",
            ]
            self.read_count = 0
            self.position_values = [
                torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]]),
            ]
            self.orientation_values = [
                torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
                torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
            ]

        def get_world_poses(self):
            position_w = self.position_values[self.read_count]
            orientation_w = self.orientation_values[self.read_count]
            self.read_count += 1
            return SimpleNamespace(torch=position_w), SimpleNamespace(torch=orientation_w)

    scene = SimpleNamespace(
        num_envs=2,
        device="cpu",
        stage=object(),
        extras={"reference": object()},
        cfg=SimpleNamespace(reference=SimpleNamespace(prim_path="{ENV_REGEX_NS}/reference")),
        env_regex_ns="/World/envs/env_.*",
        env_prim_paths=["/World/envs/env_0", "/World/envs/env_1"],
    )
    frame_view = FrameViewDouble()
    with patch.object(entity_access_module, "FrameView", return_value=frame_view) as make_frame_view:
        pose_reader = entity_access_module.AssetBaseCfgPoseReader(scene, "reference")
        first_pose_w = pose_reader.get_pose_w()
        second_pose_w = pose_reader.get_pose_w()

    make_frame_view.assert_called_once_with(
        "/World/envs/env_.*/reference",
        device="cpu",
        stage=scene.stage,
        validate_xform_ops=False,
    )
    assert frame_view.read_count == 2
    torch.testing.assert_close(
        first_pose_w,
        torch.cat((frame_view.position_values[0], frame_view.orientation_values[0]), dim=-1),
    )
    torch.testing.assert_close(
        second_pose_w,
        torch.cat((frame_view.position_values[1], frame_view.orientation_values[1]), dim=-1),
    )


def _test_arena_world_entity_access(_simulation_app) -> bool:
    import isaaclab_arena.environments.arena_world as arena_world
    import isaaclab_arena.environments.arena_world_entity_access as entity_access
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    _check_geometry_bounds_in_prim_frame(entity_access)
    _check_rigid_object_reads_and_local_aabb_cache(
        arena_world,
        entity_access,
        AxisAlignedBoundingBox,
    )
    _check_arena_world_reuses_asset_base_cfg_pose_reader(arena_world, entity_access)
    _check_asset_base_cfg_pose_reader_uses_current_frame_view_poses(entity_access)
    return True


def test_arena_world_entity_access():
    assert run_function_with_persistent_simulation_app(_test_arena_world_entity_access)


if __name__ == "__main__":
    test_arena_world_entity_access()
