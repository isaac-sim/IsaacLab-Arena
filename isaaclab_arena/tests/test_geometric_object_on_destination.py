# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from types import SimpleNamespace
from unittest.mock import patch

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _check_bounds_center_over_destination(spatial, axis_aligned_bounding_box_type) -> None:
    """Exercise translation, rotation, open-top behavior, and offset object bounds."""
    identity_quaternion = (0.0, 0.0, 0.0, 1.0)
    yaw_90_quaternion = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))

    object_pose_w = torch.tensor([
        [-0.2, 0.0, 0.2, *identity_quaternion],  # offset bounds center lands at destination center
        [0.81, 0.0, 0.2, *identity_quaternion],  # outside X
        [-0.2, 0.0, -0.01, *identity_quaternion],  # below destination bottom
        [-0.2, 0.0, 2.0, *identity_quaternion],  # above destination top, which is intentionally open
        [-0.2, 0.8, 0.2, *identity_quaternion],  # rotated destination contains center
        [0.4, 0.0, 0.2, *identity_quaternion],  # rotated destination excludes center
        [0.8, 0.0, 0.2, *identity_quaternion],  # exactly on X boundary
        [0.0, 0.31, 0.2, *yaw_90_quaternion],  # rotated object bounds center is outside Y
    ])
    destination_pose_w = torch.tensor([
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *yaw_90_quaternion],
        [0.0, 0.0, 0.0, *yaw_90_quaternion],
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *identity_quaternion],
    ])
    num_cases = object_pose_w.shape[0]
    object_bounds = axis_aligned_bounding_box_type(
        min_point=torch.tensor([0.1, -0.1, -0.1]).expand(num_cases, 3),
        max_point=torch.tensor([0.3, 0.1, 0.1]).expand(num_cases, 3),
    )
    destination_bounds = axis_aligned_bounding_box_type(
        min_point=torch.tensor([-1.0, -0.5, 0.0]).expand(num_cases, 3),
        max_point=torch.tensor([1.0, 0.5, 0.4]).expand(num_cases, 3),
    )

    result = spatial.object_bounds_center_over_destination(
        object_pose_w=object_pose_w,
        object_bounds=object_bounds,
        destination_pose_w=destination_pose_w,
        destination_bounds=destination_bounds,
    )
    torch.testing.assert_close(result, torch.tensor([True, False, False, True, True, False, True, False]))


def _check_upward_support_force(spatial) -> None:
    """Exercise the force threshold, sign, and support cone boundary."""
    contact_force_w = torch.tensor([
        [0.0, 0.0, 0.05],  # below magnitude threshold
        [0.0, 0.0, 0.1],  # exactly at magnitude threshold
        [0.0, 0.0, 0.2],  # straight up
        [0.2, 0.0, 0.0],  # horizontal
        [0.0, 0.0, -0.2],  # downward
        [1.0, 0.0, 1.0],  # exactly on 45-degree cone boundary
        [1.01, 0.0, 1.0],  # just outside cone
        [0.0, 0.0, 1.0],  # straight up and well above threshold
    ])
    result = spatial.contact_force_is_upward_support(
        contact_force_w=contact_force_w,
        force_threshold=0.1,
        support_cone_half_angle_deg=45.0,
    )
    torch.testing.assert_close(result, torch.tensor([False, True, True, False, False, True, False, True]))


def _check_aabb_relative_to_prim(live_scene_geometry) -> None:
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

    bounds = live_scene_geometry._compute_aabb_relative_to_prim(reference.GetPrim())
    torch.testing.assert_close(bounds.min_point[0], torch.tensor([1.0, -1.5, -2.0]))
    torch.testing.assert_close(bounds.max_point[0], torch.tensor([3.0, 1.5, 2.0]))


def _check_reference_pose_row_order(live_scene_geometry) -> None:
    """Check concrete-path and Newton clone-plan rows are reordered by exact environment id."""
    num_envs = 12
    environment_prim_paths = [f"/World/envs/env_{environment_id}" for environment_id in range(num_envs)]
    scene = SimpleNamespace(
        env_prim_paths=environment_prim_paths,
        clone_plan=object(),
    )
    env = SimpleNamespace(scene=scene, num_envs=num_envs, device="cpu")
    reference_prim_path = "/World/envs/env_.*/destination"

    pose_row_environment_ids = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
    expected_pose_rows = torch.tensor(
        [pose_row_environment_ids.index(environment_id) for environment_id in range(num_envs)]
    )

    concrete_path_view = SimpleNamespace(
        count=num_envs,
        prim_paths=[
            f"{environment_prim_paths[environment_id]}/destination" for environment_id in pose_row_environment_ids
        ],
    )
    concrete_path_rows = live_scene_geometry._get_reference_pose_row_order(
        env,
        "destination",
        reference_prim_path,
        concrete_path_view,
    )
    torch.testing.assert_close(concrete_path_rows, expected_pose_rows)

    clone_matches = [
        ("source_a", "destination_a", "representative_a", tuple(pose_row_environment_ids[:3])),
        ("source_b", "destination_b", "representative_b", tuple(pose_row_environment_ids[3:])),
    ]
    newton_view_without_paths = SimpleNamespace(count=num_envs)
    with patch.object(live_scene_geometry, "iter_clone_plan_matches", return_value=clone_matches):
        newton_rows = live_scene_geometry._get_reference_pose_row_order(
            env,
            "destination",
            reference_prim_path,
            newton_view_without_paths,
        )
    torch.testing.assert_close(newton_rows, expected_pose_rows)


def _check_articulation_body_pose(
    live_scene_geometry,
    scene_entity_cfg_type,
) -> None:
    """Check that an articulation pose reader follows exactly the selected body."""

    class RuntimeBufferDouble:
        def __init__(self, tensor: torch.Tensor):
            self.torch = tensor

    body_pose_w = torch.arange(3 * 2 * 7, dtype=torch.float32).reshape(3, 2, 7)
    articulation = SimpleNamespace(
        num_bodies=2,
        data=SimpleNamespace(body_pose_w=RuntimeBufferDouble(body_pose_w)),
    )

    class DummyScene:
        rigid_objects = {}
        articulations = {"microwave": articulation}
        extras = {}

        def __getitem__(self, entity_name):
            return self.articulations[entity_name]

    scene = DummyScene()
    env = SimpleNamespace(scene=scene, num_envs=3, device="cpu")
    destination_pose_cfg = scene_entity_cfg_type("microwave", body_names=["turntable"], body_ids=[1])
    pose_reader = live_scene_geometry.SceneEntityPoseReader(env, destination_pose_cfg)
    torch.testing.assert_close(pose_reader.get_pose_w(), body_pose_w[:, 1, :])


def _check_geometric_term(
    geometric_term_module,
    axis_aligned_bounding_box_type,
    scene_entity_cfg_type,
    termination_term_cfg_type,
) -> None:
    """Check combined results, live-state reads, cached bounds, and cached entity identity."""

    class DummyScene(dict):
        def __init__(self):
            super().__init__()
            self.rigid_objects = {}
            self.articulations = {}
            self.extras = {}

    class DummyEnv:
        def __init__(self):
            self.num_envs = 4
            self.device = "cpu"
            self.scene = DummyScene()

    class RuntimeBufferDouble:
        def __init__(self, tensor: torch.Tensor):
            self.torch = tensor

    class DummyRigidObject:
        def __init__(self, pose_w: torch.Tensor, linear_velocity_w: torch.Tensor):
            self.data = SimpleNamespace(
                root_pose_w=RuntimeBufferDouble(pose_w),
                root_lin_vel_w=RuntimeBufferDouble(linear_velocity_w),
            )

    class DummyContactSensor:
        def __init__(self, contact_force_w: torch.Tensor):
            self.data = SimpleNamespace(force_matrix_w=RuntimeBufferDouble(contact_force_w[:, None, None, :]))

    identity_quaternion = (0.0, 0.0, 0.0, 1.0)
    env = DummyEnv()
    object_pose_w = torch.tensor([
        [0.0, 0.0, 0.2, *identity_quaternion],
        [1.1, 0.0, 0.2, *identity_quaternion],
        [0.0, 0.0, 0.2, *identity_quaternion],
        [0.0, 0.0, 0.2, *identity_quaternion],
    ])
    destination_pose_w = torch.tensor([[0.0, 0.0, 0.0, *identity_quaternion]]).expand(4, 7)
    object_linear_velocity_w = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    contact_force_w = torch.tensor([
        [0.0, 0.0, 0.2],
        [0.0, 0.0, 0.2],
        [0.2, 0.0, 0.0],
        [0.0, 0.0, 0.2],
    ])

    object_entity = DummyRigidObject(object_pose_w, object_linear_velocity_w)
    destination_entity = DummyRigidObject(destination_pose_w, torch.zeros((4, 3)))
    env.scene["object"] = object_entity
    env.scene["destination"] = destination_entity
    env.scene["contact_sensor"] = DummyContactSensor(contact_force_w)
    env.scene.rigid_objects.update({"object": object_entity, "destination": destination_entity})

    coarse_contact_and_velocity_result = (torch.linalg.vector_norm(contact_force_w, dim=-1) > 0.1) & (
        torch.linalg.vector_norm(object_linear_velocity_w, dim=-1) < 0.1
    )
    torch.testing.assert_close(coarse_contact_and_velocity_result, torch.tensor([True, True, True, False]))

    object_cfg = scene_entity_cfg_type("object")
    destination_pose_cfg = scene_entity_cfg_type("destination")
    contact_sensor_cfg = scene_entity_cfg_type("contact_sensor")
    geometry_build_calls = []

    def compute_aabbs(_geometry_env, pose_entity_cfg, geometry_prim_path=None):
        entity_name = pose_entity_cfg.name
        geometry_build_calls.append((entity_name, geometry_prim_path))
        if entity_name == "object":
            return axis_aligned_bounding_box_type(
                min_point=torch.tensor([-0.1, -0.1, -0.1]).expand(4, 3),
                max_point=torch.tensor([0.1, 0.1, 0.1]).expand(4, 3),
            )
        return axis_aligned_bounding_box_type(
            min_point=torch.tensor([-1.0, -0.5, 0.0]).expand(4, 3),
            max_point=torch.tensor([1.0, 0.5, 0.4]).expand(4, 3),
        )

    with patch.object(
        geometric_term_module,
        "compute_spawned_geometry_aabbs_relative_to_pose",
        side_effect=compute_aabbs,
    ):
        term_cfg = termination_term_cfg_type(
            func=geometric_term_module.GeometricObjectOnDestinationTerm,
            params={
                "object_cfg": object_cfg,
                "destination_pose_cfg": destination_pose_cfg,
                "destination_prim_path": "/World/envs/env_.*/destination",
                "contact_sensor_cfg": contact_sensor_cfg,
                "force_threshold": 0.1,
                "velocity_threshold": 0.1,
                "support_cone_half_angle_deg": 45.0,
            },
        )
        term = geometric_term_module.GeometricObjectOnDestinationTerm(term_cfg, env)
        assert geometry_build_calls == [
            ("object", None),
            ("destination", "/World/envs/env_.*/destination"),
        ]

        # Each failing environment isolates one condition: geometry, force direction, or velocity.
        torch.testing.assert_close(
            term(env, **term_cfg.params),
            torch.tensor([True, False, False, False]),
        )

        # Pose remains live while geometry remains cached.
        object_entity.data.root_pose_w.torch[0, 0] = 2.0
        assert not term(env, **term_cfg.params)[0]
        assert geometry_build_calls == [
            ("object", None),
            ("destination", "/World/envs/env_.*/destination"),
        ]

        mismatched_params = dict(term_cfg.params)
        mismatched_params["object_cfg"] = scene_entity_cfg_type("another_object")
        try:
            term(env, **mismatched_params)
        except AssertionError as error:
            assert "cached geometry for object 'object'" in str(error)
        else:
            raise AssertionError("The term accepted an object that did not match its cached geometry.")


def _test_geometric_object_on_destination(_simulation_app) -> bool:
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    import isaaclab_arena.tasks.predicates.geometric_object_on_destination_term as geometric_term
    import isaaclab_arena.tasks.predicates.live_scene_geometry as live_scene_geometry
    import isaaclab_arena.tasks.predicates.spatial as spatial
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    _check_bounds_center_over_destination(spatial, AxisAlignedBoundingBox)
    _check_upward_support_force(spatial)
    _check_aabb_relative_to_prim(live_scene_geometry)
    _check_reference_pose_row_order(live_scene_geometry)
    _check_articulation_body_pose(live_scene_geometry, SceneEntityCfg)
    _check_geometric_term(geometric_term, AxisAlignedBoundingBox, SceneEntityCfg, TerminationTermCfg)
    return True


def test_geometric_object_on_destination():
    assert run_function_with_persistent_simulation_app(_test_geometric_object_on_destination)


if __name__ == "__main__":
    test_geometric_object_on_destination()
