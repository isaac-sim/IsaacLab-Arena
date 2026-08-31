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

    T_W_O = torch.tensor([
        [-0.2, 0.0, 0.2, *identity_quaternion],  # offset bounds center lands at destination center
        [0.81, 0.0, 0.2, *identity_quaternion],  # outside X
        [-0.2, 0.0, -0.01, *identity_quaternion],  # below destination bottom
        [-0.2, 0.0, 2.0, *identity_quaternion],  # above destination top, which is intentionally open
        [-0.2, 0.8, 0.2, *identity_quaternion],  # rotated destination contains center
        [0.4, 0.0, 0.2, *identity_quaternion],  # rotated destination excludes center
        [0.8, 0.0, 0.2, *identity_quaternion],  # exactly on X boundary
        [0.0, 0.31, 0.2, *yaw_90_quaternion],  # rotated object bounds center is outside Y
    ])
    T_W_D = torch.tensor([
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *yaw_90_quaternion],
        [0.0, 0.0, 0.0, *yaw_90_quaternion],
        [0.0, 0.0, 0.0, *identity_quaternion],
        [0.0, 0.0, 0.0, *identity_quaternion],
    ])
    num_cases = T_W_O.shape[0]
    object_bounds_center_O = torch.tensor([0.2, 0.0, 0.0]).expand(num_cases, 3)
    destination_bounds_D = axis_aligned_bounding_box_type(
        min_point=torch.tensor([-1.0, -0.5, 0.0]).expand(num_cases, 3),
        max_point=torch.tensor([1.0, 0.5, 0.4]).expand(num_cases, 3),
    )

    result = spatial.object_bounds_center_over_destination(
        T_W_O=T_W_O,
        object_bounds_center_O=object_bounds_center_O,
        T_W_D=T_W_D,
        destination_bounds_D=destination_bounds_D,
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


def _check_geometry_bounds_in_prim_frame(live_scene_geometry) -> None:
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

    bounds = live_scene_geometry._compute_geometry_bounds_in_prim_frame(reference.GetPrim())
    torch.testing.assert_close(bounds.min_point[0], torch.tensor([1.0, -1.5, -2.0]))
    torch.testing.assert_close(bounds.max_point[0], torch.tensor([3.0, 1.5, 2.0]))


def _check_object_on_destination_term(
    object_on_destination_term_module,
    axis_aligned_bounding_box_type,
    scene_entity_cfg_type,
    termination_term_cfg_type,
) -> None:
    """Check combined results, live-state reads, cached bounds, and cached entity identity."""

    class DummyScene(dict):
        def __init__(self):
            super().__init__()
            self.rigid_objects = {}
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
    T_W_O = torch.tensor([
        [0.0, 0.0, 0.2, *identity_quaternion],
        [1.1, 0.0, 0.2, *identity_quaternion],
        [0.0, 0.0, 0.2, *identity_quaternion],
        [0.0, 0.0, 0.2, *identity_quaternion],
    ])
    T_W_D = torch.tensor([[0.0, 0.0, 0.0, *identity_quaternion]]).expand(4, 7)
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

    object_entity = DummyRigidObject(T_W_O, object_linear_velocity_w)
    destination_entity = DummyRigidObject(T_W_D, torch.zeros((4, 3)))
    env.scene["object"] = object_entity
    env.scene["destination"] = destination_entity
    env.scene["contact_sensor"] = DummyContactSensor(contact_force_w)
    env.scene.rigid_objects.update({"object": object_entity, "destination": destination_entity})

    coarse_contact_and_velocity_result = (torch.linalg.vector_norm(contact_force_w, dim=-1) > 0.1) & (
        torch.linalg.vector_norm(object_linear_velocity_w, dim=-1) < 0.1
    )
    torch.testing.assert_close(coarse_contact_and_velocity_result, torch.tensor([True, True, True, False]))

    object_cfg = scene_entity_cfg_type("object")
    destination_cfg = scene_entity_cfg_type("destination")
    contact_sensor_cfg = scene_entity_cfg_type("contact_sensor")
    geometry_build_calls = []

    def compute_geometry_bounds(_geometry_env, entity_cfg):
        entity_name = entity_cfg.name
        geometry_build_calls.append(entity_name)
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
        object_on_destination_term_module,
        "compute_spawned_geometry_bounds_in_entity_frame",
        side_effect=compute_geometry_bounds,
    ):
        term_cfg = termination_term_cfg_type(
            func=object_on_destination_term_module.ObjectOnDestinationTerm,
            params={
                "object_cfg": object_cfg,
                "destination_cfg": destination_cfg,
                "contact_sensor_cfg": contact_sensor_cfg,
                "force_threshold": 0.1,
                "velocity_threshold": 0.1,
                "support_cone_half_angle_deg": 45.0,
            },
        )
        term = object_on_destination_term_module.ObjectOnDestinationTerm(term_cfg, env)
        assert geometry_build_calls == ["object", "destination"]

        # Each failing environment isolates one condition: geometry, force direction, or velocity.
        torch.testing.assert_close(
            term(env, **term_cfg.params),
            torch.tensor([True, False, False, False]),
        )

        # Pose remains live while geometry remains cached.
        object_entity.data.root_pose_w.torch[0, 0] = 2.0
        assert not term(env, **term_cfg.params)[0]
        assert geometry_build_calls == ["object", "destination"]

        mismatched_params = dict(term_cfg.params)
        mismatched_params["object_cfg"] = scene_entity_cfg_type("another_object")
        try:
            term(env, **mismatched_params)
        except AssertionError as error:
            assert "cached geometry for object 'object'" in str(error)
        else:
            raise AssertionError("The term accepted an object that did not match its cached geometry.")

        mismatched_params = dict(term_cfg.params)
        mismatched_params["destination_cfg"] = scene_entity_cfg_type("another_destination")
        try:
            term(env, **mismatched_params)
        except AssertionError as error:
            assert "cached geometry for destination 'destination'" in str(error)
        else:
            raise AssertionError("The term accepted a destination that did not match its cached geometry.")


def _check_owned_resource_cleanup(
    live_scene_geometry,
    object_on_destination_term_module,
    arena_env_type,
    manager_based_rl_env_type,
) -> None:
    """Check that term-owned views close once and before the termination manager is deleted."""

    class DummyClosable:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    frame_view = DummyClosable()
    pose_reader = object.__new__(live_scene_geometry.AssetBaseCfgPoseReader)
    pose_reader._frame_view = frame_view
    pose_reader._is_closed = False
    pose_reader.close()
    pose_reader.close()
    assert frame_view.close_calls == 1

    term_reader = DummyClosable()
    term = object.__new__(object_on_destination_term_module.ObjectOnDestinationTerm)
    term._destination_asset_base_pose_reader = term_reader
    term.close()
    term.close()
    assert term_reader.close_calls == 1

    closable_term = DummyClosable()

    class DummyTerminationManager:
        active_terms = ["success", "time_out"]

        def get_term_cfg(self, term_name):
            if term_name == "success":
                return SimpleNamespace(func=closable_term)
            return SimpleNamespace(func=lambda: None)

    env = object.__new__(arena_env_type)
    env._is_closed = False
    env.termination_manager = DummyTerminationManager()

    def close_parent(parent_env):
        assert closable_term.close_calls == 1
        parent_env._is_closed = True

    with patch.object(manager_based_rl_env_type, "close", autospec=True, side_effect=close_parent) as parent_close:
        env.close()
        env.close()

    assert closable_term.close_calls == 1
    assert parent_close.call_count == 2


def _test_object_on_destination_term(_simulation_app) -> bool:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    import isaaclab_arena.tasks.predicates.live_scene_geometry as live_scene_geometry
    import isaaclab_arena.tasks.predicates.object_on_destination_term as object_on_destination_term
    import isaaclab_arena.tasks.predicates.spatial as spatial
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    _check_bounds_center_over_destination(spatial, AxisAlignedBoundingBox)
    _check_upward_support_force(spatial)
    _check_geometry_bounds_in_prim_frame(live_scene_geometry)
    _check_object_on_destination_term(
        object_on_destination_term,
        AxisAlignedBoundingBox,
        SceneEntityCfg,
        TerminationTermCfg,
    )
    _check_owned_resource_cleanup(
        live_scene_geometry,
        object_on_destination_term,
        IsaacLabArenaManagerBasedRLEnv,
        ManagerBasedRLEnv,
    )
    return True


def test_object_on_destination_term():
    assert run_function_with_persistent_simulation_app(_test_object_on_destination_term)


if __name__ == "__main__":
    test_object_on_destination_term()
