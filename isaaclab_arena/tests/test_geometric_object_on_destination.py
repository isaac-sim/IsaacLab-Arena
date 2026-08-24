# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from types import SimpleNamespace
from unittest.mock import patch

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _check_bounds_center_over_destination(spatial, pose_frame_aabb_type) -> None:
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
    object_bounds = pose_frame_aabb_type(
        lower=torch.tensor([0.1, -0.1, -0.1]).expand(num_cases, 3),
        upper=torch.tensor([0.3, 0.1, 0.1]).expand(num_cases, 3),
    )
    destination_bounds = pose_frame_aabb_type(
        lower=torch.tensor([-1.0, -0.5, 0.0]).expand(num_cases, 3),
        upper=torch.tensor([1.0, 0.5, 0.4]).expand(num_cases, 3),
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


def _check_geometric_term(spatial, pose_frame_aabb_type, scene_entity_cfg_type, termination_term_cfg_type) -> None:
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

    class DummyRigidObject:
        def __init__(self, pose_w: torch.Tensor, linear_velocity_w: torch.Tensor):
            self.data = SimpleNamespace(root_pose_w=pose_w, root_lin_vel_w=linear_velocity_w)

    class DummyContactSensor:
        def __init__(self, contact_force_w: torch.Tensor):
            self.data = SimpleNamespace(force_matrix_w=contact_force_w[:, None, None, :])

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
    destination_cfg = scene_entity_cfg_type("destination")
    contact_sensor_cfg = scene_entity_cfg_type("contact_sensor")
    geometry_build_calls = []

    def fake_build_entity_pose_frame_aabbs(_env, entity_name):
        geometry_build_calls.append(entity_name)
        if entity_name == "object":
            return pose_frame_aabb_type(
                lower=torch.tensor([-0.1, -0.1, -0.1]).expand(4, 3),
                upper=torch.tensor([0.1, 0.1, 0.1]).expand(4, 3),
            )
        return pose_frame_aabb_type(
            lower=torch.tensor([-1.0, -0.5, 0.0]).expand(4, 3),
            upper=torch.tensor([1.0, 0.5, 0.4]).expand(4, 3),
        )

    with patch.object(spatial, "build_entity_pose_frame_aabbs", side_effect=fake_build_entity_pose_frame_aabbs):
        term_cfg = termination_term_cfg_type(
            func=spatial.GeometricObjectOnDestinationTerm,
            params={
                "object_cfg": object_cfg,
                "destination_cfg": destination_cfg,
                "contact_sensor_cfg": contact_sensor_cfg,
                "force_threshold": 0.1,
                "velocity_threshold": 0.1,
                "support_cone_half_angle_deg": 45.0,
            },
        )
        term = spatial.GeometricObjectOnDestinationTerm(term_cfg, env)
        assert geometry_build_calls == ["object", "destination"]

        # Each failing environment isolates one condition: geometry, force direction, or velocity.
        torch.testing.assert_close(
            term(env, **term_cfg.params),
            torch.tensor([True, False, False, False]),
        )

        # Pose remains live while geometry remains cached.
        object_entity.data.root_pose_w[0, 0] = 2.0
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


def _test_geometric_object_on_destination(_simulation_app) -> bool:
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    import isaaclab_arena.tasks.predicates.spatial as spatial
    from isaaclab_arena.tasks.predicates.live_scene_geometry import PoseFrameAabb

    _check_bounds_center_over_destination(spatial, PoseFrameAabb)
    _check_upward_support_force(spatial)
    _check_geometric_term(spatial, PoseFrameAabb, SceneEntityCfg, TerminationTermCfg)
    return True


def test_geometric_object_on_destination():
    assert run_function_with_persistent_simulation_app(_test_geometric_object_on_destination)


if __name__ == "__main__":
    test_geometric_object_on_destination()
