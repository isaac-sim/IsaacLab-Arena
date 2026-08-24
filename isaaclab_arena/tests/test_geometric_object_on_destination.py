# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_geometric_object_on_destination(_simulation_app) -> bool:
    from types import SimpleNamespace

    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    import isaaclab_arena.tasks.predicates.spatial as spatial

    identity_quaternion = (0.0, 0.0, 0.0, 1.0)
    yaw_90_quaternion = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))

    object_pose_w = torch.tensor([
        [-0.2, 0.0, 0.2, *identity_quaternion],
        [0.81, 0.0, 0.2, *identity_quaternion],
        [-0.2, 0.0, -0.01, *identity_quaternion],
        [-0.2, 0.0, 2.0, *identity_quaternion],
        [-0.2, 0.8, 0.2, *identity_quaternion],
        [0.4, 0.0, 0.2, *identity_quaternion],
        [0.8, 0.0, 0.2, *identity_quaternion],
        [0.0, 0.31, 0.2, *yaw_90_quaternion],
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
    num_geometry_cases = object_pose_w.shape[0]
    object_aabb_lower = torch.tensor([0.1, -0.1, -0.1]).expand(num_geometry_cases, 3)
    object_aabb_upper = torch.tensor([0.3, 0.1, 0.1]).expand(num_geometry_cases, 3)
    destination_aabb_lower = torch.tensor([-1.0, -0.5, 0.0]).expand(num_geometry_cases, 3)
    destination_aabb_upper = torch.tensor([1.0, 0.5, 0.4]).expand(num_geometry_cases, 3)
    torch.testing.assert_close(
        spatial.object_centroid_in_open_top_bounds(
            object_pose_w=object_pose_w,
            object_aabb_lower=object_aabb_lower,
            object_aabb_upper=object_aabb_upper,
            destination_pose_w=destination_pose_w,
            destination_aabb_lower=destination_aabb_lower,
            destination_aabb_upper=destination_aabb_upper,
        ),
        torch.tensor([True, False, False, True, True, False, True, False]),
    )

    contact_force_w = torch.tensor([
        [0.0, 0.0, 0.05],
        [0.0, 0.0, 0.1],
        [0.0, 0.0, 0.2],
        [0.2, 0.0, 0.0],
        [0.0, 0.0, -0.2],
        [1.0, 0.0, 1.0],
        [1.01, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    torch.testing.assert_close(
        spatial.contact_force_is_upward_support(
            contact_force_w=contact_force_w,
            force_threshold=0.1,
            support_cone_half_angle_deg=45.0,
        ),
        torch.tensor([False, True, True, False, False, True, False, True]),
    )

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
        def __init__(self, pose_w: torch.Tensor, velocity_w: torch.Tensor):
            self.data = SimpleNamespace(root_pose_w=pose_w, root_lin_vel_w=velocity_w)

    class DummyContactSensor:
        def __init__(self, force_w: torch.Tensor):
            self.data = SimpleNamespace(force_matrix_w=force_w[:, None, None, :])

    combined_env = DummyEnv()
    combined_object_pose_w = torch.tensor([
        [0.0, 0.0, 0.2, *identity_quaternion],
        [1.1, 0.0, 0.2, *identity_quaternion],
        [0.0, 0.0, 0.2, *identity_quaternion],
        [0.0, 0.0, 0.2, *identity_quaternion],
    ])
    combined_destination_pose_w = torch.tensor([[0.0, 0.0, 0.0, *identity_quaternion]]).expand(4, 7)
    combined_velocity_w = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    combined_force_w = torch.tensor([
        [0.0, 0.0, 0.2],
        [0.0, 0.0, 0.2],
        [0.2, 0.0, 0.0],
        [0.0, 0.0, 0.2],
    ])
    combined_object = DummyRigidObject(combined_object_pose_w, combined_velocity_w)
    combined_destination = DummyRigidObject(combined_destination_pose_w, torch.zeros((4, 3)))
    combined_env.scene["object"] = combined_object
    combined_env.scene["destination"] = combined_destination
    combined_env.scene["contact_sensor"] = DummyContactSensor(combined_force_w)
    combined_env.scene.rigid_objects.update({"object": combined_object, "destination": combined_destination})

    legacy_contact_and_velocity = (torch.linalg.vector_norm(combined_force_w, dim=-1) > 0.1) & (
        torch.linalg.vector_norm(combined_velocity_w, dim=-1) < 0.1
    )
    torch.testing.assert_close(legacy_contact_and_velocity, torch.tensor([True, True, True, False]))

    object_cfg = SceneEntityCfg("object")
    destination_cfg = SceneEntityCfg("destination")
    contact_sensor_cfg = SceneEntityCfg("contact_sensor")
    geometry_build_calls = []
    original_get_pose_frame_aabb = spatial._get_pose_frame_aabb

    def fake_get_pose_frame_aabb(_env, entity_cfg):
        geometry_build_calls.append(entity_cfg.name)
        if entity_cfg.name == "object":
            return torch.tensor([-0.1, -0.1, -0.1]).expand(4, 3), torch.tensor([0.1, 0.1, 0.1]).expand(4, 3)
        return torch.tensor([-1.0, -0.5, 0.0]).expand(4, 3), torch.tensor([1.0, 0.5, 0.4]).expand(4, 3)

    spatial._get_pose_frame_aabb = fake_get_pose_frame_aabb
    try:
        term_cfg = TerminationTermCfg(
            func=spatial.ObjectOnDestinationTerm,
            params={
                "object_cfg": object_cfg,
                "destination_cfg": destination_cfg,
                "contact_sensor_cfg": contact_sensor_cfg,
                "force_threshold": 0.1,
                "velocity_threshold": 0.1,
                "support_cone_half_angle_deg": 45.0,
            },
        )
        term = spatial.ObjectOnDestinationTerm(term_cfg, combined_env)
        assert geometry_build_calls == ["object", "destination"]
        torch.testing.assert_close(
            term(combined_env, **term_cfg.params),
            torch.tensor([True, False, False, False]),
        )

        combined_object.data.root_pose_w[0, 0] = 2.0
        assert not term(combined_env, **term_cfg.params)[0]
        assert geometry_build_calls == ["object", "destination"]
    finally:
        spatial._get_pose_frame_aabb = original_get_pose_frame_aabb

    return True


def test_geometric_object_on_destination():
    assert run_function_with_persistent_simulation_app(_test_geometric_object_on_destination)


if __name__ == "__main__":
    test_geometric_object_on_destination()
