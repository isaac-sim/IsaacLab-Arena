# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from types import SimpleNamespace

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


def _check_object_on_destination(
    spatial,
    axis_aligned_bounding_box_type,
    scene_entity_cfg_type,
) -> None:
    """Check combined results and the entity state read by the predicate."""

    class ArenaWorldDouble:
        def __init__(self, T_W_E_by_entity_name, aabbs_E_by_entity_name, linear_velocities_w_by_entity_name):
            self.T_W_E_by_entity_name = T_W_E_by_entity_name
            self.aabbs_E_by_entity_name = aabbs_E_by_entity_name
            self.linear_velocities_w_by_entity_name = linear_velocities_w_by_entity_name
            self.pose_queries = []
            self.aabb_in_entity_frame_queries = []
            self.linear_velocity_queries = []

        def get_pose_w(self, entity_name):
            self.pose_queries.append(entity_name)
            return self.T_W_E_by_entity_name[entity_name]

        def get_aabb_in_entity_frame(self, entity_name):
            self.aabb_in_entity_frame_queries.append(entity_name)
            return self.aabbs_E_by_entity_name[entity_name]

        def get_linear_velocity_w(self, entity_name):
            self.linear_velocity_queries.append(entity_name)
            return self.linear_velocities_w_by_entity_name[entity_name]

    class EnvironmentDouble:
        def __init__(self, arena_world, contact_sensor):
            self.num_envs = 4
            self.arena_world = arena_world
            self.scene = {"contact_sensor": contact_sensor}

    class RuntimeBufferDouble:
        def __init__(self, tensor: torch.Tensor):
            self.torch = tensor

    class ContactSensorDouble:
        def __init__(self, contact_force_w: torch.Tensor):
            self.data = SimpleNamespace(force_matrix_w=RuntimeBufferDouble(contact_force_w[:, None, None, :]))

    identity_quaternion = (0.0, 0.0, 0.0, 1.0)
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

    coarse_contact_and_velocity_result = (torch.linalg.vector_norm(contact_force_w, dim=-1) > 0.1) & (
        torch.linalg.vector_norm(object_linear_velocity_w, dim=-1) < 0.1
    )
    torch.testing.assert_close(coarse_contact_and_velocity_result, torch.tensor([True, True, True, False]))

    arena_world = ArenaWorldDouble(
        T_W_E_by_entity_name={"object": T_W_O, "destination": T_W_D},
        aabbs_E_by_entity_name={
            "object": axis_aligned_bounding_box_type(
                min_point=torch.tensor([-0.1, -0.1, -0.1]).expand(4, 3),
                max_point=torch.tensor([0.1, 0.1, 0.1]).expand(4, 3),
            ),
            "destination": axis_aligned_bounding_box_type(
                min_point=torch.tensor([-1.0, -0.5, 0.0]).expand(4, 3),
                max_point=torch.tensor([1.0, 0.5, 0.4]).expand(4, 3),
            ),
        },
        linear_velocities_w_by_entity_name={"object": object_linear_velocity_w},
    )
    env = EnvironmentDouble(arena_world, ContactSensorDouble(contact_force_w))
    wrapped_env = SimpleNamespace(unwrapped=env)
    object_cfg = scene_entity_cfg_type("object")
    destination_cfg = scene_entity_cfg_type("destination")
    contact_sensor_cfg = scene_entity_cfg_type("contact_sensor")
    predicate_parameters = {
        "object_cfg": object_cfg,
        "destination_cfg": destination_cfg,
        "contact_sensor_cfg": contact_sensor_cfg,
        "force_threshold": 0.1,
        "velocity_threshold": 0.1,
        "support_cone_half_angle_deg": 45.0,
    }

    # Each failing environment isolates one condition: geometry, force direction, or velocity.
    predicate_result = spatial.object_on_destination(wrapped_env, **predicate_parameters)
    torch.testing.assert_close(predicate_result, torch.tensor([True, False, False, False]))

    # Exercise the unwrapped call path with changed live state.
    T_W_O[0, 0] = 2.0
    assert not spatial.object_on_destination(env, **predicate_parameters)[0]
    assert arena_world.pose_queries == ["object", "destination", "object", "destination"]
    assert arena_world.aabb_in_entity_frame_queries == ["object", "destination", "object", "destination"]
    assert arena_world.linear_velocity_queries == ["object", "object"]


def _test_object_on_destination(_simulation_app) -> bool:
    from isaaclab.managers import SceneEntityCfg

    import isaaclab_arena.tasks.predicates.spatial as spatial
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    _check_bounds_center_over_destination(spatial, AxisAlignedBoundingBox)
    _check_upward_support_force(spatial)
    _check_object_on_destination(
        spatial,
        AxisAlignedBoundingBox,
        SceneEntityCfg,
    )
    return True


def test_object_on_destination():
    assert run_function_with_persistent_simulation_app(_test_object_on_destination)


if __name__ == "__main__":
    test_object_on_destination()
