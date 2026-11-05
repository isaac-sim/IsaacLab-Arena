# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify cached hull geometry and live containment through ArenaWorld."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _box_mesh(half_extents=(1.0, 1.0, 1.0)):
    import numpy as np

    from isaaclab_arena.utils.mesh_geometry import LocalMeshGeometry

    vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    vertices *= np.asarray(half_extents, dtype=np.float64)
    return LocalMeshGeometry(
        vertices=vertices,
        face_vertex_counts=np.full(6, 4, dtype=np.int64),
        face_vertex_indices=np.array(
            [0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5],
            dtype=np.int64,
        ),
    )


def _tetrahedron_mesh(axis_lengths=(4.0, 2.0, 1.0)):
    import numpy as np

    from isaaclab_arena.utils.mesh_geometry import LocalMeshGeometry

    return LocalMeshGeometry(
        vertices=np.array(
            [
                [0, 0, 0],
                [axis_lengths[0], 0, 0],
                [0, axis_lengths[1], 0],
                [0, 0, axis_lengths[2]],
            ],
            dtype=np.float64,
        ),
        face_vertex_counts=np.full(4, 3, dtype=np.int64),
        face_vertex_indices=np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=np.int64),
    )


def _scene_with_rigid_poses(poses_by_name):
    import torch

    first_poses = next(iter(poses_by_name.values()))
    num_envs = first_poses.shape[0]
    rigid_objects = {}
    for scene_key, root_poses in poses_by_name.items():
        rigid_objects[scene_key] = SimpleNamespace(
            data=SimpleNamespace(
                root_pose_w=SimpleNamespace(torch=root_poses.clone()),
                root_lin_vel_w=SimpleNamespace(torch=torch.full((num_envs, 3), 100.0, device=root_poses.device)),
                root_ang_vel_w=SimpleNamespace(torch=torch.full((num_envs, 3), 100.0, device=root_poses.device)),
            )
        )
    return SimpleNamespace(
        num_envs=num_envs,
        device=first_poses.device,
        rigid_objects=rigid_objects,
        articulations={},
        deformable_objects={},
        extras={},
        sensors={},
    )


def _test_hull_geometry_cache_and_live_poses(simulation_app):
    import math
    import torch

    import isaaclab_arena.environments.arena_world as arena_world_module
    import isaaclab_arena.environments.arena_world_scene_access as scene_access
    from isaaclab_arena.tasks.predicates.containment import object_hull_centroid_in_closed_hull
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    # O is the subject frame and C is the container frame. Both poses map into W.
    quarter_turn_component = math.sqrt(0.5)
    T_W_O = torch.tensor([
        [10, -3, 2, 0, 0, quarter_turn_component, quarter_turn_component],
        [-4, 5, 1, 0, 1, 0, 0],
    ])
    expected_hull_centers_W = torch.tensor([[9.5, -2.0, 2.25], [-5.0, 5.5, 0.75]])
    expected_aabb_centers_W = torch.tensor([[9.0, -1.0, 2.5], [-6.0, 6.0, 0.5]])
    T_W_C = torch.zeros((2, 7))
    T_W_C[:, :3] = expected_hull_centers_W
    T_W_C[:, 6] = 1.0
    scene = _scene_with_rigid_poses({"object": T_W_O, "container": T_W_C})
    mesh_groups = {
        "object": ((_tetrahedron_mesh(), torch.tensor([0, 1])),),
        "container": ((_box_mesh((0.5, 0.5, 0.5)), torch.tensor([0, 1])),),
    }
    extraction_calls = []

    def read_local_meshes(requested_scene, scene_key):
        assert requested_scene is scene
        extraction_calls.append(scene_key)
        return mesh_groups[scene_key]

    object_bounds = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(4.0, 2.0, 1.0))
    with (
        patch.object(
            scene_access,
            "read_spawned_meshes_in_local_frame",
            side_effect=read_local_meshes,
        ),
        patch.object(
            scene_access,
            "compute_spawned_geometry_bounds_in_local_frame",
            return_value=object_bounds,
        ) as compute_bounds,
        patch.object(
            arena_world_module,
            "build_convex_hull",
            wraps=arena_world_module.build_convex_hull,
        ) as build_hull,
    ):
        arena_world = arena_world_module.ArenaWorld(scene)
        env = SimpleNamespace(arena_world=arena_world, num_envs=scene.num_envs)
        torch.testing.assert_close(arena_world.get_centroid_w("object"), expected_aabb_centers_W)
        object_meshes = arena_world.get_meshes_in_local_frame("object")
        assert arena_world.get_meshes_in_local_frame("object") is object_meshes
        object_hulls = arena_world.get_convex_hulls_in_local_frame("object")
        assert arena_world.get_convex_hulls_in_local_frame("object") is object_hulls
        torch.testing.assert_close(arena_world.get_hull_vertex_centroid_w("object"), expected_hull_centers_W)
        torch.testing.assert_close(arena_world.get_centroid_w("object"), expected_aabb_centers_W)
        assert compute_bounds.call_count == 1
        assert extraction_calls == ["object"]
        assert build_hull.call_count == 1
        assert object_hull_centroid_in_closed_hull(env, "object", "container").tolist() == [True, True]

        scene.rigid_objects["object"].data.root_pose_w.torch[:, 0] += 2.0
        torch.testing.assert_close(
            arena_world.get_hull_vertex_centroid_w("object"),
            expected_hull_centers_W + torch.tensor([2.0, 0, 0]),
        )
        assert object_hull_centroid_in_closed_hull(env, "object", "container").tolist() == [False, False]
        scene.rigid_objects["object"].data.root_pose_w.torch = T_W_O.clone()
        assert object_hull_centroid_in_closed_hull(env, "object", "container").tolist() == [True, True]

        scene.rigid_objects["container"].data.root_pose_w.torch[:, 1] += 2.0
        assert object_hull_centroid_in_closed_hull(env, "object", "container").tolist() == [False, False]
        scene.rigid_objects["container"].data.root_pose_w.torch = T_W_C.clone()
        assert object_hull_centroid_in_closed_hull(env, "object", "container").tolist() == [True, True]
        assert extraction_calls == ["object", "container"]
        assert build_hull.call_count == 2

        env.arena_world = arena_world_module.ArenaWorld(scene)
        assert object_hull_centroid_in_closed_hull(env, "object", "container").tolist() == [True, True]
        assert extraction_calls == ["object", "container", "object", "container"]
        assert build_hull.call_count == 4
        assert env.arena_world.get_convex_hulls_in_local_frame("object") is not object_hulls
    return True


def test_hull_geometry_cache_and_live_poses():
    assert run_function_with_persistent_simulation_app(_test_hull_geometry_cache_and_live_poses)


def _test_containment_with_heterogeneous_meshes(simulation_app, device):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    import isaaclab_arena.environments.arena_world_scene_access as scene_access
    from isaaclab_arena.environments.arena_world import ArenaWorld
    from isaaclab_arena.tasks.predicates.containment import object_hull_centroid_in_closed_hull

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    container_rotations = Rotation.from_euler("xyz", [[0, 0, 0], [0, 0, 90], [0, 90, 0], [180, 0, 0]], degrees=True)
    object_rotations = Rotation.from_euler("xyz", [[0, 0, 90], [90, 0, 0], [0, 180, 0], [0, 90, 0]], degrees=True)
    container_positions_W = np.array([[0, 0, 0], [10, -3, 5], [-7, 8, 2], [4, 4, -2]], dtype=np.float64)
    object_centers_C = np.array([[1.2, 0.4, 0.4], [0.5, 0.5, 0.5], [1.2, 0.4, 0.4], [-0.5, 0, 0]])
    object_centers_W = container_positions_W + container_rotations.apply(object_centers_C)
    object_centers_O = np.array([[1, 0.5, 0.25], [0, 0, 0], [1, 0.5, 0.25], [0, 0, 0]])
    object_positions_W = object_centers_W - object_rotations.apply(object_centers_O)
    T_W_O = torch.tensor(
        np.column_stack((object_positions_W, object_rotations.as_quat())),
        dtype=torch.float32,
        device=device,
    )
    T_W_C = torch.tensor(
        np.column_stack((container_positions_W, container_rotations.as_quat())),
        dtype=torch.float32,
        device=device,
    )
    scene = _scene_with_rigid_poses({"object": T_W_O, "container": T_W_C})
    # The two entities have different, noncontiguous assignments of mesh variants.
    mesh_groups = {
        "object": (
            (_tetrahedron_mesh(), torch.tensor([2, 0], device=scene.device)),
            (_box_mesh(), torch.tensor([3, 1], device=scene.device)),
        ),
        "container": (
            (_box_mesh(), torch.tensor([1, 2], device=scene.device)),
            (_tetrahedron_mesh((4, 4, 4)), torch.tensor([3, 0], device=scene.device)),
        ),
    }
    with patch.object(
        scene_access,
        "read_spawned_meshes_in_local_frame",
        side_effect=lambda _scene, scene_key: mesh_groups[scene_key],
    ):
        arena_world = ArenaWorld(scene)
        env = SimpleNamespace(arena_world=arena_world, num_envs=scene.num_envs)
        torch.testing.assert_close(
            arena_world.get_hull_vertex_centroid_w("object"),
            torch.tensor(object_centers_W, dtype=torch.float32, device=scene.device),
        )
        container_hulls = arena_world.get_convex_hulls_in_local_frame("container")
        assert container_hulls[0][0].vertices.shape[0] == 8
        assert container_hulls[1][0].vertices.shape[0] == 4
        assert container_hulls[0][0].planes.shape[0] != container_hulls[1][0].planes.shape[0]
        contained = object_hull_centroid_in_closed_hull(env, "object", "container")
        assert contained.shape == (4,)
        assert contained.dtype == torch.bool
        assert contained.device == scene.device
        assert contained.tolist() == [True, True, False, False]
    return True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_containment_with_heterogeneous_meshes(device):
    assert run_function_with_persistent_simulation_app(_test_containment_with_heterogeneous_meshes, device=device)


def _test_open_container_follows_live_orientation(simulation_app):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    import isaaclab_arena.environments.arena_world_scene_access as scene_access
    from isaaclab_arena.environments.arena_world import ArenaWorld
    from isaaclab_arena.tasks.predicates.containment import (
        object_hull_centroid_in_closed_hull,
        object_hull_centroid_in_open_container,
    )

    num_envs = 6
    container_rotations = Rotation.from_euler("y", [[90], [90], [90], [90], [180], [90]], degrees=True)
    container_positions_W = np.column_stack((np.arange(num_envs) * 10.0, np.zeros(num_envs), np.zeros(num_envs)))
    object_centers_C = np.array([[0, 0, 3], [1.1, 0, 3], [0, 0, -1.1], [0, 0, 0], [0, 0, 3], [-3, 0, 0]])
    object_centers_W = container_positions_W + container_rotations.apply(object_centers_C)
    T_W_O = torch.zeros((num_envs, 7))
    T_W_O[:, :3] = torch.tensor(object_centers_W, dtype=torch.float32)
    T_W_O[:, 6] = 1.0
    T_W_C = torch.tensor(
        np.column_stack((container_positions_W, container_rotations.as_quat())),
        dtype=torch.float32,
    )
    scene = _scene_with_rigid_poses({"object": T_W_O, "container": T_W_C})
    mesh_groups = {
        "object": ((_box_mesh((2.0, 0.25, 0.25)), torch.arange(num_envs)),),
        "container": ((_box_mesh(), torch.arange(num_envs)),),
    }
    with patch.object(
        scene_access,
        "read_spawned_meshes_in_local_frame",
        side_effect=lambda _scene, scene_key: mesh_groups[scene_key],
    ):
        arena_world = ArenaWorld(scene)
        env = SimpleNamespace(arena_world=arena_world, num_envs=num_envs)
        closed_result = object_hull_centroid_in_closed_hull(env, "object", "container")
        open_result = object_hull_centroid_in_open_container(env, "object", "container")
        assert closed_result.tolist() == [False, False, False, True, False, False]
        assert open_result.tolist() == [True, False, False, True, True, False]

        # Environment 3 succeeds despite protruding vertices and high velocity, with no contact sensor.
        object_hull = arena_world.get_convex_hulls_in_local_frame("object")[0][0]
        vertices_C = container_rotations[3].inv().apply(object_hull.vertices.numpy())
        assert (np.abs(vertices_C) > 1.0).any(axis=1).all()

        # Rotating only the destination closes the previous world-X opening for environment 0.
        scene.rigid_objects["container"].data.root_pose_w.torch[0, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        assert object_hull_centroid_in_open_container(env, "object", "container").tolist() == [
            False,
            False,
            False,
            True,
            True,
            False,
        ]
    return True


def test_open_container_follows_live_orientation():
    assert run_function_with_persistent_simulation_app(_test_open_container_follows_live_orientation)
