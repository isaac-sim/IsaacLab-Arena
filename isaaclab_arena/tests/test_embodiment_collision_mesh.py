# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


def _test_embodiment_provides_robot_collision_mesh(simulation_app) -> bool:
    """Check the embodiment exposes its robot mesh so MESH mode does not fall back to the bbox proxy."""

    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment

    try:
        emb = DroidAbsoluteJointPositionEmbodiment()

        mesh = emb.get_collision_mesh()
        assert mesh is not None, "embodiment must expose a collision mesh; None forces the loose bbox fallback"
        assert len(mesh.vertices) > 0

        # The default prim scopes extraction to the arm. The Droid USD also bakes in a 50 m ground
        # plane and stray props; leaking those would blow the mesh up to scene scale.
        extents = mesh.extents
        assert all(e < 2.0 for e in extents), f"mesh leaked non-robot geometry: extents {extents}"

        # The mesh and the bounding box describe the same body, so their extents track each other.
        bbox = emb.get_bounding_box()
        bbox_size = (bbox.max_point - bbox.min_point)[0].tolist()
        for mesh_extent, box_extent in zip(extents, bbox_size):
            assert abs(mesh_extent - box_extent) < 0.2, f"mesh extents {extents} disagree with bbox {bbox_size}"

        # Extraction opens the USD, so the result is cached rather than recomputed per solve.
        assert emb.get_collision_mesh() is mesh

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def test_embodiment_provides_robot_collision_mesh():
    """Pytest entry point for the embodiment collision-mesh test."""
    result = run_simulation_app_function(_test_embodiment_provides_robot_collision_mesh, headless=True)
    assert result, f"Test {test_embodiment_provides_robot_collision_mesh.__name__} failed"


if __name__ == "__main__":
    test_embodiment_provides_robot_collision_mesh()
