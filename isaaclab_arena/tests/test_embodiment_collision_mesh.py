# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
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


def _test_droid_exposes_robot_and_stand_compound_geometry(simulation_app) -> bool:
    """Check Droid splits its geometry into robot + stand components and uses the stand for relations."""

    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.utils.pose import Pose

    try:
        emb = DroidAbsoluteJointPositionEmbodiment()

        components = emb.get_collision_components()
        assert [c.name for c in components] == ["robot", "stand"], f"unexpected components: {components}"
        # Both prims are placed at the solved base pose, so each component sits at the Droid root.
        assert all(c.local_pose == Pose.identity() for c in components), "components must share the root frame"

        robot_component, stand_component = components
        assert robot_component.mesh is not None, "robot component must carry its mesh for MESH-mode collision"

        # The relation bbox is the stand footprint, not the arm's wider envelope: On/NextTo measure the
        # base that sits by the support surface.
        relation_bbox = emb.get_relation_bounding_box()
        torch.testing.assert_close(relation_bbox.min_point, stand_component.bounding_box.min_point)
        torch.testing.assert_close(relation_bbox.max_point, stand_component.bounding_box.max_point)

        # The stand footprint must differ from the plain (arm) bounding box, otherwise the override is a no-op.
        arm_bbox = emb.get_bounding_box()
        relation_size = (relation_bbox.max_point - relation_bbox.min_point)[0]
        arm_size = (arm_bbox.max_point - arm_bbox.min_point)[0]
        assert not torch.allclose(relation_size, arm_size), "stand footprint should differ from the arm bbox"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def test_droid_exposes_robot_and_stand_compound_geometry():
    """Pytest entry point for the Droid compound-geometry test."""
    result = run_simulation_app_function(_test_droid_exposes_robot_and_stand_compound_geometry, headless=True)
    assert result, f"Test {test_droid_exposes_robot_and_stand_compound_geometry.__name__} failed"


if __name__ == "__main__":
    test_embodiment_provides_robot_collision_mesh()
    test_droid_exposes_robot_and_stand_compound_geometry()
