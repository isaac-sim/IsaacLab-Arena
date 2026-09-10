# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Final-pose checks must not reuse release-pose geometry or pre-settle verdicts."""

import math

import pytest

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_validators import (
    FaceToValidator,
    NoOverlapValidator,
    OnRelationValidator,
    PlacementValidator,
    validate_position_constraints,
)
from isaaclab_arena.relations.relations import AtPosition, FaceTo, On, PositionLimitsBox, PositionLimitsCylindrical
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _box(name, lower=(-0.1, -0.1, -0.1), upper=(0.1, 0.1, 0.1)):
    return DummyObject(name, AxisAlignedBoundingBox(min_point=lower, max_point=upper))


def test_neighbor_on_relation_is_rechecked_after_tipping_or_displacement():
    support = _box("support", (-1, -1, -0.1), (1, 1, 0))
    neighbor = _box("neighbor", (-0.1, -0.1, 0), (0.1, 0.1, 0.4))
    neighbor.add_relation(On(support))
    boxes = {obj: obj.get_bounding_box() for obj in (support, neighbor)}
    validator = OnRelationValidator(ObjectPlacerParams())
    poses = {support: Pose.identity(), neighbor: Pose.identity()}
    assert validator.validate_poses(poses, boxes, [])
    poses[neighbor] = Pose((1.2, 0, 0))
    assert not validator.validate_poses(poses, boxes, [])
    poses[neighbor] = Pose((0, 0, 0), (math.sqrt(0.5), 0, 0, math.sqrt(0.5)))
    assert not validator.validate_poses(poses, boxes, [])


@pytest.mark.parametrize(
    "relation",
    [
        AtPosition(x=0),
        PositionLimitsBox(x_max=0.2),
        PositionLimitsCylindrical(center_x=0, center_y=0, radius_max=0.2),
    ],
)
def test_unary_constraints_reject_a_settled_neighbor_pushed_out_of_place(relation):
    neighbor = _box("neighbor")
    neighbor.add_relation(relation)
    assert validate_position_constraints({neighbor: Pose.identity()}, 0.005)
    assert not validate_position_constraints({neighbor: Pose((0.3, 0, 0))}, 0.005)


def test_final_facing_checks_actual_yaw():
    subject, target = _box("subject"), _box("target")
    subject.add_relation(FaceTo(target))
    validator = FaceToValidator(ObjectPlacerParams())
    poses = {subject: Pose.identity(), target: Pose((1, 0, 0))}
    assert validator.validate_poses(poses, {}, [])
    poses[subject] = Pose(rotation_xyzw=(0, 0, 1, 0))
    assert not validator.validate_poses(poses, {}, [])


def test_intentional_pile_contacts_do_not_exempt_unrelated_obstacles():
    a, b, neighbor = (_box(name) for name in ("a", "b", "neighbor"))
    validator = NoOverlapValidator(ObjectPlacerParams())
    poses = {a: Pose.identity(), b: Pose.identity(), neighbor: Pose((1, 0, 0))}
    boxes = {obj: obj.get_bounding_box() for obj in poses}
    allowed = {frozenset((a, b))}
    assert validator.validate_poses(poses, boxes, [], allowed)
    poses[neighbor] = Pose.identity()
    assert not validator.validate_poses(poses, boxes, [], allowed)


def test_mesh_collision_uses_full_rotation_about_the_object_origin():
    import trimesh

    from isaaclab_arena.relations.collision_mode import CollisionMode

    a = _box("off_origin", (-0.1, -0.1, 0.2), (0.1, 0.1, 0.4))
    b = _box("obstacle")
    for obj in (a, b):
        obj.collision_mode = CollisionMode.MESH
        obj._collision_mesh = trimesh.creation.box(extents=obj.bounding_box.size[0].numpy())
        obj._collision_mesh.apply_translation(obj.bounding_box.center[0].numpy())
    poses = {a: Pose(rotation_xyzw=(0, math.sqrt(0.5), 0, math.sqrt(0.5))), b: Pose((0.3, 0, 0))}
    boxes = {obj: obj.get_bounding_box() for obj in poses}
    validator = NoOverlapValidator(ObjectPlacerParams())
    assert not validator.validate_poses(poses, boxes, [])
    poses[a] = Pose.identity()
    assert validator.validate_poses(poses, boxes, [])


def test_extension_validator_cannot_silently_ignore_full_rotations():
    class LegacyValidator(PlacementValidator):
        check = "legacy"

        def validate_batch(self, positions, orientations, bboxes, collision_objects):
            return [True] * len(positions)

    with pytest.raises(NotImplementedError, match="validate_poses"):
        LegacyValidator(ObjectPlacerParams()).validate_poses({}, {}, [])


@pytest.mark.parametrize("previous,settled", [(True, False), (False, True)])
def test_physics_settle_replaces_previous_verdict(previous, settled):
    from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
    from isaaclab_arena.relations.placement_pool_validation import (
        _compute_physics_settled_and_add_to_validation_results,
    )
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.placement_validation import PlacementCheck, PlacementValidationResults

    checks = PlacementValidationResults({PlacementCheck.PHYSICS_SETTLED: previous})
    layout = PlacementResult(checks, {}, 0.0, 1)
    _compute_physics_settled_and_add_to_validation_results(
        None, [(0, layout)], [], PhysicsSettleParams(), settled_per_env_override=[settled]
    )
    assert checks.validation_results[PlacementCheck.PHYSICS_SETTLED] is settled
