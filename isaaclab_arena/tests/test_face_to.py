# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry
from isaaclab_arena.environment_spec.arena_env_graph_types import SpatialRelationSpec
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import AtPosition, FaceTo, IsAnchor, RandomAroundSolution, RotateAroundSolution
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange
from isaaclab_arena.utils.yaw import wrap_angle_to_pi, yaw_from_quat_xyzw, yaw_toward_positions

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


def _box(name: str, half_extents: tuple[float, float, float] = (0.1, 0.1, 0.1)) -> DummyObject:
    hx, hy, hz = half_extents
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(min_point=(-hx, -hy, -hz), max_point=(hx, hy, hz)),
    )


@dataclass(frozen=True)
class _FaceToPair:
    target: DummyObject
    subject: DummyObject


def _face_to_pair(
    target_position: tuple[float, float, float] = (1.0, 0.0, 0.0),
    subject_half_extents: tuple[float, float, float] = (0.1, 0.1, 0.1),
) -> _FaceToPair:
    target = _box("target")
    target.set_initial_pose(Pose(position_xyz=target_position))
    target.add_relation(IsAnchor())
    subject = _box("subject", half_extents=subject_half_extents)
    subject.add_relation(FaceTo(target))
    return _FaceToPair(target=target, subject=subject)


def _set_solver_results(
    monkeypatch: pytest.MonkeyPatch,
    placer: ObjectPlacer,
    layouts: list[dict[ObjectBase, tuple[float, float, float]]],
    losses: list[float] | None = None,
) -> None:
    def _solve(
        objects: list[ObjectBase],
        initial_positions: list[dict[ObjectBase, tuple[float, float, float]]],
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox] | None,
        env_bboxes_include_yaw: bool = False,
        orientations: list[dict[ObjectBase, float]] | None = None,
        collision_objects=None,
    ) -> list[dict[ObjectBase, tuple[float, float, float]]]:
        assert len(initial_positions) == len(layouts)
        placer._solver._last_loss_per_env = torch.tensor(losses or [0.0] * len(layouts))
        return layouts

    monkeypatch.setattr(placer._solver, "solve", _solve)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ((1.0, 0.0, 0.0), 0.0),
        ((0.0, 1.0, 0.0), math.pi / 2),
        ((-1.0, -1.0, 0.0), -3 * math.pi / 4),
    ],
)
def test_yaw_toward_positions(target, expected):
    yaws, is_defined = yaw_toward_positions(torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([target]))

    torch.testing.assert_close(yaws, torch.tensor([expected]))
    assert is_defined.tolist() == [True]


def test_yaw_toward_positions_is_translation_invariant_and_batched():
    subjects = torch.tensor([[3.0, 4.0, 0.0], [-2.0, 5.0, 1.0]])
    targets = torch.tensor([[4.0, 5.0, 8.0], [-3.0, 5.0, -2.0]])

    yaws, is_defined = yaw_toward_positions(subjects, targets)

    torch.testing.assert_close(yaws, torch.tensor([math.pi / 4, math.pi]))
    assert is_defined.tolist() == [True, True]


def test_yaw_toward_positions_rejects_near_coincident_xy():
    from isaaclab_arena.utils.yaw import MINIMUM_FACING_DIRECTION_XY_M

    epsilon = MINIMUM_FACING_DIRECTION_XY_M
    subjects = torch.zeros((3, 3))
    targets = torch.tensor([[epsilon * 0.1, 0.0, 1.0], [epsilon, 0.0, 1.0], [epsilon * 1.1, 0.0, 1.0]])

    _, is_defined = yaw_toward_positions(subjects, targets)

    assert is_defined.tolist() == [False, False, True]


def test_face_to_uses_candidate_positions_for_anchor_and_foreground_targets():
    pair = _face_to_pair(target_position=(0.0, -2.0, 0.0))
    positions = [
        {pair.subject: (0.0, 0.0, 0.0), pair.target: pair.target.get_initial_pose().position_xyz},
        {pair.subject: (2.0, -2.0, 0.0), pair.target: (2.0, 0.0, 0.0)},
    ]
    orientations = [{}, {}]

    ObjectPlacer._apply_face_to_orientations(positions, orientations)

    assert orientations[0][pair.subject] == pytest.approx(-math.pi / 2)
    assert orientations[1][pair.subject] == pytest.approx(math.pi / 2)


def test_coincident_face_to_fails_candidate_validation():
    target = _box("target")
    subject = _box("subject")
    subject.add_relation(FaceTo(target))
    positions = {subject: (0.0, 0.0, 0.0), target: (0.0, 0.0, 2.0)}
    orientations = [{}]

    ObjectPlacer._apply_face_to_orientations([positions], orientations)
    validation = ObjectPlacer()._validate_placement(
        positions,
        {obj: obj.get_bounding_box() for obj in positions},
        {subject: 0.0},
    )

    assert orientations == [{}]
    assert validation.validation_results[PlacementCheck.FACE_TO] is False
    assert validation.do_all_required_validation_checks_pass() is False


def test_face_to_rebuilds_rotated_footprint_before_validation():
    target = _box("target")
    blocker = _box("blocker")
    subject = _box("subject", half_extents=(1.0, 0.1, 0.1))
    subject.add_relation(FaceTo(target))
    objects = [subject, target, blocker]
    positions = {subject: (0.0, 0.0, 0.0), target: (0.0, 2.0, 0.0), blocker: (0.0, 0.75, 0.0)}
    unrotated = {obj: obj.get_bounding_box() for obj in objects}
    orientations = [{}]
    placer = ObjectPlacer()

    assert placer._validate_placement(positions, unrotated).validation_results[PlacementCheck.NO_OVERLAP]
    ObjectPlacer._apply_face_to_orientations([positions], orientations)
    rotated = ObjectPlacer._rotate_candidate_bboxes(objects, unrotated, orientations)
    validation = placer._validate_placement(positions, rotated, orientations[0])

    assert orientations[0][subject] == pytest.approx(math.pi / 2)
    assert validation.validation_results[PlacementCheck.NO_OVERLAP] is False


def test_face_to_suppresses_initial_random_yaw_and_rejects_rotate_marker():
    pair = _face_to_pair()
    placer = ObjectPlacer(ObjectPlacerParams(random_yaw_init=True))

    assert placer._generate_initial_orientations([pair.target, pair.subject], {pair.target}) == {}

    pair.subject.add_relation(RotateAroundSolution(yaw_rad=0.5))
    with pytest.raises(AssertionError, match="cannot combine FaceTo"):
        placer._prepare_placement([pair.target, pair.subject])


def test_face_to_rejects_target_outside_placement():
    target = _box("target")
    subject = _box("subject")
    subject.add_relation(FaceTo(target))

    with pytest.raises(AssertionError, match="must participate in placement"):
        ObjectPlacer()._prepare_placement([subject])


@pytest.mark.parametrize(
    ("invalid_case", "message"),
    [
        ("duplicate", "more than one FaceTo"),
        ("anchor", "Anchor object"),
        ("self", "cannot face itself"),
    ],
)
def test_face_to_rejects_invalid_relation_configuration(invalid_case, message):
    pair = _face_to_pair()
    if invalid_case == "duplicate":
        pair.subject.add_relation(FaceTo(pair.target))
    elif invalid_case == "anchor":
        pair.target.add_relation(FaceTo(pair.subject))
    else:
        pair.subject.relations = [FaceTo(pair.subject)]

    with pytest.raises(AssertionError, match=message):
        ObjectPlacer()._prepare_placement([pair.target, pair.subject])


@pytest.mark.parametrize(
    ("randomized_object", "randomization", "message"),
    [
        ("subject", RandomAroundSolution(x_half_m=0.1), "cannot randomize XY"),
        ("target", RandomAroundSolution(y_half_m=0.1), "cannot randomize XY"),
    ],
)
def test_face_to_rejects_reset_randomization_that_changes_direction(randomized_object, randomization, message):
    pair = _face_to_pair()
    {"subject": pair.subject, "target": pair.target}[randomized_object].add_relation(randomization)

    with pytest.raises(AssertionError, match=message):
        ObjectPlacer()._prepare_placement([pair.target, pair.subject])


def test_face_to_allows_reset_rotation_around_facing_yaw():
    pair = _face_to_pair(target_position=(0.0, 2.0, 0.0))
    pair.subject.add_relation(RandomAroundSolution(roll_half_rad=0.1, pitch_half_rad=0.2, yaw_half_rad=0.3))
    placer = ObjectPlacer()
    positions = {pair.target: (0.0, 2.0, 0.0), pair.subject: (0.0, 0.0, 0.0)}

    anchors, _ = placer._prepare_placement([pair.target, pair.subject])
    placer._apply_poses([positions], anchors, [{pair.subject: math.pi / 2}])
    pose = pair.subject.get_initial_pose()

    assert isinstance(pose, PoseRange)
    assert pose.rpy_min == pytest.approx((-0.1, -0.2, math.pi / 2 - 0.3))
    assert pose.rpy_max == pytest.approx((0.1, 0.2, math.pi / 2 + 0.3))


def test_relation_solver_ignores_face_to_marker():
    pair = _face_to_pair()
    pair.subject.add_relation(AtPosition(x=0.0, y=0.0, z=0.0))
    positions = {pair.target: (1.0, 0.0, 0.0), pair.subject: (0.0, 0.0, 0.0)}

    assert all(not isinstance(relation, FaceTo) for relation in pair.subject.get_spatial_relations())

    RelationSolver(RelationSolverParams(max_iters=1, verbose=False)).solve([pair.target, pair.subject], [positions])


def test_face_to_applies_absolute_world_yaw_with_default_orientation_params():
    pair = _face_to_pair(target_position=(0.0, 2.0, 0.0))
    pair.subject.add_relation(AtPosition(x=0.0, y=0.0, z=0.0))
    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(max_iters=200, verbose=False),
        max_placement_attempts=1,
    )

    (result,) = ObjectPlacer(params).place([pair.target, pair.subject])
    pose = pair.subject.get_initial_pose()

    assert isinstance(pose, Pose)
    assert result.orientations[pair.subject] == pytest.approx(math.pi / 2)
    assert abs(wrap_angle_to_pi(yaw_from_quat_xyzw(pose.rotation_xyzw) - math.pi / 2)) < 1e-5


def test_face_to_applies_independent_multi_env_yaws(monkeypatch):
    pair = _face_to_pair(target_position=(10.0, 0.0, 0.0))
    subject_positions = [(0.0, 0.0, 0.0), (10.0, -10.0, 0.0), (20.0, 0.0, 0.0), (10.0, 10.0, 0.0)]
    layouts = [{pair.target: (10.0, 0.0, 0.0), pair.subject: position} for position in subject_positions]
    placer = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1))
    _set_solver_results(monkeypatch, placer, layouts)

    results = placer.place([pair.target, pair.subject], num_envs=4)
    pose = pair.subject.get_initial_pose()

    assert isinstance(pose, PosePerEnv)
    expected = [0.0, math.pi / 2, math.pi, -math.pi / 2]
    for result, env_pose, expected_yaw in zip(results, pose.poses, expected, strict=True):
        assert abs(wrap_angle_to_pi(result.orientations[pair.subject] - expected_yaw)) < 1e-5
        assert abs(wrap_angle_to_pi(yaw_from_quat_xyzw(env_pose.rotation_xyzw) - expected_yaw)) < 1e-5


def test_face_to_reranks_candidate_with_rotated_overlap(monkeypatch):
    pair = _face_to_pair(target_position=(0.0, 2.0, 0.0), subject_half_extents=(1.0, 0.1, 0.1))
    blocker = _box("blocker")
    blocker.add_relation(AtPosition(x=0.0, y=0.75, z=0.0))
    layouts = [
        {pair.target: (0.0, 2.0, 0.0), blocker: (0.0, 0.75, 0.0), pair.subject: (0.0, 0.0, 0.0)},
        {pair.target: (0.0, 2.0, 0.0), blocker: (0.0, 0.75, 0.0), pair.subject: (2.0, 0.0, 0.0)},
    ]
    placer = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=2, apply_positions_to_objects=False))
    _set_solver_results(monkeypatch, placer, layouts, losses=[0.0, 1.0])

    (result,) = placer.place([pair.target, blocker, pair.subject])

    assert result.success
    assert result.positions[pair.subject] == (2.0, 0.0, 0.0)


def test_face_to_fallback_warning_names_failed_check(monkeypatch, capsys):
    pair = _face_to_pair(target_position=(0.0, 0.0, 2.0))
    layout = {pair.target: (0.0, 0.0, 2.0), pair.subject: (0.0, 0.0, 0.0)}
    placer = ObjectPlacer(ObjectPlacerParams(max_placement_attempts=1, apply_positions_to_objects=False, verbose=True))
    _set_solver_results(monkeypatch, placer, [layout])

    (result,) = placer.place([pair.target, pair.subject])

    assert not result.success
    assert "(failed: ['face_to'])" in capsys.readouterr().out


def test_face_to_registers_as_binary_graph_relation():
    subject = _box("subject")
    target = _box("target")
    spec = SpatialRelationSpec(kind="face_to", subject="camera", reference="object")
    registry = ObjectRelationLibraryRegistry()
    subject.add_relation(FaceTo(target, **spec.params))

    assert registry.is_registered(spec.kind, ensure_loaded=False)
    (relation,) = subject.get_relations()
    assert relation.parent is target
    with pytest.raises(ValidationError, match="requires relation.reference"):
        SpatialRelationSpec(kind="face_to", subject="camera")


def test_mesh_validation_receives_final_face_to_yaw(monkeypatch):
    pair = _face_to_pair(target_position=(0.0, 1.0, 0.0))
    pair.subject.add_relation(AtPosition(x=0.0, y=0.0, z=0.0))
    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=200, verbose=False),
        max_placement_attempts=1,
        apply_positions_to_objects=False,
    )
    placer = ObjectPlacer(params)
    received = {}

    def _capture_orientations(candidate_positions, env_bboxes, candidate_orientations=None, collision_objects=None):
        received.update(candidate_orientations or {})
        return True

    monkeypatch.setattr(placer, "_validate_no_overlap_mesh", _capture_orientations)
    (result,) = placer.place([pair.target, pair.subject])

    assert received[pair.subject] == pytest.approx(math.pi / 2)
    assert result.orientations[pair.subject] == pytest.approx(math.pi / 2)
