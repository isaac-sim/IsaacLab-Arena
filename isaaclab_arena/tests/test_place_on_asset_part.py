# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for On relations whose parent is part of a placed asset, such as a drawer link of a cabinet."""

import math

import pytest

from isaaclab_arena.relations.geometry.asset_parts import get_part_bounding_box_relative_to_owner
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.optimization.initializers.on_tree_initializer import OnTreeInitializer
from isaaclab_arena.relations.optimization.losses.no_overlap_bbox import get_on_linked_pairs
from isaaclab_arena.relations.relations import FaceTo, IsAnchor, On, PositionLimitsBox, RotateAroundSolution
from isaaclab_arena.relations.validation.types import PlacementCheck
from isaaclab_arena.tests.dummy_object import DummyObject, make_candidate_batch
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

CLEARANCE_M = 0.01
EDGE_MARGIN_M = 0.02
TOLERANCE_M = 2e-3

# A cabinet whose box encloses a drawer link, relative to the cabinet origin: the drawer sits inside the cabinet,
# so a child in the drawer overlaps the cabinet's box.
CABINET_BBOX = AxisAlignedBoundingBox(min_point=(-0.3, -0.2, 0.0), max_point=(0.3, 0.2, 0.8))
DRAWER_BBOX_IN_CABINET = AxisAlignedBoundingBox(min_point=(-0.25, -0.15, 0.30), max_point=(0.15, 0.15, 0.40))
APPLE_BBOX = AxisAlignedBoundingBox(min_point=(-0.03, -0.03, -0.03), max_point=(0.03, 0.03, 0.03))


class DummyPart(DummyObject):
    """Part of another dummy asset, such as a link of an articulated object, with bounds fixed relative to it."""

    def __init__(self, name: str, root: DummyObject, bounding_box_relative_to_root: AxisAlignedBoundingBox):
        super().__init__(name=name, bounding_box=bounding_box_relative_to_root)
        self._root = root

    def get_placement_root(self) -> DummyObject:
        return self._root

    def get_bounding_box_relative_to_placement_root(self) -> AxisAlignedBoundingBox:
        return self.bounding_box


def _make_table() -> DummyObject:
    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(-1.0, -1.0, -0.05), max_point=(1.0, 1.0, 0.0)),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    )
    table.add_relation(IsAnchor())
    return table


def _make_cabinet_with_drawer() -> tuple[DummyObject, DummyPart]:
    cabinet = DummyObject(name="cabinet", bounding_box=CABINET_BBOX)
    drawer = DummyPart(name="drawer", root=cabinet, bounding_box_relative_to_root=DRAWER_BBOX_IN_CABINET)
    return cabinet, drawer


def _make_apple_on(parent: DummyObject) -> DummyObject:
    apple = DummyObject(name="apple", bounding_box=APPLE_BBOX)
    apple.add_relation(On(parent, clearance_m=CLEARANCE_M, edge_margin_m=EDGE_MARGIN_M))
    return apple


def _anchor(asset: DummyObject, position_xyz, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)) -> None:
    asset.set_initial_pose(Pose(position_xyz=position_xyz, rotation_xyzw=rotation_xyzw))
    asset.add_relation(IsAnchor())


def _assert_on_part(apple_xyz, part_world: AxisAlignedBoundingBox) -> None:
    """Assert the apple rests on the part's top, inside its footprint inset by the edge margin."""
    for axis in range(2):
        lower = float(part_world.min_point[0, axis]) + EDGE_MARGIN_M - TOLERANCE_M
        upper = float(part_world.max_point[0, axis]) - EDGE_MARGIN_M + TOLERANCE_M
        assert lower <= apple_xyz[axis] + float(APPLE_BBOX.min_point[0, axis])
        assert apple_xyz[axis] + float(APPLE_BBOX.max_point[0, axis]) <= upper
    apple_bottom = apple_xyz[2] + float(APPLE_BBOX.min_point[0, 2])
    part_top = float(part_world.max_point[0, 2])
    assert part_top - TOLERANCE_M <= apple_bottom <= part_top + CLEARANCE_M + TOLERANCE_M


def _validate_one(positions, bboxes):
    placer = ObjectPlacer(ObjectPlacerParams())
    batch = make_candidate_batch([positions], [{}], [bboxes])
    placer._validation.validate_candidates(batch, [])
    return batch.candidates[0].validation


def _drawer_world_bbox_at(cabinet_xyz) -> AxisAlignedBoundingBox:
    return DRAWER_BBOX_IN_CABINET.translated(cabinet_xyz)


def test_validators_accept_child_inside_owner_bounds_on_part():
    """A child on a drawer passes On and no_overlap although the cabinet's box encloses it."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    cabinet.add_relation(On(table, clearance_m=0.0))
    apple = _make_apple_on(drawer)
    cabinet_xyz = (0.2, 0.1, 0.0)
    apple_xyz = (0.15, 0.1, 0.40 + CLEARANCE_M + 0.03)
    positions = {table: (0.0, 0.0, 0.0), cabinet: cabinet_xyz, apple: apple_xyz}
    bboxes = {table: table.get_bounding_box(), cabinet: CABINET_BBOX, apple: APPLE_BBOX}

    validation = _validate_one(positions, bboxes)

    assert validation.validation_results[PlacementCheck.ON_RELATION] is True
    assert validation.validation_results[PlacementCheck.NO_OVERLAP] is True


def test_on_validator_uses_part_footprint_not_owner_footprint():
    """A child inside the cabinet's footprint but beside the drawer fails On."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    cabinet.add_relation(On(table, clearance_m=0.0))
    apple = _make_apple_on(drawer)
    # x = 0.25 is within the cabinet ([-0.3, 0.3]) but beyond the drawer's +x edge at 0.15.
    positions = {table: (0.0, 0.0, 0.0), cabinet: (0.0, 0.0, 0.0), apple: (0.25, 0.0, 0.40 + CLEARANCE_M + 0.03)}
    bboxes = {table: table.get_bounding_box(), cabinet: CABINET_BBOX, apple: APPLE_BBOX}

    validation = _validate_one(positions, bboxes)

    assert validation.validation_results[PlacementCheck.ON_RELATION] is False


def test_on_validator_uses_part_top_not_owner_top():
    """A child on the cabinet's top fails On to the drawer, whose top is lower."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    cabinet.add_relation(On(table, clearance_m=0.0))
    apple = _make_apple_on(drawer)
    positions = {table: (0.0, 0.0, 0.0), cabinet: (0.0, 0.0, 0.0), apple: (0.0, 0.0, 0.80 + CLEARANCE_M + 0.03)}
    bboxes = {table: table.get_bounding_box(), cabinet: CABINET_BBOX, apple: APPLE_BBOX}

    validation = _validate_one(positions, bboxes)

    assert validation.validation_results[PlacementCheck.ON_RELATION] is False


def test_no_overlap_still_checks_other_objects_against_owner():
    """Only the child on the part is exempt from the owner; another object inside the cabinet still overlaps."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    cabinet.add_relation(On(table, clearance_m=0.0))
    apple = _make_apple_on(drawer)
    pear = DummyObject(name="pear", bounding_box=APPLE_BBOX)
    pear.add_relation(On(table))
    positions = {
        table: (0.0, 0.0, 0.0),
        cabinet: (0.0, 0.0, 0.0),
        apple: (0.0, 0.0, 0.40 + CLEARANCE_M + 0.03),
        pear: (0.2, 0.1, 0.5),
    }
    bboxes = {table: table.get_bounding_box(), cabinet: CABINET_BBOX, apple: APPLE_BBOX, pear: APPLE_BBOX}

    validation = _validate_one(positions, bboxes)

    assert validation.validation_results[PlacementCheck.NO_OVERLAP] is False


def test_on_linked_pairs_link_child_to_owner_of_part():
    """The solver's no-overlap loss skips the child and the placed owner of its part parent."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    cabinet.add_relation(On(table))
    apple = _make_apple_on(drawer)

    on_pairs = get_on_linked_pairs([table, cabinet, apple])

    assert (id(apple), id(cabinet)) in on_pairs
    assert (id(cabinet), id(apple)) in on_pairs


def test_on_tree_initializer_seeds_child_on_part_after_owner():
    """The child is seeded on the drawer at the cabinet's seeded position, whatever the input order."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    cabinet.add_relation(On(table, clearance_m=0.0))
    apple = _make_apple_on(drawer)
    objects = [apple, cabinet, table]
    bboxes = {obj: obj.get_bounding_box() for obj in objects}

    positions = OnTreeInitializer().generate_initial_positions(objects, {table}, bboxes)

    _assert_on_part(positions[apple], _drawer_world_bbox_at(positions[cabinet]))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_child_on_part_moves_with_relation_placed_owner(seed):
    """Solving places the cabinet anywhere in its band and keeps the apple on the drawer at that pose."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    cabinet.add_relation(On(table, clearance_m=0.0, edge_margin_m=0.0))
    cabinet.add_relation(PositionLimitsBox(x_limits=(-0.5, 0.5), y_limits=(-0.6, 0.6)))
    apple = _make_apple_on(drawer)
    placer = ObjectPlacer(ObjectPlacerParams(placement_seed=seed, apply_positions_to_objects=False))

    (result,) = placer.place([table, cabinet, apple])

    assert result.success, result.validation_results.get_failed_validation_check_names
    _assert_on_part(result.positions[apple], _drawer_world_bbox_at(result.positions[cabinet]))


def test_child_on_part_follows_owner_across_layouts():
    """Different cabinet positions carry the apple along: its offset from the cabinet stays inside the drawer."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    cabinet.add_relation(On(table, clearance_m=0.0, edge_margin_m=0.0))
    apple = _make_apple_on(drawer)
    placer = ObjectPlacer(ObjectPlacerParams(placement_seed=7, apply_positions_to_objects=False))

    results = placer.place([table, cabinet, apple], num_envs=4)

    cabinet_x_values = {round(result.positions[cabinet][0], 3) for result in results}
    assert len(cabinet_x_values) > 1, "The cabinet should land at different positions across environments"
    for result in results:
        assert result.success
        _assert_on_part(result.positions[apple], _drawer_world_bbox_at(result.positions[cabinet]))


@pytest.mark.parametrize("cabinet_xyz", [(0.0, 0.0, 0.0), (0.4, -0.3, 0.0)])
def test_child_on_part_follows_anchored_owner_pose(cabinet_xyz):
    """Moving an anchored cabinet moves the apple with it."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    _anchor(cabinet, cabinet_xyz)
    apple = _make_apple_on(drawer)
    placer = ObjectPlacer(ObjectPlacerParams(placement_seed=3, apply_positions_to_objects=False))

    (result,) = placer.place([table, cabinet, apple])

    assert result.success
    _assert_on_part(result.positions[apple], _drawer_world_bbox_at(cabinet_xyz))


def test_part_bounds_turn_with_anchored_owner_quarter_turn():
    """A cabinet anchored at +90 degrees yaw turns its drawer's bounds about the cabinet origin."""
    cabinet, drawer = _make_cabinet_with_drawer()
    _anchor(cabinet, (0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)))

    bounds = get_part_bounding_box_relative_to_owner(drawer, cabinet)

    # +90 degrees about Z maps (x, y) to (-y, x).
    assert bounds.min_point[0].tolist() == pytest.approx([-0.15, -0.25, 0.30])
    assert bounds.max_point[0].tolist() == pytest.approx([0.15, 0.15, 0.40])


def test_child_on_part_of_owner_turned_by_rotate_around_solution():
    """A movable cabinet turned -90 degrees by its marker carries the turned drawer bounds."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    cabinet.add_relation(On(table, clearance_m=0.0, edge_margin_m=0.0))
    cabinet.add_relation(RotateAroundSolution(yaw_rad=-math.pi / 2))
    apple = _make_apple_on(drawer)
    placer = ObjectPlacer(ObjectPlacerParams(placement_seed=5, apply_positions_to_objects=False))

    (result,) = placer.place([table, cabinet, apple])

    assert result.success
    turned_drawer = DRAWER_BBOX_IN_CABINET.rotated_90_around_z(3).translated(result.positions[cabinet])
    _assert_on_part(result.positions[apple], turned_drawer)


def test_anchored_part_of_relation_placed_owner_is_rejected():
    """Anchoring a part pins it while its owner moves, so placement rejects the configuration."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    cabinet.add_relation(On(table))
    _anchor(drawer, (0.0, 0.0, 0.0))
    apple = _make_apple_on(drawer)

    with pytest.raises(AssertionError, match="Remove is_anchor"):
        ObjectPlacer(ObjectPlacerParams()).place([table, cabinet, drawer, apple])


def test_on_parent_outside_placement_is_rejected():
    """An On parent that neither takes part in placement nor belongs to a placed asset is rejected."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    apple = _make_apple_on(drawer)

    with pytest.raises(AssertionError, match="must take part in placement"):
        ObjectPlacer(ObjectPlacerParams()).place([table, apple])


def test_random_yaw_init_with_movable_owner_of_part_is_rejected():
    """Random yaw would turn the owner away from the fixed part bounds."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    cabinet.add_relation(On(table))
    apple = _make_apple_on(drawer)

    with pytest.raises(AssertionError, match="disable random_yaw_init"):
        ObjectPlacer(ObjectPlacerParams(random_yaw_init=True)).place([table, cabinet, apple])


def test_face_to_on_owner_of_part_is_rejected():
    """FaceTo turns the owner after solving, so an owner with placed parts cannot use it."""
    table = _make_table()
    cabinet, drawer = _make_cabinet_with_drawer()
    cabinet.add_relation(On(table))
    pear = DummyObject(name="pear", bounding_box=APPLE_BBOX)
    pear.add_relation(On(table))
    cabinet.add_relation(FaceTo(pear))
    apple = _make_apple_on(drawer)

    with pytest.raises(AssertionError, match="cannot use FaceTo"):
        ObjectPlacer(ObjectPlacerParams()).place([table, cabinet, pear, apple])
