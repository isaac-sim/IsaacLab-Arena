# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ObjectPlacer.place() tests for placing an object On a container's interior surface.

Container support (shelf, fridge/microwave interior) resolves "in/on the <container>" to a concrete
support surface modeled as an anchor whose bbox is the interior region. These tests pin that a plain
On(interior_surface) confines the solved placement to that surface's footprint (inset by the
production edge margin) with no new solver mechanism, and that an object too large for the surface is
rejected rather than placed spilling over the rim.
"""

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.relations import DEFAULT_ON_EDGE_MARGIN_M, IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

# A small, thin, world-offset support surface: a fridge shelf / microwave turntable recessed inside a
# larger appliance. Local footprint 0.30 x 0.30, top face 0.02 above its origin.
_SURFACE_HALF_XY_M = 0.15
_SURFACE_TOP_LOCAL_Z_M = 0.02
_SURFACE_WORLD_POSE = Pose(position_xyz=(2.0, -0.4, 1.10), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))


def _make_interior_surface() -> DummyObject:
    surface = DummyObject(
        name="interior_surface",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-_SURFACE_HALF_XY_M, -_SURFACE_HALF_XY_M, 0.0),
            max_point=(_SURFACE_HALF_XY_M, _SURFACE_HALF_XY_M, _SURFACE_TOP_LOCAL_Z_M),
        ),
    )
    surface.set_initial_pose(_SURFACE_WORLD_POSE)
    surface.add_relation(IsAnchor())
    return surface


def _make_item(half_xy_m: float) -> DummyObject:
    return DummyObject(
        name="item",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-half_xy_m, -half_xy_m, 0.0), max_point=(half_xy_m, half_xy_m, 0.08)
        ),
    )


def _placer(random_yaw_init: bool = False) -> ObjectPlacer:
    # Deterministic seed; don't touch sim (apply_positions_to_objects=False keeps this CPU-only).
    return ObjectPlacer(
        params=ObjectPlacerParams(
            placement_seed=7, random_yaw_init=random_yaw_init, apply_positions_to_objects=False
        )
    )


def _assert_validated(result) -> None:
    failed = result.validation_results.get_failed_validation_check_names
    assert result.success, f"expected a validated layout, failed checks: {failed}"


def test_on_interior_surface_confines_within_inset_footprint():
    """An item On the interior surface is solved fully within the surface footprint, inset by the edge margin."""
    surface = _make_interior_surface()
    item_half_xy = 0.03
    item = _make_item(item_half_xy)
    item.add_relation(On(surface))  # default edge_margin_m == DEFAULT_ON_EDGE_MARGIN_M

    (result,) = _placer().place([surface, item])
    _assert_validated(result)

    x, y, z = result.positions[item]
    surface_world = surface.get_world_bounding_box()
    margin = DEFAULT_ON_EDGE_MARGIN_M
    eps = 1e-4

    # Whole item footprint sits inside the surface rim, inset by the production margin on every side.
    assert x - item_half_xy >= surface_world.min_point[0, 0].item() + margin - eps
    assert x + item_half_xy <= surface_world.max_point[0, 0].item() - margin + eps
    assert y - item_half_xy >= surface_world.min_point[0, 1].item() + margin - eps
    assert y + item_half_xy <= surface_world.max_point[0, 1].item() - margin + eps
    # Bottom face rests just above the surface top (default clearance 1 cm).
    assert z >= surface_world.max_point[0, 2].item() - eps


def test_on_interior_surface_rejects_item_too_large_for_margin():
    """An item nearly as wide as the surface can't honor the edge margin, so placement is not validated."""
    surface = _make_interior_surface()
    # Footprint 0.28 x 0.28: free span 0.30 - 0.28 = 0.02 < 2 * margin (0.10), so the inset band inverts.
    item = _make_item(half_xy_m=0.14)
    item.add_relation(On(surface))

    (result,) = _placer().place([surface, item])
    assert not result.success
    assert PlacementCheck.ON_RELATION in result.validation_results.get_failed_validation_check_names


def test_on_interior_surface_confinement_holds_under_random_yaw_init():
    """With random_yaw_init, the item's yaw-inflated footprint still lands within the inset surface.

    random_yaw_init assigns a fixed yaw and confines the conservative box enclosing the rotated item,
    so an elongated item must fit the surface at its sampled orientation, not just axis-aligned.
    """
    surface = _make_interior_surface()
    # Elongated footprint so yaw materially changes the enclosing box (0.10 x 0.06); its worst-case
    # rotated half-extent sqrt(0.05^2 + 0.03^2) ~= 0.058 still fits the 0.30 surface inset by 0.05.
    item_half_x, item_half_y = 0.05, 0.03
    item = DummyObject(
        name="item",
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-item_half_x, -item_half_y, 0.0), max_point=(item_half_x, item_half_y, 0.08)
        ),
    )
    item.add_relation(On(surface))

    (result,) = _placer(random_yaw_init=True).place([surface, item])
    _assert_validated(result)

    x, y, _ = result.positions[item]
    yaw = result.orientations[item]
    # Confinement must hold against the same yaw-inflated footprint the solver/validation used.
    rotated_footprint = item.get_bounding_box().rotated_around_z(yaw).translated(result.positions[item])
    surface_world = surface.get_world_bounding_box()
    margin = DEFAULT_ON_EDGE_MARGIN_M
    eps = 1e-4

    assert rotated_footprint.min_point[0, 0].item() >= surface_world.min_point[0, 0].item() + margin - eps
    assert rotated_footprint.max_point[0, 0].item() <= surface_world.max_point[0, 0].item() - margin + eps
    assert rotated_footprint.min_point[0, 1].item() >= surface_world.min_point[0, 1].item() + margin - eps
    assert rotated_footprint.max_point[0, 1].item() <= surface_world.max_point[0, 1].item() - margin + eps
