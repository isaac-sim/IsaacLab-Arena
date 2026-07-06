# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena_environments.pick_and_place_maple_table_environment import _compute_controlled_object_bin_layout

PICK_BBOX = AxisAlignedBoundingBox(
    (-0.0355603, -0.0330279, -0.0417773),
    (0.0355603, 0.0330279, 0.0417773),
)
DESTINATION_BBOX = AxisAlignedBoundingBox(
    (-0.21, -0.14, 0.0),
    (0.21, 0.14, 0.105),
)
TABLE_WORLD_BBOX = AxisAlignedBoundingBox(
    (0.1985909, -0.4779370, -0.6970),
    (0.8985909, 0.5220630, 0.0030007),
)


@pytest.mark.parametrize("gap_m", [0.001, 0.2])
@pytest.mark.parametrize("side", ["positive_y", "negative_y"])
def test_controlled_layout_realizes_exact_gap(gap_m, side):
    pick_position, destination_position, actual_gap_m = _compute_controlled_object_bin_layout(
        PICK_BBOX,
        DESTINATION_BBOX,
        TABLE_WORLD_BBOX,
        gap_m=gap_m,
        side=side,
        pick_center_x=0.38,
        destination_center_x=0.46,
        pair_midpoint_y=0.01,
    )

    assert actual_gap_m == pytest.approx(gap_m, abs=1e-7)
    assert pick_position[0] == pytest.approx(0.38)
    assert destination_position[0] == pytest.approx(0.46)
    pick_center_y = pick_position[1] + PICK_BBOX.center[0, 1].item()
    destination_center_y = destination_position[1] + DESTINATION_BBOX.center[0, 1].item()
    assert (pick_center_y + destination_center_y) * 0.5 == pytest.approx(0.01)
    assert pick_position[2] + PICK_BBOX.min_point[0, 2].item() == pytest.approx(
        destination_position[2] + DESTINATION_BBOX.min_point[0, 2].item(),
        abs=1e-7,
    )


def test_controlled_layout_accepts_full_campaign_grid():
    for gap_m in (0.001, 0.002, 0.005, 0.010, 0.020, 0.040, 0.070, 0.100, 0.150, 0.200):
        for side in ("positive_y", "negative_y"):
            for pick_center_x in (0.30, 0.34, 0.38, 0.42, 0.45):
                for pair_midpoint_y in (-0.03, -0.01, 0.01, 0.03, 0.05):
                    _, _, actual_gap_m = _compute_controlled_object_bin_layout(
                        PICK_BBOX,
                        DESTINATION_BBOX,
                        TABLE_WORLD_BBOX,
                        gap_m=gap_m,
                        side=side,
                        pick_center_x=pick_center_x,
                        destination_center_x=0.46,
                        pair_midpoint_y=pair_midpoint_y,
                    )
                    assert actual_gap_m == pytest.approx(gap_m, abs=1e-7)


@pytest.mark.parametrize(
    ("pick_center_x", "destination_center_x", "pair_midpoint_y", "error"),
    [
        (0.38, 0.44, 0.01, "tabletop margin"),
        (0.46, 0.50, 0.01, "DROID workspace"),
        (0.38, 0.46, 0.30, "DROID workspace"),
    ],
)
def test_controlled_layout_rejects_invalid_geometry(pick_center_x, destination_center_x, pair_midpoint_y, error):
    with pytest.raises(ValueError, match=error):
        _compute_controlled_object_bin_layout(
            PICK_BBOX,
            DESTINATION_BBOX,
            TABLE_WORLD_BBOX,
            gap_m=0.001,
            side="positive_y",
            pick_center_x=pick_center_x,
            destination_center_x=destination_center_x,
            pair_midpoint_y=pair_midpoint_y,
        )


def test_controlled_layout_rejects_unknown_side():
    with pytest.raises(ValueError, match="unsupported"):
        _compute_controlled_object_bin_layout(
            PICK_BBOX,
            DESTINATION_BBOX,
            TABLE_WORLD_BBOX,
            gap_m=0.001,
            side="positive_x",
            pick_center_x=0.38,
            destination_center_x=0.46,
            pair_midpoint_y=0.01,
        )


def test_controlled_layout_rejects_non_overlapping_x_aabbs():
    with pytest.raises(ValueError, match="overlapping.*X AABBs"):
        _compute_controlled_object_bin_layout(
            PICK_BBOX,
            DESTINATION_BBOX,
            TABLE_WORLD_BBOX,
            gap_m=0.001,
            side="positive_y",
            pick_center_x=0.10,
            destination_center_x=0.46,
            pair_midpoint_y=0.01,
        )
