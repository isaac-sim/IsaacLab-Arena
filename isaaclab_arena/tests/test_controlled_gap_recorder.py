# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from isaaclab_arena.recording.common_terms import (
    compute_controlled_gap_observation,
    record_controlled_gap_observation,
    world_xy_aabb_from_local_bbox,
)

PICK_BBOX_MIN = [-0.1, -0.1, -0.1]
PICK_BBOX_MAX = [0.1, 0.1, 0.1]
DESTINATION_BBOX_MIN = [-0.2, -0.2, -0.1]
DESTINATION_BBOX_MAX = [0.2, 0.2, 0.1]
IDENTITY_QUATERNION_XYZW = [0.0, 0.0, 0.0, 1.0]


def _pose(x=0.0, y=0.0, z=0.0, quaternion=IDENTITY_QUATERNION_XYZW):
    return {"pos_w": [x, y, z], "quat_w_xyzw": quaternion}


def _observation(pick_pose, destination_pose, side="positive_y"):
    return compute_controlled_gap_observation(
        {"pick": pick_pose, "destination": destination_pose},
        pick_asset_name="pick",
        destination_asset_name="destination",
        side=side,
        pick_local_bbox_min=PICK_BBOX_MIN,
        pick_local_bbox_max=PICK_BBOX_MAX,
        destination_local_bbox_min=DESTINATION_BBOX_MIN,
        destination_local_bbox_max=DESTINATION_BBOX_MAX,
    )


@pytest.mark.parametrize(
    ("side", "pick_y"),
    [("positive_y", 0.4), ("negative_y", -0.4)],
)
def test_controlled_gap_observation_reports_requested_side_gap(side, pick_y):
    observation = _observation(_pose(y=pick_y), _pose(), side)

    assert observation == {
        "pick_asset_name": "pick",
        "destination_asset_name": "destination",
        "side": side,
        "axis_gap_m": pytest.approx([0.0, 0.1]),
        "planar_aabb_gap_m": pytest.approx(0.1),
        "signed_side_gap_m": pytest.approx(0.1),
    }


def test_controlled_gap_observation_reports_diagonal_distance_and_overlap():
    separated = _observation(_pose(x=0.5, y=0.6), _pose())
    assert separated["axis_gap_m"] == pytest.approx([0.2, 0.3])
    assert separated["planar_aabb_gap_m"] == pytest.approx(math.hypot(0.2, 0.3))
    assert separated["signed_side_gap_m"] == pytest.approx(0.3)

    overlapping = _observation(_pose(), _pose())
    assert overlapping["axis_gap_m"] == [0.0, 0.0]
    assert overlapping["planar_aabb_gap_m"] == 0.0
    assert overlapping["signed_side_gap_m"] == pytest.approx(-0.3)


def test_world_xy_aabb_applies_xyzw_rotation_and_normalizes_quaternion():
    sine = math.sin(math.pi / 4)
    cosine = math.cos(math.pi / 4)
    world_min, world_max = world_xy_aabb_from_local_bbox(
        _pose(x=1.0, y=2.0, quaternion=[0.0, 0.0, 2.0 * sine, 2.0 * cosine]),
        [-0.2, -0.1, -0.05],
        [0.2, 0.1, 0.05],
    )

    assert world_min == pytest.approx([0.9, 1.8])
    assert world_max == pytest.approx([1.1, 2.2])


class _SnapshotEnv:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.requested_env_id = None

    def get_initial_object_pose_snapshot(self, env_id):
        self.requested_env_id = env_id
        return self.snapshot


def test_recorder_wraps_observation_from_actual_initial_snapshot():
    env = _SnapshotEnv({"pick": _pose(y=0.4), "destination": _pose()})
    fields = record_controlled_gap_observation(
        env,
        3,
        pick_asset_name="pick",
        destination_asset_name="destination",
        side="positive_y",
        pick_local_bbox_min=PICK_BBOX_MIN,
        pick_local_bbox_max=PICK_BBOX_MAX,
        destination_local_bbox_min=DESTINATION_BBOX_MIN,
        destination_local_bbox_max=DESTINATION_BBOX_MAX,
    )

    assert env.requested_env_id == 3
    assert fields["controlled_gap_observation"]["planar_aabb_gap_m"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("snapshot", "kwargs", "error"),
    [
        ({"pick": _pose()}, {}, "missing required assets"),
        (
            {"pick": _pose(), "destination": _pose()},
            {"side": "positive_x"},
            "side must be",
        ),
        (
            {"pick": _pose(quaternion=[0.0, 0.0, 0.0, 0.0]), "destination": _pose()},
            {},
            "non-zero norm",
        ),
        (
            {"pick": _pose(), "destination": _pose()},
            {"pick_local_bbox_max": [-0.2, 0.1, 0.1]},
            "strictly less",
        ),
    ],
)
def test_controlled_gap_observation_rejects_malformed_input(snapshot, kwargs, error):
    params = {
        "pick_asset_name": "pick",
        "destination_asset_name": "destination",
        "side": "positive_y",
        "pick_local_bbox_min": PICK_BBOX_MIN,
        "pick_local_bbox_max": PICK_BBOX_MAX,
        "destination_local_bbox_min": DESTINATION_BBOX_MIN,
        "destination_local_bbox_max": DESTINATION_BBOX_MAX,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=error):
        compute_controlled_gap_observation(snapshot, **params)


def test_recorder_requires_initial_pose_snapshot_api():
    with pytest.raises(ValueError, match="get_initial_object_pose_snapshot"):
        record_controlled_gap_observation(
            object(),
            0,
            pick_asset_name="pick",
            destination_asset_name="destination",
            side="positive_y",
            pick_local_bbox_min=PICK_BBOX_MIN,
            pick_local_bbox_max=PICK_BBOX_MAX,
            destination_local_bbox_min=DESTINATION_BBOX_MIN,
            destination_local_bbox_max=DESTINATION_BBOX_MAX,
        )
