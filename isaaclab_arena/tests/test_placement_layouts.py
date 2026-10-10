# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Placement records from offline generation and episode recording."""

import json

import pytest

from isaaclab_arena.relations.placement_sampler import (
    PlacementSample,
    deserialize_placement_samples,
    placement_samples_from_pose_columns,
    placement_samples_to_pose_columns,
    write_placement_samples,
)
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.recorded_variation_samples import load_rebuild_variation_record


def _load_placement_samples(path):
    record = load_rebuild_variation_record(path, build_time_variation_keys=set())
    return deserialize_placement_samples(
        [episode.placement_sample for episode in record.episode_records if episode.placement_sample is not None]
    )


def test_offline_and_episode_records_share_layout_order(tmp_path):
    poses = {"cup": [Pose((1, 2, 3)), Pose((4, 5, 6))]}
    samples = placement_samples_from_pose_columns(poses)
    path = tmp_path / "layouts.jsonl"
    write_placement_samples(path, samples)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1].update(env_id=7, success=True)
    records[1]["variations"] = {"light.brightness": 2}
    path.write_text("\n".join(json.dumps(record) for record in records))
    assert placement_samples_to_pose_columns(_load_placement_samples(path)) == poses


def test_episode_records_allow_repeated_layout_ids(tmp_path):
    poses = [Pose((1, 2, 3)), Pose((4, 5, 6))]
    samples = [PlacementSample(layout_id="fixed", poses={"cup": pose}) for pose in poses]
    path = tmp_path / "layouts.jsonl"

    write_placement_samples(path, samples)
    loaded = _load_placement_samples(path)

    assert [sample.layout_id for sample in loaded] == ["fixed", "fixed"]
    assert placement_samples_to_pose_columns(loaded) == {"cup": poses}


@pytest.mark.parametrize("failure", ["missing_object", "invalid_rotation", "missing_rotation", "short_position"])
def test_replay_rejects_incomplete_or_invalid_records(tmp_path, failure):
    path = tmp_path / "layouts.jsonl"
    poses = {"cup": Pose().to_dict(), "bowl": Pose().to_dict()}
    first = {"placement": {"layout_id": "layout_0", "poses": poses}}
    first_line = json.dumps(first)
    first["placement"]["layout_id"] = "layout_1"
    if failure == "missing_object":
        del poses["bowl"]
    elif failure == "invalid_rotation":
        poses["cup"]["rotation_xyzw"] = [0, 0, 0, 0]
    elif failure == "missing_rotation":
        del poses["cup"]["rotation_xyzw"]
    else:
        poses["cup"]["position_xyz"] = [0, 0]
    path.write_text(first_line + "\n" + json.dumps(first))
    with pytest.raises(AssertionError):
        _load_placement_samples(path)


def test_replay_rejects_duplicate_pose_fields(tmp_path):
    path = tmp_path / "layouts.jsonl"
    path.write_text(
        '{"placement":{"poses":{"cup":{"position_xyz":[1,0,0],"position_xyz":[2,0,0],"rotation_xyzw":[0,0,0,1]}}}'
    )
    with pytest.raises(AssertionError, match="Duplicate key"):
        _load_placement_samples(path)
