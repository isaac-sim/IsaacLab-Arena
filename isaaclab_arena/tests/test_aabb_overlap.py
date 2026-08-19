# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test reading annotated bounding boxes back out of a dataset and comparing footprints.

``aabb_overlap`` is vendorable: it depends on h5py and numpy alone, so it restates the HDF5 layout it
reads and carries its own bounding-box type. The last two tests here pin those restatements against
their in-tree counterparts, so the duplication cannot silently drift.
"""

import h5py
import math
import numpy as np

import pytest

from isaaclab_arena.analysis.aabb_overlap import (
    BOUNDING_BOXES_GROUP,
    AxisAlignedBoundingBox,
    find_demo_name,
    read_object_aabbs,
    xy_overlap_fraction,
)

_IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)


def _box(min_point: tuple[float, float, float], max_point: tuple[float, float, float]) -> AxisAlignedBoundingBox:
    return AxisAlignedBoundingBox(min_point=min_point, max_point=max_point)


def _write_annotated_dataset(
    path,
    poses: dict[str, list[tuple[float, ...]]],
    boxes: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    demos: tuple[tuple[str, int, int], ...] = (("demo_0", 0, 0),),
    format_version: int = 1,
) -> None:
    """Write a dataset already carrying a bounding-box group, as recording_annotation would leave it.

    Args:
        path: Destination file.
        poses: Per-object list of ``root_pose`` rows, one per frame.
        boxes: Per-object ``(min_point, max_point)``, each of shape ``(rows, 3)``.
        demos: ``(demo_name, env_id, episode_in_env)`` per demo; every demo gets the same poses.
        format_version: Written to the root attrs; only 1 (XYZW) is supported by the reader.
    """
    with h5py.File(path, "w") as hdf5_file:
        hdf5_file.attrs["format_version"] = format_version
        data = hdf5_file.create_group("data")
        for demo_name, env_id, episode_in_env in demos:
            demo = data.create_group(demo_name)
            episode = demo.create_group("episode_id")
            episode.create_dataset("env_id", data=np.array([env_id]))
            episode.create_dataset("episode_in_env", data=np.array([episode_in_env]))
            rigid_objects = demo.create_group("states").create_group("rigid_object")
            for name, rows in poses.items():
                rigid_objects.create_group(name).create_dataset("root_pose", data=np.array(rows, dtype=np.float32))
        group = hdf5_file.create_group(BOUNDING_BOXES_GROUP)
        for name, (min_point, max_point) in boxes.items():
            object_group = group.create_group(name)
            object_group.create_dataset("min_point", data=np.asarray(min_point, dtype=np.float64))
            object_group.create_dataset("max_point", data=np.asarray(max_point, dtype=np.float64))


def _unit_boxes() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "apple": (np.array([[-0.05, -0.05, 0.0]]), np.array([[0.05, 0.05, 0.10]])),
        "wooden_bowl": (np.array([[-0.10, -0.10, 0.0]]), np.array([[0.10, 0.10, 0.06]])),
    }


# --------------------------------------------------------------------------------------------------
# xy_overlap_fraction
# --------------------------------------------------------------------------------------------------


def test_child_entirely_over_destination_scores_one():
    child = _box((-0.05, -0.05, 0.10), (0.05, 0.05, 0.20))
    destination = _box((-0.10, -0.10, 0.0), (0.10, 0.10, 0.06))

    assert xy_overlap_fraction(child, destination) == pytest.approx(1.0)


def test_disjoint_footprints_score_zero():
    child = _box((0.50, 0.50, 0.0), (0.60, 0.60, 0.10))
    destination = _box((-0.10, -0.10, 0.0), (0.10, 0.10, 0.06))

    assert xy_overlap_fraction(child, destination) == pytest.approx(0.0)


def test_child_hanging_over_one_edge_scores_the_overlapping_fraction():
    # The child spans x in [0.05, 0.15] against a destination ending at x = 0.10: half its footprint.
    child = _box((0.05, -0.05, 0.10), (0.15, 0.05, 0.20))
    destination = _box((-0.10, -0.10, 0.0), (0.10, 0.10, 0.06))

    assert xy_overlap_fraction(child, destination) == pytest.approx(0.5)


def test_touching_edges_score_zero_not_a_sliver():
    # Exactly abutting at x = 0.10, the rim-contact case the runtime success signal accepts.
    child = _box((0.10, -0.05, 0.06), (0.20, 0.05, 0.16))
    destination = _box((-0.10, -0.10, 0.0), (0.10, 0.10, 0.06))

    assert xy_overlap_fraction(child, destination) == pytest.approx(0.0)


def test_fraction_is_relative_to_the_child_not_the_destination():
    # A small child inside a large destination scores 1.0; the same pair swapped scores far less.
    child = _box((-0.05, -0.05, 0.0), (0.05, 0.05, 0.10))
    destination = _box((-0.50, -0.50, 0.0), (0.50, 0.50, 0.06))

    assert xy_overlap_fraction(child, destination) == pytest.approx(1.0)
    assert xy_overlap_fraction(destination, child) == pytest.approx(0.01)


def test_zero_area_child_is_rejected():
    with pytest.raises(AssertionError, match="no horizontal area"):
        xy_overlap_fraction(_box((0.0, 0.0, 0.0), (0.0, 0.0, 0.1)), _box((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)))


# --------------------------------------------------------------------------------------------------
# AxisAlignedBoundingBox
# --------------------------------------------------------------------------------------------------


def test_inverted_corners_are_rejected():
    with pytest.raises(AssertionError, match="min_point must not exceed max_point"):
        _box((1.0, 0.0, 0.0), (0.0, 1.0, 1.0))


def test_identity_rotation_leaves_the_box_unchanged():
    box = _box((-0.05, -0.03, 0.0), (0.05, 0.03, 0.10))

    rotated = box.rotated_by_quat(_IDENTITY_XYZW)

    np.testing.assert_allclose(rotated.min_point, box.min_point, atol=1e-12)
    np.testing.assert_allclose(rotated.max_point, box.max_point, atol=1e-12)


def test_a_quarter_turn_swaps_the_horizontal_extents_exactly():
    box = _box((-0.05, -0.03, 0.0), (0.05, 0.03, 0.10))
    yaw_90 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))

    rotated = box.rotated_by_quat(yaw_90)

    # A 90-degree multiple stays axis-aligned, so x and y extents swap with no inflation.
    np.testing.assert_allclose(rotated.size, [0.06, 0.10, 0.10], atol=1e-9)


def test_size_and_center_describe_the_box():
    box = _box((-0.05, -0.03, 0.0), (0.05, 0.03, 0.10))

    np.testing.assert_allclose(box.size, [0.10, 0.06, 0.10])
    np.testing.assert_allclose(box.center, [0.0, 0.0, 0.05])


# --------------------------------------------------------------------------------------------------
# read_object_aabbs
# --------------------------------------------------------------------------------------------------


def test_stored_boxes_are_placed_at_the_recorded_pose(tmp_path):
    dataset = tmp_path / "dataset_job_rebuild0.hdf5"
    _write_annotated_dataset(
        dataset,
        poses={
            # Frame 0 holds the apple away from the bowl; the last frame drops it in.
            "apple": [(0.9, 0.0, 0.8, *_IDENTITY_XYZW), (0.4, 0.2, 0.85, *_IDENTITY_XYZW)],
            "wooden_bowl": [(0.4, 0.2, 0.75, *_IDENTITY_XYZW)] * 2,
        },
        boxes=_unit_boxes(),
    )

    child, destination = read_object_aabbs(dataset, "apple", "wooden_bowl")

    np.testing.assert_allclose(child.min_point, [0.35, 0.15, 0.85], atol=1e-6)
    np.testing.assert_allclose(destination.min_point, [0.30, 0.10, 0.75], atol=1e-6)
    assert xy_overlap_fraction(child, destination) == pytest.approx(1.0)

    # The default frame is the last one, so the earlier off-target frame must score differently.
    early_child, early_destination = read_object_aabbs(dataset, "apple", "wooden_bowl", frame=0)
    assert xy_overlap_fraction(early_child, early_destination) == pytest.approx(0.0)


def test_a_yaw_rotated_box_is_refitted_conservatively(tmp_path):
    dataset = tmp_path / "dataset_job_rebuild0.hdf5"
    yaw_45 = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    _write_annotated_dataset(
        dataset,
        poses={"apple": [(0.0, 0.0, 0.0, *yaw_45)], "wooden_bowl": [(0.0, 0.0, 0.0, *_IDENTITY_XYZW)]},
        boxes=_unit_boxes(),
    )

    child, _ = read_object_aabbs(dataset, "apple", "wooden_bowl")

    # A 0.1 m square yawed 45 degrees needs a 0.1 * sqrt(2) axis-aligned box to contain it.
    np.testing.assert_allclose(float(child.max_point[0]), 0.05 * math.sqrt(2), atol=1e-6)


def test_a_per_env_box_is_selected_by_the_demo_env_id(tmp_path):
    dataset = tmp_path / "dataset_job_rebuild0.hdf5"
    _write_annotated_dataset(
        dataset,
        poses={"apple": [(0.0, 0.0, 0.0, *_IDENTITY_XYZW)], "wooden_bowl": [(0.0, 0.0, 0.0, *_IDENTITY_XYZW)]},
        boxes={
            # Two rows: env 0 spawned a small variant, env 1 a large one.
            "apple": (
                np.array([[-0.05, -0.05, 0.0], [-0.20, -0.20, 0.0]]),
                np.array([[0.05, 0.05, 0.1], [0.2, 0.2, 0.4]]),
            ),
            "wooden_bowl": (np.array([[-0.10, -0.10, 0.0]]), np.array([[0.10, 0.10, 0.06]])),
        },
        demos=(("demo_0", 1, 0), ("demo_1", 0, 0)),
    )

    from_env_one, _ = read_object_aabbs(dataset, "apple", "wooden_bowl", demo="demo_0")
    from_env_zero, _ = read_object_aabbs(dataset, "apple", "wooden_bowl", demo="demo_1")

    assert float(from_env_one.max_point[0]) == pytest.approx(0.2)
    assert float(from_env_zero.max_point[0]) == pytest.approx(0.05)


def test_demo_lookup_joins_an_episode_results_record_to_its_demo(tmp_path):
    dataset = tmp_path / "dataset_job_rebuild0.hdf5"
    # Demos are written in completion order, so demo_0 need not be env 0.
    _write_annotated_dataset(
        dataset,
        poses={"apple": [(0.0, 0.0, 0.0, *_IDENTITY_XYZW)], "wooden_bowl": [(0.0, 0.0, 0.0, *_IDENTITY_XYZW)]},
        boxes=_unit_boxes(),
        demos=(("demo_0", 1, 0), ("demo_1", 0, 0), ("demo_2", 1, 1)),
    )

    assert find_demo_name(dataset, env_id=0, episode_in_env=0) == "demo_1"
    assert find_demo_name(dataset, env_id=1, episode_in_env=1) == "demo_2"
    with pytest.raises(KeyError):
        find_demo_name(dataset, env_id=7, episode_in_env=0)


def test_an_unannotated_object_is_reported_clearly(tmp_path):
    dataset = tmp_path / "dataset_job_rebuild0.hdf5"
    _write_annotated_dataset(
        dataset,
        poses={"apple": [(0.0, 0.0, 0.0, *_IDENTITY_XYZW)], "plasticpail": [(0.0, 0.0, 0.0, *_IDENTITY_XYZW)]},
        boxes=_unit_boxes(),
    )
    with h5py.File(dataset, "r+") as hdf5_file:
        hdf5_file[BOUNDING_BOXES_GROUP].create_group("plasticpail").attrs["error"] = "is a RigidObjectSet"

    with pytest.raises(AssertionError, match="RigidObjectSet"):
        read_object_aabbs(dataset, "apple", "plasticpail")
    with pytest.raises(AssertionError, match="annotate the dataset"):
        read_object_aabbs(dataset, "apple", "never_annotated")


def test_an_unannotated_dataset_is_reported_clearly(tmp_path):
    dataset = tmp_path / "dataset_job_rebuild0.hdf5"
    _write_annotated_dataset(
        dataset,
        poses={"apple": [(0.0, 0.0, 0.0, *_IDENTITY_XYZW)], "wooden_bowl": [(0.0, 0.0, 0.0, *_IDENTITY_XYZW)]},
        boxes=_unit_boxes(),
    )
    with h5py.File(dataset, "r+") as hdf5_file:
        del hdf5_file[BOUNDING_BOXES_GROUP]

    with pytest.raises(AssertionError, match="No 'bounding_boxes' group"):
        read_object_aabbs(dataset, "apple", "wooden_bowl")


def test_legacy_quaternion_format_is_refused(tmp_path):
    dataset = tmp_path / "dataset_job_rebuild0.hdf5"
    _write_annotated_dataset(
        dataset,
        poses={"apple": [(0.0, 0.0, 0.0, *_IDENTITY_XYZW)], "wooden_bowl": [(0.0, 0.0, 0.0, *_IDENTITY_XYZW)]},
        boxes=_unit_boxes(),
        format_version=0,
    )

    with pytest.raises(AssertionError, match="format_version=0"):
        read_object_aabbs(dataset, "apple", "wooden_bowl")


# --------------------------------------------------------------------------------------------------
# Guards on the vendorable copy: these must fail if the in-tree originals change
# --------------------------------------------------------------------------------------------------


def test_group_name_matches_the_one_the_annotation_tool_writes():
    from isaaclab_arena.analysis.recording_annotation import BOUNDING_BOXES_GROUP as written_group

    assert BOUNDING_BOXES_GROUP == written_group, "the reader's group name has drifted from the writer's"


def test_rotation_matches_arenas_bounding_box_implementation():
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox as ArenaBoundingBox

    generator = np.random.default_rng(seed=0)
    for _ in range(32):
        lower = generator.uniform(-0.5, 0.0, size=3)
        upper = lower + generator.uniform(0.01, 0.5, size=3)
        quaternion = generator.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        offset = generator.uniform(-1.0, 1.0, size=3)

        vendored = (
            AxisAlignedBoundingBox(min_point=lower, max_point=upper)
            .rotated_by_quat(tuple(quaternion))
            .translated(tuple(offset))
        )
        arena = (
            ArenaBoundingBox(min_point=tuple(lower), max_point=tuple(upper))
            .rotated_by_quat(tuple(quaternion))
            .translated(tuple(offset))
        )

        # Arena's implementation is float32, hence the loose-ish tolerance.
        np.testing.assert_allclose(vendored.min_point, arena.min_point[0].numpy(), atol=1e-6)
        np.testing.assert_allclose(vendored.max_point, arena.max_point[0].numpy(), atol=1e-6)
