# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test recordings annotation without building a real Task (no Isaac Sim needed).

``build_task_and_assets_from_job_name`` and the two per-job lookups built on it need the Isaac Lab /
pxr environment, and are exercised indirectly through the injectable lookup functions that
``_JobLookupCache`` takes.
"""

import h5py
import json
import numpy as np

import pytest

from isaaclab_arena.analysis.recording_annotation import (
    BOUNDING_BOXES_GROUP,
    ORIGINAL_DATASET_SUFFIX,
    STATIC_PREDICATES_SUFFIX,
    _JobLookupCache,
    annotate_bounding_boxes_dir,
    annotate_episode_results_dir,
    annotate_episode_results_file,
    dataset_path_for_episode_results,
)

_FAKE_PREDICATES = {
    "pick_and_place": {
        "pick_up_object": "apple",
        "destination_location": "wooden_bowl",
        "groups": {"default_group": [{"index": 0, "predicate": "objects_settled", "score": 1.0}]},
    }
}

_FAKE_BOXES = {
    "apple": {
        "min_point": np.array([[-0.03, -0.03, -0.03]]),
        "max_point": np.array([[0.03, 0.03, 0.03]]),
        "usd_path": "apple.usd",
        "scale": np.array([1.0, 1.0, 1.0]),
    },
    "wooden_bowl": {
        "min_point": np.array([[-0.10, -0.10, 0.0]]),
        "max_point": np.array([[0.10, 0.10, 0.06]]),
        "usd_path": "wooden_bowl.usd",
        "scale": np.array([1.0, 1.0, 1.0]),
    },
    "plasticpail": {"error": "'plasticpail' is a RigidObjectSet; per-env variant assignment is not recoverable."},
    "never_recorded": {
        "min_point": np.array([[0.0, 0.0, 0.0]]),
        "max_point": np.array([[1.0, 1.0, 1.0]]),
        "usd_path": "unused.usd",
        "scale": np.array([1.0, 1.0, 1.0]),
    },
}


def _fake_predicate_lookup(job_name: str, env_package: str) -> dict:
    if job_name == "unknown_job":
        raise AssertionError(f"No task yaml for job_name={job_name!r} under env_package={env_package!r}")
    return _FAKE_PREDICATES


def _fake_boxes_lookup(job_name: str, env_package: str) -> dict:
    if job_name == "unknown_job":
        raise AssertionError(f"No task yaml for job_name={job_name!r} under env_package={env_package!r}")
    return _FAKE_BOXES


def _write_jsonl(path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_results(path, job_name: str, count: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        path,
        [json.dumps({"job_name": job_name, "env_id": index, "episode_in_env": 0}) for index in range(count)],
    )


def _write_dataset(
    path,
    object_names: tuple[str, ...] = ("apple", "wooden_bowl", "plasticpail"),
    *,
    demos: tuple[tuple[str, int, int], ...] = (("demo_0", 0, 0),),
    num_frames: int = 3,
    format_version: int = 1,
) -> None:
    """Write a minimal stand-in for a recorded trajectory dataset (identity poses at the origin)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hdf5_file:
        hdf5_file.attrs["format_version"] = format_version
        data = hdf5_file.create_group("data")
        for demo_name, env_id, episode_in_env in demos:
            demo = data.create_group(demo_name)
            episode = demo.create_group("episode_id")
            episode.create_dataset("env_id", data=np.array([env_id]))
            episode.create_dataset("episode_in_env", data=np.array([episode_in_env]))
            rigid_objects = demo.create_group("states").create_group("rigid_object")
            for name in object_names:
                root_pose = np.zeros((num_frames, 7), dtype=np.float32)
                root_pose[:, 6] = 1.0  # Identity quaternion, XYZW order.
                rigid_objects.create_group(name).create_dataset("root_pose", data=root_pose)


# --------------------------------------------------------------------------------------------------
# Static predicate annotation
# --------------------------------------------------------------------------------------------------


def test_annotate_preserves_line_correspondence(tmp_path):
    source = tmp_path / "episode_results_rebuild0.jsonl"
    _write_jsonl(
        source,
        [
            json.dumps({"job_name": "apple_and_yogurt_in_bowl", "env_id": 0, "episode_in_env": 0}),
            "",
            json.dumps({"job_name": "apple_and_yogurt_in_bowl", "env_id": 1, "episode_in_env": 0}),
        ],
    )

    cache = _JobLookupCache("robolab", lookup_fn=_fake_predicate_lookup)
    output_path = annotate_episode_results_file(source, "robolab", cache=cache)

    assert output_path == source.with_name(f"episode_results_rebuild0{STATIC_PREDICATES_SUFFIX}")
    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(output_lines) == 3
    assert output_lines[1] == ""

    first = json.loads(output_lines[0])
    assert first == {
        "job_name": "apple_and_yogurt_in_bowl",
        "env_id": 0,
        "episode_in_env": 0,
        "predicates": _FAKE_PREDICATES,
    }
    third = json.loads(output_lines[2])
    assert third["env_id"] == 1


def test_annotate_reports_per_line_errors_without_aborting(tmp_path):
    source = tmp_path / "episode_results_rebuild0.jsonl"
    _write_jsonl(
        source,
        [
            "not valid json",
            json.dumps({"no_job_name_field": True}),
            json.dumps({"job_name": "unknown_job"}),
            json.dumps({"job_name": "apple_and_yogurt_in_bowl"}),
        ],
    )

    cache = _JobLookupCache("robolab", lookup_fn=_fake_predicate_lookup)
    output_path = annotate_episode_results_file(source, "robolab", cache=cache)
    output_lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(output_lines) == 4
    for line in output_lines[:3]:
        assert "error" in json.loads(line)
    assert "predicates" in json.loads(output_lines[3])


def test_job_lookup_cache_is_reused_across_files(tmp_path):
    calls: list[str] = []

    def _counting_lookup(job_name: str, env_package: str) -> dict:
        calls.append(job_name)
        return _FAKE_PREDICATES

    first = tmp_path / "a" / "episode_results_rebuild0.jsonl"
    second = tmp_path / "b" / "episode_results_rebuild0.jsonl"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    for path in (first, second):
        _write_jsonl(path, [json.dumps({"job_name": "apple_and_yogurt_in_bowl", "env_id": 0})])

    cache = _JobLookupCache("robolab", lookup_fn=_counting_lookup)
    annotate_episode_results_file(first, "robolab", cache=cache)
    annotate_episode_results_file(second, "robolab", cache=cache)

    assert calls == ["apple_and_yogurt_in_bowl"]


def test_annotate_dir_finds_and_annotates_every_results_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "isaaclab_arena.analysis.recording_annotation.static_predicates_for_job", _fake_predicate_lookup
    )

    task_a = tmp_path / "task_a"
    task_b = tmp_path / "task_b"
    task_a.mkdir()
    task_b.mkdir()
    _write_jsonl(task_a / "episode_results_rebuild0.jsonl", [json.dumps({"job_name": "task_a", "env_id": 0})])
    _write_jsonl(task_b / "episode_results_rank1.jsonl", [json.dumps({"job_name": "task_b", "env_id": 0})])
    # A report page under a "report" dir must not be picked up as a source file.
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "episode_results_rebuild0.jsonl").write_text("{}\n", encoding="utf-8")

    output_paths = annotate_episode_results_dir(tmp_path, "robolab")

    assert {path.name for path in output_paths} == {
        f"episode_results_rebuild0{STATIC_PREDICATES_SUFFIX}",
        f"episode_results_rank1{STATIC_PREDICATES_SUFFIX}",
    }


# --------------------------------------------------------------------------------------------------
# Bounding box annotation
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "results_name,expected",
    [
        ("episode_results_rebuild0.jsonl", "dataset_my_job_rebuild0.hdf5"),
        ("episode_results_rebuild2.jsonl", "dataset_my_job_rebuild2.hdf5"),
        # A rankless or rebuildless name still maps to the rebuild-0 dataset the recorder wrote.
        ("episode_results.jsonl", "dataset_my_job_rebuild0.hdf5"),
        ("episode_results_rebuild1_rank3.jsonl", "dataset_my_job_rebuild1.hdf5"),
    ],
)
def test_dataset_path_pairs_results_file_with_its_rebuild(tmp_path, results_name, expected):
    assert dataset_path_for_episode_results(tmp_path / results_name, "my_job").name == expected


def _annotate(tmp_path, **kwargs) -> list[str]:
    return annotate_bounding_boxes_dir(tmp_path, "robolab", boxes_lookup_fn=_fake_boxes_lookup, **kwargs)


def test_bounding_boxes_are_written_and_the_original_is_preserved(tmp_path):
    job = tmp_path / "apple_and_yogurt_in_bowl"
    _write_results(job / "episode_results_rebuild0.jsonl", "apple_and_yogurt_in_bowl")
    dataset = job / "dataset_apple_and_yogurt_in_bowl_rebuild0.hdf5"
    _write_dataset(dataset)
    original_bytes = dataset.read_bytes()

    reports = _annotate(tmp_path)
    assert len(reports) == 1 and reports[0].startswith("wrote")

    preserved = dataset.with_suffix(ORIGINAL_DATASET_SUFFIX)
    assert preserved.read_bytes() == original_bytes, "the untouched original must be kept byte-for-byte"

    with h5py.File(dataset, "r") as hdf5_file:
        group = hdf5_file[BOUNDING_BOXES_GROUP]
        assert group.attrs["job_name"] == "apple_and_yogurt_in_bowl"
        # Only recorded objects are stored: "never_recorded" has no root_pose in this dataset.
        assert set(group.keys()) == {"apple", "wooden_bowl", "plasticpail"}
        np.testing.assert_allclose(group["apple"]["min_point"][:], [[-0.03, -0.03, -0.03]])
        assert group["apple"]["min_point"].shape == (1, 3)
        assert group["apple"].attrs["usd_path"] == "apple.usd"
        # The recorder's own data must survive the rewrite untouched.
        assert hdf5_file["data"]["demo_0"]["states"]["rigid_object"]["apple"]["root_pose"].shape == (3, 7)


def test_object_set_is_reported_as_an_error_instead_of_a_box(tmp_path):
    job = tmp_path / "job"
    _write_results(job / "episode_results_rebuild0.jsonl", "job")
    dataset = job / "dataset_job_rebuild0.hdf5"
    _write_dataset(dataset)

    reports = _annotate(tmp_path)

    assert "1 unresolved (plasticpail)" in reports[0]
    with h5py.File(dataset, "r") as hdf5_file:
        pail = hdf5_file[BOUNDING_BOXES_GROUP]["plasticpail"]
        assert "RigidObjectSet" in pail.attrs["error"]
        assert "min_point" not in pail


def test_second_run_skips_an_annotated_dataset_unless_overwritten(tmp_path):
    job = tmp_path / "job"
    _write_results(job / "episode_results_rebuild0.jsonl", "job")
    dataset = job / "dataset_job_rebuild0.hdf5"
    _write_dataset(dataset)

    assert _annotate(tmp_path)[0].startswith("wrote")
    assert _annotate(tmp_path)[0].startswith("skip (already annotated")
    assert _annotate(tmp_path, overwrite=True)[0].startswith("wrote")


def test_overwrite_recomputes_from_the_preserved_original(tmp_path):
    job = tmp_path / "job"
    _write_results(job / "episode_results_rebuild0.jsonl", "job")
    dataset = job / "dataset_job_rebuild0.hdf5"
    _write_dataset(dataset)

    _annotate(tmp_path)
    _annotate(tmp_path, overwrite=True)

    # However many times the tool runs, the backup must stay the pristine, unannotated recording.
    with h5py.File(dataset.with_suffix(ORIGINAL_DATASET_SUFFIX), "r") as preserved:
        assert BOUNDING_BOXES_GROUP not in preserved
    assert len(list(job.glob("*.hdf5"))) == 2, "no stray temporary or doubly-backed-up files"


def test_dry_run_leaves_the_dataset_untouched(tmp_path):
    job = tmp_path / "job"
    _write_results(job / "episode_results_rebuild0.jsonl", "job")
    dataset = job / "dataset_job_rebuild0.hdf5"
    _write_dataset(dataset)
    original_bytes = dataset.read_bytes()

    reports = _annotate(tmp_path, dry_run=True)

    assert reports[0].startswith("dry-run: would store 3 boxes")
    assert dataset.read_bytes() == original_bytes
    assert not dataset.with_suffix(ORIGINAL_DATASET_SUFFIX).exists()


def test_rank_results_files_annotate_their_shared_dataset_once(tmp_path):
    job = tmp_path / "job"
    _write_results(job / "episode_results_rebuild0_rank0.jsonl", "job")
    _write_results(job / "episode_results_rebuild0_rank1.jsonl", "job")
    _write_dataset(job / "dataset_job_rebuild0.hdf5")

    reports = _annotate(tmp_path)

    assert len(reports) == 1, f"both rank files name one dataset, so it is annotated once: {reports}"


def test_results_without_a_recorded_dataset_are_skipped(tmp_path):
    # A run with trajectory recording disabled leaves results and videos but no dataset.
    _write_results(tmp_path / "job" / "episode_results_rebuild0.jsonl", "job")

    reports = _annotate(tmp_path)

    assert len(reports) == 1 and reports[0].startswith("skip (no dataset recorded)")


def test_unresolvable_job_is_reported_without_aborting_the_run(tmp_path):
    for job_name in ("unknown_job", "job"):
        _write_results(tmp_path / job_name / "episode_results_rebuild0.jsonl", job_name)
        _write_dataset(tmp_path / job_name / f"dataset_{job_name}_rebuild0.hdf5")

    reports = sorted(_annotate(tmp_path))

    assert len(reports) == 2
    assert any("No task yaml" in report for report in reports)
    assert any(report.startswith("wrote") for report in reports)


def test_legacy_quaternion_format_is_refused(tmp_path):
    job = tmp_path / "job"
    _write_results(job / "episode_results_rebuild0.jsonl", "job")
    dataset = job / "dataset_job_rebuild0.hdf5"
    _write_dataset(dataset, format_version=0)
    original_bytes = dataset.read_bytes()

    reports = _annotate(tmp_path)

    assert reports[0].startswith("failed (AssertionError") and "format_version=0" in reports[0]
    assert dataset.read_bytes() == original_bytes
