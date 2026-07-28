# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for storing and reloading an articulation's collision mesh, keyed by joint pose."""

import contextlib
import numpy as np
import os
import shutil
import tempfile
import time
import trimesh
from pathlib import Path

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _build_arm_usd(tmp_dir: str, joint_type: str = "revolute") -> str:
    """Export the two-link fixture arm from the articulation tests to a USD file."""
    from isaaclab_arena.tests.test_usd_articulation import _build_two_link_arm

    usd_path = f"{tmp_dir}/arm.usda"
    _build_two_link_arm(joint_type=joint_type).Export(usd_path)
    return usd_path


@contextlib.contextmanager
def _published_dir(path: str):
    """Read published artifacts from path, keeping tests off Nucleus and away from real artifacts."""
    from isaaclab_arena.utils.collision_mesh_store import ROBOT_LIBRARY_DIR_ENV_VAR

    previous = os.environ.get(ROBOT_LIBRARY_DIR_ENV_VAR)
    os.environ[ROBOT_LIBRARY_DIR_ENV_VAR] = path
    try:
        yield
    finally:
        if previous is None:
            del os.environ[ROBOT_LIBRARY_DIR_ENV_VAR]
        else:
            os.environ[ROBOT_LIBRARY_DIR_ENV_VAR] = previous


def _test_trimming_evicts_least_recently_used_artifacts(simulation_app) -> bool:
    """Trimming drops artifacts oldest-first down to the budget, and leaves in-flight writes alone."""
    from isaaclab_arena.utils import collision_mesh_store as store

    with tempfile.TemporaryDirectory() as cache_dir:
        # Three 1 KiB artifacts, aged a day apart, plus a staging file another process is writing.
        paths = {}
        for age_days, name in enumerate(["newest.usd", "middle.usd", "oldest.usd"]):
            path = Path(cache_dir) / "robot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * 1024)
            os.utime(path, (0, time.time() - age_days * 86400))
            paths[name] = path
        staging = Path(cache_dir) / "robot" / f"{store._STAGING_PREFIX}half_written.usd"
        staging.write_bytes(b"x" * 4096)
        os.utime(staging, (0, time.time() - 7 * 86400))

        previous_budget = store.CACHE_BUDGET_BYTES
        store.CACHE_BUDGET_BYTES = 2048
        try:
            store._trim_cache(Path(cache_dir))
        finally:
            store.CACHE_BUDGET_BYTES = previous_budget

        assert paths["newest.usd"].is_file(), "the most recently used artifact must survive"
        assert paths["middle.usd"].is_file(), "the budget must be filled before evicting"
        assert not paths["oldest.usd"].is_file(), "the least recently used artifact must be evicted"
        # Unlinking a staging file would fail the rename of whichever process is writing it.
        assert staging.is_file(), "a half-written artifact must not be evicted despite being oldest"
    return True


def _test_pose_keys_identify_the_pose(simulation_app) -> bool:
    """Every all-zero spelling shares the readable zero key; distinct poses key apart."""
    from isaaclab_arena.utils.collision_mesh_store import is_zero_pose, pose_key

    assert is_zero_pose({}) and is_zero_pose({"elbow": 0.0, "wrist": -0.0})
    assert not is_zero_pose({"elbow": 0.5})

    assert pose_key({}) == "zero"
    assert pose_key({"elbow": 0.0, "wrist": -0.0}) == "zero"
    assert pose_key({"elbow": 0.5}) != "zero"
    # Order must not matter, but the values must.
    assert pose_key({"a": 0.5, "b": 0.25}) == pose_key({"b": 0.25, "a": 0.5})
    assert pose_key({"elbow": 0.5}) != pose_key({"elbow": 0.6})
    assert pose_key({"elbow": 0.5}) != pose_key({"wrist": 0.5})
    return True


def _test_distinct_poses_get_distinct_artifacts(simulation_app) -> bool:
    """Posing at one configuration must never serve the mesh for another."""
    from isaaclab_arena.utils import usd_helpers
    from isaaclab_arena.utils.collision_mesh_store import mesh_cache_path

    extended = {"elbow": 0.5}
    with tempfile.TemporaryDirectory() as tmp_dir, contextlib.ExitStack() as published:
        usd_path = _build_arm_usd(tmp_dir, joint_type="prismatic")
        published.enter_context(_published_dir(tmp_dir))
        for joint_pos in ({}, extended):
            mesh_cache_path(usd_path, joint_pos).unlink(missing_ok=True)

        usd_helpers._extract_trimesh_from_usd_at_joint_pos.cache_clear()
        zero_mesh = usd_helpers.extract_trimesh_from_usd_at_joint_pos(usd_path, {})
        usd_helpers._extract_trimesh_from_usd_at_joint_pos.cache_clear()
        posed_mesh = usd_helpers.extract_trimesh_from_usd_at_joint_pos(usd_path, extended)

        assert mesh_cache_path(usd_path, {}).is_file() and mesh_cache_path(usd_path, extended).is_file()
        assert mesh_cache_path(usd_path, {}) != mesh_cache_path(usd_path, extended)
        assert posed_mesh.extents[0] > zero_mesh.extents[0] + 0.4, f"prismatic pose should extend: {posed_mesh.extents}"

        # Reloading from the store must preserve that difference.
        usd_helpers._extract_trimesh_from_usd_at_joint_pos.cache_clear()
        reloaded_zero = usd_helpers.extract_trimesh_from_usd_at_joint_pos(usd_path, {})
        np.testing.assert_allclose(reloaded_zero.extents, zero_mesh.extents, atol=1e-6)
        for joint_pos in ({}, extended):
            mesh_cache_path(usd_path, joint_pos).unlink(missing_ok=True)
    return True


def _test_stored_mesh_keeps_its_vertices_verbatim(simulation_app) -> bool:
    """Reloading must not merge coincident vertices, which trimesh does by default on construction."""
    from isaaclab_arena.utils.collision_mesh_store import load_mesh, mesh_cache_path, save_mesh

    # Two coincident triangles: merging would collapse six vertices into three.
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] * 2)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array([[0, 1, 2], [3, 4, 5]]), process=False)

    with tempfile.TemporaryDirectory() as tmp_dir, contextlib.ExitStack() as published:
        source = f"{tmp_dir}/synthetic.usda"
        published.enter_context(_published_dir(tmp_dir))
        save_mesh(source, {}, mesh)
        loaded = load_mesh(source, {}, (1.0, 1.0, 1.0))
        assert loaded is not None
        assert len(loaded.vertices) == len(vertices), f"round trip changed vertex count: {len(loaded.vertices)}"

        scaled = load_mesh(source, {}, (2.0, 2.0, 2.0))
        assert scaled is not None
        assert len(scaled.vertices) == len(vertices), f"scaling changed vertex count: {len(scaled.vertices)}"
        mesh_cache_path(source, {}).unlink(missing_ok=True)
    return True


def _test_stored_mesh_is_scale_independent(simulation_app) -> bool:
    """One unit-scale artifact serves every spawn scale."""
    from isaaclab_arena.utils import usd_helpers
    from isaaclab_arena.utils.collision_mesh_store import mesh_cache_path

    with tempfile.TemporaryDirectory() as tmp_dir, contextlib.ExitStack() as published:
        usd_path = _build_arm_usd(tmp_dir)
        published.enter_context(_published_dir(tmp_dir))
        mesh_cache_path(usd_path, {}).unlink(missing_ok=True)

        usd_helpers._extract_trimesh_from_usd_at_joint_pos.cache_clear()
        unit = usd_helpers.extract_trimesh_from_usd_at_joint_pos(usd_path, {})
        # Served from the store now, and must still honour the requested scale.
        usd_helpers._extract_trimesh_from_usd_at_joint_pos.cache_clear()
        doubled = usd_helpers.extract_trimesh_from_usd_at_joint_pos(usd_path, {}, scale=(2.0, 1.0, 1.0))
        np.testing.assert_allclose(doubled.extents[0], unit.extents[0] * 2.0, atol=1e-6)
        np.testing.assert_allclose(doubled.extents[1:], unit.extents[1:], atol=1e-6)
        mesh_cache_path(usd_path, {}).unlink(missing_ok=True)
    return True


def _test_artifact_from_another_asset_or_pose_is_ignored(simulation_app) -> bool:
    """A mesh recording a different source or pose is refused, so it cannot misplace a robot."""
    from isaaclab_arena.utils.collision_mesh_store import load_mesh, mesh_cache_path, save_mesh
    from isaaclab_arena.utils.usd_helpers import extract_trimesh_from_usd_at_joint_pos

    posed = {"elbow": 0.5}
    with tempfile.TemporaryDirectory() as tmp_dir, contextlib.ExitStack() as published:
        usd_path = _build_arm_usd(tmp_dir)
        published.enter_context(_published_dir(tmp_dir))
        mesh = extract_trimesh_from_usd_at_joint_pos(usd_path, posed)

        # Recorded under a foreign source USD, then moved into this asset's slot.
        save_mesh(f"{tmp_dir}/other_arm.usda", posed, mesh).replace(mesh_cache_path(usd_path, posed))
        assert load_mesh(usd_path, posed, (1.0, 1.0, 1.0)) is None, "foreign source must be ignored"

        # Recorded for this asset but at another pose, then moved into this pose's slot.
        save_mesh(usd_path, {"elbow": 0.9}, mesh).replace(mesh_cache_path(usd_path, posed))
        assert load_mesh(usd_path, posed, (1.0, 1.0, 1.0)) is None, "foreign pose must be ignored"
        mesh_cache_path(usd_path, posed).unlink(missing_ok=True)
    return True


def _test_published_artifact_is_found_by_its_name(simulation_app) -> bool:
    """An exported artifact is loaded from the published directory when the local cache is empty."""
    from isaaclab_arena.utils.collision_mesh_store import (
        export_ready_pose_mesh,
        load_mesh,
        mesh_cache_path,
        published_ready_pose_path,
        ready_pose_artifact_name,
    )
    from isaaclab_arena.utils.usd_helpers import extract_trimesh_from_usd_at_joint_pos

    posed = {"elbow": 0.5}
    with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as publish_dir:
        usd_path = _build_arm_usd(tmp_dir)
        mesh = extract_trimesh_from_usd_at_joint_pos(usd_path, posed)
        artifact = export_ready_pose_mesh(usd_path, posed, mesh, Path(publish_dir), "test_arm")

        # The artifact lands in the robot's own library folder, under a name derived from the asset.
        assert artifact.name == "arm_ready_pose.usd" == ready_pose_artifact_name(usd_path)
        assert artifact.parent.name == "test_arm", artifact

        with _published_dir(publish_dir):
            assert published_ready_pose_path(usd_path, "test_arm") == str(artifact)
            # Empty the local cache so only the published copy can answer.
            mesh_cache_path(usd_path, posed).unlink(missing_ok=True)
            loaded = load_mesh(usd_path, posed, (1.0, 1.0, 1.0), "test_arm")
            assert loaded is not None, f"published artifact {artifact} was not found"
            np.testing.assert_allclose(loaded.vertices, mesh.vertices, atol=1e-6)

            # A robot that names no folder must not be served another robot's artifact.
            mesh_cache_path(usd_path, posed).unlink(missing_ok=True)
            assert load_mesh(usd_path, posed, (1.0, 1.0, 1.0)) is None, "published lookup needs a folder"

            # Published artifacts outlive config changes, so a repose must not be served the old shape.
            assert load_mesh(usd_path, {"elbow": 0.9}, (1.0, 1.0, 1.0), "test_arm") is None, "stale pose"
        mesh_cache_path(usd_path, posed).unlink(missing_ok=True)
    return True


def _test_published_artifact_loads_for_a_relocated_source(simulation_app) -> bool:
    """An artifact stays usable when the robot USD sits elsewhere, as on another machine.

    Arena composes the robot-on-stand USDs into a per-user cache directory, so validating against the
    full source path would reject every artifact anyone else exported.
    """
    from isaaclab_arena.utils.collision_mesh_store import export_ready_pose_mesh, load_mesh, mesh_cache_path
    from isaaclab_arena.utils.usd_helpers import extract_trimesh_from_usd_at_joint_pos

    posed = {"elbow": 0.5}
    with (
        tempfile.TemporaryDirectory() as exporter_dir,
        tempfile.TemporaryDirectory() as consumer_dir,
        tempfile.TemporaryDirectory() as publish_dir,
    ):
        exporter_usd = _build_arm_usd(exporter_dir)
        mesh = extract_trimesh_from_usd_at_joint_pos(exporter_usd, posed)
        export_ready_pose_mesh(exporter_usd, posed, mesh, Path(publish_dir), "test_arm")

        # The same asset reached through another absolute path, as a second machine's cache would.
        consumer_usd = f"{consumer_dir}/arm.usda"
        shutil.copy(exporter_usd, consumer_usd)

        with _published_dir(publish_dir):
            mesh_cache_path(consumer_usd, posed).unlink(missing_ok=True)
            loaded = load_mesh(consumer_usd, posed, (1.0, 1.0, 1.0), "test_arm")
            assert loaded is not None, "an artifact must load for the same asset at another path"
            np.testing.assert_allclose(loaded.vertices, mesh.vertices, atol=1e-6)
        for path in (exporter_usd, consumer_usd):
            mesh_cache_path(path, posed).unlink(missing_ok=True)
    return True


def _test_truncated_artifact_is_ignored(simulation_app) -> bool:
    """A half-written or hand-edited artifact falls back to extraction rather than raising."""
    from pxr import Usd, UsdGeom

    from isaaclab_arena.utils.collision_mesh_store import load_mesh, save_mesh
    from isaaclab_arena.utils.usd_helpers import extract_trimesh_from_usd_at_joint_pos

    posed = {"elbow": 0.5}
    with tempfile.TemporaryDirectory() as tmp_dir, contextlib.ExitStack() as published:
        usd_path = _build_arm_usd(tmp_dir)
        published.enter_context(_published_dir(tmp_dir))
        cache_path = save_mesh(usd_path, posed, extract_trimesh_from_usd_at_joint_pos(usd_path, posed))
        assert load_mesh(usd_path, posed, (1.0, 1.0, 1.0)) is not None

        # Drop the geometry but keep the provenance, as a truncated write would.
        stage = Usd.Stage.Open(str(cache_path))
        mesh_prim = UsdGeom.Mesh(stage.GetPrimAtPath("/CollisionMesh"))
        mesh_prim.GetPointsAttr().Clear()
        mesh_prim.GetFaceVertexIndicesAttr().Clear()
        stage.GetRootLayer().Save()

        assert load_mesh(usd_path, posed, (1.0, 1.0, 1.0)) is None, "an artifact without geometry must be ignored"
        cache_path.unlink(missing_ok=True)
    return True


def _test_colliding_published_names_are_refused(simulation_app) -> bool:
    """Two robots sharing a stem within one folder cannot both be published, as one name serves both."""
    from isaaclab_arena.embodiments.embodiment_base import PlacementGeometrySource
    from isaaclab_arena.scripts.export_ready_pose_collision_meshes import plan_artifacts

    def source(usd_path: str, library_folder: str | None) -> PlacementGeometrySource:
        return PlacementGeometrySource(usd_path, (1.0, 1.0, 1.0), {}, library_folder)

    # Same stem in one folder, different assets: the published name cannot tell them apart.
    colliding = {
        "robot_a": source("/assets/vendor_a/robot.usd", "shared"),
        "robot_b": source("/assets/vendor_b/robot.usd", "shared"),
    }
    try:
        plan_artifacts(colliding)
    except AssertionError as error:
        assert "rename one of the source USDs" in str(error), error
    else:
        raise AssertionError("a colliding published name must be refused")

    # The same stem in separate folders is fine, which is what per-robot folders buy.
    separate = {
        "robot_a": source("/assets/vendor_a/robot.usd", "robot_a"),
        "robot_b": source("/assets/vendor_b/robot.usd", "robot_b"),
    }
    assert sorted(plan_artifacts(separate)) == ["robot_a/robot_ready_pose.usd", "robot_b/robot_ready_pose.usd"]

    # Action-space variants of one robot share a USD, and so legitimately share one artifact.
    shared_usd = source("/assets/vendor_a/robot.usd", "robot_a")
    assert sorted(plan_artifacts({"robot_ik": shared_usd, "robot_joint_pos": shared_usd})) == [
        "robot_a/robot_ready_pose.usd"
    ]

    # A robot with nowhere to publish is left out rather than written to a guessed folder.
    assert plan_artifacts({"unpublished": source("/assets/vendor_a/robot.usd", None)}) == {}
    return True


def _test_export_writes_one_named_artifact_per_robot(simulation_app) -> bool:
    """The exporter deduplicates embodiments sharing a USD and reports nothing as failed."""
    from isaaclab_arena.scripts.export_ready_pose_collision_meshes import export_ready_pose_meshes

    # Droid's three action-space variants share one USD and one ready pose.
    droid_variants = ["droid_abs_joint_pos", "droid_rel_joint_pos", "droid_differential_ik"]
    with tempfile.TemporaryDirectory() as out_dir:
        written, failed = export_ready_pose_meshes(Path(out_dir), droid_variants)

        assert not failed, f"export reported failures: {failed}"
        assert len(written) == 1, f"variants of one robot must share an artifact: {written}"
        # Relative paths, because the staging tree is uploaded into the robot library as it stands.
        artifacts = sorted(str(path.relative_to(out_dir)) for path in Path(out_dir).rglob("*.usd"))
        assert artifacts == ["droid/droid_franka_robotiq_on_stand_1.350_ready_pose.usd"], artifacts
    return True


def test_trimming_evicts_least_recently_used_artifacts():
    assert run_simulation_app_function(_test_trimming_evicts_least_recently_used_artifacts, headless=HEADLESS)


def test_pose_keys_identify_the_pose():
    assert run_simulation_app_function(_test_pose_keys_identify_the_pose, headless=HEADLESS)


def test_truncated_artifact_is_ignored():
    assert run_simulation_app_function(_test_truncated_artifact_is_ignored, headless=HEADLESS)


def test_colliding_published_names_are_refused():
    assert run_simulation_app_function(_test_colliding_published_names_are_refused, headless=HEADLESS)


def test_export_writes_one_named_artifact_per_robot():
    assert run_simulation_app_function(_test_export_writes_one_named_artifact_per_robot, headless=HEADLESS)


def test_published_artifact_is_found_by_its_name():
    assert run_simulation_app_function(_test_published_artifact_is_found_by_its_name, headless=HEADLESS)


def test_published_artifact_loads_for_a_relocated_source():
    assert run_simulation_app_function(_test_published_artifact_loads_for_a_relocated_source, headless=HEADLESS)


def test_distinct_poses_get_distinct_artifacts():
    assert run_simulation_app_function(_test_distinct_poses_get_distinct_artifacts, headless=HEADLESS)


def test_stored_mesh_keeps_its_vertices_verbatim():
    assert run_simulation_app_function(_test_stored_mesh_keeps_its_vertices_verbatim, headless=HEADLESS)


def test_stored_mesh_is_scale_independent():
    assert run_simulation_app_function(_test_stored_mesh_is_scale_independent, headless=HEADLESS)


def test_artifact_from_another_asset_or_pose_is_ignored():
    assert run_simulation_app_function(_test_artifact_from_another_asset_or_pose_is_ignored, headless=HEADLESS)
