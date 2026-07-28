# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Persist an articulation's posed collision mesh as a USD, so extraction runs once per pose.

Walking a robot USD to merge its meshes costs 0.1-2.5 s depending on the robot, which is paid again
in every new process. Artifacts are keyed by the joint pose they were extracted at, because that is
what makes a mesh reusable: embodiments spawn at their configured pose rather than at zero, so keying
on the asset alone would store a mesh nobody asks for.

Lookup order is the local cache, then the robot's own folder under ``ARENA_ROBOT_LIBRARY_DIR`` on
Nucleus, which holds one published artifact per robot at the pose that robot spawns in, written by
``scripts/export_ready_pose_collision_meshes.py``. A miss falls back to extraction.

Artifacts are named and validated by the source USD's stem, so a file exported on one machine loads
on another even though Arena composes the robot-on-stand USDs into a per-user cache directory. The
stem only has to be unique within a robot's folder, and a full export refuses to write a pair that
would collide there, since at a shared pose one robot would otherwise be served the other's mesh.
"""

from __future__ import annotations

import contextlib
import hashlib
import numpy as np
import os
import tempfile
import trimesh
from collections.abc import Mapping
from pathlib import Path

from pxr import Usd, UsdGeom

from isaaclab_arena.assets.asset_cache import get_arena_usd_cache_dir
from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR

_MESH_PRIM_PATH = "/CollisionMesh"
"""Prim the merged mesh is authored at, also the artifact's default prim."""

_ASSET_KEY = "arenaAssetKey"
"""customData key recording which asset the mesh describes, validated on load."""

_SOURCE_KEY = "arenaSourceUsdPath"
"""customData key recording the exporting machine's source path, kept to trace an artifact back."""

_POSE_KEY = "arenaJointPoseKey"
"""customData key recording which joint pose the mesh was extracted at, validated on load."""

_ZERO_POSE_KEY = "zero"
"""Pose key for an all-zero pose, spelled out so cached artifacts stay readable."""

_READY_POSE_SUFFIX = "_ready_pose.usd"
"""Suffix naming a published artifact, one per robot, at the pose that robot spawns in."""

ROBOT_LIBRARY_DIR_ENV_VAR = "ISAACLAB_ARENA_ROBOT_LIBRARY_DIR"
"""Environment variable redirecting which robot library published artifacts are read from."""

ARENA_ROBOT_LIBRARY_DIR = f"{ARENA_NUCLEUS_DIR}/Arena/assets/robot_library"
"""Nucleus robot library, holding each robot's assets in a folder of its own."""

_LOCAL_CACHE_DIR = "collision_mesh"
"""Local cache root under ``get_arena_usd_cache_dir()``, with one subfolder per robot when known."""

_STAGING_PREFIX = "_staging_"
"""Prefix marking a half-written artifact, so trimming leaves another process's write alone.

Staging has to share the destination's directory for the rename to stay atomic, and the suffix has
to stay a USD one for ``Usd.Stage.CreateNew``, so the prefix is what keeps the two apart.
"""

CACHE_BUDGET_BYTES = 1 << 30
"""Disk the local cache may occupy before its least recently used artifacts are dropped.

A robot's mesh runs 1-9 MB, so this holds a few hundred: every robot Arena ships at several poses
each, while bounding what a run against throwaway USDs can leave behind.
"""


def local_collision_mesh_cache_dir() -> Path:
    """Return the local collision-mesh cache root (``~/.cache/.../usd/collision_mesh``)."""
    return get_arena_usd_cache_dir() / _LOCAL_CACHE_DIR


def scaled_mesh(mesh: trimesh.Trimesh, scale: tuple[float, float, float]) -> trimesh.Trimesh:
    """Return mesh with vertices scaled per axis in its own frame.

    Scaling in the root frame commutes with merging prims, so scaling a merged mesh matches scaling
    each prim's vertices during extraction.
    """
    if tuple(scale) == (1.0, 1.0, 1.0):
        return mesh
    # process=False keeps trimesh from merging vertices, which would make a scaled or reloaded mesh
    # differ from the one extraction produced.
    return trimesh.Trimesh(
        vertices=mesh.vertices * np.asarray(scale, dtype=np.float64), faces=mesh.faces, process=False
    )


def is_zero_pose(joint_pos: Mapping[str, float]) -> bool:
    """Whether joint_pos poses every joint at zero, including by naming no joints at all.

    Omitted joints are posed at zero, so an empty mapping is the zero pose rather than the asset's
    authored configuration.
    """
    return all(float(value) == 0.0 for value in joint_pos.values())


def pose_key(joint_pos: Mapping[str, float]) -> str:
    """Return a filename-safe key identifying a joint pose.

    Keys are derived from the joint names and values as given, so two spellings of one pose, a regex
    and the names it expands to, key differently and are extracted separately.
    """
    if is_zero_pose(joint_pos):
        return _ZERO_POSE_KEY
    canonical = ";".join(f"{name}={float(value) + 0.0:.9g}" for name, value in sorted(joint_pos.items()))
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def asset_key(source_usd_path: str) -> str:
    """Return the identity a published artifact is named and validated by: the source USD's stem."""
    # The stem rather than the full path, so an artifact exported from a per-user cache directory
    # still matches elsewhere. Stems already spell out the variant: droid_franka_robotiq_on_stand_1.350.
    return Path(str(source_usd_path)).stem


def ready_pose_artifact_name(source_usd_path: str) -> str:
    """Return the filename a robot USD's ready-pose mesh is published under."""
    return f"{asset_key(source_usd_path)}{_READY_POSE_SUFFIX}"


def published_ready_pose_dir() -> str:
    """Return the robot library that published ready-pose artifacts are read from.

    Defaults to the Arena Nucleus robot library and is redirected by
    ``ISAACLAB_ARENA_ROBOT_LIBRARY_DIR``, which lets an export be checked locally before upload.
    """
    return os.environ.get(ROBOT_LIBRARY_DIR_ENV_VAR) or ARENA_ROBOT_LIBRARY_DIR


def published_ready_pose_path(source_usd_path: str, library_folder: str) -> str:
    """Return the full path a robot USD's ready-pose mesh is published at.

    The artifact sits in the robot's own library folder rather than beside its source USD, because a
    robot that spawns a composed asset sources from a per-user cache path that is nobody else's.
    """
    return f"{published_ready_pose_dir().rstrip('/')}/{library_folder}/{ready_pose_artifact_name(source_usd_path)}"


def mesh_cache_path(source_usd_path: str, joint_pos: Mapping[str, float], library_folder: str | None = None) -> Path:
    """Return the local cache path for a robot USD's mesh at joint_pos.

    Layout is ``collision_mesh/{library_folder}/{filename}`` when ``library_folder`` is set (the
    embodiment's ``robot_library_folder``), otherwise ``collision_mesh/{filename}`` for throwaway
    fixtures that have no published robot folder. The full source path is hashed into the filename
    so assets sharing a stem stay distinct on one machine even though they would share a published
    name.
    """
    source_digest = hashlib.sha1(str(source_usd_path).encode()).hexdigest()[:12]
    name = f"{asset_key(source_usd_path)}_{source_digest}_collision_{pose_key(joint_pos)}_pose.usd"
    cache_root = local_collision_mesh_cache_dir()
    if library_folder:
        return cache_root / library_folder / name
    return cache_root / name


def load_mesh(
    source_usd_path: str,
    joint_pos: Mapping[str, float],
    scale: tuple[float, float, float],
    library_folder: str | None = None,
) -> trimesh.Trimesh | None:
    """Return the stored mesh for a robot USD at joint_pos scaled to scale, or None if unavailable.

    Reads the local cache first and the published robot library second, copying a published hit
    into the local cache on the way out. An artifact recording another asset or another pose is
    ignored rather than trusted.

    Args:
        source_usd_path: Robot USD the mesh should describe.
        joint_pos: Joint positions the mesh should be posed at.
        scale: Per-axis scale to apply to the stored vertices.
        library_folder: Robot's folder in the published library (and local cache), or None to
            skip the published lookup and use the flat local-cache path.
    """
    cached = mesh_cache_path(source_usd_path, joint_pos, library_folder)
    mesh = _read_mesh_usd(str(cached), expected_asset=asset_key(source_usd_path), expected_pose=pose_key(joint_pos))
    if mesh is not None:
        # Mark the artifact used, so trimming evicts by last use rather than by write time.
        with contextlib.suppress(OSError):
            os.utime(cached)
        return scaled_mesh(mesh, scale)

    if library_folder is None:
        return None
    published = published_ready_pose_path(source_usd_path, library_folder)
    mesh = _read_mesh_usd(published, expected_asset=asset_key(source_usd_path), expected_pose=pose_key(joint_pos))
    if mesh is None:
        return None
    # Copy the published artifact locally, so only the first process pays the Nucleus round trip.
    with contextlib.suppress(OSError):
        save_mesh(source_usd_path, joint_pos, mesh, library_folder)
    return scaled_mesh(mesh, scale)


def save_mesh(
    source_usd_path: str,
    joint_pos: Mapping[str, float],
    mesh: trimesh.Trimesh,
    library_folder: str | None = None,
) -> Path:
    """Write a robot USD's unscaled mesh at joint_pos to the local cache and return its path.

    Least recently used artifacts are dropped to keep the cache within ``CACHE_BUDGET_BYTES``.

    Args:
        source_usd_path: Robot USD the mesh was extracted from, recorded for validation on load.
        joint_pos: Joint positions the mesh was posed at, recorded for validation on load.
        mesh: Merged mesh at unit scale, in the robot's default-prim frame.
        library_folder: Robot's folder under the local cache, or None for a flat cache path.
    """
    out_path = _write_mesh_usd(
        mesh_cache_path(source_usd_path, joint_pos, library_folder), source_usd_path, joint_pos, mesh
    )
    _trim_cache(local_collision_mesh_cache_dir())
    return out_path


def _trim_cache(cache_dir: Path) -> None:
    """Delete cached artifacts, least recently used first, until the cache fits its budget.

    Artifacts another process is still staging are left alone, since unlinking one would fail that
    process's rename.
    """
    if not cache_dir.is_dir():
        return
    artifacts = (path for path in cache_dir.rglob("*.usd") if not path.name.startswith(_STAGING_PREFIX))
    cumulative_bytes = 0
    for path in sorted(artifacts, key=_last_used, reverse=True):
        with contextlib.suppress(OSError):
            # Counts what has been walked rather than what survives, so eviction keeps a strict
            # most-recently-used prefix instead of backfilling with whatever happens to fit.
            cumulative_bytes += path.stat().st_size
            if cumulative_bytes > CACHE_BUDGET_BYTES:
                path.unlink()


def _last_used(path: Path) -> float:
    """Return when an artifact was last read or written, or 0.0 if it has since gone."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def export_ready_pose_mesh(
    source_usd_path: str, joint_pos: Mapping[str, float], mesh: trimesh.Trimesh, out_dir: Path, library_folder: str
) -> Path:
    """Write a robot's ready-pose mesh into out_dir under its published name, for upload.

    Args:
        source_usd_path: Robot USD the mesh was extracted from.
        joint_pos: The robot's configured joint positions, recorded for validation on load.
        mesh: Merged mesh at unit scale, in the robot's default-prim frame.
        out_dir: Staging directory whose layout mirrors ``ARENA_ROBOT_LIBRARY_DIR`` for upload as-is.
        library_folder: Robot's folder in the published library, created under out_dir.
    """
    artifact_path = out_dir / library_folder / ready_pose_artifact_name(source_usd_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_mesh_usd(artifact_path, source_usd_path, joint_pos, mesh)


def _write_mesh_usd(
    out_path: Path, source_usd_path: str, joint_pos: Mapping[str, float], mesh: trimesh.Trimesh
) -> Path:
    """Author a mesh artifact at out_path, staged and renamed so no reader sees a partial file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=_STAGING_PREFIX, suffix=".usd", dir=out_path.parent, delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        stage = Usd.Stage.CreateNew(str(tmp_path))
        mesh_prim = UsdGeom.Mesh.Define(stage, _MESH_PRIM_PATH)
        stage.SetDefaultPrim(mesh_prim.GetPrim())
        faces = np.asarray(mesh.faces, dtype=np.int32)
        mesh_prim.GetPointsAttr().Set(np.asarray(mesh.vertices, dtype=np.float32))
        mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(faces))
        mesh_prim.GetFaceVertexIndicesAttr().Set(faces.reshape(-1))
        mesh_prim.GetPrim().SetCustomDataByKey(_ASSET_KEY, asset_key(source_usd_path))
        mesh_prim.GetPrim().SetCustomDataByKey(_POSE_KEY, pose_key(joint_pos))
        mesh_prim.GetPrim().SetCustomDataByKey(_SOURCE_KEY, str(source_usd_path))
        assert stage.GetRootLayer().Save(), f"failed to save collision mesh to {tmp_path}"
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return out_path


def _read_mesh_usd(usd_path: str, expected_asset: str, expected_pose: str) -> trimesh.Trimesh | None:
    """Read a stored artifact, returning None when it is absent or describes another asset or pose."""
    from isaaclab.utils.assets import check_file_path

    # Stage.Open raises rather than returning None on a missing remote path, and a published artifact
    # is missing for every robot nobody has exported yet.
    if check_file_path(usd_path) == 0:
        return None
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return None
    mesh_prim = UsdGeom.Mesh(stage.GetPrimAtPath(_MESH_PRIM_PATH))
    if not mesh_prim:
        return None

    recorded = (
        mesh_prim.GetPrim().GetCustomDataByKey(_ASSET_KEY),
        mesh_prim.GetPrim().GetCustomDataByKey(_POSE_KEY),
    )
    if recorded != (expected_asset, expected_pose):
        print(f"Ignoring collision mesh {usd_path}: recorded {recorded} != {(expected_asset, expected_pose)}")
        return None

    points = mesh_prim.GetPointsAttr().Get()
    indices = mesh_prim.GetFaceVertexIndicesAttr().Get()
    if points is None or indices is None:
        return None
    return trimesh.Trimesh(
        vertices=np.asarray(points, dtype=np.float64),
        faces=np.asarray(indices, dtype=np.int32).reshape(-1, 3),
        process=False,
    )
