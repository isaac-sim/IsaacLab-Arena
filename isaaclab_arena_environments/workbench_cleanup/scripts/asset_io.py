# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read original USD packages and export portable, independently editable assets."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from pxr import Ar, Sdf, Usd, UsdGeom, UsdUtils

from isaaclab_arena.assets.simready_central_asset_loader import download_central_asset_package, is_central_asset_url
from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections


def file_sha256(path: Path) -> str:
    """Return a file's SHA-256 without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file(usd_path: str, relative_to: Path) -> Path:
    """Download a pinned Central package or return an existing local USD."""
    if is_central_asset_url(usd_path):
        return Path(download_central_asset_package(usd_path))
    assert "://" not in usd_path, "Use a pinned Central URL or a local USD package."
    path = Path(usd_path).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    assert path.is_file(), f"Missing source USD: {path}"
    return path.absolute()


def open_asset(source: Path, variants: dict[str, str]) -> tuple[Usd.Stage, dict[str, str]]:
    """Create a /Asset reference wrapper, select variants, and flatten into an anonymous editable layer."""
    original = Usd.Stage.Open(str(source))
    assert original and original.GetDefaultPrim(), f"Source requires a default prim: {source}"
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
    apply_usd_variant_selections(stage, variants)
    assert not stage.GetCompositionErrors(), stage.GetCompositionErrors()
    source_hashes = {}
    for layer in stage.GetUsedLayers():
        if layer.realPath:
            source_hashes[layer.realPath] = file_sha256(Path(layer.realPath))
    # Copies of instance contents must be writable, including material subsets.
    while True:
        instances = [prim for prim in stage.Traverse() if prim.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    return Usd.Stage.Open(stage.Flatten()), source_hashes


def bounds_m(stage: Usd.Stage) -> list[list[float]]:
    """Return the default prim's world bounds in metres."""
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    aligned = bounds.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    assert not aligned.IsEmpty(), "Asset has no bounded geometry."
    metres = UsdGeom.GetStageMetersPerUnit(stage)
    return [[float(value) * metres for value in vector] for vector in (aligned.GetMin(), aligned.GetMax())]


def copy_dependencies(layer: Sdf.Layer, output_dir: Path) -> list[str]:
    """Copy flattened asset dependencies, including sibling MDL imports, and author relative paths."""
    copied: set[str] = set()
    copied_files: set[Path] = set()
    copied_mdl_directories: set[Path] = set()

    def copy_asset(asset_path):
        if not asset_path:
            return asset_path
        resolved = str(Ar.GetResolver().Resolve(asset_path))
        source = Path(resolved)
        assert resolved and source.is_file(), f"Unresolved file dependency: {asset_path}"
        # Source directory identity preserves MDL module/texture sibling relationships.
        key = hashlib.sha256(str(source.parent).encode()).hexdigest()[:16]
        folder = output_dir / "dependencies" / key
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / source.name
        assert source.resolve() != destination.resolve(), "Cannot replace a source dependency."
        if source.suffix.lower() == ".mdl":
            if source.parent not in copied_mdl_directories:
                for sibling in source.parent.iterdir():
                    if sibling.is_file():
                        shutil.copy2(sibling, folder / sibling.name)
                    elif sibling.is_dir():
                        shutil.copytree(sibling, folder / sibling.name, dirs_exist_ok=True)
                copied_mdl_directories.add(source.parent)
        elif source not in copied_files:
            shutil.copy2(source, destination)
            copied_files.add(source)
        relative = destination.relative_to(output_dir).as_posix()
        copied.add(relative)
        return "./" + relative

    UsdUtils.ModifyAssetPaths(layer, copy_asset)
    return sorted(copied)


def validate_package(path: Path) -> dict:
    """Check a flat USD's dependencies, topology, bindings and joint targets after export."""
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    assert not unresolved, f"Unresolved dependencies: {unresolved}"
    assert len(layers) == 1, "Output must be a single flattened USD layer."
    for asset in assets:
        assert Path(asset).resolve().is_relative_to(path.parent.resolve()), f"External dependency: {asset}"

    def check_path(asset_path):
        assert not os.path.isabs(asset_path) and "://" not in asset_path, f"Nonportable path: {asset_path}"
        return asset_path

    # Inspect an anonymous copy: validation must never modify the saved output.
    copy = Sdf.Layer.CreateAnonymous()
    copy.TransferContent(layers[0])
    UsdUtils.ModifyAssetPaths(copy, check_path)
    stage = Usd.Stage.Open(str(path))
    assert not stage.GetCompositionErrors(), stage.GetCompositionErrors()
    meshes = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            valid, reason = UsdGeom.Mesh.ValidateTopology(indices, counts, len(points))
            assert valid, f"{prim.GetPath()}: {reason}"
            meshes += 1
        for relationship in prim.GetRelationships():
            if relationship.GetName().startswith(("physics:body", "material:binding")):
                for target in relationship.GetTargets():
                    target_prim = stage.GetPrimAtPath(target.GetPrimPath())
                    assert (
                        target_prim and target_prim.IsActive()
                    ), f"Dangling target: {relationship.GetPath()} -> {target}"
    return {"usd_layers": 1, "file_dependencies": len(assets), "meshes": meshes, "bounds_m": bounds_m(stage)}
