# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic derivation and caching for USD packages."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.resources
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir

if TYPE_CHECKING:
    from isaaclab_arena.assets.sources.base import UsdSource

DERIVATION_PIPELINE_VERSION = 1
"""Version of the derived-package layout and execution contract."""

OVERLAY_PIPELINE_VERSION = 1
"""Version of the cached USDA overlay contract."""


def derive_usd_package(source: str | UsdSource, derivation_spec: str, *, source_identity: str) -> str:
    """Build a complete package atomically on a cache miss and return its derived USD entry."""
    recipe_bytes = _read_derivation_spec(derivation_spec)
    recipe = _parse_recipe(recipe_bytes)
    assert (
        recipe["source_identity"] == source_identity
    ), f"recipe source identity {recipe['source_identity']!r} does not match {source_identity!r}"
    digest = _cache_digest(source_identity, recipe_bytes)
    cache_root = get_arena_asset_cache_dir() / "derived"
    cache_root.mkdir(parents=True, exist_ok=True)
    package_dir = cache_root / digest
    if not _is_valid_cache_entry(package_dir, digest, DERIVATION_PIPELINE_VERSION):
        with _exclusive_lock(cache_root / f"{digest}.lock"):
            if not _is_valid_cache_entry(package_dir, digest, DERIVATION_PIPELINE_VERSION):
                _build_derived_package(
                    source=source,
                    source_identity=source_identity,
                    recipe=recipe,
                    recipe_bytes=recipe_bytes,
                    digest=digest,
                    cache_root=cache_root,
                    package_dir=package_dir,
                )
    return str(package_dir / recipe["output_entry"])


def derivation_identity(source_identity: str, derivation_spec: str) -> str:
    """Return the stable identity of a derived USD without resolving its source."""
    recipe_bytes = _read_derivation_spec(derivation_spec)
    return f"derived-usd:{_cache_digest(source_identity, recipe_bytes)}"


def cache_usd_overlay(
    source: str | UsdSource,
    overlay_spec: str,
    *,
    source_identity: str,
    expected_source_sha256: str | None = None,
) -> str:
    """Cache a USDA overlay whose root reference is replaced with the resolved source."""
    overlay_bytes = _read_spec(overlay_spec)
    output_name = PurePosixPath(overlay_spec).name
    assert output_name.endswith(".usda"), "overlay_spec must name a USDA file"
    identity_bytes = b"\0".join((
        b"usd-overlay",
        str(OVERLAY_PIPELINE_VERSION).encode(),
        source_identity.encode(),
        (expected_source_sha256 or "").encode(),
        overlay_bytes,
    ))
    digest = hashlib.sha256(identity_bytes).hexdigest()
    cache_root = get_arena_asset_cache_dir() / "overlays"
    cache_root.mkdir(parents=True, exist_ok=True)
    package_dir = cache_root / digest
    if not _is_valid_overlay_entry(package_dir, digest):
        with _exclusive_lock(cache_root / f"{digest}.lock"):
            if not _is_valid_overlay_entry(package_dir, digest):
                _build_overlay(
                    source=source,
                    source_identity=source_identity,
                    expected_source_sha256=expected_source_sha256,
                    overlay_bytes=overlay_bytes,
                    output_name=output_name,
                    digest=digest,
                    cache_root=cache_root,
                    package_dir=package_dir,
                )
    return str(package_dir / output_name)


def _build_derived_package(
    *,
    source: str | UsdSource,
    source_identity: str,
    recipe: dict[str, Any],
    recipe_bytes: bytes,
    digest: str,
    cache_root: Path,
    package_dir: Path,
) -> None:
    # Preserve the Hugging Face snapshot path. Resolving its symlink would discard
    # the package-relative location and leave only the content-addressed blob path.
    source_path = Path(source if isinstance(source, str) else source.resolve()).expanduser().absolute()
    assert source_path.is_file(), f"USD source did not resolve to a file: {source_path}"
    source_entry = PurePosixPath(recipe["source_entry"])
    source_root = _package_root(source_path, source_entry)
    expected_source_hash = recipe["expected_source_sha256"]
    source_hash = _sha256_file(source_path)
    assert (
        source_hash == expected_source_hash
    ), f"source entry hash {source_hash} does not match recipe hash {expected_source_hash}"

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=cache_root))
    try:
        shutil.copytree(source_root, staging_dir, dirs_exist_ok=True, symlinks=False)
        from isaaclab_arena.assets.derivation.pipeline import derive_asset

        derive_asset(staging_dir, recipe)
        output_path = staging_dir / recipe["output_entry"]
        assert output_path.is_file(), f"derivation did not create output entry: {output_path}"
        files = _package_file_hashes(staging_dir)
        manifest = {
            "cache_key": digest,
            "pipeline_version": DERIVATION_PIPELINE_VERSION,
            "source_identity": source_identity,
            "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
            "source_entry": recipe["source_entry"],
            "source_entry_sha256": source_hash,
            "output_entry": recipe["output_entry"],
            "output_sha256": files[recipe["output_entry"]],
            "dependencies": sorted(
                path for path in files if path not in (recipe["source_entry"], recipe["output_entry"])
            ),
            "files": files,
        }
        (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if package_dir.exists():
            shutil.rmtree(package_dir)
        os.replace(staging_dir, package_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _build_overlay(
    *,
    source: str | UsdSource,
    source_identity: str,
    expected_source_sha256: str | None,
    overlay_bytes: bytes,
    output_name: str,
    digest: str,
    cache_root: Path,
    package_dir: Path,
) -> None:
    from pxr import Sdf, Usd

    source_path = Path(source if isinstance(source, str) else source.resolve()).expanduser().absolute()
    assert source_path.is_file(), f"USD source did not resolve to a file: {source_path}"
    source_hash = _sha256_file(source_path)
    if expected_source_sha256 is not None:
        assert (
            source_hash == expected_source_sha256
        ), f"source entry hash {source_hash} does not match expected hash {expected_source_sha256}"

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=cache_root))
    try:
        output = staging_dir / output_name
        layer = Sdf.Layer.CreateAnonymous()
        assert layer.ImportFromString(overlay_bytes.decode()), f"Cannot parse USDA overlay: {output_name}"
        root_path = Sdf.Path.absoluteRootPath.AppendChild(layer.defaultPrim)
        assert layer.GetPrimAtPath(root_path), "USDA overlay requires a valid default prim"
        reference_owners = []
        pending = list(layer.rootPrims)
        while pending:
            prim_spec = pending.pop()
            if prim_spec.referenceList.prependedItems:
                reference_owners.append(prim_spec)
            pending.extend(prim_spec.nameChildren)
        assert len(reference_owners) == 1, "USDA overlay requires exactly one prim with a prepended reference"
        reference_owner = reference_owners[0]
        references = list(reference_owner.referenceList.prependedItems)
        assert len(references) == 1, "USDA overlay requires exactly one prepended reference"
        reference = references[0]
        reference_owner.referenceList.prependedItems = [
            Sdf.Reference(str(source_path), reference.primPath, reference.layerOffset, reference.customData)
        ]
        assert layer.Export(str(output)), f"Cannot export USDA overlay: {output}"
        stage = Usd.Stage.Open(str(output))
        assert stage and stage.GetDefaultPrim(), f"Cannot open USDA overlay: {output}"
        assert not stage.GetCompositionErrors(), stage.GetCompositionErrors()
        files = {output_name: _sha256_file(output)}
        manifest = {
            "cache_key": digest,
            "pipeline_version": OVERLAY_PIPELINE_VERSION,
            "source_identity": source_identity,
            "source_path": str(source_path),
            "source_entry_sha256": source_hash,
            "output_entry": output_name,
            "output_sha256": files[output_name],
            "files": files,
        }
        (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if package_dir.exists():
            shutil.rmtree(package_dir)
        os.replace(staging_dir, package_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _parse_recipe(recipe_bytes: bytes) -> dict[str, Any]:
    """Parse and validate a derivation recipe."""
    recipe = json.loads(recipe_bytes)
    assert isinstance(recipe, dict), "derivation recipe must be a JSON object"
    required_strings = (
        "source_identity",
        "source_entry",
        "output_entry",
        "expected_source_sha256",
    )
    assert all(
        isinstance(recipe.get(key), str) and recipe[key] for key in required_strings
    ), f"derivation recipe requires non-empty strings: {required_strings}"
    for key in ("source_entry", "output_entry"):
        path = PurePosixPath(recipe[key])
        assert not path.is_absolute() and ".." not in path.parts, f"{key} must stay inside the package"
    source_digest = recipe["expected_source_sha256"]
    assert len(source_digest) == 64 and all(
        character in "0123456789abcdef" for character in source_digest
    ), "expected_source_sha256 must be a lowercase SHA-256 digest"
    return recipe


def _read_derivation_spec(spec: str) -> bytes:
    """Read a filesystem or package-relative derivation specification."""
    return _read_spec(spec)


def _read_spec(spec: str) -> bytes:
    """Read a filesystem or package-relative specification."""
    prefix = "package://"
    if not spec.startswith(prefix):
        return Path(spec).read_bytes()
    package_and_path = spec.removeprefix(prefix)
    package, separator, relative_path = package_and_path.partition("/")
    assert separator and package and relative_path, "package recipe must be package://<package>/<path>"
    resource = importlib.resources.files(package).joinpath(relative_path)
    return resource.read_bytes()


def _package_root(source_path: Path, source_entry: PurePosixPath) -> Path:
    """Return the package root whose relative entry resolves to ``source_path``."""
    package_root = source_path
    for _ in source_entry.parts:
        package_root = package_root.parent
    assert (
        package_root.joinpath(*source_entry.parts).absolute() == source_path
    ), f"resolved source {source_path} does not match recipe entry {source_entry}"
    return package_root


def _is_valid_cache_entry(package_dir: Path, digest: str, pipeline_version: int) -> bool:
    """Return whether a derived package and all recorded files are intact."""
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest["cache_key"] != digest or manifest["pipeline_version"] != pipeline_version:
            return False
        files = manifest["files"]
        if not isinstance(files, dict) or not files:
            return False
        for relative_path, expected_hash in files.items():
            relative = PurePosixPath(relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                return False
            path = package_dir / relative_path
            if not path.is_file() or _sha256_file(path) != expected_hash:
                return False
        return manifest["output_sha256"] == files[manifest["output_entry"]]
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _is_valid_overlay_entry(package_dir: Path, digest: str) -> bool:
    if not _is_valid_cache_entry(package_dir, digest, OVERLAY_PIPELINE_VERSION):
        return False
    try:
        manifest = json.loads((package_dir / "manifest.json").read_text())
        return Path(manifest["source_path"]).is_file()
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _package_file_hashes(package_dir: Path) -> dict[str, str]:
    """Return SHA-256 hashes for every derived package file."""
    return {
        path.relative_to(package_dir).as_posix(): _sha256_file(path)
        for path in sorted(package_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_digest(source_identity: str, recipe_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(f"derived-usd-v{DERIVATION_PIPELINE_VERSION}\0".encode())
    digest.update(source_identity.encode())
    digest.update(b"\0")
    digest.update(recipe_bytes)
    return digest.hexdigest()


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an inter-process lock for one derived cache key."""
    with path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
