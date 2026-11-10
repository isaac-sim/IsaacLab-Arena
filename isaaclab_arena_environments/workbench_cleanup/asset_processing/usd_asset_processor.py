# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Process remote USD assets into local compositional wrapper layers."""

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
from typing import Any

from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir

PROCESSING_PIPELINE_VERSION = 1
"""Version of the USD asset processing contract."""

PINNED_USD_LOCALIZATION_VERSION = 1
"""Version of the persistent pinned-USD localization contract."""


def retrieve_pinned_usd_path(path: str) -> str:
    """Return a persistent localized copy of a USD with immutable remote dependencies.

    Args:
        path: Local USD wrapper whose remote asset paths are revision-pinned.

    Returns:
        Path to the localized USD entry layer.
    """
    wrapper_path = Path(path).expanduser().absolute()
    assert wrapper_path.is_file(), f"Pinned USD wrapper does not exist: {wrapper_path}"
    wrapper_bytes = wrapper_path.read_bytes()
    digest = hashlib.sha256(
        b"\0".join((
            f"pinned-usd-v{PINNED_USD_LOCALIZATION_VERSION}".encode(),
            wrapper_path.name.encode(),
            wrapper_bytes,
        ))
    ).hexdigest()
    cache_root = get_arena_asset_cache_dir() / "pinned_usd"
    cache_root.mkdir(parents=True, exist_ok=True)
    package_dir = cache_root / digest
    output = _pinned_usd_output(package_dir, digest)
    if output is None:
        with _exclusive_lock(cache_root / f"{digest}.lock"):
            output = _pinned_usd_output(package_dir, digest)
            if output is None:
                output = _build_pinned_usd(wrapper_path, package_dir, digest)
    return str(output)


def _build_pinned_usd(wrapper_path: Path, package_dir: Path, digest: str) -> Path:
    """Localize one pinned USD dependency graph into its persistent cache directory."""
    from isaaclab.utils.assets import retrieve_file_path

    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    try:
        output = Path(retrieve_file_path(str(wrapper_path), download_dir=str(package_dir))).absolute()
        assert output.is_relative_to(package_dir), "Pinned USD localization must return a cached copy."
        relative_output = output.relative_to(package_dir)
        manifest = {
            "cache_key": digest,
            "localization_version": PINNED_USD_LOCALIZATION_VERSION,
            "output_entry": relative_output.as_posix(),
        }
        (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return output
    except Exception:
        shutil.rmtree(package_dir, ignore_errors=True)
        raise


def _pinned_usd_output(package_dir: Path, digest: str) -> Path | None:
    """Return a valid localized entry layer from a pinned USD cache entry."""
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        relative = PurePosixPath(manifest["output_entry"])
        if (
            manifest["cache_key"] != digest
            or manifest["localization_version"] != PINNED_USD_LOCALIZATION_VERSION
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            return None
        output = package_dir.joinpath(*relative.parts)
        return output if output.is_file() else None
    except (KeyError, OSError, TypeError, ValueError):
        return None


class UsdAssetProcessor:
    """Process a pinned remote USD into a local USDA wrapper layer.

    Processing flow::

        UsdAssetProcessor
        ├── retrieve source
        ├── optionally author_structural_layer()
        │   ├── partition_mesh()
        │   └── create_fixed_articulations()
        └── write final USDA wrapper layer
    """

    def __init__(
        self,
        source_url: str,
        wrapper_path: str,
        *,
        structural_recipe: str | None = None,
    ):
        """Configure processing without retrieving the remote source.

        Args:
            source_url: Revision-pinned remote USD URL accepted by Isaac Lab.
            wrapper_path: Filesystem or package-resource path to the USDA wrapper template.
            structural_recipe: Optional recipe for computed structural edits.
        """
        assert source_url, "source_url must not be empty"
        self.source_url = source_url
        self.wrapper_path = wrapper_path
        self.structural_recipe = structural_recipe

    def process(self) -> str:
        """Return the processed USDA wrapper path, building or reusing it as needed."""
        wrapper_bytes = _read_spec(self.wrapper_path)
        wrapper_name = PurePosixPath(self.wrapper_path).name
        assert wrapper_name.endswith(".usda"), "wrapper_path must name a USDA file"
        recipe_bytes = _read_spec(self.structural_recipe) if self.structural_recipe is not None else b""
        recipe = _parse_structural_recipe(recipe_bytes) if self.structural_recipe is not None else None

        identity_bytes = b"\0".join((
            f"processed-usd-v{PROCESSING_PIPELINE_VERSION}".encode(),
            self.source_url.encode(),
            wrapper_name.encode(),
            wrapper_bytes,
            recipe_bytes,
        ))
        digest = hashlib.sha256(identity_bytes).hexdigest()
        cache_root = get_arena_asset_cache_dir() / "processed"
        cache_root.mkdir(parents=True, exist_ok=True)
        package_dir = cache_root / digest
        if not _is_valid_cache_entry(package_dir, digest):
            with _exclusive_lock(cache_root / f"{digest}.lock"):
                if not _is_valid_cache_entry(package_dir, digest):
                    _build_processed_usd(
                        source_url=self.source_url,
                        wrapper_bytes=wrapper_bytes,
                        wrapper_name=wrapper_name,
                        recipe=recipe,
                        digest=digest,
                        cache_root=cache_root,
                        package_dir=package_dir,
                    )
        return str(package_dir / wrapper_name)


def _build_processed_usd(
    *,
    source_url: str,
    wrapper_bytes: bytes,
    wrapper_name: str,
    recipe: dict[str, Any] | None,
    digest: str,
    cache_root: Path,
    package_dir: Path,
) -> None:
    """Build and atomically publish one processed USD cache entry."""
    from isaaclab.utils.assets import retrieve_file_path

    source_path = Path(retrieve_file_path(source_url)).expanduser().absolute()
    assert source_path.is_file(), f"USD source did not resolve to a file: {source_path}"

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=cache_root))
    try:
        reference_path = source_path
        files: dict[str, str] = {}
        if recipe is not None:
            from isaaclab_arena_environments.workbench_cleanup.asset_processing.structural_layer import (
                author_structural_layer,
            )

            structure_path = staging_dir / "structure.usda"
            author_structural_layer(source_path, structure_path, recipe)
            reference_path = structure_path
            files[structure_path.name] = _sha256_file(structure_path)

        output = staging_dir / wrapper_name
        _write_wrapper_layer(wrapper_bytes, output, reference_path)
        files[wrapper_name] = _sha256_file(output)
        manifest = {
            "cache_key": digest,
            "pipeline_version": PROCESSING_PIPELINE_VERSION,
            "source_url": source_url,
            "source_path": str(source_path),
            "output_entry": wrapper_name,
            "output_sha256": files[wrapper_name],
            "files": files,
        }
        (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if package_dir.exists():
            shutil.rmtree(package_dir)
        os.replace(staging_dir, package_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _write_wrapper_layer(wrapper_bytes: bytes, output: Path, reference_path: Path) -> None:
    """Write a USDA wrapper with its reference retargeted to the processed source."""
    from pxr import Sdf, Usd

    layer = Sdf.Layer.CreateAnonymous()
    assert layer.ImportFromString(wrapper_bytes.decode()), f"Cannot parse USDA wrapper: {output.name}"
    root_path = Sdf.Path.absoluteRootPath.AppendChild(layer.defaultPrim)
    assert layer.GetPrimAtPath(root_path), "USDA wrapper requires a valid default prim"
    reference_owners = []
    pending = list(layer.rootPrims)
    while pending:
        prim_spec = pending.pop()
        if prim_spec.referenceList.prependedItems:
            reference_owners.append(prim_spec)
        pending.extend(prim_spec.nameChildren)
    assert len(reference_owners) == 1, "USDA wrapper requires exactly one prim with a prepended reference"
    reference_owner = reference_owners[0]
    references = list(reference_owner.referenceList.prependedItems)
    assert len(references) == 1, "USDA wrapper requires exactly one prepended reference"
    reference = references[0]
    asset_path = (
        os.path.relpath(reference_path, output.parent)
        if reference_path.parent == output.parent
        else str(reference_path)
    )
    reference_owner.referenceList.prependedItems = [
        Sdf.Reference(asset_path, reference.primPath, reference.layerOffset, reference.customData)
    ]
    assert layer.Export(str(output)), f"Cannot export USDA wrapper: {output}"
    stage = Usd.Stage.Open(str(output))
    assert stage and stage.GetDefaultPrim(), f"Cannot open USDA wrapper: {output}"
    assert not stage.GetCompositionErrors(), stage.GetCompositionErrors()


def _parse_structural_recipe(recipe_bytes: bytes) -> dict[str, Any]:
    """Parse and validate a structural recipe."""
    recipe = json.loads(recipe_bytes)
    assert isinstance(recipe, dict), "structural recipe must be a JSON object"
    assert isinstance(recipe.get("steps"), list), "structural recipe requires a steps list"
    return recipe


def _read_spec(spec: str) -> bytes:
    """Read a filesystem or package-resource specification."""
    prefix = "package://"
    if not spec.startswith(prefix):
        return Path(spec).read_bytes()
    package_and_path = spec.removeprefix(prefix)
    package, separator, relative_path = package_and_path.partition("/")
    assert separator and package and relative_path, "package spec must be package://<package>/<path>"
    return importlib.resources.files(package).joinpath(relative_path).read_bytes()


def _is_valid_cache_entry(package_dir: Path, digest: str) -> bool:
    """Return whether a cache entry and its referenced source remain intact."""
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest["cache_key"] != digest or manifest["pipeline_version"] != PROCESSING_PIPELINE_VERSION:
            return False
        if not Path(manifest["source_path"]).is_file():
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


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive interprocess lock for the context duration."""
    with path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
