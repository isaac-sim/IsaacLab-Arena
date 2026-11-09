# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SimReady Central USD acquisition."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from isaaclab_arena.assets.simready_constants import SIMREADY_PHYSICS_VARIANTS
from isaaclab_arena.assets.sources.base import UsdSource


class SimReadyCentralUsdSource(UsdSource):
    """Download a complete pinned SimReady Central package on first resolution."""

    def __init__(self, url: str):
        _parse_central_asset_url(url)
        self.url = url
        self._asset_path: str | None = None

    def resolve(self) -> str:
        """Download the package and return its original USD entrypoint."""
        if self._asset_path is None:
            self._asset_path = download_central_asset_package(self.url)
        return self._asset_path


class SimReadyCentralRigidSource(SimReadyCentralUsdSource):
    """Return a validated single-rigid-body wrapper for a Central package."""

    def __init__(self, url: str):
        super().__init__(url)
        self._wrapper_path: str | None = None

    def resolve(self) -> str:
        """Download the package and return its compatibility wrapper."""
        if self._wrapper_path is None:
            asset_path = super().resolve()
            self._wrapper_path = create_central_rigid_wrapper(asset_path, source_url=self.url)
        return self._wrapper_path


def is_central_asset_url(url: str) -> bool:
    """Identify Hugging Face dataset URLs handled by the Central asset loader."""
    parsed = urlsplit(url)
    return (
        parsed.scheme in ("https", "hf") and parsed.netloc == "huggingface.co" and parsed.path.startswith("/datasets/")
    )


def download_central_asset_package(url: str) -> str:
    """Download every file in a pinned asset package and return its cached USD entrypoint."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    repo_id, revision, filename, folder = _parse_central_asset_url(url)
    # List only the selected package, avoiding a scan of the entire dataset.
    files = [
        entry.path
        for entry in HfApi().list_repo_tree(
            repo_id=repo_id, repo_type="dataset", revision=revision, path_in_repo=folder, recursive=True
        )
        if isinstance(entry, RepoFile)
    ]
    assert filename in files, "asset package is missing its USD entrypoint"

    def download_file(path: str) -> str:
        return hf_hub_download(repo_id=repo_id, repo_type="dataset", revision=revision, filename=path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        cached_paths = dict(zip(files, executor.map(download_file, files), strict=True))
    return cached_paths[filename]


def create_central_rigid_wrapper(asset_path: str, *, source_url: str) -> str:
    """Validate a Central USD as one rigid body and cache its PhysX wrapper."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.utils.usd.variants import apply_usd_variant_selections

    asset_stage = Usd.Stage.Open(asset_path)
    assert asset_stage and asset_stage.GetDefaultPrim(), "asset requires a default prim"
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(asset_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(asset_stage))
    root = stage.DefinePrim("/Asset")
    root.GetReferences().AddReference(asset_path)
    stage.SetDefaultPrim(root)
    # TODO(xinjieyao, 20261022): Support Newton.
    apply_usd_variant_selections(stage, {**SIMREADY_PHYSICS_VARIANTS, "PhysX": "Enabled"})
    assert not stage.GetCompositionErrors(), "USD has unresolved composition errors"
    bodies = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        assert not prim.HasAPI(UsdPhysics.ArticulationRootAPI), "asset has an articulation root"
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(prim)
    assert len(bodies) == 1, f"expected one rigid body, found {len(bodies)}"
    assert UsdPhysics.RigidBodyAPI(bodies[0]).GetRigidBodyEnabledAttr().Get(), "rigid body is disabled"
    cache = Path.home() / ".cache" / "isaaclab_arena" / "simready-central"
    cache.mkdir(parents=True, exist_ok=True)
    wrapper = cache / f"{hashlib.sha256(source_url.encode()).hexdigest()}.usda"
    assert stage.GetRootLayer().Export(str(wrapper)), f"failed to export Central wrapper to {wrapper}"
    return str(wrapper)


def _parse_central_asset_url(url: str) -> tuple[str, str, str, str]:
    """Parse a revision-pinned Central dataset URL and its asset package folder."""
    assert is_central_asset_url(url), "expected a Hugging Face dataset URL"
    parsed = urlsplit(url)
    assert not parsed.query and not parsed.fragment, "expected a plain asset URL"
    match = re.fullmatch(r"/datasets/([^/]+/[^/]+)/(?:resolve|blob)/([0-9a-f]{40})/(.+)", parsed.path)
    assert match, "expected a pinned Hugging Face dataset URL"
    repo_id, revision, filename = match.groups()
    path = PurePosixPath(filename)
    assert not path.is_absolute() and ".." not in path.parts, "invalid asset path"
    assert not any(char in filename for char in "*?[]\\"), "invalid asset path"
    assert path.suffix in (".usd", ".usda", ".usdc", ".usdz"), "expected USD asset"
    assert "simready_usd" in path.parts, "unsupported Central package layout"
    # Central keeps the entrypoint in <asset>/simready_usd/; fetch that asset's package and dependencies.
    folder = PurePosixPath(*path.parts[: path.parts.index("simready_usd")])
    assert str(folder) != ".", "asset must have a package folder"
    return repo_id, revision, filename, str(folder)
