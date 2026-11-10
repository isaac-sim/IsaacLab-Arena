# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Retrieve Central assets through Isaac Lab and create validated PhysX USD wrappers."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


def is_central_asset_url(url: str) -> bool:
    """Identify Hugging Face dataset URLs handled by the Central asset loader."""
    parsed = urlsplit(url)
    return (
        parsed.scheme in ("https", "hf") and parsed.netloc == "huggingface.co" and parsed.path.startswith("/datasets/")
    )


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
    # Central packages place their entrypoint in <asset>/simready_usd/.
    assert "simready_usd" in path.parts, "unsupported Central package layout"
    folder = PurePosixPath(*path.parts[: path.parts.index("simready_usd")])
    assert str(folder) != ".", "asset must have a package folder"
    return repo_id, revision, filename, str(folder)


def download_and_wrap_central_asset(url: str) -> str:
    """Retrieve a Central asset through Isaac Lab and write a validated PhysX USD wrapper."""
    from isaaclab.utils.assets import retrieve_file_path
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections

    repo_id, revision, filename, _ = _parse_central_asset_url(url)
    asset_path = retrieve_file_path(f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}")
    asset_stage = Usd.Stage.Open(asset_path)
    assert asset_stage and asset_stage.GetDefaultPrim(), "asset requires a default prim"
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(asset_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(asset_stage))
    root = stage.DefinePrim("/Asset")
    root.GetReferences().AddReference(asset_path)
    stage.SetDefaultPrim(root)
    # TODO(xinjieyao, 20261022): Support Newton.
    apply_usd_variant_selections(stage, {"Physics": "physics", "PhysX": "Enabled"})
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
    wrapper = cache / f"{hashlib.sha256(url.encode()).hexdigest()}.usda"
    stage.GetRootLayer().Export(str(wrapper))
    return str(wrapper)
