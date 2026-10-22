# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Discover and prepare single-body props from Hugging Face SimReady Central."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path, PurePosixPath

import requests

from isaaclab_arena.agentic_environment_generation.simready_asset_search import (
    MIN_INSPECTED_MATCHES_PER_PHRASE,
    SimReadyObjectCandidate,
)

CENTRAL_SEARCH_URL = "https://nvidia-simready-central.hf.space/search_hybrid"
_logger = logging.getLogger(__name__)


def search_central_objects(phrase: str, *, max_results: int) -> list[SimReadyObjectCandidate]:
    """Search Central and retain only locally validated single-body props."""
    assert max_results > 0, "max_results must be positive"
    response = requests.post(
        CENTRAL_SEARCH_URL,
        headers={"x-usdsearch-storage-backend": "usdsearch-dev"},
        json={
            "hybrid_text_query": phrase,
            "file_extension_include": "usd*",
            "filter_by_properties_numeric": "simready_metadata_rigid_body_count=1",
            "return_usd_properties": True,
            "limit": max(MIN_INSPECTED_MATCHES_PER_PHRASE, max_results),
        },
        timeout=30,
    )
    response.raise_for_status()
    candidates = []
    for hit in response.json()["hits"]:
        try:
            source = hit["source"]
            # Fail closed even if a backend ignores the numeric filter.
            counts = [
                prop.get("value_numeric", prop.get("value"))
                for prop in source.get("usd_properties", [])
                if prop.get("name") == "simready_metadata_rigid_body_count"
            ]
            if not counts or any(float(count) != 1 for count in counts):
                continue
            usd_path = _prepare_central_asset(source["path"])
            candidates.append(
                SimReadyObjectCandidate(
                    search_phrase=phrase,
                    usd_path=usd_path,
                    tags=("sim-ready", "single-rigid-body"),
                    relevance_score=hit.get("rrf_score"),
                )
            )
        except Exception as exc:
            _logger.info("Skipping Central asset %s: %s", hit.get("id"), exc)
            continue
        if len(candidates) >= max_results:
            break
    return candidates


def _central_asset_location(url: str) -> tuple[str, str, str, str]:
    """Parse a revision-pinned Central dataset URL and its asset package folder."""
    match = re.fullmatch(r"(?:hf|https)://huggingface\.co/datasets/([^/]+/[^/]+)/resolve/([0-9a-f]{40})/(.+)", url)
    assert match, "expected a pinned Hugging Face dataset URL"
    repo_id, revision, filename = match.groups()
    path = PurePosixPath(filename)
    assert not path.is_absolute() and ".." not in path.parts, "invalid asset path"
    assert not any(char in filename for char in "*?[]\\"), "invalid asset path"
    assert path.suffix in (".usd", ".usda", ".usdc", ".usdz"), "expected USD asset"
    # Central packages place the entrypoint in <asset>/simready_usd/. Fetch its
    # siblings too (physics layers, materials, textures), never the entire dataset.
    assert "simready_usd" in path.parts, "unsupported Central package layout"
    folder = PurePosixPath(*path.parts[: path.parts.index("simready_usd")])
    assert str(folder) != ".", "asset must have a package folder"
    return repo_id, revision, filename, str(folder)


def _prepare_central_asset(url: str) -> str:
    """Cache a pinned asset package and write a validated PhysX reference wrapper."""
    from huggingface_hub import snapshot_download
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections

    repo_id, revision, filename, folder = _central_asset_location(url)
    snapshot = snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=revision, allow_patterns=[f"{folder}/*"]
    )
    asset_path = str(Path(snapshot) / filename)
    asset_stage = Usd.Stage.Open(asset_path)
    assert asset_stage and asset_stage.GetDefaultPrim(), "asset requires a default prim"
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(asset_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(asset_stage))
    root = stage.DefinePrim("/Asset")
    root.GetReferences().AddReference(asset_path)
    stage.SetDefaultPrim(root)
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
