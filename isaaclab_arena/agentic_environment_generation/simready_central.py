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
    SimReadyCandidateCatalogue,
    SimReadyObjectCandidate,
)

CENTRAL_SEARCH_URL = "https://nvidia-simready-central.hf.space/search_hybrid"
_logger = logging.getLogger(__name__)


def load_pinned_central_objects(urls: tuple[str, ...]) -> SimReadyCandidateCatalogue:
    """Look up exact pinned Central assets and validate each before offering them to the agent."""
    catalogue = SimReadyCandidateCatalogue()
    for url in dict.fromkeys(urls):
        repo_id, revision, filename, _ = _central_asset_location(url)
        source_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"
        response = requests.post(
            CENTRAL_SEARCH_URL,
            headers={"x-usdsearch-storage-backend": "usdsearch-dev"},
            json={
                "filter_url_regexp": re.escape(source_url.replace("https://", "hf://", 1)),
                "filter_by_properties_numeric": "simready_metadata_rigid_body_count=1",
                "return_usd_properties": True,
                "limit": 1,
            },
            timeout=30,
        )
        response.raise_for_status()
        hits = response.json()["hits"]
        assert hits, f"Pinned asset is not indexed as a single rigid body in Central: {source_url}"
        source = hits[0]["source"]
        assert source["path"].replace("hf://", "https://", 1) == source_url, "Central returned a different asset"
        properties = {prop["name"]: prop for prop in source.get("usd_properties", [])}
        count = properties.get("simready_metadata_rigid_body_count", {})
        assert float(count.get("value_numeric", count.get("value", 0))) == 1, "expected one indexed rigid body"
        name = properties.get("simready_metadata_asset_name", {}).get("value")
        assert name, f"Central asset has no descriptive name: {source_url}"
        catalogue.candidates.append(
            SimReadyObjectCandidate(
                search_phrase=name,
                usd_path=_prepare_central_asset(source_url),
                source_url=source_url,
                tags=("sim-ready", "single-rigid-body", name.lower()),
            )
        )
    return catalogue


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
                    source_url=source["path"].replace("hf://", "https://", 1),
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
    match = re.fullmatch(
        r"(?:hf|https)://huggingface\.co/datasets/([^/]+/[^/]+)/(?:resolve|blob)/([0-9a-f]{40})/(.+)", url
    )
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


def _download_central_asset_to_cache(url: str) -> str:
    """Download the pinned asset package and return its cached USD entrypoint."""
    from huggingface_hub import snapshot_download

    repo_id, revision, filename, folder = _central_asset_location(url)
    snapshot = snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=revision, allow_patterns=[f"{folder}/*"]
    )
    return str(Path(snapshot) / filename)


def _prepare_central_asset(url: str) -> str:
    """Write a validated PhysX reference wrapper for a cached Central asset."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections

    asset_path = _download_central_asset_to_cache(url)
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
