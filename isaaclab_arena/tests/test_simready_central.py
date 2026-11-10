# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Central search filtering and pinned asset preparation tests."""

from unittest.mock import Mock, patch

import pytest

from isaaclab_arena.agentic_environment_generation.simready_asset_search import search_simready_objects
from isaaclab_arena.agentic_environment_generation.simready_central_search import search_central_objects
from isaaclab_arena.agentic_environment_generation.simready_search_interface import (
    SimReadySearchConfig,
    SimReadySourceKind,
)
from isaaclab_arena.assets.sources.simready_central import _parse_central_asset_url, is_central_asset_url

MODULE = "isaaclab_arena.agentic_environment_generation.simready_central_search"
LOADER = "isaaclab_arena.assets.sources.simready_central"
REVISION = "a" * 40
URL = f"hf://huggingface.co/datasets/nvidia/simready-assets/resolve/{REVISION}/pallet/simready_usd/pallet.usd"


def _hit(count, path=URL):
    properties = [] if count is None else [{"name": "simready_metadata_rigid_body_count", "value_numeric": count}]
    return {"source": {"path": path, "usd_properties": properties}, "rrf_score": 0.5}


def test_central_requires_single_body_metadata_and_skips_invalid_usd():
    response = Mock()
    response.json.return_value = {"hits": [_hit(None), _hit(2), _hit(1), _hit(1)]}
    with (
        patch(f"{MODULE}.requests.post", return_value=response) as post,
        patch(f"{MODULE}.SimReadyCentralUsdSource") as source,
        patch(f"{MODULE}.validate_usd_object_type", side_effect=[ValueError("articulation"), None]) as validate,
    ):
        source.return_value.to_local_usd_path.side_effect = ["/cache/first.usd", "/cache/second.usd"]
        candidates = search_central_objects("pallet", max_results=1)
    assert [candidate.usd_path for candidate in candidates] == [URL.replace("hf://", "https://", 1)]
    assert all(call.args == (URL.replace("hf://", "https://", 1),) for call in source.call_args_list)
    assert source.call_count == 2
    assert validate.call_count == 2
    assert post.call_args.kwargs["json"]["filter_by_properties_numeric"] == "simready_metadata_rigid_body_count=1"
    assert post.call_args.kwargs["timeout"] == 30


def test_central_failure_leaves_phrase_unmatched():
    with patch(f"{MODULE}.requests.post", side_effect=TimeoutError("offline")):
        catalogue = search_simready_objects(["pallet"], SimReadySearchConfig(sources=[SimReadySourceKind.CENTRAL]))
    assert catalogue.candidates == []
    assert catalogue.unmatched_phrases == ["pallet"]


def test_central_package_location_preserves_revision_and_folder():
    assert _parse_central_asset_url(URL) == (
        "nvidia/simready-assets",
        REVISION,
        "pallet/simready_usd/pallet.usd",
        "pallet",
    )


@pytest.mark.parametrize(
    "url",
    [
        URL.replace(REVISION, "main"),
        URL.replace("pallet/simready_usd", "../simready_usd"),
        URL.replace("pallet/simready_usd/", ""),
    ],
)
def test_central_rejects_unpinned_or_unbounded_packages(url):
    with pytest.raises(AssertionError):
        _parse_central_asset_url(url)


def test_central_usd_source_downloads_package_once():
    from isaaclab_arena.assets.sources.simready_central import _ASSET_PATH_BY_URL, SimReadyCentralUsdSource

    _ASSET_PATH_BY_URL.clear()
    first_source = SimReadyCentralUsdSource(URL)
    second_source = SimReadyCentralUsdSource(URL)
    with patch(f"{LOADER}.download_central_asset_package", return_value="/cache/pallet.usd") as download:
        assert first_source.to_local_usd_path() == "/cache/pallet.usd"
        assert second_source.to_local_usd_path() == "/cache/pallet.usd"
    download.assert_called_once_with(URL)
    _ASSET_PATH_BY_URL.clear()


def _test_central_usd_validation():
    import tempfile
    from pathlib import Path

    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.simready_constants import SIMREADY_CENTRAL_PHYSICS_VARIANTS
    from isaaclab_arena.utils.usd.object_type import validate_usd_object_type

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        asset = folder / "pallet/simready_usd/pallet.usd"
        asset.parent.mkdir(parents=True)
        stage = Usd.Stage.CreateNew(str(asset))
        root = UsdGeom.Xform.Define(stage, "/Pallet").GetPrim()
        stage.SetDefaultPrim(root)
        body = UsdPhysics.RigidBodyAPI.Apply(root)
        physx_variants = root.GetVariantSets().AddVariantSet("PhysX")
        for selection, enabled in (("Enabled", True), ("Disabled", False)):
            physx_variants.AddVariant(selection)
            physx_variants.SetVariantSelection(selection)
            with physx_variants.GetVariantEditContext():
                body.CreateRigidBodyEnabledAttr(enabled)
        physx_variants.SetVariantSelection("Disabled")
        stage.GetRootLayer().Save()
        validate_usd_object_type(str(asset), ObjectType.RIGID, SIMREADY_CENTRAL_PHYSICS_VARIANTS)
        physx_variants.SetVariantSelection("Enabled")
        with physx_variants.GetVariantEditContext():
            body.GetRigidBodyEnabledAttr().Set(False)
        stage.GetRootLayer().Save()
        with pytest.raises(ValueError, match="disabled"):
            validate_usd_object_type(str(asset), ObjectType.RIGID, SIMREADY_CENTRAL_PHYSICS_VARIANTS)
        with physx_variants.GetVariantEditContext():
            body.GetRigidBodyEnabledAttr().Set(True)
        UsdPhysics.ArticulationRootAPI.Apply(root)
        stage.GetRootLayer().Save()
        with pytest.raises(ValueError, match="articulation root"):
            validate_usd_object_type(str(asset), ObjectType.RIGID, SIMREADY_CENTRAL_PHYSICS_VARIANTS)
        root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Pallet/Other").GetPrim())
        stage.GetRootLayer().Save()
        with pytest.raises(ValueError, match="2 rigid bodies"):
            validate_usd_object_type(str(asset), ObjectType.RIGID, SIMREADY_CENTRAL_PHYSICS_VARIANTS)
    return True


def test_central_usd_validation():
    assert _test_central_usd_validation()


def test_central_download_lists_only_the_selected_package():
    from huggingface_hub.hf_api import RepoFile

    from isaaclab_arena.assets.sources.simready_central import download_central_asset_package

    filenames = ["pallet/simready_usd/pallet.usd", "pallet/textures/color.png"]
    entries = [RepoFile(path=path, size=1, oid="a" * 40) for path in filenames]
    with (
        patch("huggingface_hub.HfApi.list_repo_tree", return_value=iter(entries)) as list_tree,
        patch("huggingface_hub.hf_hub_download", side_effect=lambda **kw: "/cache/" + kw["filename"]) as download,
    ):
        assert download_central_asset_package(URL) == "/cache/" + filenames[0]
    list_tree.assert_called_once_with(
        repo_id="nvidia/simready-assets",
        repo_type="dataset",
        revision=REVISION,
        path_in_repo="pallet",
        recursive=True,
    )
    assert {call.kwargs["filename"] for call in download.call_args_list} == set(filenames)
    assert all(call.kwargs["revision"] == REVISION for call in download.call_args_list)


@pytest.mark.parametrize("scheme", ["https", "hf"])
def test_central_url_recognition_and_parsing_share_supported_schemes(scheme):
    url = URL.replace("hf://", f"{scheme}://")
    assert is_central_asset_url(url)
    assert _parse_central_asset_url(url) == _parse_central_asset_url(URL)


@pytest.mark.parametrize(
    "url",
    [
        "/cache/pallet.usda",
        "s3://bucket/pallet.usd",
        URL.replace("huggingface.co", "huggingface.co.example.com"),
        URL.replace("/datasets/", "/models/"),
        URL.replace("hf://", "http://"),
    ],
)
def test_central_url_recognition_leaves_other_asset_paths_unchanged(url):
    assert not is_central_asset_url(url)


def test_central_rejects_unrelated_hit_before_download_and_tries_next(caplog):
    import logging

    cabinet = URL.replace("pallet.usd", "sm_cabinet_grey_a01.usd")
    bin_url = URL.replace("pallet.usd", "sm_bin_grey_a01.usd")
    response = Mock()
    response.json.return_value = {"hits": [_hit(1, cabinet), _hit(1, bin_url)]}
    with (
        patch(f"{MODULE}.requests.post", return_value=response),
        patch(f"{MODULE}.SimReadyCentralUsdSource") as source,
        patch(f"{MODULE}.validate_usd_object_type"),
        caplog.at_level(logging.INFO, logger=MODULE),
    ):
        candidates = search_central_objects("grey bin", max_results=1)
    assert [candidate.usd_path for candidate in candidates] == [bin_url.replace("hf://", "https://", 1)]
    source.assert_called_once_with(candidates[0].usd_path)
    source.return_value.to_local_usd_path.assert_called_once_with()
    assert "asset name does not match the object" in caplog.text


@pytest.mark.parametrize("filename", ["grey_cabinet", "grey_cabin", "binary_box"])
def test_central_unrelated_hits_leave_phrase_unmatched(filename):
    # A matching word in the dataset/package path must not qualify the asset itself.
    path = URL.replace("/pallet/", "/grey_bin/").replace("pallet.usd", filename + ".usd")
    response = Mock()
    response.json.return_value = {"hits": [_hit(1, path)]}
    with (
        patch(f"{MODULE}.requests.post", return_value=response),
        patch(f"{MODULE}.SimReadyCentralUsdSource") as source,
    ):
        result = search_simready_objects(["grey bin"], SimReadySearchConfig(sources=[SimReadySourceKind.CENTRAL]))
    assert result.candidates == []
    assert result.unmatched_phrases == ["grey bin"]
    source.assert_not_called()


@pytest.mark.parametrize(
    ("phrase", "filename"),
    [
        ("grey bin", "sm_greyBins_a01"),
        ("grey bins", "sm_bin_grey_a01"),
        ("spray gun", "sm_paint_supplies_spraygun_b01_01"),
        ("spray gun stand", "sm_serviceTool_sprayGunStand_benchMount_a01_01"),
        ("ball bearing", "sm_consumable_bearing_ball_a01_01"),
    ],
)
def test_central_accepts_matching_asset_names(phrase, filename):
    path = URL.replace("pallet.usd", filename + ".usd")
    response = Mock()
    response.json.return_value = {"hits": [_hit(1, path)]}
    with (
        patch(f"{MODULE}.requests.post", return_value=response),
        patch(f"{MODULE}.SimReadyCentralUsdSource") as source,
        patch(f"{MODULE}.validate_usd_object_type"),
    ):
        candidates = search_central_objects(phrase, max_results=1)
    assert [candidate.usd_path for candidate in candidates] == [path.replace("hf://", "https://", 1)]
    source.assert_called_once_with(candidates[0].usd_path)
    source.return_value.to_local_usd_path.assert_called_once_with()
