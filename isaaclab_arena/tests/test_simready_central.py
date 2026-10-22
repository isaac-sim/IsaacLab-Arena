# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Central search filtering and pinned asset preparation tests."""

from unittest.mock import Mock, patch

import pytest

from isaaclab_arena.agentic_environment_generation.simready_asset_search import (
    SimReadySearchConfig,
    SimReadySourceKind,
    search_simready_objects,
)
from isaaclab_arena.agentic_environment_generation.simready_central import (
    _central_asset_location,
    search_central_objects,
)

MODULE = "isaaclab_arena.agentic_environment_generation.simready_central"
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
        patch(
            f"{MODULE}._prepare_central_asset", side_effect=[ValueError("articulation"), "/cache/pallet.usda"]
        ) as prepare,
    ):
        candidates = search_central_objects("pallet", max_results=1)
    assert [candidate.usd_path for candidate in candidates] == ["/cache/pallet.usda"]
    assert prepare.call_count == 2
    assert post.call_args.kwargs["json"]["filter_by_properties_numeric"] == "simready_metadata_rigid_body_count=1"
    assert post.call_args.kwargs["timeout"] == 30


def test_central_failure_leaves_phrase_unmatched():
    with patch(f"{MODULE}.requests.post", side_effect=TimeoutError("offline")):
        catalogue = search_simready_objects(["pallet"], SimReadySearchConfig(source=SimReadySourceKind.CENTRAL))
    assert catalogue.candidates == []
    assert catalogue.unmatched_phrases == ["pallet"]


def test_central_package_location_preserves_revision_and_folder():
    assert _central_asset_location(URL) == (
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
        _central_asset_location(url)


def _test_central_usd_validation(simulation_app):
    import tempfile
    from pathlib import Path

    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.agentic_environment_generation.simready_central import _prepare_central_asset

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        asset = folder / "pallet/simready_usd/pallet.usd"
        asset.parent.mkdir(parents=True)
        stage = Usd.Stage.CreateNew(str(asset))
        root = UsdGeom.Xform.Define(stage, "/Pallet").GetPrim()
        stage.SetDefaultPrim(root)
        body = UsdPhysics.RigidBodyAPI.Apply(root)
        body.CreateRigidBodyEnabledAttr(True)
        stage.GetRootLayer().Save()
        with (
            patch("huggingface_hub.snapshot_download", return_value=directory),
            patch(f"{MODULE}.Path.home", return_value=folder),
        ):
            wrapper = _prepare_central_asset(URL)
            assert Usd.Stage.Open(wrapper).GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
            body.GetRigidBodyEnabledAttr().Set(False)
            stage.GetRootLayer().Save()
            with pytest.raises(AssertionError, match="disabled"):
                _prepare_central_asset(URL)
            body.GetRigidBodyEnabledAttr().Set(True)
            UsdPhysics.ArticulationRootAPI.Apply(root)
            stage.GetRootLayer().Save()
            with pytest.raises(AssertionError, match="articulation root"):
                _prepare_central_asset(URL)
            root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Pallet/Other").GetPrim())
            stage.GetRootLayer().Save()
            with pytest.raises(AssertionError, match="found 2"):
                _prepare_central_asset(URL)
    return True


def test_central_usd_validation():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_central_usd_validation)
