# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for URL-based USD asset processing."""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from isaaclab_arena_environments.workbench_cleanup.asset_processing.usd_asset_processor import (
    UsdAssetProcessor,
    retrieve_pinned_usd_path,
)

MODULE = "isaaclab_arena_environments.workbench_cleanup.asset_processing.usd_asset_processor"
SOURCE_URL = "https://example.com/revision/item.usda"


@pytest.fixture
def source(tmp_path: Path) -> Path:
    from pxr import Usd, UsdGeom, UsdPhysics

    path = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Source").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Source/Body").GetPrim())
    stage.GetRootLayer().Save()
    return path


@pytest.fixture
def wrapper(tmp_path: Path) -> Path:
    path = tmp_path / "wrapper.usda"
    path.write_text("""#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset" (
    prepend references = @source.usda@</Source>
)
{
    over "Body" (
        prepend apiSchemas = ["PhysicsMassAPI"]
    )
    {
        float physics:mass = 2
    }
}
""")
    return path


def _patch_retrieval(monkeypatch, source: Path) -> Mock:
    retrieve = Mock(return_value=str(source))
    monkeypatch.setattr("isaaclab.utils.assets.retrieve_file_path", retrieve)
    return retrieve


def test_pinned_usd_localization_reuses_persistent_cache(tmp_path, monkeypatch):
    wrapper = tmp_path / "wrapper.usda"
    wrapper.write_text("#usda 1.0\n")
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")

    def localize(path, download_dir):
        output = Path(download_dir) / "localized" / Path(path).name
        output.parent.mkdir()
        output.write_bytes(Path(path).read_bytes())
        return str(output)

    retrieve = Mock(side_effect=localize)
    monkeypatch.setattr("isaaclab.utils.assets.retrieve_file_path", retrieve)

    first = retrieve_pinned_usd_path(str(wrapper))
    assert retrieve_pinned_usd_path(str(wrapper)) == first
    assert retrieve.call_count == 1

    Path(first).unlink()
    assert retrieve_pinned_usd_path(str(wrapper)) == first
    assert retrieve.call_count == 2

    wrapper.write_text("#usda 1.0\n# changed\n")
    assert retrieve_pinned_usd_path(str(wrapper)) != first
    assert retrieve.call_count == 3


def test_wrapper_rewrites_reference_and_reuses_cache(tmp_path, monkeypatch, source, wrapper):
    from pxr import Usd, UsdPhysics

    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    retrieve = _patch_retrieval(monkeypatch, source)
    first = Path(UsdAssetProcessor(SOURCE_URL, str(wrapper)).process())

    stage = Usd.Stage.Open(str(first))
    assert not stage.GetCompositionErrors()
    assert UsdPhysics.MassAPI(stage.GetPrimAtPath("/Asset/Body")).GetMassAttr().Get() == 2

    retrieve.side_effect = AssertionError("cache hit must not retrieve the source")
    second = UsdAssetProcessor(SOURCE_URL, str(wrapper)).process()
    assert second == str(first)
    assert retrieve.call_count == 1


def test_wrapper_name_content_and_pipeline_version_change_cache_key(tmp_path, monkeypatch, source, wrapper):
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    _patch_retrieval(monkeypatch, source)
    first = UsdAssetProcessor(SOURCE_URL, str(wrapper)).process()

    renamed_wrapper = wrapper.with_name("renamed.usda")
    renamed_wrapper.write_bytes(wrapper.read_bytes())
    renamed = UsdAssetProcessor(SOURCE_URL, str(renamed_wrapper)).process()
    assert renamed != first
    assert Path(renamed).is_file()

    wrapper.write_text(wrapper.read_text().replace("physics:mass = 2", "physics:mass = 3"))
    changed = UsdAssetProcessor(SOURCE_URL, str(wrapper)).process()
    assert changed != first

    monkeypatch.setattr(f"{MODULE}.PROCESSING_PIPELINE_VERSION", 2)
    versioned = UsdAssetProcessor(SOURCE_URL, str(wrapper)).process()
    assert versioned != changed


def test_corrupt_cached_wrapper_is_rebuilt(tmp_path, monkeypatch, source, wrapper):
    from pxr import Usd

    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    retrieve = _patch_retrieval(monkeypatch, source)
    output = Path(UsdAssetProcessor(SOURCE_URL, str(wrapper)).process())
    output.write_text("corrupt")

    rebuilt = UsdAssetProcessor(SOURCE_URL, str(wrapper)).process()
    assert Usd.Stage.Open(rebuilt)
    assert retrieve.call_count == 2


def test_retrieval_failure_does_not_publish_cache_entry(tmp_path, monkeypatch, wrapper):
    cache = tmp_path / "cache"
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: cache)
    monkeypatch.setattr(
        "isaaclab.utils.assets.retrieve_file_path",
        Mock(side_effect=RuntimeError("download failed")),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        UsdAssetProcessor(SOURCE_URL, str(wrapper)).process()
    assert not list((cache / "processed").glob("*/manifest.json"))


def test_structural_operation_failure_removes_staging_directory(tmp_path, monkeypatch, source, wrapper):
    cache = tmp_path / "cache"
    recipe = tmp_path / "structure.json"
    recipe.write_text(json.dumps({"steps": [{"operation": "missing"}]}))
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: cache)
    _patch_retrieval(monkeypatch, source)

    with pytest.raises(AssertionError, match="Unknown structural operation"):
        UsdAssetProcessor(SOURCE_URL, str(wrapper), structural_recipe=str(recipe)).process()
    assert not [path for path in (cache / "processed").iterdir() if path.is_dir()]


@pytest.mark.parametrize("group_path,remainder_path", [("Left", "Right"), ("Structure/Left", "Structure/Right")])
def test_structural_recipe_partitions_mesh_without_flattening_source(tmp_path, monkeypatch, group_path, remainder_path):
    from pxr import Usd, UsdGeom

    source = tmp_path / "mesh.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Shape").GetPrim()
    stage.SetDefaultPrim(root)
    mesh = UsdGeom.Mesh.Define(stage, "/Shape/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (3, 0, 0), (2, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5])
    stage.GetRootLayer().Save()

    from isaaclab_arena_environments.workbench_cleanup.asset_processing.mesh_partition import inspect_mesh_components

    geometry_hash = inspect_mesh_components(stage, "Mesh")["geometry_sha256"]
    recipe = tmp_path / "structure.json"
    recipe.write_text(
        json.dumps({
            "required_prims": [f"Mesh/{group_path}", f"Mesh/{remainder_path}"],
            "steps": [{
                "operation": "partition_mesh",
                "prim": "Mesh",
                "groups": {group_path: [0]},
                "remainder": remainder_path,
                "expected_geometry_sha256": geometry_hash,
            }],
        })
    )
    wrapper = tmp_path / "wrapper.usda"
    wrapper.write_text("""#usda 1.0
(defaultPrim = "Asset")
def Xform "Asset" (prepend references = @structure.usda@</Asset>) {}
""")
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    _patch_retrieval(monkeypatch, source)
    output = Path(
        UsdAssetProcessor(
            SOURCE_URL,
            str(wrapper),
            structural_recipe=str(recipe),
        ).process()
    )

    processed = Usd.Stage.Open(str(output))
    left = UsdGeom.Mesh(processed.GetPrimAtPath(f"/Asset/Mesh/{group_path}"))
    right = UsdGeom.Mesh(processed.GetPrimAtPath(f"/Asset/Mesh/{remainder_path}"))
    assert left and right
    assert list(left.GetPointsAttr().Get()) == [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    assert list(right.GetPointsAttr().Get()) == [(2, 0, 0), (3, 0, 0), (2, 1, 0)]
    assert left.GetPrim().GetParent().IsA(UsdGeom.Xform)
    structure_text = (output.parent / "structure.usda").read_text()
    assert str(source) in structure_text
    assert "points" in structure_text


def test_workbench_asset_processing_configuration(monkeypatch):
    from isaaclab_arena_environments.workbench_cleanup import assets

    retrieve_wrapper = Mock(return_value="/cache/socket_set.usda")
    monkeypatch.setattr(assets, "_retrieve_wrapper", retrieve_wrapper)
    socket_set = assets.WorkbenchCleanupSocketSet(initial_joint_positions={"case_lid": 1.5})
    assert socket_set.object_cfg.init_state.joint_pos == {"case_lid": 1.5}
    retrieve_wrapper.assert_called_once_with("socket_set.usda")

    package_path = assets._WORKBENCH.structural_recipe.removeprefix("package://")
    package, relative_path = package_path.split("/", 1)
    recipe = json.loads(importlib.resources.files(package).joinpath(relative_path).read_bytes())
    assert {step["operation"] for step in recipe["steps"]} == {"partition_mesh", "fixed_articulations"}
    fixed_articulations = next(step for step in recipe["steps"] if step["operation"] == "fixed_articulations")
    assert set(fixed_articulations["groups"]) == {"Cabinet"}
    assert len(fixed_articulations["groups"]["Cabinet"]["joints"]) == 5


@pytest.mark.parametrize(
    "wrapper_name",
    ["white_bin.usda", "socket_set.usda", "oscilloscope.usda", "slotted_screwdriver.usda", "small_wrench.usda"],
)
def test_non_structural_wrappers_reference_revision_pinned_urls(wrapper_name):
    resource = importlib.resources.files("isaaclab_arena_environments").joinpath(
        "workbench_cleanup/asset_processing/wrappers", wrapper_name
    )
    wrapper = resource.read_text()
    reference = wrapper.split("prepend references = @", 1)[1].split("@", 1)[0]
    revision = reference.split("/resolve/", 1)[1].split("/", 1)[0]

    assert reference.startswith("https://huggingface.co/")
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)


def test_environment_discovery_registers_workbench_assets_without_processing():
    import subprocess
    import sys

    code = """
import sys
import isaaclab.utils.assets
from isaaclab_arena.assets.registries import AssetRegistry

registry = AssetRegistry()
assert not registry.is_registered('workbench_cleanup_workbench', ensure_loaded=False)
assert 'isaaclab_arena_environments.workbench_cleanup.assets' not in sys.modules

def unexpected_retrieval(*args, **kwargs):
    raise AssertionError('Registration must not retrieve or process assets')

isaaclab.utils.assets.retrieve_file_path = unexpected_retrieval

from isaaclab_arena_environments.cli import ensure_environments_registered
ensure_environments_registered()

expected = {
    'workbench_cleanup_workbench', 'workbench_cleanup_white_bin',
    'workbench_cleanup_socket_set', 'workbench_cleanup_oscilloscope',
    'workbench_cleanup_slotted_screwdriver', 'workbench_cleanup_small_wrench',
}
for name in expected:
    asset_type = registry.get_asset_by_name(name)
    assert asset_type.__module__ == 'isaaclab_arena_environments.workbench_cleanup.assets'

assert 'isaaclab_arena_environments.workbench_cleanup.prebuild_assets' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
