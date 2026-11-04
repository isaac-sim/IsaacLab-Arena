# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for immutable sources, relocatable packages and ordered YAML edits."""

import shutil
import subprocess
import sys
import yaml
from pathlib import Path
from unittest.mock import patch

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab_arena_environments.workbench_cleanup.scripts.asset_io import (
    open_asset,
    source_file,
    validate_package,
)
from isaaclab_arena_environments.workbench_cleanup.scripts.asset_operations import (
    apply_operations,
    configure_articulation,
    edit_prims,
)
from isaaclab_arena_environments.workbench_cleanup.scripts.prepare_assets import build_asset


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "original"
    folder.mkdir()
    texture = folder / "texture.png"
    texture.write_bytes(b"test dependency, not rendered")
    stage = Usd.Stage.CreateNew(str(folder / "source.usda"))
    root = UsdGeom.Xform.Define(stage, "/Original")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Cube.Define(stage, "/Original/Body/Cube")
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/Original/Body"))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/Original/Body/Cube"))
    shader = UsdShade.Shader.Define(stage, "/Original/Material/Texture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("texture.png"))
    stage.GetRootLayer().Save()
    return folder / "source.usda"


def _recipe(source, tmp_path, operations=()):
    path = tmp_path / "asset.yaml"
    path.write_text(
        yaml.safe_dump({
            "asset": {"id": "example", "usd_path": str(source)},
            "operations": list(operations),
            "output": "example.usdc",
        })
    )
    return path


def test_central_source_uses_shared_package_downloader(tmp_path):
    revision = "a" * 40
    url = f"hf://huggingface.co/datasets/nvidia/simready-assets/resolve/{revision}/asset/simready_usd/asset.usd"
    with patch(
        "isaaclab_arena_environments.workbench_cleanup.scripts.asset_io.download_central_asset_package",
        return_value="/cache/asset/simready_usd/asset.usd",
    ) as download:
        assert source_file(url, tmp_path) == Path("/cache/asset/simready_usd/asset.usd")
    download.assert_called_once_with(url)


def test_deactivate_matching_parent_and_children(source):
    stage, _ = open_asset(source, {})
    report = edit_prims(stage, select={"paths": ["Body", "Body/*"]}, active=False)
    assert report["prims"] == ["/Asset/Body", "/Asset/Body/Cube"]
    assert not stage.GetPrimAtPath("/Asset/Body").IsActive()
    assert Usd.Stage.Open(str(source)).GetPrimAtPath("/Original/Body").IsActive()


def test_package_reopens_after_sources_are_removed_and_output_moved(source, tmp_path):
    before = {p.name: p.read_bytes() for p in source.parent.iterdir()}
    recipe = _recipe(source, tmp_path)
    report = build_asset(recipe, tmp_path / "output")
    assert report["validation"]["file_dependencies"] == 1
    assert {p.name: p.read_bytes() for p in source.parent.iterdir()} == before
    with pytest.raises(AssertionError, match="already exists"):
        build_asset(recipe, tmp_path / "output")
    moved = tmp_path / "relocated"
    shutil.move(tmp_path / "output" / "example", moved)
    # These are synthetic pytest fixture files only.
    shutil.rmtree(source.parent)
    code = (
        "from pathlib import Path; "
        "from isaaclab_arena_environments.workbench_cleanup.scripts.asset_io import validate_package; "
        "validate_package(Path(__import__('sys').argv[1]))"
    )
    subprocess.run([sys.executable, "-c", code, str(moved / "example.usdc")], check=True, timeout=60)


def test_failed_recipe_never_publishes_output(source, tmp_path):
    recipe = _recipe(source, tmp_path, [{"kind": "misspelled_operation", "params": {}}])
    with pytest.raises(AssertionError, match="Unknown USD operation"):
        build_asset(recipe, tmp_path / "output")
    assert not (tmp_path / "output" / "example").exists()


def test_articulation_welds_only_unselected_joints(source):
    stage, _ = open_asset(source, {})
    for name in ("Keep", "Weld"):
        body = UsdGeom.Xform.Define(stage, f"/Asset/{name}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Asset/Joints/{name}")
        joint.CreateBody0Rel().SetTargets(["/Asset/Body"])
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    configure_articulation(stage, fixed_base_body="Body", movable_joints=["Joints/Keep"])
    assert stage.GetPrimAtPath("/Asset/Joints/Keep").IsA(UsdPhysics.RevoluteJoint)
    welded = stage.GetPrimAtPath("/Asset/Joints/Weld")
    assert welded.IsA(UsdPhysics.FixedJoint)
    assert "PhysicsDriveAPI:angular" not in welded.GetAppliedSchemas()
    assert stage.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI)
    fixed = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/Asset/ArenaFixedBase"))
    assert list(fixed.GetBody1Rel().GetTargets()) == [Sdf.Path("/Asset/Body")]


def test_cli_build(source, tmp_path):
    recipe = _recipe(source, tmp_path)
    scripts = Path(__file__).parents[1]
    subprocess.run(
        [
            sys.executable,
            str(scripts / "prepare_assets.py"),
            "--spec",
            str(recipe),
            "--output-dir",
            str(tmp_path / "cli"),
        ],
        check=True,
        timeout=60,
    )
    validate_package(tmp_path / "cli/example/example.usdc")


def test_edit_typed_vectors_and_reject_empty_selector(source):
    stage, _ = open_asset(source, {})
    edit_prims(
        stage,
        select={"paths": ["Body"]},
        attributes={
            "physics:centerOfMass": {"type": "point3f", "value": [1, 2, 3]},
        },
    )
    assert list(stage.GetPrimAtPath("/Asset/Body").GetAttribute("physics:centerOfMass").Get()) == [1, 2, 3]
    with pytest.raises(AssertionError, match="matched no prims"):
        edit_prims(stage, select={"paths": ["DoesNotExist"]})
    with pytest.raises(AssertionError, match="Unknown selector fields"):
        apply_operations(stage, [{"kind": "edit_prims", "params": {"select": {"path": "Body"}}}])
