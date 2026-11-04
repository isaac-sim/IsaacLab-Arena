# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for immutable sources, relocatable packages and ordered YAML edits."""

import json
import numpy as np
import shutil
import subprocess
import sys
import yaml
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab_arena_environments.workbench_cleanup.scripts.asset_io import bounds_m, open_asset, validate_package
from isaaclab_arena_environments.workbench_cleanup.scripts.asset_operations import (
    apply_operations,
    bake_scale,
    configure_articulation,
    edit_prims,
)
from isaaclab_arena_environments.workbench_cleanup.scripts.prepare_assets import build_asset
from isaaclab_arena_environments.workbench_cleanup.scripts.validate_assets import check_scene


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


def test_scale_survives_replacing_spawn_root_transform(source):
    stage, _ = open_asset(source, {})
    bake_scale(stage, scale=[0.75, 0.75, 0.75])
    root = UsdGeom.Xformable(stage.GetDefaultPrim())
    root.ClearXformOpOrder()
    root.AddTranslateOp().Set(Gf.Vec3d(10, 0, 0))
    np.testing.assert_allclose(bounds_m(stage), [[9.25, -0.75, -0.75], [10.75, 0.75, 0.75]])


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


def test_cli_build_and_inspect(source, tmp_path):
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
    report = tmp_path / "inspection.json"
    subprocess.run(
        [
            sys.executable,
            str(scripts / "inspect_asset.py"),
            "--spec",
            str(recipe),
            "--output",
            str(report),
        ],
        check=True,
        timeout=60,
    )
    assert json.loads(report.read_text())["prims"]
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


def test_scene_validation_follows_nested_references_without_editing_input(source):
    scene = {
        "background": {"id": "asset", "params": {"usd_path": "/old/source.usda"}},
        "object_references": [
            {"id": "cube", "parent_id": "body", "prim_path": "Cube", "object_type": "base"},
            {"id": "body", "parent_id": "asset", "prim_path": "Body", "object_type": "rigid"},
        ],
    }
    updated, count = check_scene(scene, {"source.usda": source})
    assert count == 3
    assert updated["background"]["params"]["usd_path"] == str(source)
    assert scene["background"]["params"]["usd_path"] == "/old/source.usda"
    scene["object_references"][0]["prim_path"] = "Missing"
    with pytest.raises(AssertionError, match="Missing scene reference"):
        check_scene(scene, {"source.usda": source})


def test_scene_validation_rejects_cyclic_references():
    scene = {
        "object_references": [
            {"id": "a", "parent_id": "b", "prim_path": "Body", "object_type": "base"},
            {"id": "b", "parent_id": "a", "prim_path": "Body", "object_type": "base"},
        ]
    }
    with pytest.raises(AssertionError, match="cyclic"):
        check_scene(scene, {})
