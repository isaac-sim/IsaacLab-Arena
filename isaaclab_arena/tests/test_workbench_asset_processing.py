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
    PROCESSING_PIPELINE_VERSION,
    UsdAssetProcessor,
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

    monkeypatch.setattr(f"{MODULE}.PROCESSING_PIPELINE_VERSION", PROCESSING_PIPELINE_VERSION + 1)
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


@pytest.mark.parametrize("physics_binding", ["inherited", "direct", "none"])
def test_workbench_wrapper_preserves_bodies_joints_and_materials(tmp_path, monkeypatch, physics_binding):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Source").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    body = UsdGeom.Xform.Define(stage, "/Source/Geometry/table_01_obj_00").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body)
    mesh = UsdGeom.Mesh.Define(stage, body.GetPath().AppendChild("table_01_mesh_00"))
    # One triangle per component exercises the shipped partition groups without downloading the asset.
    points = []
    for index in range(58):
        points.extend([(2 * index, 0, 0), (2 * index + 1, 0, 0), (2 * index, 1, 0)])
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * 58)
    mesh.CreateFaceVertexIndicesAttr(list(range(174)))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    visual = UsdShade.Material.Define(stage, "/Source/Looks/Paint")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(visual)
    if physics_binding != "none":
        steel = UsdShade.Material.Define(stage, "/Source/Looks/Steel")
        physics = UsdPhysics.MaterialAPI.Apply(steel.GetPrim())
        physics.CreateStaticFrictionAttr(0.5)
        physics.CreateDynamicFrictionAttr(0.4)
        physics.CreateRestitutionAttr(0.15)
        binding_owner = body if physics_binding == "inherited" else mesh.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(binding_owner).Bind(
            steel, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
        )
    moving_bodies = []
    joints = []
    for index, name in enumerate(("drawersA", "drawersB", "drawersC", "drawersD", "door")):
        moving_body = UsdGeom.Xform.Define(stage, f"/Source/Geometry/{name}_01_obj_01")
        moving_body.AddTranslateOp().Set(Gf.Vec3d(0.2, -0.5, 0.1 * index))
        UsdPhysics.RigidBodyAPI.Apply(moving_body.GetPrim())
        collider = UsdGeom.Cube.Define(stage, moving_body.GetPath().AppendChild(f"{name}_01_mesh_01"))
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        joint_type = UsdPhysics.RevoluteJoint if name == "door" else UsdPhysics.PrismaticJoint
        joint = joint_type.Define(stage, f"/Source/Joints/joint_{name}_01_joint_00")
        joint.CreateBody0Rel().SetTargets([mesh.GetPath()])
        joint.CreateBody1Rel().SetTargets([collider.GetPath()])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(-0.1)
        joint.CreateUpperLimitAttr(0.4)
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.1, 0.2, 0.3))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.4, 0.5, 0.6))
        moving_bodies.append(moving_body)
        joints.append(joint)
    stage.GetRootLayer().Save()
    source_bytes = source.read_bytes()

    specs = importlib.resources.files("isaaclab_arena_environments").joinpath("workbench_cleanup/asset_processing")
    recipe = json.loads(specs.joinpath("recipes/workbench.json").read_bytes())
    recipe["steps"][0].pop("expected_geometry_sha256")  # The synthetic fixture has different mesh coordinates.
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    retrieve = _patch_retrieval(monkeypatch, source)
    processor = UsdAssetProcessor(
        SOURCE_URL,
        "package://isaaclab_arena_environments/workbench_cleanup/asset_processing/wrappers/workbench.usda",
        structural_recipe=str(recipe_path),
    )
    output = processor.process()
    processed = Usd.Stage.Open(output)
    assert not processed.GetCompositionErrors()
    asset_path = Sdf.Path("/Asset/Workbench")
    assert not processed.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI)
    assert processed.GetPrimAtPath(asset_path).HasAPI(UsdPhysics.ArticulationRootAPI)
    table_path = body.GetPath().ReplacePrefix(root.GetPath(), asset_path)
    assert not processed.GetPrimAtPath(table_path).HasAPI(UsdPhysics.RigidBodyAPI)
    assert not processed.GetPrimAtPath(table_path).HasAPI(UsdPhysics.MassAPI)
    frame_path = table_path.AppendPath("table_01_mesh_00/CabinetFrame")
    frame = processed.GetPrimAtPath(frame_path)
    assert frame.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not UsdPhysics.RigidBodyAPI(frame).GetKinematicEnabledAttr().Get()
    assert not processed.GetPrimAtPath("/Asset/Articulations")
    for moving_body, joint in zip(moving_bodies, joints, strict=True):
        output_body = processed.GetPrimAtPath(moving_body.GetPath().ReplacePrefix(root.GetPath(), asset_path))
        assert output_body.HasAPI(UsdPhysics.RigidBodyAPI)
        assert UsdGeom.Xformable(output_body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) == (
            moving_body.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )
        output_joint = UsdPhysics.Joint(
            processed.GetPrimAtPath(joint.GetPath().ReplacePrefix(root.GetPath(), asset_path))
        )
        assert output_joint.GetPrim().GetTypeName() == joint.GetPrim().GetTypeName()
        assert output_joint.GetBody0Rel().GetTargets() == [frame_path]
        assert output_joint.GetBody1Rel().GetTargets() == [
            joint.GetBody1Rel().GetTargets()[0].ReplacePrefix(root.GetPath(), asset_path)
        ]
        for attr in ("axis", "localPos0", "localPos1", "localRot0", "localRot1", "upperLimit"):
            assert output_joint.GetPrim().GetAttribute(f"physics:{attr}").Get() == (
                joint.GetPrim().GetAttribute(f"physics:{attr}").Get()
            )
        if joint.GetPrim().IsA(UsdPhysics.PrismaticJoint):
            assert UsdPhysics.PrismaticJoint(output_joint).GetLowerLimitAttr().Get() == 0
    fixed = UsdPhysics.FixedJoint(processed.GetPrimAtPath(asset_path.AppendPath("Joints/FixedBase")))
    assert fixed and not fixed.GetBody0Rel().GetTargets()
    assert fixed.GetBody1Rel().GetTargets() == [frame_path]
    for support in recipe["steps"][0]["groups"]:
        if support == "CabinetFrame":
            continue
        prim = processed.GetPrimAtPath(table_path.AppendPath(f"table_01_mesh_00/{support}"))
        assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
        assert "PhysxContactReportAPI" in prim.GetMetadata("apiSchemas").GetAppliedItems()

    binding_api = UsdShade.MaterialBindingAPI(frame)
    material, relationship = binding_api.ComputeBoundMaterial("physics")
    if physics_binding == "none":
        assert not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI)
    else:
        physics = UsdPhysics.MaterialAPI(material.GetPrim())
        assert physics.GetStaticFrictionAttr().Get() == pytest.approx(0.5)
        assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(0.4)
        assert physics.GetRestitutionAttr().Get() == pytest.approx(0.15)
        assert binding_api.GetMaterialBindingStrength(relationship) == UsdShade.Tokens.strongerThanDescendants
        assert material.GetPath() == asset_path.AppendPath("Looks/Steel")
    assert binding_api.ComputeBoundMaterial()[0].GetPath() == asset_path.AppendPath("Looks/Paint")
    assert source.read_bytes() == source_bytes
    retrieve.side_effect = AssertionError("cache hit must not retrieve the source")
    assert processor.process() == output


def test_workbench_asset_processing_configuration(monkeypatch, source):
    from isaaclab_arena.assets.sources.simready_central import SimReadyCentralUsdSource
    from isaaclab_arena_environments.workbench_cleanup import assets

    assert isinstance(assets.WorkbenchCleanupWhiteBin.usd_source, SimReadyCentralUsdSource)
    monkeypatch.setattr(assets.WorkbenchCleanupWhiteBin.usd_source, "to_local_usd_path", lambda: "/cache/white_bin.usd")
    white_bin = assets.WorkbenchCleanupWhiteBin()
    assert white_bin.usd_path == "/cache/white_bin.usd"

    tool_assets = (
        (
            assets.WorkbenchCleanupSlottedScrewdriver.usd_source,
            assets.WorkbenchCleanupSlottedScrewdriver,
            "/cache/slotted_screwdriver.usd",
        ),
        (
            assets.WorkbenchCleanupSmallWrench.usd_source,
            assets.WorkbenchCleanupSmallWrench,
            "/cache/small_wrench.usd",
        ),
    )
    for usd_source, asset_type, usd_path in tool_assets:
        assert isinstance(asset_type.usd_source, SimReadyCentralUsdSource)
        monkeypatch.setattr(usd_source, "to_local_usd_path", lambda path=usd_path: path)
        tool = asset_type()
        spawn = tool.object_cfg.spawn
        assert tool.usd_path == usd_path
        assert spawn.make_uninstanceable
        assert spawn.activate_contact_sensors
        assert spawn.rigid_props.linear_damping == 2.0
        assert spawn.rigid_props.angular_damping == 2.0
        assert spawn.rigid_props.sleep_threshold == 0.0
        assert spawn.collision_props.contact_offset == 0.001
        assert spawn.collision_props.rest_offset == 0.0
        assert spawn.collision_props.mesh_collision_property.mesh_approximation_name == "convexHull"

    monkeypatch.setattr(assets.WorkbenchCleanupSocketSet.processor, "process", lambda: "/cache/socket_set.usda")
    socket_set = assets.WorkbenchCleanupSocketSet(initial_joint_positions={"case_lid": 1.5})
    assert socket_set.object_cfg.init_state.joint_pos == {"case_lid": 1.5}

    package_path = assets.WorkbenchCleanupWorkbench.processor.structural_recipe.removeprefix("package://")
    package, relative_path = package_path.split("/", 1)
    recipe = json.loads(importlib.resources.files(package).joinpath(relative_path).read_bytes())
    assert [step["operation"] for step in recipe["steps"]] == ["partition_mesh"]
    monkeypatch.setattr(assets.WorkbenchCleanupWorkbench.processor, "process", lambda: str(source))
    assert assets.WorkbenchCleanupWorkbench().collision_mode.value == "mesh"
    assert assets.WorkbenchCleanupWorkbench(collision_mode="bbox").collision_mode.value == "bbox"


def test_environment_discovery_defers_asset_registration_and_processing():
    import subprocess
    import sys

    code = """
import sys
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena_environments.cli import ensure_environments_registered
from isaaclab_arena_environments.workbench_cleanup.asset_processing.usd_asset_processor import UsdAssetProcessor

ensure_environments_registered()
forbidden_prefixes = ('pxr', 'isaaclab.assets', 'isaaclab.scene', 'isaaclab_assets', 'isaaclab_tasks')
assert not [name for name in sys.modules if name.startswith(forbidden_prefixes)]

registry = AssetRegistry()
assert not registry.is_registered('workbench_cleanup_workbench', ensure_loaded=False)
assert 'isaaclab_arena_environments.asset_library' not in sys.modules
assert 'isaaclab_arena_environments.workbench_cleanup.assets' not in sys.modules

def unexpected_retrieval(*args, **kwargs):
    raise AssertionError('Registration must not retrieve or process assets')

import isaaclab.utils.assets
isaaclab.utils.assets.retrieve_file_path = unexpected_retrieval
UsdAssetProcessor.process = unexpected_retrieval

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
