# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate workbench asset registration, spawn settings, and uploaded layers."""


def test_environment_discovery_defers_asset_registration():
    import subprocess
    import sys

    code = """
import sys
from isaaclab_arena_environments.cli import ensure_environments_registered

ensure_environments_registered()
forbidden = ('pxr', 'isaaclab.assets', 'isaaclab.scene', 'isaaclab_assets', 'isaaclab_tasks')
assert not [name for name in sys.modules if name.startswith(forbidden)]
assert 'isaaclab_arena_environments.asset_library' not in sys.modules
assert 'isaaclab_arena_environments.workbench_cleanup.assets' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def _test_workbench_registration_does_not_download(simulation_app, monkeypatch):
    from isaaclab_arena.assets.registries import AssetRegistry

    def unexpected_retrieval(*args, **kwargs):
        raise AssertionError("Asset registration must not download assets")

    monkeypatch.setattr("isaaclab.utils.assets.retrieve_file_path", unexpected_retrieval)
    registry = AssetRegistry()
    for name in ("workbench", "white_bin", "socket_set", "oscilloscope", "slotted_screwdriver", "small_wrench"):
        asset_type = registry.get_asset_by_name(f"workbench_cleanup_{name}")
        assert asset_type.__module__ == "isaaclab_arena_environments.workbench_cleanup.assets"
    return True


def test_workbench_registration_does_not_download(monkeypatch):
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(
        _test_workbench_registration_does_not_download, monkeypatch=monkeypatch
    )


def _test_workbench_asset_configuration(simulation_app, monkeypatch, tmp_path):
    from pxr import Usd, UsdGeom

    source = tmp_path / "background.usda"
    stage = Usd.Stage.CreateNew(str(source))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Asset").GetPrim())
    stage.GetRootLayer().Save()
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

    monkeypatch.setattr(assets.WorkbenchCleanupSocketSet, "usd_path", "/cache/socket_set.usda")
    socket_set = assets.WorkbenchCleanupSocketSet(initial_joint_positions={"case_lid": 1.5})
    assert socket_set.object_cfg.init_state.joint_pos == {"case_lid": 1.5}

    monkeypatch.setattr(assets.WorkbenchCleanupWorkbench, "usd_path", str(source))
    assert assets.WorkbenchCleanupWorkbench().collision_mode is None
    assert assets.WorkbenchCleanupWorkbench(collision_mode="bbox").collision_mode.value == "bbox"

    return True


def test_workbench_asset_configuration(monkeypatch, tmp_path):
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(
        _test_workbench_asset_configuration, monkeypatch=monkeypatch, tmp_path=tmp_path
    )


def _test_workbench_uploaded_articulations(simulation_app):
    from pxr import Usd, UsdPhysics

    from isaaclab_arena_environments.workbench_cleanup.assets import (
        WorkbenchCleanupOscilloscope,
        WorkbenchCleanupSocketSet,
        WorkbenchCleanupWorkbench,
    )

    stage = Usd.Stage.Open(WorkbenchCleanupWorkbench.usd_path)
    cabinet = stage.GetPrimAtPath("/Asset/Workbench")
    articulation_roots = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert articulation_roots == [cabinet]
    movable_joints = {
        prim.GetName()
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint)
    }
    assert movable_joints == {
        "joint_drawersA_01_joint_00",
        "joint_drawersB_01_joint_00",
        "joint_drawersC_01_joint_00",
        "joint_drawersD_01_joint_00",
        "joint_door_01_joint_00",
    }
    for path in (
        "/Asset/Workbench/Geometry/table_01_obj_00/table_01_mesh_00/Tabletop",
        "/Asset/Workbench/Geometry/table_01_obj_00/table_01_mesh_00/Structure/OverheadShelf",
    ):
        assert stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI)

    for asset_type, moving_joint in (
        (WorkbenchCleanupSocketSet, "Physics/Joints/case_lid"),
        (WorkbenchCleanupOscilloscope, "Joints/joint_button_01_joint_00"),
    ):
        stage = Usd.Stage.Open(asset_type.usd_path)
        root = stage.GetDefaultPrim()
        articulation_roots = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        assert articulation_roots == [root]
        movable_joints = [
            prim.GetPath().MakeRelativePath(root.GetPath()).pathString
            for prim in stage.Traverse()
            if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint)
        ]
        assert movable_joints == [moving_joint]
        if asset_type is WorkbenchCleanupSocketSet:
            body_names = {prim.GetName() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
            assert body_names == {"Case_base", "Case_lid"}
    return True


def test_workbench_uploaded_articulations():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_uploaded_articulations)
