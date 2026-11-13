# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate workbench asset configuration and uploaded layers."""


def _test_workbench_asset_configuration(simulation_app, monkeypatch):
    from isaaclab_arena.assets.object_library import (
        SimReadyCentralSlottedScrewdriver,
        SimReadyCentralSmallWrench,
        SimReadyCentralSocketSet,
        SimReadyCentralWhiteBin,
    )
    from isaaclab_arena.assets.simready_constants import SIMREADY_CENTRAL_PHYSICS_VARIANTS

    for asset_type in (
        SimReadyCentralWhiteBin,
        SimReadyCentralSlottedScrewdriver,
        SimReadyCentralSmallWrench,
    ):
        assert asset_type.spawn_cfg_addon["variants"] == SIMREADY_CENTRAL_PHYSICS_VARIANTS

    monkeypatch.setattr(SimReadyCentralSocketSet, "usd_path", "/cache/socket_set.usda")
    socket_set = SimReadyCentralSocketSet(initial_joint_positions={"case_lid": 1.5})
    assert socket_set.object_cfg.init_state.joint_pos == {"case_lid": 1.5}

    return True


def test_workbench_asset_configuration(monkeypatch):
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_asset_configuration, monkeypatch=monkeypatch)


def _test_workbench_uploaded_articulations(simulation_app):
    from pxr import Usd, UsdPhysics

    from isaaclab_arena.assets.background_library import SimReadyCentralWorkbench
    from isaaclab_arena.assets.object_library import SimReadyCentralOscilloscope, SimReadyCentralSocketSet

    stage = Usd.Stage.Open(SimReadyCentralWorkbench.usd_path)
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
        (SimReadyCentralSocketSet, "Physics/Joints/case_lid"),
        (SimReadyCentralOscilloscope, "Joints/joint_button_01_joint_00"),
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
        if asset_type is SimReadyCentralSocketSet:
            body_names = {prim.GetName() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
            assert body_names == {"Case_base", "Case_lid"}
    return True


def test_workbench_uploaded_articulations():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_uploaded_articulations)
