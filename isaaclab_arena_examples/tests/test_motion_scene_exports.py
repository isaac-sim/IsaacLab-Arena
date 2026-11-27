# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the robot-free USD scenes without creating an environment or a robot."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_motion_scene_exports(simulation_app):
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena_environments.spatial_memory.workcells.catalog import SCENES, build_stage
    from isaaclab_arena_examples.spatial_memory.export_motion_scenes import export_scene

    moving_components = {"closing_door": "Door", "moving_carriage": "Carriage", "descending_hood": "Hood"}
    assert len(SCENES) == 30
    with tempfile.TemporaryDirectory() as directory:
        for name in SCENES:
            path = Path(directory) / f"{name}.usda"
            authored_stage = build_stage(name, 3847)
            export_scene(authored_stage, path)
            stage = Usd.Stage.Open(str(path))
            assert stage.GetDefaultPrim().GetPath() == "/World"
            assert UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1
            for view in ("Overview", "Approach"):
                camera = UsdGeom.Camera.Get(stage, "/World/" + view)
                assert camera.GetHorizontalApertureAttr().Get() > 0
            parts = stage.GetPrimAtPath("/World/Workcell")
            shapes = [prim for prim in Usd.PrimRange(parts) if prim.IsA(UsdGeom.Gprim)]
            assert shapes and all(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in shapes)
            if name in moving_components:
                mover = UsdGeom.Xformable(parts.GetChild(moving_components[name]))
                assert mover.GetLocalTransformation(0) != mover.GetLocalTransformation(stage.GetEndTimeCode() / 2)
                assert mover.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
            else:
                assert stage.GetEndTimeCode() == 0
    return True


def test_motion_scene_exports():
    assert run_function_with_persistent_simulation_app(_test_motion_scene_exports)


def test_failed_export_preserves_existing_scene(tmp_path):
    from isaaclab_arena_examples.spatial_memory.export_motion_scenes import export_scene

    path = tmp_path / "workcell.usda"
    path.write_text("previous scene")

    def interrupted_export(filename):
        Path(filename).write_text("partial scene")
        return False

    stage = Mock()
    stage.GetRootLayer.return_value.Export.side_effect = interrupted_export
    with pytest.raises(OSError, match="Could not export"):
        export_scene(stage, path)
    assert list(tmp_path.iterdir()) == [path]
    assert path.read_text() == "previous scene"
