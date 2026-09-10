# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pathlib
import torch
from types import SimpleNamespace

import pytest
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_cable_reset_event_runtime(_, device: str):
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg

    from isaaclab_arena.assets.cable import Cable
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.configclass import combine_configclass_instances

    cable_asset = Cable(
        name="test_cable",
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=sim_utils.CableCfg(
            positions=((0.0, 0.0, 1.0), (0.0, 0.2, 1.0), (0.0, 0.4, 1.0), (0.0, 0.6, 1.0)),
            physics_material=sim_utils.CableMaterialCfg(),
        ),
    )
    arena_scene = Scene(assets=[cable_asset])
    scene_cfg = combine_configclass_instances(
        "CableSceneCfg",
        InteractiveSceneCfg(num_envs=2, env_spacing=1.0, replicate_physics=True),
        arena_scene.get_scene_cfg(),
    )
    sim_cfg = SimulationCfg(
        device=device,
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=2),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.update(0.0)
        cable = scene[cable_asset.name]
        default_pose = cable.data.default_segment_pose_w.torch.clone()
        default_velocity = cable.data.default_segment_velocity_w.torch.clone()
        moved_pose = default_pose.clone()
        moved_pose[..., 0] += 0.25
        moved_velocity = torch.ones_like(default_velocity)
        all_env_ids = torch.arange(2, device=sim.device, dtype=torch.int32)
        cable.write_segment_pose_to_sim_index(segment_pose=moved_pose, env_ids=all_env_ids)
        cable.write_segment_velocity_to_sim_index(segment_velocity=moved_velocity, env_ids=all_env_ids)
        cable.update(0.0)

        event_cfg = arena_scene.get_events_cfg().test_cable
        reset_env_ids = torch.tensor([1], device=sim.device, dtype=torch.int32)
        event_cfg.func(SimpleNamespace(scene=scene), reset_env_ids, **event_cfg.params)
        cable.update(0.0)

        torch.testing.assert_close(cable.data.segment_pose_w.torch[0], moved_pose[0])
        torch.testing.assert_close(cable.data.segment_velocity_w.torch[0], moved_velocity[0])
        torch.testing.assert_close(cable.data.segment_pose_w.torch[1], default_pose[1])
        torch.testing.assert_close(cable.data.segment_velocity_w.torch[1], default_velocity[1])

    return True


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_cable_reset_event_runtime(device: str):
    assert run_function_with_persistent_simulation_app(_test_cable_reset_event_runtime, device=device)


def _test_cable_scene_export(_, output_path: pathlib.Path) -> bool:
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom

    from isaaclab_arena.assets.cable import Cable
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    positions = ((0.0, 0.0, 0.0), (0.1, 0.05, 0.0), (0.2, 0.0, 0.0))
    initial_pose = Pose(
        position_xyz=(0.1, 0.2, 0.3),
        rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068),
    )
    cable = Cable(
        name="test_cable",
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=sim_utils.CableCfg(
            positions=positions,
            physics_material=sim_utils.CableMaterialCfg(thickness=0.01),
        ),
        initial_pose=initial_pose,
    )

    Scene(assets=[cable]).export_to_usd(output_path)

    stage = Usd.Stage.Open(output_path.as_posix())
    assert stage is not None
    assert stage.GetDefaultPrim().GetPath() == "/World"
    cable_prim = stage.GetPrimAtPath("/World/test_cable")
    assert cable_prim.GetTypeName() == "Xform"
    assert tuple(cable_prim.GetAttribute("xformOp:translate").Get()) == pytest.approx(initial_pose.position_xyz)
    orientation = cable_prim.GetAttribute("xformOp:orient").Get()
    orientation_xyzw = (*orientation.GetImaginary(), orientation.GetReal())
    assert orientation_xyzw == pytest.approx(initial_pose.rotation_xyzw)

    curve_prim = stage.GetPrimAtPath("/World/test_cable/geometry/mesh")
    curves = UsdGeom.BasisCurves(curve_prim)
    assert curves
    for point, expected in zip(curves.GetPointsAttr().Get(), positions, strict=True):
        assert tuple(point) == pytest.approx(expected)
    assert list(curves.GetWidthsAttr().Get()) == pytest.approx([0.01])
    assert "PhysicsCurvesDeformableSimAPI" in curve_prim.GetPrimTypeInfo().GetAppliedAPISchemas()
    return True


def test_cable_scene_export(tmp_path: pathlib.Path):
    output_path = tmp_path / "cable.usd"
    assert run_function_with_persistent_simulation_app(_test_cable_scene_export, output_path=output_path)


def test_cable_asset_config():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import CableObjectCfg
    from isaaclab.managers import EventTermCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    from isaaclab_arena.assets.cable import Cable
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.terms.events import reset_cable_to_default
    from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange

    initial_pose = Pose(position_xyz=(0.1, 0.2, 0.3))
    cable = Cable(
        name="test_cable",
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=sim_utils.CableCfg(
            positions=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
            physics_material=sim_utils.CableMaterialCfg(),
        ),
        initial_pose=initial_pose,
    )

    assert isinstance(cable, ObjectBase)
    assert cable.object_type == ObjectType.CABLE
    assert cable.get_prim_path() == "{ENV_REGEX_NS}/Cable"

    scene_key, cable_cfg = cable.get_object_cfg()
    assert scene_key == "test_cable"
    assert isinstance(cable_cfg, CableObjectCfg)
    assert cable_cfg.prim_path == "{ENV_REGEX_NS}/Cable"
    assert cable_cfg.init_state.pos == initial_pose.position_xyz
    assert cable_cfg.init_state.rot == initial_pose.rotation_xyzw

    event_name, event_cfg = cable.get_event_cfg()
    assert event_name == "test_cable"
    assert isinstance(event_cfg, EventTermCfg)
    assert event_cfg.func is reset_cable_to_default
    assert event_cfg.mode == "reset"
    assert event_cfg.params["asset_cfg"].name == "test_cable"

    with pytest.raises(NotImplementedError, match="relation-based placement"):
        cable.add_relation(IsAnchor())
    with pytest.raises(NotImplementedError, match="bounding boxes"):
        cable.get_bounding_box()
    with pytest.raises(NotImplementedError, match="segment-state writes"):
        cable.layout_pose_to_scene_writes(Pose())
    with pytest.raises(NotImplementedError, match="fixed initial Pose"):
        cable.set_initial_pose(PosePerEnv([Pose()]))
    with pytest.raises(NotImplementedError, match="fixed initial Pose"):
        cable.set_initial_pose(PoseRange())

    scene = Scene(assets=[cable])
    with pytest.raises(AssertionError, match="requires the Newton physics backend"):
        scene.validate_simulation_cfg(SimulationCfg())
    scene.validate_simulation_cfg(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))
