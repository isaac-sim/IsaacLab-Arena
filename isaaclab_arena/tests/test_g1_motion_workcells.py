# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_workcell_physics(simulation_app):
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena_environments.spatial_memory.workcells.bodies import BODIES
    from isaaclab_arena_environments.spatial_memory.workcells.catalog import SCENES
    from isaaclab_arena_environments.spatial_memory.workcells.layouts import LAYOUTS
    from isaaclab_arena_environments.spatial_memory.workcells.physics import physical_stage

    assert SCENES.keys() == BODIES.keys() == LAYOUTS.keys()
    for scene, bodies in BODIES.items():
        for layout in (0, 1):
            stage = physical_stage(scene, layout)
            for body in bodies:
                prim = stage.GetPrimAtPath("/Workcell/Body_" + body.name)
                assert prim.HasAPI(UsdPhysics.RigidBodyAPI), (scene, body.name)
                assert any(child.HasAPI(UsdPhysics.CollisionAPI) for child in prim.GetChildren())
            feature = LAYOUTS[scene].feature
            matches = [p for p in stage.Traverse() if p.GetName() == feature]
            assert len(matches) == 1, (scene, feature)
            center = UsdGeom.XformCache(Usd.TimeCode(0)).GetLocalToWorldTransform(matches[0]).ExtractTranslation()
            expected = LAYOUTS[scene].centers[layout]
            assert max(abs(a - b) for a, b in zip(center, expected, strict=True)) < 1e-5
    return True


def test_workcell_physics():
    assert run_function_with_persistent_simulation_app(_test_workcell_physics)


def _test_registered_workcells(simulation_app, layout):
    import torch

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.cli import ensure_environments_registered
    from isaaclab_arena_environments.g1_motion_environment import G1MotionEnvironmentCfg
    from isaaclab_arena_environments.spatial_memory.workcells.bodies import BODIES
    from isaaclab_arena_environments.spatial_memory.workcells.outcomes import reached_goal
    from isaaclab_arena_environments.spatial_memory.workcells.task import move_fixture

    ensure_environments_registered()
    factory = EnvironmentRegistry().get_component_by_name("g1_spatial_motion")()
    for name in BODIES:
        print(f"Building G1 workcell {name} / {layout}", flush=True)
        arena = factory.build(G1MotionEnvironmentCfg(scene=name, layout=layout))
        env = ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
        try:
            env.reset()
            base = env.unwrapped
            if name != "seam_inspection":
                assert not reached_goal(base, name, int(layout == "B")).any(), name
            if name == "dishwasher":
                # The stocked plate must settle without tripping the rack finger.
                actions = torch.zeros(env.action_space.shape, device=base.device)
                for _ in range(5):
                    _, _, terminated, truncated, _ = env.step(actions)
                    assert not (terminated | truncated).any(), (name, layout)
            if name == "panel_catches":
                for body in ("left", "right"):
                    asset = base.scene[body]
                    target = asset.data.root_pose_w.torch.clone()
                    target[1, 1] += 0.009
                    asset.write_root_pose_to_sim(target)
                    expected = body == "right"
                    assert reached_goal(base, name, int(layout == "B")).tolist() == [False, expected]
            if arena.task.animation:
                ids = torch.tensor([1], device=base.device)
                move_fixture(base, ids, **arena.task.animation)
            if name == "display_cover":
                from isaaclab_arena_environments.spatial_memory.workcells.task import reset_adhesion, resist_peel

                ids = torch.tensor([1], device=base.device)
                resist_peel(base, ids)
                base.cover_attached[:] = False
                reset_adhesion(base, ids)
                assert base.cover_attached[1].all() and not base.cover_attached[0].any()
        finally:
            env.close()
    return True


@pytest.mark.parametrize("layout", ["A", "B"])
def test_registered_workcells(layout):
    assert run_function_with_persistent_simulation_app(_test_registered_workcells, layout=layout)


def _test_measured_point_cloud(simulation_app):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.g1_motion_environment import G1MotionEnvironment, G1MotionEnvironmentCfg
    from isaaclab_arena_environments.spatial_memory.observations import get_online_map_provider

    arena = G1MotionEnvironment().build(
        G1MotionEnvironmentCfg(scene="shelf_bay", layout="B", enable_cameras=True, mapped=True)
    )
    env = ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    try:
        obs, _ = env.reset()
        assert "robot_head_cam_distance_to_image_plane" in obs["camera_obs"]
        assert not obs["mapped"]["online_map"][..., 3].any()
        base = env.unwrapped
        with torch.no_grad():
            obs, *_ = env.step(torch.zeros(env.action_space.shape, device=base.device))
        assert obs["mapped"]["online_map"][..., 3].any(dim=1).all()
        provider = get_online_map_provider(base)
        base.reset(env_ids=torch.tensor([1], device=base.device))
        assert len(provider.voxels_E[0]) and not len(provider.voxels_E[1])
    finally:
        env.close()
    return True


@pytest.mark.with_cameras
def test_measured_point_cloud():
    assert run_function_with_persistent_simulation_app(_test_measured_point_cloud, enable_cameras=True)


def _test_fixture_playback(simulation_app):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.g1_motion_environment import G1MotionEnvironment, G1MotionEnvironmentCfg
    from isaaclab_arena_environments.spatial_memory.workcells.outcomes import pose_e

    for scene in ("moving_carriage", "closing_door", "descending_hood"):
        arena = G1MotionEnvironment().build(G1MotionEnvironmentCfg(scene=scene))
        env = ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
        try:
            env.reset()
            base = env.unwrapped
            start = pose_e(base, "mover")
            if scene == "moving_carriage":
                load_start = pose_e(base, "part")
            with torch.no_grad():
                actions = torch.zeros(env.action_space.shape, device=base.device)
                for _ in range(10):
                    env.step(actions)
            moved = pose_e(base, "mover")
            if scene == "moving_carriage":
                assert (pose_e(base, "part")[:, 0] - load_start[:, 0]).min() > 0.015
            assert (moved - start).abs().amax(dim=1).min() > 0.008, scene
            base.reset(env_ids=torch.tensor([1], device=base.device))
            after = pose_e(base, "mover")
            assert torch.allclose(after[1], start[1], atol=0.003), scene
            assert torch.allclose(after[0], moved[0], atol=0.003), scene
        finally:
            env.close()
    return True


def test_fixture_playback():
    assert run_function_with_persistent_simulation_app(_test_fixture_playback)


def _test_shelf_stop_occlusion(simulation_app):
    import torch

    from pxr import UsdGeom

    from isaaclab_arena_environments.spatial_memory.geometry import segments_blocked_by_boxes
    from isaaclab_arena_environments.spatial_memory.workcells.physics import physical_stage

    # The eye pair and gaze are the close shelf pose in the guide, expressed in workcell coordinates.
    eye = torch.tensor([-1.24036, -0.54081, 1.15790])
    gaze = torch.tensor([-1.50632, -0.45692, 0.985]) - eye
    right = torch.linalg.cross(gaze, torch.tensor([0.0, 0.0, 1.0]))
    right /= right.norm()
    eyes = torch.stack((eye - 0.032 * right, eye + 0.032 * right))
    for layout in (0, 1):
        stage = physical_stage("shelf_bay", layout)
        bbox = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        carton = bbox.ComputeWorldBound(stage.GetPrimAtPath("/Workcell/Body_load")).ComputeAlignedRange()
        stop = bbox.ComputeWorldBound(stage.GetPrimAtPath("/Workcell/Guard")).ComputeAlignedRange()
        corners = torch.tensor([tuple(stop.GetCorner(i)) for i in range(8)])
        carton_pose = torch.tensor([*carton.GetMidpoint(), 0.0, 0.0, 0.0, 1.0])
        assert segments_blocked_by_boxes(
            eyes,
            corners.expand(2, -1, -1),
            carton_pose.expand(2, 1, 7),
            torch.tensor([tuple(carton.GetSize() / 2)]),
        ).all()
    return True


def test_shelf_stop_occlusion():
    assert run_function_with_persistent_simulation_app(_test_shelf_stop_occlusion)


def _test_library_visuals(simulation_app):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena_environments.spatial_memory.workcells.physics import workcell_usd

    for scene, body, shape in (("shelf_bay", "load", "load"), ("dishwasher", "plate", "HeldPlate")):
        _, path = workcell_usd(scene, 0)
        stage = Usd.Stage.Open(path)
        actor = stage.GetPrimAtPath("/Workcell/Body_" + body)
        proxy = actor.GetChild(shape)
        visual = actor.GetChild("Appearance")
        assert proxy.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdGeom.Imageable(proxy).ComputeVisibility() == UsdGeom.Tokens.invisible
        assert any(p.HasAPI(UsdShade.MaterialBindingAPI) for p in Usd.PrimRange(visual))
        assert not any(
            p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(visual)
        )
        cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], False, True)
        actual = cache.ComputeWorldBound(visual).ComputeAlignedRange()
        expected = cache.ComputeWorldBound(proxy).ComputeAlignedRange()
        for a, b in ((actual.GetMin(), expected.GetMin()), (actual.GetMax(), expected.GetMax())):
            assert (a - b).GetLength() < 0.001, (scene, a, b)
    return True


def test_library_visuals():
    assert run_function_with_persistent_simulation_app(_test_library_visuals)
