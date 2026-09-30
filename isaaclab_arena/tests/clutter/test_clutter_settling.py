# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Offline clutter settling, acceptance and candidate isolation."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _assert_scene_state_equal(actual, expected):
    import torch

    for kind, states in expected.items():
        for name, state in states.items():
            for field, value in state.items():
                torch.testing.assert_close(actual[kind][name][field], value, atol=1e-6, rtol=0)


def _test_settling_rejects_object_sets_before_reset(simulation_app, tmp_path):
    from unittest.mock import patch

    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
    from isaaclab_arena.relations.relations import ClutterOn, get_relation

    arena_env = _make_primitive_clutter_scene(tmp_path)
    assets = arena_env.get_placement_assets()
    cube = next(asset for asset in assets if get_relation(asset, ClutterOn) is not None)
    variants = RigidObjectSet("cube_variants", [cube])
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1)).make_registered()
    try:
        with patch.object(env.unwrapped, "reset", side_effect=AssertionError("must reject before resetting")):
            with pytest.raises(AssertionError, match="Resolve object sets"):
                collect_settled_placements(env, 1, scene_assets=[*assets, variants])
    finally:
        env.close()
    return True


def test_settling_rejects_object_sets_before_reset(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_settling_rejects_object_sets_before_reset, tmp_path=tmp_path
    )


def _make_primitive_clutter_scene(tmp_path, raised_support=False):
    import yaml
    from unittest.mock import patch

    from isaaclab_arena.assets.registries import AssetRegistry, ensure_assets_registered
    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.tests.test_settled_placement import _write_scene

    source = tmp_path / "clutter.yaml"
    _write_scene(source)
    if raised_support:
        import trimesh

        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.Open(str(tmp_path / "table.usda"))
        stage.RemovePrim("/Body/geometry")
        frame = UsdGeom.Xform.Define(stage, "/Body/tabletop")
        frame.AddTranslateOp().Set((0, 0, 0))
        frame.AddOrientOp().Set(Gf.Quatf(1))
        frame.AddScaleOp().Set((1, 1, 1))
        box = trimesh.creation.box(extents=(0.8, 0.8, 0.04))
        tabletop = UsdGeom.Mesh.Define(stage, "/Body/tabletop/geometry")
        tabletop.CreatePointsAttr(box.vertices.tolist())
        tabletop.CreateFaceVertexCountsAttr([3] * len(box.faces))
        tabletop.CreateFaceVertexIndicesAttr(box.faces.flatten().tolist())
        UsdPhysics.CollisionAPI.Apply(tabletop.GetPrim())
        rail = UsdGeom.Cube.Define(stage, "/Body/rail")
        rail.CreateSizeAttr(1)
        rail.AddTranslateOp().Set((0, 0.38, 0.12))
        rail.AddScaleOp().Set((0.8, 0.04, 0.2))
        UsdPhysics.CollisionAPI.Apply(rail.GetPrim())
        stage.GetRootLayer().Save()
    data = yaml.safe_load(source.read_text())
    data["relations"][1] = {
        "kind": "clutter_on",
        "subject": "cube",
        "reference": "table",
        "params": {"clearance_m": 0.2, "random_yaw": False},
    }
    ensure_assets_registered()
    with patch.dict(AssetRegistry()._components, {"recording_no_embodiment": NoEmbodiment}):
        return ArenaEnvGraphSpec.model_validate(data).to_arena_env()


def _test_settling_rejects_static_geometry_not_restored_by_reset(simulation_app, tmp_path):
    import torch
    from contextlib import closing
    from unittest.mock import patch

    import warp as wp
    from isaaclab.sim.views import FrameView
    from pxr import Usd, UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
    from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams

    arena_env = _make_primitive_clutter_scene(tmp_path)
    floor = arena_env.scene.assets["floor"]
    stage = Usd.Stage.Open(floor.usd_path)
    stage.GetDefaultPrim().RemoveAPI(UsdPhysics.RigidBodyAPI)
    stage.GetDefaultPrim().RemoveAPI(UsdPhysics.MassAPI)
    stage.GetRootLayer().Save()
    arena_env.scene.add_asset(
        Object(
            name=floor.name,
            usd_path=floor.usd_path,
            object_type=ObjectType.BASE,
            initial_pose=floor.get_initial_pose(),
        )
    )
    assets = arena_env.get_placement_assets()
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    try:
        env.reset()
        base = env.unwrapped
        assert "floor" in base.scene.extras
        original = base.arena_world.get_pose_e("floor")
        with closing(
            FrameView(
                floor.prim_path.format(ENV_REGEX_NS=base.scene.env_regex_ns),
                device=base.device,
                stage=base.scene.stage,
                validate_xform_ops=False,
            )
        ) as floor_view:
            original_positions = floor_view.get_world_poses()[0].torch.clone()
            moved_positions = original_positions.clone()
            moved_positions[1, 0] += 2.0
            floor_view.set_world_poses(positions=wp.from_torch(moved_positions))
            base.sim.forward()
            # BASE geometry has no root reset event; normal reset cannot repair this scene edit.
            env.reset()
            moved = original.clone()
            moved[1, 0] += 2.0
            torch.testing.assert_close(base.arena_world.get_pose_e("floor"), moved)
            before = base.scene.get_state()
            with patch.object(base, "reset", side_effect=AssertionError("must reject before sampling again")):
                with pytest.raises(AssertionError, match="'floor' differs from its configured pose"):
                    collect_settled_placements(env, 1, SettledPlacementParams(), scene_assets=assets)
            _assert_scene_state_equal(base.scene.get_state(), before)
            torch.testing.assert_close(base.arena_world.get_pose_e("floor"), moved)
            floor_view.set_world_poses(positions=wp.from_torch(original_positions))
            base.sim.forward()
            # The same scene at its authored poses remains usable across both environments.
            results = collect_settled_placements(
                env,
                1,
                SettledPlacementParams(num_steps=480, validators=default_clutter_validators()),
                scene_assets=assets,
            )
            assert len(results.accepted_indices) == 2
            assert all(abs(pose.position_xyz[0]) < 0.4 for pose in results.poses["cube_body"])
    finally:
        env.close()
    return True


def test_settling_rejects_static_geometry_not_restored_by_reset(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_settling_rejects_static_geometry_not_restored_by_reset, tmp_path=tmp_path
    )


def _test_clutter_collection_uses_shared_batches(simulation_app, tmp_path):
    import torch
    from copy import deepcopy
    from unittest.mock import patch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
    from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams
    from isaaclab_arena.relations.placement_events import get_placement_pool
    from isaaclab_arena.relations.relations import ClutterOn, RequiresReachability, get_relation
    from isaaclab_arena.utils.physics_settle import step_physics

    arena_env = _make_primitive_clutter_scene(tmp_path)
    arena_env.placer_params.min_unique_layouts_per_env = 2
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    try:
        base = env.unwrapped
        pool = get_placement_pool(base)
        params = SettledPlacementParams(num_steps=480, validators=default_clutter_validators())
        before = base.arena_world.get_pose_e("cube_body").clone()
        assets = arena_env.get_placement_assets()
        clutter = next(asset for asset in assets if get_relation(asset, ClutterOn) is not None)
        clutter.add_relation(RequiresReachability())
        with patch.object(base, "reset", side_effect=AssertionError("must reject before resetting")):
            with pytest.raises(AssertionError, match="reachability after clutter drops"):
                collect_settled_placements(env, 1, params, scene_assets=assets)
        clutter.relations.remove(get_relation(clutter, RequiresReachability))
        # A fixed target can retain its pre-physics reachability requirement.
        get_relation(clutter, ClutterOn).parent.add_relation(RequiresReachability())
        with patch.object(pool, "sample_for_envs", wraps=pool.sample_for_envs) as sample:
            result = collect_settled_placements(env, 2, params, scene_assets=arena_env.get_placement_assets())
        assert sample.call_count == 2
        assert pool.remaining == 0
        assert result.attempted == 4
        assert result.accepted_indices == [(0, 0), (1, 0), (0, 1), (1, 1)]
        assert not result.rejections
        for outcome in result.validation:
            reports = {report.check: report for report in outcome.post_physics}
            assert reports["physics_settled"].passed
            assert reports["pose_shift"].passed
            assert reports["support_containment"].passed
            assert reports["articulation_link_shift"].passed is None
            assert reports["support_containment"].configuration["fall_through_tolerance_m"] == 0.01
        after = base.arena_world.get_pose_e("cube_body")
        assert torch.all(before[:, 2] - after[:, 2] > 0.15)
        for env_id in range(2):
            torch.testing.assert_close(after[env_id], result.poses["cube_body"][2 + env_id].to_tensor(base.device))
        step_physics(base, 200)
        torch.testing.assert_close(base.arena_world.get_pose_e("cube_body"), after, atol=0.005, rtol=0)

        # The same pool/reset path reports rejections and permits explicit check configuration.
        # A one-step drop retains high downward speed after release.
        from isaaclab_arena.offline_placement.post_physics_validation import evaluate_settled_batch

        checked_outcomes = []

        def evaluate(batch, validators):
            outcomes = evaluate_settled_batch(batch, validators)
            checked_outcomes.append(outcomes)
            return outcomes

        short = SettledPlacementParams(num_steps=1)
        short.validators["physics_settled"]["lin_vel_thresh"] = 0.0001
        explicit_settings = deepcopy(short.validators)
        with patch("isaaclab_arena.offline_placement.settled_placement.evaluate_settled_batch", side_effect=evaluate):
            defaults = collect_settled_placements(env, 1, scene_assets=arena_env.get_placement_assets())
            rejected = collect_settled_placements(env, 1, short, scene_assets=arena_env.get_placement_assets())
        assert defaults.attempted == 2
        default_reports = {report.check: report for report in checked_outcomes[0][0].post_physics}
        explicit_reports = {report.check: report for report in checked_outcomes[1][0].post_physics}
        assert default_reports["support_containment"].passed
        assert default_reports["pose_shift"].passed  # Intentional drops are not displacement failures.
        assert "support_containment" not in explicit_reports  # Explicit check selection is preserved.
        assert short.validators == explicit_settings
        assert rejected.attempted == 2
        assert not rejected.accepted_indices
        assert all("physics_settled" in reason for reason in rejected.rejections.values())
    finally:
        env.close()
    return True


def test_clutter_collection_uses_shared_batches(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_clutter_collection_uses_shared_batches, tmp_path=tmp_path)


def _test_raised_support_requires_explicit_surface(simulation_app, tmp_path):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
    from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor, get_relation
    from isaaclab_arena.utils.physics_settle import step_physics

    params = SettledPlacementParams(num_steps=480, validators=default_clutter_validators())
    for support_kind in ("whole_table", "unprepared_tabletop", "tabletop", "configured_table"):
        params.validators["support_containment"]["minimum_resting_heights_m"] = {}
        directory = tmp_path / support_kind
        directory.mkdir()
        arena_env = _make_primitive_clutter_scene(directory, raised_support=True)
        assets = arena_env.get_placement_assets()
        cube = next(asset for asset in assets if get_relation(asset, ClutterOn) is not None)
        relation = get_relation(cube, ClutterOn)
        if support_kind in ("unprepared_tabletop", "tabletop"):
            if support_kind == "unprepared_tabletop":
                from pxr import Usd, UsdGeom

                stage = Usd.Stage.Open(str(directory / "table.usda"))
                UsdGeom.Xformable(stage.GetPrimAtPath("/Body/tabletop")).ClearXformOpOrder()
                stage.GetRootLayer().Save()
            surface = ObjectReference(
                name="tabletop",
                parent_asset=relation.parent,
                prim_path="{ENV_REGEX_NS}/table/tabletop",
                object_type=ObjectType.BASE,
            )
            surface.add_relation(IsAnchor())
            relation.parent = surface
            arena_env.scene.add_asset(surface)
        env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1)).make_registered()
        try:
            base = env.unwrapped
            assets = arena_env.get_placement_assets()
            if support_kind == "configured_table":
                params.validators["support_containment"]["minimum_resting_heights_m"] = {"table": 0.02}
            if support_kind in ("whole_table", "unprepared_tabletop"):
                reason = "flat rectangular" if support_kind == "whole_table" else "translate, orient and scale"
                before = base.scene.get_state()
                with patch.object(base, "reset", side_effect=AssertionError("must reject before resetting")):
                    with pytest.raises(AssertionError, match=reason):
                        collect_settled_placements(env, 1, params, scene_assets=assets)
                _assert_scene_state_equal(base.scene.get_state(), before)
            else:
                result = collect_settled_placements(env, 1, params, scene_assets=assets)
                assert result.accepted_indices == [(0, 0)], result.rejections
                pose = base.arena_world.get_pose_e("cube_body").clone()
                # The tabletop is at 0.52 m; the rail is at 0.72 m. The cube is 0.1 m tall.
                assert abs(float(pose[0, 2]) - 0.57) < 0.002
                step_physics(base, 200)
                torch.testing.assert_close(base.arena_world.get_pose_e("cube_body"), pose, atol=0.002, rtol=0)
        finally:
            env.close()
    return True


def test_raised_support_requires_explicit_surface(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_raised_support_requires_explicit_surface, tmp_path=tmp_path
    )


def _test_bowl_collection_uses_local_resting_height(simulation_app, tmp_path):
    import torch
    from unittest.mock import patch

    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_library import BowlYcbRobolab
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
    from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.physics_settle import step_physics
    from isaaclab_arena.utils.pose import Pose

    arena_env = _make_primitive_clutter_scene(tmp_path)
    floor = next(asset for asset in arena_env.get_placement_assets() if asset.name == "floor")
    bowl = BowlYcbRobolab(instance_name="bowl", initial_pose=Pose((0.3, -0.2, 0.5), (0, 0, 2**-0.5, 2**-0.5)))
    bowl.object_cfg.spawn.rigid_props = sim_utils.RigidBodyBaseCfg(kinematic_enabled=True)
    bowl.add_relation(IsAnchor())
    cube = Object(name="cube_body", usd_path=str(tmp_path / "cube.usda"), scale=(0.4, 0.2, 0.2))
    cube.add_relation(ClutterOn(bowl, clearance_m=0.01, spread=0.15, random_yaw=False))
    arena_env.scene = Scene([bowl, cube, floor])
    arena_env.placer_params.placement_seed = 42
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    params = SettledPlacementParams(num_steps=240, validators=default_clutter_validators())
    containment = params.validators["support_containment"]
    # This YCB bowl's inner floor is at about -0.025 m in its local frame; its rim is +0.0275 m.
    containment["minimum_resting_heights_m"] = {"bowl": -0.025}
    containment["fall_through_tolerance_m"] = 0.002
    try:
        assets = arena_env.get_placement_assets()
        for heights, reason in (
            ({"misspelled_bowl": -0.025}, "unknown ClutterOn supports"),
            ({"bowl": -1}, "local Z bounds"),
        ):
            containment["minimum_resting_heights_m"] = heights
            with patch.object(env.unwrapped, "reset", side_effect=AssertionError("must reject before resetting")):
                with pytest.raises(AssertionError, match=reason):
                    collect_settled_placements(env, 1, params, scene_assets=assets)
        containment["minimum_resting_heights_m"] = {"bowl": -0.025}
        result = collect_settled_placements(env, 1, params, scene_assets=assets)
        poses = env.unwrapped.arena_world.get_pose_e("cube_body").clone()
        assert result.accepted_indices, (result.rejections, poses.tolist())
        assert result.attempted == 2
        accepted_envs = [env_id for env_id, _ in result.accepted_indices]
        accepted_poses = poses[accepted_envs]
        # Entire accepted cubes are below the rim, yet above the configured interior floor.
        assert torch.all(accepted_poses[:, 2] + 0.01 < 0.5 + 0.0275)
        assert torch.all(accepted_poses[:, 2] - 0.01 >= 0.5 - 0.025 - 0.002)
        for outcome in result.validation:
            report = next(report for report in outcome.post_physics if report.check == "support_containment")
            assert report.passed
            assert report.configuration["minimum_resting_heights_m"] == {"bowl": -0.025}
        step_physics(env.unwrapped, 200)
        torch.testing.assert_close(
            env.unwrapped.arena_world.get_pose_e("cube_body")[accepted_envs], accepted_poses, atol=0.002, rtol=0
        )
    finally:
        env.close()
    return True


def test_bowl_collection_uses_local_resting_height(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_bowl_collection_uses_local_resting_height, tmp_path=tmp_path
    )
