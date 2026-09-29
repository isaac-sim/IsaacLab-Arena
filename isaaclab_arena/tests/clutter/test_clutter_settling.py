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


def _test_settling_rejects_kinematic_variant_before_release(simulation_app, tmp_path):
    from unittest.mock import patch

    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.background_library import OfficeTableBackground
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.arena_world_scene_access import get_representative_rigid_body_prims
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.offline_placement.clutter_settling import settle_clutter
    from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    objects = []
    for index, kinematic in enumerate((False, True)):
        path = tmp_path / f"cube_{index}.usda"
        stage = Usd.Stage.CreateNew(str(path))
        root = UsdGeom.Xform.Define(stage, "/Cube").GetPrim()
        stage.SetDefaultPrim(root)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        body = UsdPhysics.RigidBodyAPI.Apply(root)
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(kinematic)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.1)
        cube = UsdGeom.Cube.Define(stage, "/Cube/geometry")
        cube.CreateSizeAttr(0.06)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        stage.GetRootLayer().Save()
        objects.append(Object(name=f"cube_{index}", usd_path=str(path)))

    table = OfficeTableBackground()
    table.set_initial_pose(Pose.identity())
    table.add_relation(IsAnchor())
    variants = RigidObjectSet("mixed_cube", objects, initial_pose=Pose((0, 0, 1.2)))
    variants.assign_variants(2)
    assignments = list(variants.variant_indices_by_env)
    variants.add_relation(ClutterOn(table, clearance_m=0.2, random_yaw=False))
    arena_env = IsaacLabArenaEnvironment("mixed_mobility", Scene([table, variants]))
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    try:
        env.reset()
        bodies = get_representative_rigid_body_prims(env.unwrapped.scene, "mixed_cube")
        assert sorted(UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get() for body in bodies) == [
            False,
            True,
        ]
        initial = env.unwrapped.scene.get_state()
        with patch(
            "isaaclab_arena.offline_placement.clutter_settling.collect_settled_placements",
            side_effect=AssertionError("release must not run for kinematic clutter"),
        ) as release:
            with pytest.raises(AssertionError, match="mixed_cube.*must be dynamic"):
                settle_clutter(env, arena_env.get_placement_assets(), 1, SettledPlacementParams())
            release.assert_not_called()
        assert variants.variant_indices_by_env == assignments
        _assert_scene_state_equal(env.unwrapped.scene.get_state(), initial)
    finally:
        env.close()
    return True


def test_settling_rejects_kinematic_variant_before_release(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_settling_rejects_kinematic_variant_before_release, tmp_path=tmp_path
    )


def _make_primitive_clutter_scene(tmp_path):
    import yaml
    from unittest.mock import patch

    from isaaclab_arena.assets.registries import AssetRegistry, ensure_assets_registered
    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.tests.test_settled_placement import _write_scene

    source = tmp_path / "clutter.yaml"
    _write_scene(source)
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


def _test_settling_rejects_moved_fixed_assets_before_solving(simulation_app, tmp_path):
    from unittest.mock import patch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.clutter_settling import settle_clutter
    from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
    from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams

    arena_env = _make_primitive_clutter_scene(tmp_path)
    assets = arena_env.get_placement_assets()
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    try:
        env.reset()
        base = env.unwrapped
        for key in ("table", "floor"):
            body = base.scene.rigid_objects[key]
            original = body.data.root_pose_w.torch.clone()
            moved = original.clone()
            moved[1, 0] += 2.0
            body.write_root_pose_to_sim_index(root_pose=moved)
            base.scene.write_data_to_sim()
            base.sim.forward()
            before = base.scene.get_state()
            with patch(
                "isaaclab_arena.offline_placement.clutter_settling.collect_settled_placements",
                side_effect=AssertionError("must reject before solving or releasing"),
            ):
                with pytest.raises(AssertionError, match=f"{key!r} differs from its configured pose"):
                    settle_clutter(env, assets, 1, SettledPlacementParams())
            _assert_scene_state_equal(base.scene.get_state(), before)
            body.write_root_pose_to_sim_index(root_pose=original)
            base.scene.write_data_to_sim()
            base.sim.forward()
        # The same scene at its authored poses remains usable across both environments.
        results = settle_clutter(
            env, assets, 1, SettledPlacementParams(num_steps=480, validators=default_clutter_validators())
        )
        assert len(results.accepted_indices) == 2
        assert all(abs(pose.position_xyz[0]) < 0.4 for pose in results.poses["cube_body"])
    finally:
        env.close()
    return True


def test_settling_rejects_moved_fixed_assets_before_solving(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_settling_rejects_moved_fixed_assets_before_solving, tmp_path=tmp_path
    )


def _test_clutter_collection_uses_shared_batches(simulation_app, tmp_path):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.clutter_settling import settle_clutter
    from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
    from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams
    from isaaclab_arena.relations.placement_events import get_placement_pool
    from isaaclab_arena.utils.physics_settle import step_physics

    arena_env = _make_primitive_clutter_scene(tmp_path)
    arena_env.placer_params.min_unique_layouts_per_env = 2
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    try:
        base = env.unwrapped
        pool = get_placement_pool(base)
        params = SettledPlacementParams(num_steps=480, validators=default_clutter_validators())
        before = base.arena_world.get_pose_e("cube_body").clone()
        with patch.object(pool, "sample_for_envs", wraps=pool.sample_for_envs) as sample:
            result = settle_clutter(env, arena_env.get_placement_assets(), 2, params)
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
        short = SettledPlacementParams(num_steps=1, validators=default_clutter_validators())
        short.validators["physics_settled"]["lin_vel_thresh"] = 0.0001
        rejected = settle_clutter(env, arena_env.get_placement_assets(), 1, short)
        assert rejected.attempted == 2
        assert not rejected.accepted_indices
        assert all("physics_settled" in reason for reason in rejected.rejections.values())
    finally:
        env.close()
    return True


def test_clutter_collection_uses_shared_batches(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_clutter_collection_uses_shared_batches, tmp_path=tmp_path)
