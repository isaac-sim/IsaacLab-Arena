# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Offline ClutterOn settling and pose-file generation."""

import json
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

SOURCE = Path(__file__).parent / "data/clutter_cubes.yaml"


def _arguments(output):
    from isaaclab_arena.offline_placement.clutter_generation import ClutterGenerationCfg

    return ClutterGenerationCfg(env_spec=str(SOURCE), output=str(output), num_envs=2, attempts=3)


def _assert_scene_state_equal(actual, expected):
    import torch

    for kind, states in expected.items():
        for name, state in states.items():
            for field, value in state.items():
                torch.testing.assert_close(actual[kind][name][field], value, atol=1e-6, rtol=0)


def _test_generation_writes_complete_layouts(simulation_app, tmp_path):
    import torch
    import yaml
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.clutter_generation import generate_clutter_layouts
    from isaaclab_arena.offline_placement.settle import settle_clutter
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.utils.pose import Pose

    data = yaml.safe_load(SOURCE.read_text())
    data["objects"][0]["params"] = {"instance_name": "first_cube"}
    data["placer_params"] = {
        "enabled_checks": ["no_overlap", "on_relation"],
        "required_checks": ["no_overlap", "on_relation"],
    }
    data["external_yaml"] = "physics.yaml"
    physics = tmp_path / "physics.yaml"
    physics.write_text(
        yaml.safe_dump({
            "default_physics_backend": "physx",
            "env_cfg_override": {"sim": {"dt": 0.01}},
        })
    )
    source = tmp_path / "scene.yaml"
    source.write_text(yaml.safe_dump(data))
    original = source.read_bytes()
    path = tmp_path / "episodes.jsonl"
    args = _arguments(path)
    args.env_spec = str(source)
    args.num_layouts = 4

    def settle_with_graph_settings(env, *args, **kwargs):
        assert env.unwrapped.sim.get_physics_dt() == pytest.approx(0.01)
        return settle_clutter(env, *args, **kwargs)

    with patch("isaaclab_arena.offline_placement.settle.settle_clutter", wraps=settle_with_graph_settings):
        assert generate_clutter_layouts(ArenaEnvGraphSpec.from_yaml(args.env_spec).to_arena_env(), args) == path
    assert source.read_bytes() == original
    records = [json.loads(line)["variations"]["scene.relation_placement"] for line in path.read_text().splitlines()]
    assert [record["layout_id"] for record in records] == [f"layout_{i:06d}" for i in range(4)]
    assert records[0]["poses"]["first_cube"] != records[1]["poses"]["first_cube"]
    for record in records:
        assert record["source"] == "settled"
        reports = record["validation"]["post_physics"]
        assert {report["check"] for report in reports} == {"rest", "support_containment", "passive_drift"}
        assert all(report["passed"] is True for report in reports)
        rest_report = next(report for report in reports if report["check"] == "rest")
        assert rest_report["configuration"]["move_thresh_m"] == 0.002
        assert record["validation"]["sampling"]["physics_dt_s"] == pytest.approx(0.01)
        assert set(record["poses"]) == {"first_cube", "cube_1", "cube_2", "cube_3"}
        for value in record["poses"].values():
            Pose.from_dict(value)
    layouts = PlacementLayouts.from_episode_jsonl(path)
    assert layouts.num_layouts == 4
    replay_env = ArenaEnvGraphSpec.from_yaml(args.env_spec).to_arena_env()
    with patch.object(RelationSolver, "solve", side_effect=AssertionError("Replay must not solve")):
        env = ArenaEnvBuilder(
            replay_env, ArenaEnvBuilderCfg(num_envs=3, placement_layouts_path=str(path))
        ).make_registered()
        try:
            env.reset()
            for name, poses in layouts.poses.items():
                expected = torch.stack([pose.to_tensor(env.unwrapped.device) for pose in poses[:3]])
                torch.testing.assert_close(env.unwrapped.arena_world.get_pose_e(name), expected, atol=2e-5, rtol=0)
        finally:
            env.close()
    replay_env.placement_layouts = layouts
    args.output = str(tmp_path / "regenerated.jsonl")
    with pytest.raises(AssertionError, match="Remove cached placement layouts"):
        generate_clutter_layouts(replay_env, args)
    assert not Path(args.output).exists()
    args.output = str(path)
    saved = path.read_bytes()
    with pytest.raises(AssertionError, match="Output already exists"):
        generate_clutter_layouts(ArenaEnvGraphSpec.from_yaml(args.env_spec).to_arena_env(), args)
    assert path.read_bytes() == saved
    return True


def test_generation_writes_complete_layouts(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_generation_writes_complete_layouts, tmp_path=tmp_path)


def _test_uncached_clutter_drops_at_simulation_start(simulation_app):
    import torch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.relations import RotateAroundSolution
    from isaaclab_arena.utils.pose import Pose

    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    marker = RotateAroundSolution(roll_rad=0.4, pitch_rad=0.3, yaw_rad=0.7)
    arena_env.scene.assets["cube_0"].add_relation(marker)
    arena_env.placer_params.min_unique_layouts_per_env = 1
    arena_env.placer_params.max_placement_attempts = 1
    arena_env.placer_params.resolve_on_reset = False
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, placement_seed=42)).make_registered()
    try:
        env.reset()
        pose = env.unwrapped.arena_world.get_pose_e("cube_0")[0].cpu()
        rotation = Pose(rotation_xyzw=tuple(pose[3:].tolist())).to_transform_matrix("cpu")[:3, :3]
        base = Pose(rotation_xyzw=marker.get_rotation_xyzw()).to_transform_matrix("cpu")[:3, :3]
        # Random world-Z yaw must preserve the authored tilt.
        torch.testing.assert_close((rotation @ base.T)[2], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=0)
        for _ in range(200):
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())
        after = env.unwrapped.arena_world.get_pose_e("cube_0")[0, 2]
        assert float(pose[2] - after.cpu()) > 0.005
    finally:
        env.close()
    return True


def test_uncached_clutter_drops_at_simulation_start():
    assert run_function_with_persistent_simulation_app(_test_uncached_clutter_drops_at_simulation_start)


def _test_settling_restores_scene_and_retries_only_rejected_layouts(simulation_app):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.settle import _release_objects, settle_clutter
    from isaaclab_arena.offline_placement.validators import (
        PassiveDriftValidator,
        RestValidator,
        SupportContainmentValidator,
    )
    from isaaclab_arena.utils.pose import Pose

    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    assets = arena_env.get_placement_assets()
    arena_env.scene.assets["office_table_background"].set_initial_pose(
        Pose((1.0, 0.0, 0.0), (0.0, 0.0, 2**-0.5, 2**-0.5))
    )
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    checked = []

    class RejectSecondContainment(SupportContainmentValidator):
        def validate(self, data):
            checked.append(data.poses)
            return self.report("fell off: cube_0") if len(checked) == 2 else super().validate(data)

    containment = RejectSecondContainment()
    validators = [RestValidator(), containment, PassiveDriftValidator()]

    try:
        env.reset()
        initial = env.unwrapped.scene.get_state()
        with patch("isaaclab_arena.offline_placement.settle._release_objects", wraps=_release_objects) as release:
            results = settle_clutter(env, assets, attempts=3, validators=validators)
            layouts = [result.poses for result in results]
        # Only the rejected environment receives a second release.
        assert [call.args[1] for call in release.call_args_list] == [0, 1, 1]
        assert layouts == [checked[0], checked[2]]
        _assert_scene_state_equal(env.unwrapped.scene.get_state(), initial)
        with patch.object(containment, "validate", return_value=containment.report("fell off: cube_0")):
            with pytest.raises(AssertionError, match="fell off: cube_0"):
                settle_clutter(env, assets, attempts=1, validators=validators)
        _assert_scene_state_equal(env.unwrapped.scene.get_state(), initial)
        world = env.unwrapped.arena_world
        by_scene_key = {asset.get_scene_key(): asset for asset in assets}
        for env_id, layout in enumerate(layouts):
            for name, pose in layout.items():
                by_scene_key[name].write_layout_pose_to_sim(env.unwrapped, env_id, pose)
        before = {name: world.get_pose_e(name).clone() for name in layouts[0]}
        for _ in range(200):
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())
        for name, pose in before.items():
            torch.testing.assert_close(world.get_pose_e(name), pose, atol=0.005, rtol=0)
    finally:
        env.close()
    return True


def test_settling_restores_scene_and_retries_only_rejected_layouts():
    assert run_function_with_persistent_simulation_app(_test_settling_restores_scene_and_retries_only_rejected_layouts)


def _test_generation_rejects_invalid_scene(simulation_app, failure, expected_error):
    import tempfile
    import yaml

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.offline_placement.clutter_generation import generate_clutter_layouts

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source.yaml"
        data = yaml.safe_load(SOURCE.read_text())
        if failure == "passive":
            data["relations"] = [relation for relation in data["relations"] if relation["subject"] != "cube_3"]
            data["objects"][3]["params"] = {"initial_pose": {"position_xyz": [0.5, 0.0, 3.0]}}
        else:
            for relation in data["relations"]:
                if relation["kind"] == "clutter_on":
                    relation["params"] = {"spread": 0.001}
        source.write_text(yaml.safe_dump(data))
        output = Path(directory) / "episodes.jsonl"
        args = _arguments(output)
        args.env_spec, args.num_envs, args.attempts = str(source), 1, 1
        with pytest.raises(AssertionError, match=expected_error):
            generate_clutter_layouts(ArenaEnvGraphSpec.from_yaml(args.env_spec).to_arena_env(), args)
        assert not output.exists()
    return True


@pytest.mark.parametrize(
    "failure, expected_error",
    [
        ("passive", "cube_3: passive drift"),
        ("release", "release placement failed"),
    ],
)
def test_generation_rejects_invalid_scene(failure, expected_error):
    assert run_function_with_persistent_simulation_app(
        _test_generation_rejects_invalid_scene, failure=failure, expected_error=expected_error
    )


def _test_generation_honors_requested_validators(simulation_app, tmp_path, available, expected_error):
    import yaml
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.offline_placement.clutter_generation import generate_clutter_layouts
    from isaaclab_arena.offline_placement.settle import _release_objects
    from isaaclab_arena.relations.placement_validation import PlacementCheck
    from isaaclab_arena.relations.placement_validator_registry import PlacementValidatorRegistry
    from isaaclab_arena.relations.placement_validators import PrePhysicsPlacementValidator

    validated_batches = []

    class RejectRelease(PrePhysicsPlacementValidator):
        check = "reject_release"

        def validate_batch(self, positions, orientations, bboxes, collision_objects):
            validated_batches.append(len(positions))
            return [False] * len(positions)

    class SettledPoseCheck(PrePhysicsPlacementValidator):
        check = PlacementCheck.IK_REACHABLE

        def validate_batch(self, positions, orientations, bboxes, collision_objects):
            pytest.fail("Release validation must not run settled-pose checks")

    data = yaml.safe_load(SOURCE.read_text())
    if not available:
        data["placer_params"] = {
            "enabled_checks": [RejectRelease.check],
            "required_checks": [RejectRelease.check],
        }
    source = tmp_path / "scene.yaml"
    source.write_text(yaml.safe_dump(data))
    output = tmp_path / "episodes.jsonl"
    args = _arguments(output)
    args.env_spec = str(source)
    args.num_envs = 1
    args.attempts = 1
    registry = PlacementValidatorRegistry()
    with (
        patch.dict(registry._components, reject_release=RejectRelease, ik_reachable=SettledPoseCheck),
        patch.object(RejectRelease, "is_available", return_value=available),
        patch("isaaclab_arena.offline_placement.settle._release_objects", wraps=_release_objects) as release,
    ):
        with pytest.raises(AssertionError, match=expected_error):
            generate_clutter_layouts(ArenaEnvGraphSpec.from_yaml(args.env_spec).to_arena_env(), args)
        release.assert_not_called()
        assert not output.exists()
    assert bool(validated_batches) == available
    return True


@pytest.mark.parametrize(
    "available, expected_error",
    [
        pytest.param(True, "release placement failed", id="default_checks"),
        pytest.param(False, "validators did not run", id="unavailable"),
    ],
)
def test_generation_honors_requested_validators(tmp_path, available, expected_error):
    assert run_function_with_persistent_simulation_app(
        _test_generation_honors_requested_validators,
        tmp_path=tmp_path,
        available=available,
        expected_error=expected_error,
    )


def _test_generation_from_python_environment(simulation_app, tmp_path):
    from isaaclab_arena.assets.background_library import OfficeTableBackground
    from isaaclab_arena.assets.object_library import DexCube, DomeLight
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.offline_placement.clutter_generation import ClutterGenerationCfg, generate_clutter_layouts
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    table = OfficeTableBackground()
    table.set_initial_pose(Pose.identity())
    table.add_relation(IsAnchor())
    cube = DexCube(instance_name="python_cube")
    cube.add_relation(ClutterOn(table))
    arena_env = IsaacLabArenaEnvironment(name="python_clutter", scene=Scene(assets=[table, cube, DomeLight()]))
    output = tmp_path / "python.jsonl"
    generate_clutter_layouts(arena_env, ClutterGenerationCfg(output=str(output)))
    record = json.loads(output.read_text())["variations"]["scene.relation_placement"]
    assert set(record["poses"]) == {"python_cube"}
    assert record["poses"]["python_cube"]["position_xyz"][2] > 0
    return True


def test_generation_from_python_environment(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_generation_from_python_environment, tmp_path=tmp_path)


def _test_post_physics_checks_gate_output(simulation_app, tmp_path):
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.offline_placement.clutter_generation import generate_clutter_layouts

    output = tmp_path / "checked.jsonl"
    cfg = _arguments(output)
    cfg.num_envs = 1
    cfg.attempts = 1
    cfg.post_physics["reject_for_test"] = {
        "_target_": "isaaclab_arena.tests.clutter.test_clutter_cli.RejectPostPhysics",
        "threshold": 7.0,
    }
    with pytest.raises(AssertionError, match="reject_for_test: deliberately rejected"):
        generate_clutter_layouts(ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env(), cfg)
    assert not output.exists()
    cfg.post_physics["reject_for_test"]["enabled"] = False
    cfg.post_physics["rest"]["enabled"] = False
    cfg.settle.timeout_s = 2.0
    generate_clutter_layouts(ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env(), cfg)
    record = json.loads(output.read_text())["variations"]["scene.relation_placement"]
    reports = {report["check"]: report for report in record["validation"]["post_physics"]}
    assert reports["reject_for_test"]["passed"] is None
    assert reports["reject_for_test"]["reason"] == "disabled by configuration"
    assert reports["reject_for_test"]["configuration"]["threshold"] == 7.0
    assert record["source"] == "physics"
    assert reports["rest"]["passed"] is None
    assert all(reports[name]["passed"] for name in ("support_containment", "passive_drift"))
    return True


def test_post_physics_checks_gate_output(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_post_physics_checks_gate_output, tmp_path=tmp_path)
