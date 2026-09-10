# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline cache generation and ordinary runtime pose replay."""

from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

SOURCE = Path(__file__).parents[2] / "isaaclab_arena_examples/relations/clutter_scene.yaml"


def _test_offline_scene_round_trip(simulation_app):
    import tempfile
    import torch
    from argparse import Namespace
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_events import get_placement_pool
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena_examples.relations.generate_clutter_scene import generate_scene

    with (
        tempfile.TemporaryDirectory() as directory,
        patch.object(RelationSolver, "solve", side_effect=AssertionError("Offline clutter must not invoke the solver")),
    ):
        args = Namespace(
            env_spec=SOURCE,
            output=Path(directory) / "scene.yaml",
            num_envs=2,
            seed=42,
            device="cuda:0",
            presets=None,
            register=[],
            support="table",
            objects=[f"cube_{i}" for i in range(4)],
            spread=0.2,
            gap_m=0.03,
            clearance_m=0.01,
            keep_rotation=False,
            drop_order="as_listed",
            attempts=2,
            timeout_s=10.0,
            poll_interval_s=0.4,
            move_thresh_m=0.002,
            turn_thresh_deg=2.0,
            required_quiet_windows=2,
            fall_through_tolerance_m=0.01,
            containment_margin_m=0.0,
            passive_move_thresh_m=0.002,
            passive_turn_thresh_deg=2.0,
        )
        paths = generate_scene(args)
        assert len(paths) == 2
        specs = [ArenaEnvGraphSpec.from_yaml(path) for path in paths]
        assert all(not spec.relations for spec in specs)
        assert specs[0].objects[0].params["initial_pose"] != specs[1].objects[0].params["initial_pose"]
        spec = specs[0]
        env = ArenaEnvBuilder(spec.to_arena_env(), ArenaEnvBuilderCfg(num_envs=2)).make_registered()
        try:
            assert get_placement_pool(env) is None
            env.reset()
            robot = env.unwrapped.scene.articulations["robot"]
            robot_pose = robot.data.root_pose_w.torch.clone()
            for _ in range(8):
                moved_robot = robot_pose.clone()
                moved_robot[:, 0] += 0.25
                robot.write_root_pose_to_sim(moved_robot)
                for body in env.unwrapped.scene.rigid_objects.values():
                    displaced = body.data.root_pose_w.torch.clone()
                    displaced[:, 2] += 1.0
                    body.write_root_pose_to_sim(displaced)
                    body.write_root_velocity_to_sim(torch.ones_like(body.data.root_vel_w.torch))
                env.reset()
                torch.testing.assert_close(robot.data.root_pose_w.torch, robot_pose, atol=2e-5, rtol=0)
                for obj in spec.objects:
                    expected = obj.params["initial_pose"]
                    T_E_O = torch.tensor(
                        expected["position_xyz"] + expected["rotation_xyzw"], device=env.unwrapped.device
                    )
                    actual = env.unwrapped.arena_world.get_pose_e(obj.id)
                    torch.testing.assert_close(actual, T_E_O.expand_as(actual), atol=2e-5, rtol=0)
                before = torch.stack([env.unwrapped.arena_world.get_pose_e(obj.id)[:, :3] for obj in spec.objects])
                for _ in range(200):
                    env.unwrapped.scene.write_data_to_sim()
                    env.unwrapped.sim.step(render=False)
                    env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())
                after = torch.stack([env.unwrapped.arena_world.get_pose_e(obj.id)[:, :3] for obj in spec.objects])
                assert float((after - before).norm(dim=-1).max()) < 0.02
        finally:
            env.close()
    return True


def test_offline_scene_round_trip():
    assert run_function_with_persistent_simulation_app(_test_offline_scene_round_trip)


def _test_offline_settling_restores_scene_on_success_and_failure(simulation_app):
    import torch
    from unittest.mock import patch

    import pytest

    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import (
        build_arena_env_with_assets_from_graph_spec,
    )
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_examples.relations.clutter.settle import ClutterGroup, settle_clutter
    from isaaclab_arena_examples.relations.clutter.validation import ClutterRestVerdict

    arena_env, assets = build_arena_env_with_assets_from_graph_spec(ArenaEnvGraphSpec.from_yaml(SOURCE))
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    try:
        env.reset()
        initial = env.unwrapped.scene.get_state()
        group = ClutterGroup(assets["table"].get_scene_key(), tuple(f"cube_{i}" for i in range(4)), spread=0.2)
        for fail in (False, True):
            if fail:
                with patch(
                    "isaaclab_arena_examples.relations.clutter.settle.check_resting_poses",
                    return_value=ClutterRestVerdict(fell_off=[0]),
                ):
                    with pytest.raises(AssertionError, match="fell off: cube_0"):
                        settle_clutter(env, [group], attempts=1)
            else:
                layouts = settle_clutter(env, [group], attempts=2)
                assert len(layouts) == 2
                assert all(set(layout) == set(group.objects) for layout in layouts)
            restored = env.unwrapped.scene.get_state()
            for kind, assets_state in initial.items():
                for name, state in assets_state.items():
                    for field, value in state.items():
                        torch.testing.assert_close(restored[kind][name][field], value, atol=1e-6, rtol=0)
        from isaaclab_arena_examples.relations.clutter.settle import _release_objects

        scene = env.unwrapped.scene
        for key in group.objects:
            T_E_O = torch.stack([layout[key].to_tensor(device=env.unwrapped.device) for layout in layouts])
            T_W_O = T_E_O.clone()
            T_W_O[:, :3] += scene.env_origins
            scene.rigid_objects[key].write_root_pose_to_sim(T_W_O)
            scene.rigid_objects[key].write_root_velocity_to_sim(
                torch.zeros_like(scene.rigid_objects[key].data.root_vel_w.torch)
            )
        for _ in range(120):
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(env.unwrapped.sim.get_physics_dt())

        def displace_neighbor(env, env_id, *args):
            _release_objects(env, env_id, *args)
            neighbor = env.scene.rigid_objects["cube_3"]
            pose = neighbor.data.root_pose_w.torch[env_id : env_id + 1].clone()
            pose[:, 0] += 0.02
            neighbor.write_root_pose_to_sim(pose, env_ids=torch.tensor([env_id], device=env.device))

        group = ClutterGroup(group.support, group.objects[:3], spread=0.2)
        with patch("isaaclab_arena_examples.relations.clutter.settle._release_objects", side_effect=displace_neighbor):
            with pytest.raises(AssertionError, match="cube_3: passive drift") as failure:
                settle_clutter(env, [group], attempts=1)
            assert "still moving" not in str(failure.value)
    finally:
        env.close()
    return True


def test_offline_settling_restores_scene_on_success_and_failure():
    assert run_function_with_persistent_simulation_app(_test_offline_settling_restores_scene_on_success_and_failure)


def test_initial_pose_validation_and_reset_contract():
    import pytest

    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import _apply_initial_pose

    class Asset:
        def get_initial_pose(self):
            return None

        def set_initial_pose(self, pose, create_reset_event=True):
            self.pose = pose
            self.reset = create_reset_event

    asset = Asset()
    _apply_initial_pose(asset, {"position_xyz": [1, 2, 3], "rotation_xyzw": [0, 0, 0, 1]})
    assert asset.pose.position_xyz == (1.0, 2.0, 3.0)
    assert asset.reset
    for bad in (
        {"position_xyz": [float("nan"), 0, 0]},
        {"position_xyz": [True, 0, 0]},
        {"rotation_xyzw": [0, 0, 0, 0]},
        {"rotation_xyzw": [0, 0, 1]},
        {"unknown": 1},
    ):
        with pytest.raises(AssertionError):
            _apply_initial_pose(asset, bad)


def test_cache_uses_scene_keys_when_graph_names_differ():
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_examples.relations.clutter.cache import scene_with_cached_poses

    spec = ArenaEnvGraphSpec.from_yaml(SOURCE)
    nodes = [spec.background, spec.embodiment, *spec.objects]
    mapping = {
        node.id: DummyObject(f"runtime_{i}", AxisAlignedBoundingBox(min_point=(0, 0, 0), max_point=(1, 1, 1)))
        for i, node in enumerate(nodes)
    }
    poses = {asset.get_scene_key(): Pose((float(i), 0.0, 0.5)) for i, asset in enumerate(mapping.values())}
    result = scene_with_cached_poses(spec, poses, mapping)
    for node in [result.background, result.embodiment, *result.objects]:
        assert node.params["initial_pose"]["position_xyz"] == list(poses[mapping[node.id].get_scene_key()].position_xyz)
    assert all("initial_pose" not in node.params for node in spec.objects)


def test_cache_rejects_poses_the_runtime_loader_cannot_read():
    import pytest

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_examples.relations.clutter.cache import scene_with_cached_poses

    spec = ArenaEnvGraphSpec.from_yaml(SOURCE)
    mapping = {
        node.id: DummyObject(node.id, AxisAlignedBoundingBox(min_point=(0, 0, 0), max_point=(1, 1, 1)))
        for node in [spec.background, spec.embodiment, *spec.objects]
    }
    for pose, message in (
        (Pose((float("nan"), 0, 0)), "must be finite"),
        (Pose(rotation_xyzw=(0, 0, 0, 0)), "must be a unit quaternion"),
        (Pose(rotation_xyzw=(0, 0, 0, 2)), "must be a unit quaternion"),
    ):
        with pytest.raises(AssertionError, match=message):
            scene_with_cached_poses(spec, {"cube_0": pose}, mapping)


def test_partial_initial_pose_preserves_authored_components():
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import _apply_initial_pose
    from isaaclab_arena.utils.pose import Pose

    class Asset:
        def __init__(self):
            self.pose = Pose((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))

        def get_initial_pose(self):
            return self.pose

        def set_initial_pose(self, pose, create_reset_event=True):
            self.pose = pose
            self.reset = create_reset_event

    asset = Asset()
    _apply_initial_pose(asset, {"position_xyz": [4, 5, 6]})
    assert asset.pose.rotation_xyzw == (1.0, 0.0, 0.0, 0.0)
    _apply_initial_pose(asset, {"rotation_xyzw": [0, 0, 0, 1]})
    assert asset.pose.position_xyz == (4.0, 5.0, 6.0)
    assert asset.reset


def test_cache_preserves_scene_configuration():
    from unittest.mock import Mock

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environment_spec.arena_env_graph_types import CliOverrideSpec, PlacementValidatorSpec
    from isaaclab_arena_examples.relations.clutter.cache import scene_with_cached_poses

    spec = ArenaEnvGraphSpec.from_yaml(SOURCE)
    spec.cli_override_specs = [CliOverrideSpec(arg="object", target_node_id="cube_0")]
    spec.placement_validators = PlacementValidatorSpec(enabled_checks=["collision"])
    mapping = {node.id: Mock() for node in [spec.background, spec.embodiment, *spec.objects]}
    for key, asset in mapping.items():
        asset.get_scene_key.return_value = key
    result = scene_with_cached_poses(spec, {}, mapping)
    assert result.cli_override_specs == spec.cli_override_specs
    assert result.placement_validators == spec.placement_validators


def test_cache_write_failure_leaves_no_partial_output(tmp_path):
    from unittest.mock import Mock

    import pytest

    from isaaclab_arena_examples.relations.clutter.cache import write_scene_cache

    def fail_write(path):
        path.write_text("partial YAML")
        raise OSError("disk full")

    target = tmp_path / "scene.yaml"
    with pytest.raises(OSError, match="disk full"):
        write_scene_cache(Mock(write_yaml=fail_write), target)
    assert list(tmp_path.iterdir()) == []


def test_cache_write_does_not_replace_existing_output(tmp_path):
    import pytest

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena_examples.relations.clutter.cache import write_scene_cache

    target = tmp_path / "scene.yaml"
    spec = ArenaEnvGraphSpec.from_yaml(SOURCE)
    write_scene_cache(spec, target)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_scene_cache(spec, target)
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_no_task_accepts_base_constructor_parameters():
    from isaaclab_arena.tasks.no_task import NoTask

    task = NoTask(episode_length_s=12, task_description="inspect clutter")
    assert task.episode_length_s == 12
    assert task.task_description == "inspect clutter"


def _test_offline_generation_rejects_invalid_inputs(simulation_app):
    import errno
    import tempfile
    import yaml
    from argparse import Namespace
    from unittest.mock import patch

    import pytest

    from isaaclab_arena_examples.relations.generate_clutter_scene import generate_scene

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "input.yaml"
        output = Path(directory) / "output.yaml"
        cases = [
            (
                {"relations": [{"kind": "on", "subject": "cube_0", "reference": "table"}]},
                "table",
                ["cube_0"],
                "resolve placement relations",
            ),
            ({"object_sets": [{"id": "extras", "members": ["dex_cube"]}]}, "table", ["cube_0"], "Resolve object sets"),
            ({}, "missing", ["cube_0"], "Unknown support node"),
            ({}, "table", ["missing"], "Unknown clutter nodes"),
        ]
        for fields, support, objects, message in cases:
            data = yaml.safe_load(SOURCE.read_text())
            data.update(fields)
            source.write_text(yaml.safe_dump(data))
            args = Namespace(register=[], env_spec=source, output=output, num_envs=1, support=support, objects=objects)
            with pytest.raises(AssertionError, match=message):
                generate_scene(args)
            assert not output.exists()

        args = Namespace(register=[], env_spec=SOURCE, output=output, num_envs=1, support="table", objects=["cube_0"])
        with (
            patch(
                "isaaclab_arena_examples.relations.clutter.cache.os.link",
                side_effect=OSError(errno.EOPNOTSUPP, "Operation not supported"),
            ),
            patch(
                "isaaclab_arena.environment_spec.arena_env_graph_conversion_utils.build_arena_env_with_assets_from_graph_spec"
            ) as build,
        ):
            with pytest.raises(OSError, match="requires hard links"):
                generate_scene(args)
            build.assert_not_called()
        assert not output.exists()
    return True


def test_offline_generation_rejects_invalid_inputs():
    assert run_function_with_persistent_simulation_app(_test_offline_generation_rejects_invalid_inputs)


def _test_offline_retry_preserves_accepted_environments(simulation_app):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_examples.relations.clutter.settle import ClutterGroup, _release_objects, settle_clutter
    from isaaclab_arena_examples.relations.clutter.validation import ClutterRestVerdict, check_resting_poses

    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    checked_positions = []

    def reject_second_environment_once(positions, *args):
        checked_positions.append(positions.clone())
        if len(checked_positions) == 2:
            return ClutterRestVerdict(fell_off=[0])
        return check_resting_poses(positions, *args)

    try:
        env.reset()
        group = ClutterGroup("office_table_background", tuple(f"cube_{i}" for i in range(4)), spread=0.2)
        with (
            patch(
                "isaaclab_arena_examples.relations.clutter.settle._release_objects", wraps=_release_objects
            ) as release,
            patch(
                "isaaclab_arena_examples.relations.clutter.settle.check_resting_poses",
                side_effect=reject_second_environment_once,
            ),
        ):
            layouts = settle_clutter(env, [group], attempts=2)
        assert [call.args[1] for call in release.call_args_list] == [0, 1, 1]
        assert len(checked_positions) == 3
        for env_id, check_index in ((0, 0), (1, 2)):
            positions = torch.tensor([layouts[env_id][key].position_xyz for key in group.objects])
            torch.testing.assert_close(positions, checked_positions[check_index], atol=0, rtol=0)
    finally:
        env.close()
    return True


def test_offline_retry_preserves_accepted_environments():
    assert run_function_with_persistent_simulation_app(_test_offline_retry_preserves_accepted_environments)


def test_cache_directory_requires_hard_links(tmp_path):
    import errno
    from unittest.mock import patch

    import pytest

    from isaaclab_arena_examples.relations.clutter.cache import validate_cache_directory

    validate_cache_directory(tmp_path)
    assert list(tmp_path.iterdir()) == []
    with patch(
        "isaaclab_arena_examples.relations.clutter.cache.os.link",
        side_effect=OSError(errno.EOPNOTSUPP, "Operation not supported"),
    ):
        with pytest.raises(OSError, match="requires hard links") as failure:
            validate_cache_directory(tmp_path)
    assert failure.value.errno == errno.EOPNOTSUPP
    assert failure.value.filename == str(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_cache_link_failure_leaves_no_output(tmp_path):
    import errno
    from unittest.mock import patch

    import pytest

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena_examples.relations.clutter.cache import write_scene_cache

    spec = ArenaEnvGraphSpec.from_yaml(SOURCE)
    with patch(
        "isaaclab_arena_examples.relations.clutter.cache.os.link",
        side_effect=OSError(errno.EOPNOTSUPP, "Operation not supported"),
    ):
        with pytest.raises(OSError):
            write_scene_cache(spec, tmp_path / "scene.yaml")
    assert list(tmp_path.iterdir()) == []
