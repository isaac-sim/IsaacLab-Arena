# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ordinary placements are recorded after physics, then replayed without solving."""

from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.tests.utils.subprocess import run_subprocess


def register_no_embodiment():
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment

    AssetRegistry().register(NoEmbodiment, key="recording_no_embodiment")


def run_cli_with_test_assets():
    from unittest.mock import patch

    from isaaclab_arena.scripts.record_placement_layouts import main
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    enter = SimulationAppContext.__enter__

    def enter_with_test_assets(context):
        app = enter(context)
        register_no_embodiment()
        return app

    with patch.object(SimulationAppContext, "__enter__", enter_with_test_assets):
        main()


def _write_scene(path: Path) -> None:
    import yaml

    for name, kinematic, size in (
        ("table", "true", (0.8, 0.8, 0.04)),
        ("cube", "false", (0.05, 0.1, 0.1)),
        ("floor", "true", (4.0, 4.0, 0.04)),
    ):
        (path.parent / f"{name}.usda").write_text(f"""#usda 1.0
(
    defaultPrim = "Body"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {{
    bool physics:kinematicEnabled = {kinematic}
    float physics:mass = 1
    def Cube "geometry" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{
        double size = 1
        double3 xformOp:scale = {size}
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }}
}}
""")
    data = {
        "env_name": "placement_recording",
        "embodiment": {"id": "robot", "registry_name": "recording_no_embodiment"},
        "background": {
            "id": "table",
            "registry_name": "simready_usd_object",
            "params": {
                "usd_path": str(path.parent / "table.usda"),
                "instance_name": "table",
                "initial_pose": {"position_xyz": [0, 0, 0.5]},
            },
        },
        "objects": [
            {
                "id": "cube",
                "registry_name": "simready_usd_object",
                "params": {"usd_path": str(path.parent / "cube.usda"), "instance_name": "cube_body"},
            },
            {
                "id": "floor",
                "registry_name": "simready_usd_object",
                "params": {
                    "usd_path": str(path.parent / "floor.usda"),
                    "instance_name": "floor",
                    "initial_pose": {"position_xyz": [0, 0, -0.5]},
                },
            },
        ],
        "relations": [
            {"kind": "is_anchor", "subject": "table"},
            {"kind": "on", "subject": "cube", "reference": "table", "params": {"clearance_m": 0.001}},
        ],
        "task": {
            "composition": "atomic",
            "description": "record settled placements",
            "subtasks": [{"kind": "NoTask", "params": {}}],
        },
    }
    path.write_text(yaml.safe_dump(data))


@pytest.mark.with_subprocess
@pytest.mark.parametrize("backend", ["physx", "newton"])
def test_recording_cli_saves_final_poses(tmp_path, backend):
    import json

    source, output = tmp_path / "scene.yaml", tmp_path / "placements.jsonl"
    _write_scene(source)
    completed = run_subprocess(
        [
            TestConstants.python_path,
            "-c",
            (
                "from isaaclab_arena.tests.test_settled_placement import run_cli_with_test_assets;"
                " run_cli_with_test_assets()"
            ),
            f"env_spec={source}",
            f"output={output}",
            f"presets={backend}",
            "num_envs=2",
            "layouts_per_env=2",
            "settle.num_steps=120",
            "settle.validators.pose_shift.max_translation_m=0.0015",
            "--viz",
            "none",
        ],
        timeout_sec=180,
        capture_output=True,
    )
    assert "physics_settled: ENABLED" in completed.stdout
    assert "articulation_link_shift: SKIPPED: scene has no articulations" in completed.stdout
    records = [json.loads(line)["variations"]["scene.relation_placement"] for line in output.read_text().splitlines()]
    assert len(records) == 4
    for record in records:
        assert record["source"] == "settled"
        reports = {report["check"]: report for report in record["validation"]["post_physics"]}
        assert reports["physics_settled"]["passed"] is True
        assert reports["pose_shift"]["passed"] is True
        assert reports["pose_shift"]["configuration"]["max_translation_m"] == 0.0015
        assert reports["articulation_link_shift"]["passed"] is None
        assert reports["articulation_link_shift"]["reason"] == "scene has no articulations"
        assert set(record["poses"]) == {"cube_body", "table", "floor"}
        # Table top is 0.52, cube half-height is 0.05.
        x, y, _ = record["poses"]["cube_body"]["position_xyz"]
        assert abs(x) < 0.4 and abs(y) < 0.4
        assert record["poses"]["cube_body"]["position_xyz"][2] == pytest.approx(0.57, abs=0.005)

    positions = {tuple(record["poses"]["cube_body"]["position_xyz"]) for record in records}
    assert len(positions) == len(records)


def test_recording_cli_imports_before_simulation_startup():
    import subprocess

    script = Path(TestConstants.scripts_dir) / "record_placement_layouts.py"
    result = subprocess.run(
        [
            TestConstants.python_path,
            "-c",
            (
                "import runpy, sys; runpy.run_path(sys.argv[1]); "
                "assert 'numpy' not in sys.modules, 'Numerical libraries imported before SimulationApp startup'"
            ),
            str(script),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _test_recording_filters_layouts(simulation_app, tmp_path):
    import torch
    import yaml
    from copy import deepcopy
    from dataclasses import replace
    from unittest.mock import Mock, patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_pool_layouts
    from isaaclab_arena.relations.placement_events import get_placement_pool, make_cached_placement_event
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.placement_validation import PlacementCheck
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.reachability_config import ReachabilityConfig
    from isaaclab_arena.utils.pose import Pose, PoseRange
    from isaaclab_arena.utils.velocity import Velocity

    register_no_embodiment()
    source = tmp_path / "scene.yaml"
    _write_scene(source)
    data = yaml.safe_load(source.read_text())
    data["relations"][1]["params"].update(clearance_m=0.001, overlap=True)
    source.write_text(yaml.safe_dump(data))
    spec = ArenaEnvGraphSpec.from_yaml(source)
    arena_env = spec.to_arena_env()
    arena_env.placer_params = replace(
        arena_env.placer_params,
        min_unique_layouts_per_env=1,
        placement_seed=42,
        random_yaw_init=False,
        enabled_checks={PlacementCheck.NO_OVERLAP},
        required_checks={PlacementCheck.NO_OVERLAP},
    )
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    try:
        env.reset()
        base = env.unwrapped
        pool = get_placement_pool(env)
        assets = arena_env.get_placement_assets()
        floor = arena_env.scene.assets["floor"]
        assert floor.has_pose_reset_event()
        state = base.scene.get_state()
        layouts = PlacementLayouts({key: [Pose.identity()] for key in base.scene.rigid_objects})
        for attribute, value, reason in (
            ("reset_pose", False, "pose resets disabled"),
            ("initial_velocity", Velocity(linear_xyz=(1.0, 0.0, 0.0)), "nonzero initial velocity"),
            ("initial_pose", PoseRange(), "non-fixed pose-reset policy"),
        ):
            with (
                patch.object(floor, attribute, value),
                patch(
                    "isaaclab_arena.offline_placement.settled_placement.iter_pool_validation",
                    side_effect=AssertionError("Incompatible reset policies must fail before physics"),
                ) as simulate,
            ):
                with pytest.raises(AssertionError, match=f"floor.*{reason}"):
                    collect_settled_pool_layouts(env, pool, scene_assets=assets)
                simulate.assert_not_called()
                with pytest.raises(AssertionError, match=f"floor.*{reason}"):
                    make_cached_placement_event(layouts, assets, base.num_envs)
            torch.testing.assert_close(base.scene.get_state(), state)

        queues = pool.layouts_per_env()
        cube = arena_env.scene.assets["cube_body"]
        initial = base.arena_world.get_pose_e("cube_body").clone()
        # Both pass On(overlap=True); only the overhanging cube falls to the floor.
        queues[0][0].positions[cube] = (0.0, 0.0, 0.571)
        queues[1][0].positions[cube] = (0.42, 0.0, 0.571)
        from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes
        from isaaclab_arena.relations.placement_validators import OnRelationValidator, PlacementValidator

        boxes = build_per_env_bounding_boxes(pool.objects, 2).get_bounding_boxes_for_all_envs()
        validator = OnRelationValidator(arena_env.placer_params)
        assert validator.validate_batch([queue[0].positions for queue in queues], [{}, {}], boxes, []) == [True, True]
        # An unequal queue ends with a solver failure that must not be simulated or recorded.
        failed = replace(queues[1][0], validation_results=deepcopy(queues[1][0].validation_results))
        failed.validation_results.validation_results[PlacementCheck.NO_OVERLAP] = False
        pool._env_pools[1].append(failed)
        ik_validator = Mock(spec=PlacementValidator)
        ik_validator.check = PlacementCheck.IK_REACHABLE
        pool._placer._validators.append(ik_validator)
        saved_positions = [dict(queue[0].positions) for queue in queues]
        saved_checks = [deepcopy(layout.validation_results) for queue in pool.layouts_per_env() for layout in queue]
        result = collect_settled_pool_layouts(
            env,
            pool,
            PlacementRecordingParams(num_steps=120),
            scene_assets=arena_env.get_placement_assets(),
        )
        ik_validator.validate_batch.assert_not_called()
        layouts = result.layouts
        assert result.attempted == 3
        assert result.accepted_indices == [(0, 0)]
        assert "cube_body: moved" in result.rejections[1, 0]
        assert result.rejections[1, 1] == "solver validation failed"
        assert layouts.num_layouts == 1
        assert layouts.poses["cube_body"][0].position_xyz[2] == pytest.approx(0.57, abs=0.005)
        torch.testing.assert_close(base.arena_world.get_pose_e("cube_body"), initial)
        assert [queue[0].positions for queue in queues] == saved_positions
        assert [layout.validation_results for queue in pool.layouts_per_env() for layout in queue] == saved_checks
        queues[0][0].positions[cube] = (0.0, 0.0, 0.575)
        # A 5 mm drop exceeds the shift limit even though the centered cube comes to rest.
        with pytest.raises(AssertionError, match=r"cube_body: moved .*limits 0.002 m"):
            collect_settled_pool_layouts(
                env, pool, PlacementRecordingParams(num_steps=120), scene_assets=arena_env.get_placement_assets()
            )
        torch.testing.assert_close(base.arena_world.get_pose_e("cube_body"), initial)
        params = PlacementRecordingParams(num_steps=120)
        params.validators["pose_shift"]["enabled"] = False
        relaxed = collect_settled_pool_layouts(env, pool, params, scene_assets=arena_env.get_placement_assets())
        assert (0, 0) in relaxed.accepted_indices
        report = next(report for report in relaxed.validation[0]["post_physics"] if report["check"] == "pose_shift")
        assert report["passed"] is None
        assert report["reason"] == "disabled by configuration"
        assert PlacementRecordingParams().validators["pose_shift"]["enabled"] is True
        # An unavailable required IK check must not turn into an accepted empty checklist.
        unavailable_pool = PooledObjectPlacer(
            pool.objects,
            replace(
                arena_env.placer_params,
                enabled_checks={PlacementCheck.IK_REACHABLE},
                required_checks={PlacementCheck.IK_REACHABLE},
                reachability_config=ReachabilityConfig(),
            ),
            pool_size=2,
            num_envs=2,
        )
        for queue in unavailable_pool.layouts_per_env():
            assert PlacementCheck.IK_REACHABLE not in queue[0].validation_results.validation_results
        with patch("isaaclab_arena.offline_placement.pool_validation.physics_settle.step_physics") as step:
            with pytest.raises(AssertionError, match="missing required solver checks: ik_reachable"):
                collect_settled_pool_layouts(env, unavailable_pool, scene_assets=arena_env.get_placement_assets())
            step.assert_not_called()
        torch.testing.assert_close(base.arena_world.get_pose_e("cube_body"), initial)
    finally:
        env.close()
    return True


def test_recording_filters_layouts(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_recording_filters_layouts, tmp_path=tmp_path)


def _test_recording_with_robot(simulation_app, tmp_path):
    import torch
    import yaml
    from dataclasses import replace
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_pool_layouts
    from isaaclab_arena.relations.placement_events import get_placement_pool
    from isaaclab_arena.relations.relation_solver import RelationSolver

    source = tmp_path / "robot.yaml"
    _write_scene(source)
    data = yaml.safe_load(source.read_text())
    data["embodiment"] = {"id": "robot", "registry_name": "franka_ik"}
    data["relations"].append({"kind": "at_position", "subject": "robot", "params": {"x": -1.0, "y": 0.0, "z": 0.0}})
    source.write_text(yaml.safe_dump(data))
    spec = ArenaEnvGraphSpec.from_yaml(source)
    arena = spec.to_arena_env()
    arena.placer_params = replace(
        arena.placer_params, min_unique_layouts_per_env=1, placement_seed=42, allow_best_loss_fallbacks=False
    )
    env = ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=1)).make_registered()
    try:
        env.reset()
        scene = env.unwrapped.scene
        state = scene.get_state()
        robot = scene.articulations["robot"]
        targets = robot.data.joint_pos_target.torch.clone()
        result = collect_settled_pool_layouts(
            env,
            get_placement_pool(env),
            PlacementRecordingParams(num_steps=120),
            scene_assets=arena.get_placement_assets(),
        )
        assert result.layouts.num_layouts == 1
        assert set(result.layouts.poses) == {"cube_body", "robot", "table", "floor"}
        torch.testing.assert_close(scene.get_state(), state)
        torch.testing.assert_close(robot.data.joint_pos_target.torch, targets)
        output = tmp_path / "robot.jsonl"
        result.layouts.write_episode_jsonl(output, source="settled", validation=result.validation)
        pool = get_placement_pool(env)
        from isaaclab_arena.offline_placement.scene_snapshot import articulation_link_poses_in_root_frame

        initial_links = articulation_link_poses_in_root_frame(env.unwrapped)
        moved_links = {key: poses.clone() for key, poses in initial_links.items()}
        moved_links["robot"][:, -1, 0] += 0.01
        with (
            patch(
                "isaaclab_arena.offline_placement.settled_placement.articulation_link_poses_in_root_frame",
                side_effect=[initial_links, moved_links],
            ),
            pytest.raises(AssertionError, match="joint states are not recorded"),
        ):
            collect_settled_pool_layouts(
                env, pool, PlacementRecordingParams(num_steps=120), scene_assets=arena.get_placement_assets()
            )
        torch.testing.assert_close(scene.get_state(), state)
        torch.testing.assert_close(robot.data.joint_pos_target.torch, targets)
    finally:
        env.close()
    with patch.object(RelationSolver, "solve", side_effect=AssertionError("Replay must not solve")):
        env = ArenaEnvBuilder(
            spec.to_arena_env(), ArenaEnvBuilderCfg(num_envs=2, placement_layouts_path=str(output))
        ).make_registered()
        try:
            env.reset()
            for key, poses in result.layouts.poses.items():
                expected = poses[0].to_tensor(env.unwrapped.device).expand(2, 7)
                torch.testing.assert_close(env.unwrapped.arena_world.get_pose_e(key), expected, atol=2e-5, rtol=0)
            from isaaclab_arena.utils.physics_settle import step_physics

            step_physics(env, 200)
            for key, poses in result.layouts.poses.items():
                if key == "robot":
                    continue
                expected = poses[0].to_tensor(env.unwrapped.device)[:3]
                actual = env.unwrapped.arena_world.get_pose_e(key)[:, :3]
                assert (actual - expected).norm(dim=-1).max() < 0.02
        finally:
            env.close()
    return True


def test_recording_with_robot_restores_scene_and_replays(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_recording_with_robot, tmp_path=tmp_path)
