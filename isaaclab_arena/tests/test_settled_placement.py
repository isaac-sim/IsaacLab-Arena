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
            "env_spacing=2.0",
            "viewer_eye=[4.0,4.0,6.3]",
            "viewer_lookat=[0.6,0.6,0.3]",
            "layouts_per_env=2",
            "settle.num_steps=120",
            "settle.validators.pose_shift.max_translation_m=0.0015",
            "--viz",
            "none",
        ],
        timeout_sec=180,
        capture_output=True,
    )
    assert "batch 1/2: 480/960 physics steps" in completed.stdout
    assert "batch 2/2: 960/960 physics steps" in completed.stdout
    assert "2 solutions, 2 passed solver validation, 2 passed post-physics validation" in completed.stdout
    assert "overall 4/4 validated, 4 accepted" in completed.stdout
    assert "physics_settled: ENABLED" in completed.stdout
    assert "articulation_link_shift: SKIPPED: no articulated task objects selected" in completed.stdout
    records = [json.loads(line)["variations"]["scene.relation_placement"] for line in output.read_text().splitlines()]
    assert len(records) == 4
    for record in records:
        assert record["source"] == "settled"
        reports = {report["check"]: report for report in record["validation"]["post_physics"]}
        assert reports["physics_settled"]["passed"] is True
        assert reports["pose_shift"]["passed"] is True
        assert reports["pose_shift"]["configuration"]["max_translation_m"] == 0.0015
        assert reports["articulation_link_shift"]["passed"] is None
        assert reports["articulation_link_shift"]["reason"] == "no articulated task objects selected"
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
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
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
        min_unique_layouts_per_env=2,
        placement_seed=42,
        random_yaw_init=False,
        enabled_checks={PlacementCheck.NO_OVERLAP},
        required_checks={PlacementCheck.NO_OVERLAP},
    )
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    try:
        base = env.unwrapped
        pool = get_placement_pool(env)
        assets = arena_env.get_placement_assets()
        floor = arena_env.scene.assets["floor"]
        floor.tags = None
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
                    "isaaclab_arena.offline_placement.settled_placement._settle_reset",
                    side_effect=AssertionError("Incompatible reset policies must fail before physics"),
                ) as simulate,
            ):
                with pytest.raises(AssertionError, match=f"floor.*{reason}"):
                    collect_settled_placements(env, 1, scene_assets=assets)
                simulate.assert_not_called()
                with pytest.raises(AssertionError, match=f"floor.*{reason}"):
                    make_cached_placement_event(layouts, assets, base.num_envs)
            torch.testing.assert_close(base.scene.get_state(), state)

        queues = pool.layouts_per_env()
        cube = arena_env.scene.assets["cube_body"]
        # Both pass On(overlap=True); only the overhanging cube falls to the floor.
        queues[0][0].positions[cube] = (0.0, 0.0, 0.571)
        queues[1][0].positions[cube] = (0.42, 0.0, 0.571)
        from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes
        from isaaclab_arena.relations.placement_validators import OnRelationValidator, PlacementValidator

        boxes = build_per_env_bounding_boxes(pool.objects, 2).get_bounding_boxes_for_all_envs()
        validator = OnRelationValidator(arena_env.placer_params)
        assert validator.validate_batch([queue[0].positions for queue in queues], [{}, {}], boxes, []) == [True, True]
        # The second reset rejects a stable but excessive drop and a solver failure.
        queues[0][1].positions[cube] = (0.0, 0.0, 0.575)
        queues[1][1].validation_results.validation_results[PlacementCheck.NO_OVERLAP] = False
        ik_validator = Mock(spec=PlacementValidator)
        ik_validator.check = PlacementCheck.IK_REACHABLE
        pool._placer._validators.append(ik_validator)
        saved_positions = [dict(queue[0].positions) for queue in queues]
        saved_checks = [deepcopy(layout.validation_results) for queue in pool.layouts_per_env() for layout in queue]
        with (
            patch.object(pool, "sample_for_envs", wraps=pool.sample_for_envs) as sample,
            patch.object(pool, "_solve_and_store", side_effect=AssertionError("First batch must not be skipped")),
        ):
            result = collect_settled_placements(
                env,
                2,
                PlacementRecordingParams(num_steps=120),
                scene_assets=assets,
            )
        assert sample.call_count == 2
        assert all(call.args == ([0, 1],) for call in sample.call_args_list)
        assert pool.remaining == 0
        ik_validator.validate_batch.assert_not_called()
        layouts = result.layouts
        assert result.attempted == 4
        assert result.accepted_indices == [(0, 0)]
        assert "cube_body: moved" in result.rejections[1, 0]
        assert result.rejections[1, 1] == "solver validation failed"
        assert layouts.num_layouts == 1
        assert layouts.poses["cube_body"][0].position_xyz[2] == pytest.approx(0.57, abs=0.005)
        assert [queue[0].positions for queue in queues] == saved_positions
        assert [layout.validation_results for queue in pool.layouts_per_env() for layout in queue] == saved_checks
        assert "cube_body: moved" in result.rejections[0, 1]
        assert result.validation[0]["pre_physics"] == queues[0][0].validation_results.validation_results
        # The environment remains at its final measured state, not the release pose.
        assert base.arena_world.get_pose_e("cube_body")[0, 2].item() == pytest.approx(0.57, abs=0.002)
        pool._placer._validators.remove(ik_validator)
        params = PlacementRecordingParams(num_steps=120)
        params.validators["pose_shift"]["enabled"] = False
        for env_pool, queue in zip(pool._env_pools, queues, strict=True):
            env_pool.append(queue[1])
        with patch.object(pool, "_solve_and_store", side_effect=AssertionError("Reuse the controlled drop")):
            relaxed = collect_settled_placements(env, 1, params, scene_assets=assets)
        assert relaxed.accepted_indices == [(0, 0)]
        assert queues[0][1].positions[cube][2] - relaxed.layouts.poses["cube_body"][0].position_xyz[2] > 0.002
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
        from isaaclab_arena.relations.placement_events import PLACEMENT_RESET_EVENT_NAME

        handle = base.event_manager.get_term_cfg(PLACEMENT_RESET_EVENT_NAME).params["placement_pool"]
        with (
            patch.object(handle, "pool", unavailable_pool),
            patch("isaaclab_arena.offline_placement.pool_validation.physics_settle.step_physics") as step,
        ):
            with pytest.raises(AssertionError, match="missing required solver checks: ik_reachable"):
                collect_settled_placements(env, 1, scene_assets=arena_env.get_placement_assets())
            step.assert_not_called()
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
    from isaaclab_arena.offline_placement.post_physics_validation import articulation_link_poses_in_root_frame
    from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
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
        scene = env.unwrapped.scene
        robot = scene.articulations["robot"]
        robot.set_joint_position_target_index(target=robot.data.default_joint_pos.torch + 0.2)
        joint_positions = []

        def measure_links(env, articulation_keys):
            assert articulation_keys == []
            joint_positions.append(robot.data.joint_pos.torch.clone())
            return articulation_link_poses_in_root_frame(env, articulation_keys)

        with patch(
            "isaaclab_arena.offline_placement.settled_placement.articulation_link_poses_in_root_frame",
            side_effect=measure_links,
        ):
            result = collect_settled_placements(
                env,
                2,
                PlacementRecordingParams(num_steps=120),
                scene_assets=arena.get_placement_assets(),
            )
        assert result.layouts.num_layouts == 2
        assert set(result.layouts.poses) == {"cube_body", "robot", "table", "floor"}
        assert len(joint_positions) == 4
        # Normal resets still randomize the robot joints.
        assert not torch.allclose(joint_positions[0], joint_positions[2])
        output = tmp_path / "robot.jsonl"
        result.layouts.write_episode_jsonl(output, source="settled", validation=result.validation)
        for validation in result.validation:
            reports = {report["check"]: report for report in validation["post_physics"]}
            assert reports["articulation_link_shift"]["passed"] is None
            assert "no articulated task objects selected" in reports["articulation_link_shift"]["reason"]
            assert reports["physics_settled"]["passed"] is True
            assert reports["pose_shift"]["passed"] is True
            assert validation["sampling"]["embodiment_keys"] == ["robot"]
    finally:
        env.close()
    with patch.object(RelationSolver, "solve", side_effect=AssertionError("Replay must not solve")):
        env = ArenaEnvBuilder(
            spec.to_arena_env(), ArenaEnvBuilderCfg(num_envs=2, placement_layouts_path=str(output))
        ).make_registered()
        try:
            env.reset()
            for key, poses in result.layouts.poses.items():
                expected = torch.stack([pose.to_tensor(env.unwrapped.device) for pose in poses])
                torch.testing.assert_close(env.unwrapped.arena_world.get_pose_e(key), expected, atol=2e-5, rtol=0)
            from isaaclab_arena.utils.physics_settle import step_physics

            step_physics(env, 200)
            for key, poses in result.layouts.poses.items():
                if key == "robot":
                    continue
                expected = torch.stack([pose.to_tensor(env.unwrapped.device)[:3] for pose in poses])
                actual = env.unwrapped.arena_world.get_pose_e(key)[:, :3]
                assert (actual - expected).norm(dim=-1).max() < 0.02
        finally:
            env.close()
    return True


def test_recording_with_robot_resets_and_replays(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_recording_with_robot, tmp_path=tmp_path)


def test_link_shift_checks_task_objects_separately_from_roots():
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.offline_placement.post_physics_validation import (
        ArticulationLinkShiftValidator,
        PoseShiftValidator,
        PostPhysicsState,
    )

    root = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    links = root[:, None, :].repeat(1, 2, 1)
    env = SimpleNamespace(scene=SimpleNamespace(articulations={"arm": None, "cabinet": None}))
    state = PostPhysicsState(
        env=env,
        env_ids=[0],
        initial_poses={"arm": root.clone(), "cabinet": root.clone()},
        final_poses={"arm": root.clone(), "cabinet": root.clone()},
        initial_links={"cabinet": links.clone()},
        final_links={"cabinet": links.clone()},
    )
    validator = ArticulationLinkShiftValidator()
    assert validator.skip_reason(list(state.initial_links)) is None
    assert validator.skip_reason([]) == "no articulated task objects selected"
    assert validator.validate(state)[0].passed is True
    state.final_links["cabinet"][0, 1, 0] += 0.01
    report = validator.validate(state)[0]
    assert report.passed is False
    assert "cabinet" in report.reason and "joint states are not recorded" in report.reason
    state.final_poses["arm"][0, 0] += 0.01
    assert PoseShiftValidator().validate(state)[0].passed is False
