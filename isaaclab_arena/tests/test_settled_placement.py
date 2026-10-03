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


def run_cli_with_test_assets(summary_path: Path | None = None):
    import json
    from unittest.mock import patch

    from isaaclab_arena.scripts import record_placement_layouts
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    enter = SimulationAppContext.__enter__
    record = record_placement_layouts.record_settled_placement_layouts

    def enter_with_test_assets(context):
        app = enter(context)
        register_no_embodiment()
        return app

    def record_with_summary(cfg, *, device):
        summary = record(cfg, device=device)
        if summary_path is not None:
            summary_path.write_text(
                json.dumps({
                    "accepted": summary.accepted,
                    "attempted": summary.attempted,
                    "output": None if summary.output is None else str(summary.output),
                })
            )
        return summary

    with (
        patch.object(SimulationAppContext, "__enter__", enter_with_test_assets),
        patch.object(record_placement_layouts, "record_settled_placement_layouts", record_with_summary),
    ):
        record_placement_layouts.main()


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
    run_subprocess(
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
            "min_layouts=4",
            "max_batches=5",
            "settle.num_steps=120",
            "settle.validators.pose_shift.max_translation_m=0.0015",
            "--viz",
            "none",
        ],
        timeout_sec=180,
        capture_output=True,
    )
    records = [json.loads(line)["variations"]["scene.relation_placement"] for line in output.read_text().splitlines()]
    assert len(records) == 4
    for record in records:
        assert record["source"] == "settled"
        reports = {report["check"]: report for report in record["validation"]["post_physics"]}
        assert reports["physics_settled"]["passed"] is True
        assert reports["pose_shift"]["passed"] is True
        assert reports["pose_shift"]["configuration"]["max_translation_m"] == 0.0015
        assert reports["articulation_link_shift"]["passed"] is None
        assert reports["articulation_link_shift"]["reason"]
        assert set(record["poses"]) == {"cube_body", "table", "floor"}
        # Table top is 0.52, cube half-height is 0.05.
        x, y, _ = record["poses"]["cube_body"]["position_xyz"]
        assert abs(x) < 0.4 and abs(y) < 0.4
        assert record["poses"]["cube_body"]["position_xyz"][2] == pytest.approx(0.57, abs=0.005)

    positions = {tuple(record["poses"]["cube_body"]["position_xyz"]) for record in records}
    assert len(positions) == len(records)


def test_record_placements_to_jsonl_leaves_no_file_when_target_unmet(tmp_path):
    from unittest.mock import Mock, patch

    from isaaclab_arena.offline_placement import settled_placement
    from isaaclab_arena.scripts.record_placement_layouts import record_placements_to_jsonl

    env = Mock()
    pool = Mock(objects=[])
    with (
        patch("isaaclab_arena.relations.placement_events.get_placement_pool", return_value=pool),
        patch(
            "isaaclab_arena.offline_placement.recording.collect_layouts_until_count",
            return_value=({"cube": []}, [], 1, {(0, 0): "failed"}),
        ),
        patch("isaaclab_arena.offline_placement.recording.validate_recording_assets"),
        patch.object(settled_placement, "resolve_settle_params", return_value=Mock(num_steps=1)),
    ):
        output = tmp_path / "unused.jsonl"
        summary = record_placements_to_jsonl(env, output, min_layouts=2, max_batches=2, scene_assets=[])
    assert summary.output is None
    assert summary.accepted == 0
    assert summary.attempted == 1


def test_record_placements_to_jsonl_writes_partial_acceptance(tmp_path):
    from unittest.mock import Mock, patch

    from isaaclab_arena.offline_placement import settled_placement
    from isaaclab_arena.scripts.record_placement_layouts import record_placements_to_jsonl

    env = Mock()
    pool = Mock(objects=[])
    output = tmp_path / "partial.jsonl"

    def write_partial(_env, destination, _assets, _poses, _validation, _num_steps):
        Path(destination).write_text("accepted\n")

    with (
        patch("isaaclab_arena.relations.placement_events.get_placement_pool", return_value=pool),
        patch(
            "isaaclab_arena.offline_placement.recording.collect_layouts_until_count",
            return_value=({"cube": [Mock()]}, [Mock()], 2, {(1, 0): "failed"}),
        ),
        patch("isaaclab_arena.offline_placement.recording.validate_recording_assets"),
        patch("isaaclab_arena.offline_placement.recording.write_settled_layouts", side_effect=write_partial),
        patch.object(settled_placement, "resolve_settle_params", return_value=Mock(num_steps=1)),
    ):
        summary = record_placements_to_jsonl(env, output, min_layouts=2, max_batches=1, scene_assets=[])
    assert summary.output == output
    assert summary.accepted == 1
    assert summary.attempted == 2
    assert output.read_text() == "accepted\n"


@pytest.mark.with_subprocess
def test_recording_cli_writes_partial_acceptance(tmp_path):
    import os
    import subprocess

    source, output = tmp_path / "scene.yaml", tmp_path / "placements.jsonl"
    _write_scene(source)
    child_env = os.environ.copy()
    child_env["ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE"] = "1"
    completed = subprocess.run(
        [
            TestConstants.python_path,
            "-c",
            (
                "from isaaclab_arena.tests.test_settled_placement import run_cli_with_test_assets;"
                " run_cli_with_test_assets()"
            ),
            f"env_spec={source}",
            f"output={output}",
            "presets=physx",
            "num_envs=2",
            "layouts_per_env=1",
            "min_layouts=2",
            "max_batches=1",
            "settle.num_steps=120",
            "+settle.validators.reject_for_test._target_=isaaclab_arena.tests.clutter.test_recording.RejectPostPhysics",
            "+settle.validators.reject_for_test.reject_env_ids=[1]",
            "--viz",
            "none",
        ],
        env=child_env,
        timeout=180,
        capture_output=True,
        text=True,
        start_new_session=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Only 1 of 2 requested layouts were recorded" in completed.stdout + completed.stderr
    assert len(output.read_text().splitlines()) == 1


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


def _test_recording_with_default_placer_params(simulation_app, tmp_path):
    import json
    from unittest.mock import patch

    from isaaclab_arena.assets.registries import AssetRegistry, ensure_assets_registered
    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.offline_placement.recording_config import PlacementRecordingCfg
    from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams
    from isaaclab_arena.scripts.record_placement_layouts import record_settled_placement_layouts

    source, output = tmp_path / "scene.yaml", tmp_path / "placements.jsonl"
    _write_scene(source)
    ensure_assets_registered()
    with patch.dict(AssetRegistry()._components, {"recording_no_embodiment": NoEmbodiment}):
        scene_description = ArenaEnvGraphSpec.from_yaml(source).to_arena_env()
    arena_env = IsaacLabArenaEnvironment(
        name=scene_description.name,
        scene=scene_description.scene,
        embodiment=scene_description.embodiment,
        task=scene_description.task,
    )
    assert arena_env.placer_params is None
    cfg = PlacementRecordingCfg(
        output=str(output),
        min_layouts=2,
        layouts_per_env=1,
        max_batches=3,
        settle=SettledPlacementParams(num_steps=120),
    )
    summary = record_settled_placement_layouts(cfg, arena_env=arena_env)
    assert summary.output == output
    assert summary.accepted == 2
    records = [json.loads(line)["variations"]["scene.relation_placement"] for line in output.read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        assert record["poses"]["cube_body"]["position_xyz"][2] == pytest.approx(0.57, abs=0.002)
        reports = {report["check"]: report for report in record["validation"]["post_physics"]}
        assert reports["physics_settled"]["passed"]
        assert reports["pose_shift"]["passed"]
    return True


def test_recording_with_default_placer_params(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_recording_with_default_placer_params, tmp_path=tmp_path)


def _test_recording_filters_layouts(simulation_app, tmp_path):
    import json
    import torch
    import yaml
    from copy import deepcopy
    from dataclasses import replace
    from unittest.mock import Mock, patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
    from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams
    from isaaclab_arena.relations.placement_events import get_placement_pool, make_cached_placement_event
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.reachability_config import ReachabilityConfig
    from isaaclab_arena.relations.validation.types import PlacementCheck
    from isaaclab_arena.scripts.record_placement_layouts import record_placements_to_jsonl
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
                patch.object(
                    base, "reset", side_effect=AssertionError("Incompatible reset policies must fail before reset")
                ) as reset,
            ):
                with pytest.raises(AssertionError, match=f"floor.*{reason}"):
                    record_placements_to_jsonl(
                        env, tmp_path / "incompatible.jsonl", min_layouts=1, max_batches=1, scene_assets=assets
                    )
                reset.assert_not_called()
                with pytest.raises(AssertionError, match=f"floor.*{reason}"):
                    make_cached_placement_event(layouts, assets, base.num_envs)
            torch.testing.assert_close(base.scene.get_state(), state)

        queues = pool.layouts_per_env()
        cube = arena_env.scene.assets["cube_body"]
        # Both pass On(overlap=True); only the overhanging cube falls to the floor.
        queues[0][0].positions[cube] = (0.0, 0.0, 0.571)
        queues[1][0].positions[cube] = (0.42, 0.0, 0.571)
        from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes
        from isaaclab_arena.relations.validation.pre_physics import OnRelationValidator, PrePhysicsPlacementValidator
        from isaaclab_arena.tests.dummy_object import make_candidate_batch

        boxes = build_per_env_bounding_boxes(pool.objects, 2).get_bounding_boxes_for_all_envs()
        validator = OnRelationValidator(arena_env.placer_params)
        batch = make_candidate_batch([queue[0].positions for queue in queues], [{}, {}], boxes)
        assert validator.validate_batch(batch, []) == [True, True]
        # The second reset rejects a stable but excessive drop and a solver failure.
        queues[0][1].positions[cube] = (0.0, 0.0, 0.575)
        queues[1][1].validation_results.validation_results[PlacementCheck.NO_OVERLAP] = False
        ik_validator = Mock(spec=PrePhysicsPlacementValidator)
        ik_validator.check = PlacementCheck.IK_REACHABLE
        pool._placer._validators.append(ik_validator)
        saved_positions = [dict(queue[0].positions) for queue in queues]
        saved_checks = [deepcopy(layout.validation_results) for queue in pool.layouts_per_env() for layout in queue]
        assert floor not in pool.objects
        with (
            patch.object(pool, "sample_for_envs", wraps=pool.sample_for_envs) as sample,
            patch.object(pool, "_solve_and_store", side_effect=AssertionError("First batch must not be skipped")),
            patch.object(cube, "reset_pose", False),
        ):
            # Collection measures scene roots without requiring replay-compatible asset metadata.
            result = collect_settled_placements(env, 2, SettledPlacementParams(num_steps=120))
        assert sample.call_count == 2
        assert all(call.args == ([0, 1],) for call in sample.call_args_list)
        assert pool.remaining == 0
        ik_validator.validate_batch.assert_not_called()
        assert result.attempted == 4
        assert result.accepted_indices == [(0, 0)]
        assert "cube_body: moved" in result.rejections[1, 0]
        assert result.rejections[1, 1] == "solver validation failed"
        assert set(result.poses) == {"cube_body", "table", "floor"}
        assert all(len(poses) == 1 for poses in result.poses.values())
        assert result.poses["cube_body"][0].position_xyz[2] == pytest.approx(0.57, abs=0.005)
        assert [queue[0].positions for queue in queues] == saved_positions
        assert [layout.validation_results for queue in pool.layouts_per_env() for layout in queue] == saved_checks
        assert "cube_body: moved" in result.rejections[0, 1]
        assert result.validation[0].pre_physics == queues[0][0].validation_results.validation_results
        # The environment remains at its final measured state, not the release pose.
        assert base.arena_world.get_pose_e("cube_body")[0, 2].item() == pytest.approx(0.57, abs=0.002)
        pool._placer._validators.remove(ik_validator)
        params = SettledPlacementParams(num_steps=120)
        params.validators["pose_shift"]["enabled"] = False
        for env_pool, queue in zip(pool._env_pools, queues, strict=True):
            env_pool.append(queue[1])
        output = tmp_path / "relaxed.jsonl"
        with (
            patch.object(pool, "_solve_and_store", side_effect=AssertionError("Reuse the controlled drop")),
            patch.object(base, "close", wraps=base.close) as close,
        ):
            summary = record_placements_to_jsonl(
                env, output, min_layouts=1, max_batches=1, params=params, scene_assets=assets
            )
            close.assert_not_called()
        assert summary.output == output
        assert summary.accepted == 1 and summary.attempted == 2
        assert set(summary.rejections) == {(1, 0)}
        records = [
            json.loads(line)["variations"]["scene.relation_placement"] for line in output.read_text().splitlines()
        ]
        assert len(records) == 1
        record = records[0]
        assert record["source"] == "settled"
        saved_position = record["poses"]["cube_body"]["position_xyz"]
        assert queues[0][1].positions[cube][2] - saved_position[2] > 0.002
        assert saved_position == pytest.approx(base.arena_world.get_pose_e("cube_body")[0, :3].tolist())
        assert record["validation"]["pre_physics"] == queues[0][1].validation_results.validation_results
        sampling = record["validation"]["sampling"]
        assert sampling["num_steps"] == params.num_steps
        assert sampling["decimation"] == base.cfg.decimation
        assert sampling["physics_dt_s"] == base.sim.get_physics_dt()
        report = next(report for report in record["validation"]["post_physics"] if report["check"] == "pose_shift")
        assert report["passed"] is None
        assert report["reason"]
        assert SettledPlacementParams().validators["pose_shift"]["enabled"] is True
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
        rejected_output = tmp_path / "rejected.jsonl"
        with (
            patch.object(handle, "pool", unavailable_pool),
            patch("isaaclab_arena.offline_placement.pool_validation.physics_settle.step_physics") as step,
            patch.object(base, "close", wraps=base.close) as close,
        ):
            rejected = collect_settled_placements(env, 1)
            assert rejected.attempted == base.num_envs
            assert rejected.accepted_indices == []
            assert rejected.poses == {key: [] for key in result.poses}
            assert rejected.validation == []
            assert set(rejected.rejections) == {(0, 0), (1, 0)}
            assert all(
                "missing required solver checks: ik_reachable" in reason for reason in rejected.rejections.values()
            )
            summary = record_placements_to_jsonl(
                env, rejected_output, min_layouts=1, max_batches=1, scene_assets=assets
            )
            assert summary.output is None
            assert summary.accepted == 0 and summary.attempted == base.num_envs
            assert summary.rejections == rejected.rejections
            step.assert_not_called()
            close.assert_not_called()
        assert not rejected_output.exists()
    finally:
        env.close()
    return True


def test_recording_filters_layouts(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_recording_filters_layouts, tmp_path=tmp_path)


def _test_recording_with_robot(simulation_app, tmp_path):
    import torch
    import yaml
    from dataclasses import asdict, replace
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.settled_batch import capture_articulation_link_poses_in_root_frame
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
    from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
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
            return capture_articulation_link_poses_in_root_frame(env, articulation_keys)

        with patch(
            "isaaclab_arena.offline_placement.settled_batch.capture_articulation_link_poses_in_root_frame",
            side_effect=measure_links,
        ):
            result = collect_settled_placements(
                env,
                2,
                SettledPlacementParams(num_steps=120),
                scene_assets=arena.get_placement_assets(),
            )
        assert result.accepted_indices == [(0, 0), (0, 1)]
        assert set(result.poses) == {"cube_body", "robot", "table", "floor"}
        assert all(len(poses) == 2 for poses in result.poses.values())
        assert len(joint_positions) == 4
        # Normal resets still randomize the robot joints.
        assert not torch.allclose(joint_positions[0], joint_positions[2])
        output = tmp_path / "robot.jsonl"
        PlacementLayouts(result.poses).write_episode_jsonl(
            output, source="settled", validation=[asdict(validation) for validation in result.validation]
        )
        for validation in result.validation:
            reports = {report.check: report for report in validation.post_physics}
            assert reports["articulation_link_shift"].passed is None
            assert reports["articulation_link_shift"].reason
            assert reports["physics_settled"].passed is True
            assert reports["pose_shift"].passed is True
    finally:
        env.close()
    with patch.object(RelationSolver, "solve", side_effect=AssertionError("Replay must not solve")):
        env = ArenaEnvBuilder(
            spec.to_arena_env(), ArenaEnvBuilderCfg(num_envs=2, placement_layouts_path=str(output))
        ).make_registered()
        try:
            env.reset()
            for key, poses in result.poses.items():
                expected = torch.stack([pose.to_tensor(env.unwrapped.device) for pose in poses])
                torch.testing.assert_close(env.unwrapped.arena_world.get_pose_e(key), expected, atol=2e-5, rtol=0)
            from isaaclab_arena.utils.physics_settle import step_physics

            step_physics(env, 200)
            for key, poses in result.poses.items():
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


def _test_configured_validator_reports_write_jsonl(simulation_app, tmp_path, minimum_resting_heights):
    import json
    from dataclasses import asdict

    from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
    from isaaclab_arena.offline_placement.post_physics_validation import build_post_physics_validators
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.utils.pose import Pose

    configurations = default_clutter_validators()
    if minimum_resting_heights:
        configurations["support_containment"]["minimum_resting_heights_m"] = minimum_resting_heights
    validators = build_post_physics_validators(configurations, [])
    reports = [asdict(validator.report(True)) for validator in validators]
    output = tmp_path / "placements.jsonl"
    PlacementLayouts({"cube": [Pose.identity()]}).write_episode_jsonl(
        output, source="settled", validation=[{"post_physics": reports}]
    )
    record = json.loads(output.read_text())["variations"]["scene.relation_placement"]
    recorded_reports = {report["check"]: report for report in record["validation"]["post_physics"]}
    assert (
        recorded_reports["support_containment"]["configuration"]["minimum_resting_heights_m"] == minimum_resting_heights
    )
    assert PlacementLayouts.from_episode_jsonl(output).poses == {"cube": [Pose.identity()]}
    return True


@pytest.mark.parametrize("minimum_resting_heights", [{}, {"bowl": -0.025}], ids=["flat-support", "bowl"])
def test_configured_validator_reports_write_jsonl(tmp_path, minimum_resting_heights):
    assert run_function_with_persistent_simulation_app(
        _test_configured_validator_reports_write_jsonl,
        tmp_path=tmp_path,
        minimum_resting_heights=minimum_resting_heights,
    )


def test_link_shift_checks_task_objects_separately_from_roots():
    import torch

    from isaaclab_arena.offline_placement.post_physics_validation import (
        ArticulationLinkShiftValidator,
        PoseShiftValidator,
    )
    from isaaclab_arena.offline_placement.settled_batch import SettledBatch
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.validation.types import PlacementValidationResults

    root = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    links = root[:, None, :].repeat(1, 2, 1)
    batch = SettledBatch(
        source_layouts={0: PlacementResult(PlacementValidationResults(), {}, 0.0, 1)},
        env_ids=[0],
        initial_root_poses={"arm": root.clone(), "cabinet": root.clone()},
        final_root_poses={"arm": root.clone(), "cabinet": root.clone()},
        initial_link_poses={"cabinet": links.clone()},
        final_link_poses={"cabinet": links.clone()},
        final_root_velocities={"arm": torch.zeros((1, 6)), "cabinet": torch.zeros((1, 6))},
    )
    validator = ArticulationLinkShiftValidator()
    assert validator.skip_reason(list(batch.initial_link_poses)) is None
    assert validator.skip_reason([])
    assert validator.validate(batch)[0].passed is True
    batch.final_link_poses["cabinet"][0, 1, 0] += 0.01
    report = validator.validate(batch)[0]
    assert report.passed is False
    assert "cabinet" in report.reason
    batch.final_root_poses["arm"][0, 0] += 0.01
    assert PoseShiftValidator().validate(batch)[0].passed is False

    # Non-finite measurements reject both roots and links rather than accepting a missing drift.
    batch.final_root_poses["arm"][0, 0] = float("nan")
    root_report = PoseShiftValidator().validate(batch)[0]
    assert root_report.passed is False and root_report.reason
    batch.final_link_poses["cabinet"][0, 1, 0] = float("inf")
    link_report = validator.validate(batch)[0]
    assert link_report.passed is False and link_report.reason


def test_settled_batch_evaluation_uses_captured_state():
    import torch
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from isaaclab_arena.offline_placement.post_physics_validation import VelocityValidator, evaluate_settled_batch
    from isaaclab_arena.offline_placement.settled_batch import sample_and_settle_batch
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.validation.types import PlacementCheck, PlacementValidationResults

    root_poses = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).repeat(3, 1)
    linear_velocity = torch.zeros((3, 3))
    angular_velocity = torch.zeros((3, 3))
    world = SimpleNamespace(
        get_pose_e=Mock(side_effect=lambda key: root_poses.clone()),
        get_root_linear_velocity_w=Mock(return_value=linear_velocity),
        get_root_angular_velocity_w=Mock(return_value=angular_velocity),
    )
    env = SimpleNamespace(num_envs=3, cfg=SimpleNamespace(decimation=2), arena_world=world, reset=Mock())
    env.unwrapped = env
    source_layouts = {}
    for env_id in range(env.num_envs):
        checks = PlacementValidationResults(
            validation_results={PlacementCheck.NO_OVERLAP: env_id != 2},
            required_checks={PlacementCheck.NO_OVERLAP},
        )
        source_layouts[env_id] = PlacementResult(checks, {}, 0.0, 1)

    def advance_physics(*_args, **_kwargs):
        root_poses[0, 0] = 0.01
        linear_velocity[1, 0] = 0.2

    with (
        patch("isaaclab_arena.relations.placement_events.get_placement_pool", return_value=SimpleNamespace(num_envs=3)),
        patch("isaaclab_arena.relations.placement_events.get_reset_placement_results", return_value=source_layouts),
        patch("isaaclab_arena.utils.physics_settle.step_physics", side_effect=advance_physics),
    ):
        batch = sample_and_settle_batch(env, root_keys=["cube"], link_keys=[], num_env_steps=2)

    # Later simulator updates and source-check changes must not alter this batch's verdicts.
    root_poses[:, 0] = 1.0
    linear_velocity[0, 0] = 1.0
    linear_velocity[1, 0] = 0.0
    source_layouts[0].validation_results.validation_results[PlacementCheck.NO_OVERLAP] = False
    source_layouts[2].validation_results.validation_results[PlacementCheck.NO_OVERLAP] = True
    outcomes = evaluate_settled_batch(batch, [VelocityValidator()])
    assert set(outcomes) == {0, 1, 2}
    assert outcomes[0].passed
    assert not outcomes[1].passed
    assert outcomes[1].post_physics[0].passed is False
    assert not outcomes[2].passed
    assert outcomes[2].pre_physics[PlacementCheck.NO_OVERLAP] is False
    assert outcomes[2].post_physics == []
    assert outcomes[2].rejection_reason
    assert batch.initial_root_poses["cube"][0, 0].item() == 0.0
    assert batch.final_root_poses["cube"][0, 0].item() == pytest.approx(0.01)
    assert batch.final_root_velocities["cube"][1, 0].item() == pytest.approx(0.2)


def test_collect_layouts_until_count_trims_and_merges():
    from unittest.mock import Mock, patch

    from isaaclab_arena.offline_placement.recording import collect_layouts_until_count
    from isaaclab_arena.offline_placement.settled_placement import SettledPlacementResult
    from isaaclab_arena.utils.pose import Pose

    env = Mock()
    batch_one = SettledPlacementResult(
        poses={"cube": [Pose.identity(), Pose.identity()]},
        accepted_indices=[(0, 0), (1, 0)],
        rejections={},
        validation=[Mock(), Mock()],
    )
    batch_two = SettledPlacementResult(
        poses={"cube": [Pose.identity()]},
        accepted_indices=[(0, 0)],
        rejections={(1, 0): "pose_shift"},
        validation=[Mock()],
    )

    with patch(
        "isaaclab_arena.offline_placement.settled_placement.collect_settled_placements",
        side_effect=[batch_one, batch_two],
    ) as collect:
        poses, validation, attempted, rejections = collect_layouts_until_count(env, min_layouts=3, max_batches=5)

    assert collect.call_count == 2
    assert len(validation) == 3
    assert len(poses["cube"]) == 3
    assert attempted == batch_one.attempted + batch_two.attempted
    assert rejections == {(1, 1): "pose_shift"}
