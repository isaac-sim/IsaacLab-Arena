# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Clutter recording, configurable acceptance and reusable JSONL output."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from isaaclab_arena.offline_placement.post_physics_validation import PostPhysicsPlacementValidator
from isaaclab_arena.offline_placement.recording_config import PlacementRecordingCfg
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


@dataclass
class RejectPostPhysics(PostPhysicsPlacementValidator):
    """Reject measured candidates to exercise configured output gating."""

    check: ClassVar[str] = "reject_for_test"
    threshold: float = 0.0
    reject_env_ids: list[int] | None = None

    def validate(self, data):
        rejected = set(data.env_ids if self.reject_env_ids is None else self.reject_env_ids)
        return [
            self.report(env_id not in rejected, "deliberately rejected" if env_id in rejected else "")
            for env_id in data.env_ids
        ]


def _recording_cfg(output, scene_path, **kwargs):
    cfg = PlacementRecordingCfg(env_spec=str(scene_path), output=str(output), **kwargs)
    cfg.settle.num_steps = 480
    return cfg


def _read_records(path):
    return [json.loads(line)["variations"]["scene.relation_placement"] for line in path.read_text().splitlines()]


def _test_recording_writes_complete_layouts(simulation_app, tmp_path):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.validation.types import PlacementCheck
    from isaaclab_arena.scripts.record_placement_layouts import record_settled_placement_layouts
    from isaaclab_arena.tests.clutter.test_clutter_collection import _make_primitive_clutter_scene

    arena_env = _make_primitive_clutter_scene(tmp_path)
    scene_path = tmp_path / "scene.yaml"
    checks = {PlacementCheck.NO_OVERLAP, PlacementCheck.CLUTTER_ON_RELATION}
    arena_env.placer_params.enabled_checks = checks
    arena_env.placer_params.required_checks = checks
    original_callback = arena_env.env_cfg_callback

    def configure_physics(env_cfg):
        if original_callback is not None:
            env_cfg = original_callback(env_cfg)
        env_cfg.sim.dt = 0.01
        env_cfg.decimation = 2
        return env_cfg

    arena_env.env_cfg_callback = configure_physics
    output = tmp_path / "episodes.jsonl"
    cfg = _recording_cfg(output, scene_path, num_envs=2, min_layouts=3, max_batches=2, layouts_per_env=1)
    # A solve can supply only one layout per environment. Later refills must still
    # satisfy an output request larger than the initial pool.
    arena_env.placer_params.max_placement_attempts = 1
    solve = PooledObjectPlacer._solve_env_ranked_layouts

    def one_layout_per_env(pool, count):
        ranked, layouts_per_env = solve(pool, count)
        return [layouts[:1] for layouts in ranked], layouts_per_env

    with patch.object(PooledObjectPlacer, "_solve_env_ranked_layouts", one_layout_per_env):
        record_settled_placement_layouts(cfg, arena_env=arena_env)
    records = _read_records(output)
    # Three requested records exercise a final batch with more accepted candidates than needed.
    assert len(records) == 3
    assert records[0]["poses"]["cube_body"] != records[1]["poses"]["cube_body"]
    for record in records:
        assert set(record["poses"]) == {"cube_body", "table", "floor"}
        assert record["poses"]["cube_body"]["position_xyz"][2] == pytest.approx(0.57, abs=0.002)
        assert record["validation"]["pre_physics"] == {check: True for check in checks}
        reports = {report["check"]: report for report in record["validation"]["post_physics"]}
        assert all(reports[check]["passed"] for check in ("physics_settled", "pose_shift", "support_containment"))

    layouts = PlacementLayouts.from_episode_jsonl(output)
    assert layouts.num_layouts == 3
    replay_env = _make_primitive_clutter_scene(tmp_path)
    with patch.object(RelationSolver, "solve", side_effect=AssertionError("Replay must not solve")):
        env = ArenaEnvBuilder(
            replay_env, ArenaEnvBuilderCfg(num_envs=2, placement_layouts_path=str(output))
        ).make_registered()
        try:
            env.reset()
            for name, poses in layouts.poses.items():
                expected = torch.stack([pose.to_tensor(env.unwrapped.device) for pose in poses[:2]])
                torch.testing.assert_close(env.unwrapped.arena_world.get_pose_e(name), expected, atol=2e-5, rtol=0)
        finally:
            env.close()

    replay_env.placement_layouts = layouts
    cfg.output = str(tmp_path / "regenerated.jsonl")
    with pytest.raises(AssertionError, match="Remove cached placement layouts"):
        record_settled_placement_layouts(cfg, arena_env=replay_env)
    assert not Path(cfg.output).exists()
    cfg.output = str(output)
    saved = output.read_bytes()
    with pytest.raises(AssertionError, match="Output already exists"):
        record_settled_placement_layouts(cfg, arena_env=_make_primitive_clutter_scene(tmp_path))
    assert output.read_bytes() == saved
    return True


def test_recording_writes_complete_layouts(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_recording_writes_complete_layouts, tmp_path=tmp_path)


def _test_post_physics_checks_gate_output(simulation_app, tmp_path):
    from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
    from isaaclab_arena.scripts.record_placement_layouts import record_settled_placement_layouts
    from isaaclab_arena.tests.clutter.test_clutter_collection import _make_primitive_clutter_scene

    scene_path = tmp_path / "scene.yaml"
    output = tmp_path / "checked.jsonl"
    cfg = _recording_cfg(output, scene_path, num_envs=2, min_layouts=1, max_batches=1, layouts_per_env=1)
    cfg.settle.validators = dict(default_clutter_validators())
    cfg.settle.validators["reject_for_test"] = {
        "_target_": "isaaclab_arena.tests.clutter.test_recording.RejectPostPhysics",
        "threshold": 7.0,
    }
    summary = record_settled_placement_layouts(cfg, arena_env=_make_primitive_clutter_scene(tmp_path))
    assert summary.output is None
    assert any("reject_for_test: deliberately rejected" in text for text in summary.rejections.values())
    assert not output.exists()
    cfg.settle.validators["reject_for_test"]["enabled"] = False
    cfg.settle.validators["physics_settled"]["enabled"] = False
    cfg.settle.validators["support_containment"]["minimum_resting_heights_m"] = {"table": 0.02}
    assert record_settled_placement_layouts(cfg, arena_env=_make_primitive_clutter_scene(tmp_path)).output == output
    records = _read_records(output)
    assert len(records) == 1
    record = records[0]
    reports = {report["check"]: report for report in record["validation"]["post_physics"]}
    assert reports["reject_for_test"]["passed"] is None
    assert reports["reject_for_test"]["reason"]
    assert reports["reject_for_test"]["configuration"]["threshold"] == 7.0
    assert reports["physics_settled"]["passed"] is None
    assert reports["support_containment"]["passed"]
    assert reports["support_containment"]["configuration"]["minimum_resting_heights_m"] == {"table": 0.02}
    return True


def test_post_physics_checks_gate_output(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_post_physics_checks_gate_output, tmp_path=tmp_path)
