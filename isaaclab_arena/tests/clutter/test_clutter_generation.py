# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Clutter generation, configurable acceptance and reusable JSONL output."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from isaaclab_arena.offline_placement.post_physics_validation import PostPhysicsPlacementValidator
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


@dataclass
class RejectPostPhysics(PostPhysicsPlacementValidator):
    """Reject measured candidates to exercise configured output gating."""

    check: ClassVar[str] = "reject_for_test"
    threshold: float = 0.0

    def validate(self, data):
        return [self.report(False, "deliberately rejected") for _ in data.env_ids]


def _arguments(output, **kwargs):
    from isaaclab_arena.offline_placement.clutter_generation import ClutterGenerationCfg

    cfg = ClutterGenerationCfg(output=str(output), **kwargs)
    cfg.settle.num_steps = 480
    return cfg


def _read_records(path):
    return [json.loads(line)["variations"]["scene.relation_placement"] for line in path.read_text().splitlines()]


def _test_generation_writes_complete_layouts(simulation_app, tmp_path):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.clutter_generation import generate_clutter_layouts
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.validation.types import PlacementCheck
    from isaaclab_arena.tests.clutter.test_clutter_collection import _make_primitive_clutter_scene
    from isaaclab_arena.utils.pose import Pose

    arena_env = _make_primitive_clutter_scene(tmp_path)
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
    cfg = _arguments(output, num_envs=2, num_layouts=3, max_batches=2)
    # A solve can supply only one layout per environment. Later refills must still
    # satisfy an output request larger than the initial pool.
    arena_env.placer_params.max_placement_attempts = 1
    solve = PooledObjectPlacer._solve_env_ranked_layouts

    def one_layout_per_env(pool, count):
        ranked, layouts_per_env = solve(pool, count)
        return [layouts[:1] for layouts in ranked], layouts_per_env

    with patch.object(PooledObjectPlacer, "_solve_env_ranked_layouts", one_layout_per_env):
        assert generate_clutter_layouts(arena_env, cfg) == output
    records = _read_records(output)
    # Three requested records exercise a final batch with more accepted candidates than needed.
    assert [record["layout_id"] for record in records] == [f"layout_{i:06d}" for i in range(3)]
    assert records[0]["poses"]["cube_body"] != records[1]["poses"]["cube_body"]
    for record in records:
        assert record["source"] == "settled"
        assert set(record["poses"]) == {"cube_body", "table", "floor"}
        for value in record["poses"].values():
            Pose.from_dict(value)
        assert record["poses"]["cube_body"]["position_xyz"][2] == pytest.approx(0.57, abs=0.002)
        assert record["validation"]["pre_physics"] == {check: True for check in checks}
        reports = {report["check"]: report for report in record["validation"]["post_physics"]}
        assert all(reports[check]["passed"] for check in ("physics_settled", "pose_shift", "support_containment"))
        assert reports["articulation_link_shift"]["passed"] is None
        assert reports["support_containment"]["configuration"]["minimum_resting_heights_m"] == {}
        sampling = record["validation"]["sampling"]
        assert sampling["num_steps"] == 480
        assert sampling["decimation"] == 2
        assert sampling["physics_dt_s"] == pytest.approx(0.01)
        assert sampling["embodiment_keys"] == []

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
        generate_clutter_layouts(replay_env, cfg)
    assert not Path(cfg.output).exists()
    cfg.output = str(output)
    saved = output.read_bytes()
    with pytest.raises(AssertionError, match="Output already exists"):
        generate_clutter_layouts(_make_primitive_clutter_scene(tmp_path), cfg)
    assert output.read_bytes() == saved
    return True


def test_generation_writes_complete_layouts(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_generation_writes_complete_layouts, tmp_path=tmp_path)


def _test_generation_honors_required_solver_checks(simulation_app, tmp_path, available):
    from unittest.mock import patch

    from isaaclab_arena.offline_placement.clutter_generation import generate_clutter_layouts
    from isaaclab_arena.relations.validation.pre_physics import PrePhysicsPlacementValidator
    from isaaclab_arena.relations.validation.registry import PlacementValidatorRegistry
    from isaaclab_arena.relations.validation.types import PlacementCheck
    from isaaclab_arena.tests.clutter.test_clutter_collection import _make_primitive_clutter_scene

    validated_batches = []

    class RejectRelease(PrePhysicsPlacementValidator):
        check = "reject_release"

        def validate_batch(self, batch, collision_objects):
            validated_batches.append(len(batch))
            return [False] * len(batch)

    arena_env = _make_primitive_clutter_scene(tmp_path)
    checks = {PlacementCheck.NO_OVERLAP, PlacementCheck.CLUTTER_ON_RELATION, RejectRelease.check}
    arena_env.placer_params.enabled_checks = checks
    arena_env.placer_params.required_checks = checks
    arena_env.placer_params.max_placement_attempts = 1
    output = tmp_path / "rejected.jsonl"
    cfg = _arguments(output, max_batches=1)
    reason = "solver validation failed" if available else "missing required solver checks: reject_release"
    with (
        patch.dict(PlacementValidatorRegistry()._components, reject_release=RejectRelease),
        patch.object(RejectRelease, "is_available", return_value=available),
        patch("isaaclab_arena.offline_placement.pool_validation.physics_settle.step_physics") as step,
    ):
        with pytest.raises(AssertionError, match=reason):
            generate_clutter_layouts(arena_env, cfg)
        step.assert_not_called()
    assert bool(validated_batches) == available
    assert not output.exists()
    return True


@pytest.mark.parametrize("available", [True, False], ids=["rejected", "unavailable"])
def test_generation_honors_required_solver_checks(tmp_path, available):
    assert run_function_with_persistent_simulation_app(
        _test_generation_honors_required_solver_checks, tmp_path=tmp_path, available=available
    )


def _test_post_physics_checks_gate_output(simulation_app, tmp_path):
    from isaaclab_arena.offline_placement.clutter_generation import generate_clutter_layouts
    from isaaclab_arena.tests.clutter.test_clutter_collection import _make_primitive_clutter_scene

    output = tmp_path / "checked.jsonl"
    cfg = _arguments(output, num_envs=2, num_layouts=1, max_batches=1)
    cfg.settle.validators["reject_for_test"] = {
        "_target_": "isaaclab_arena.tests.clutter.test_clutter_generation.RejectPostPhysics",
        "threshold": 7.0,
    }
    with pytest.raises(AssertionError, match="reject_for_test: deliberately rejected"):
        generate_clutter_layouts(_make_primitive_clutter_scene(tmp_path), cfg)
    assert not output.exists()
    cfg.settle.validators["reject_for_test"]["enabled"] = False
    cfg.settle.validators["physics_settled"]["enabled"] = False
    cfg.settle.validators["support_containment"]["minimum_resting_heights_m"] = {"table": 0.02}
    generate_clutter_layouts(_make_primitive_clutter_scene(tmp_path), cfg)
    records = _read_records(output)
    assert len(records) == 1
    record = records[0]
    reports = {report["check"]: report for report in record["validation"]["post_physics"]}
    assert reports["reject_for_test"]["passed"] is None
    assert reports["reject_for_test"]["reason"]
    assert reports["reject_for_test"]["configuration"]["threshold"] == 7.0
    assert record["source"] == "settled"
    assert reports["physics_settled"]["passed"] is None
    assert reports["support_containment"]["passed"]
    assert reports["support_containment"]["configuration"]["minimum_resting_heights_m"] == {"table": 0.02}
    return True


def test_post_physics_checks_gate_output(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_post_physics_checks_gate_output, tmp_path=tmp_path)


def _test_generation_exhausts_settling_budget(simulation_app, tmp_path):
    from unittest.mock import patch

    from isaaclab_arena.offline_placement.clutter_generation import generate_clutter_layouts
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
    from isaaclab_arena.tests.clutter.test_clutter_collection import _make_primitive_clutter_scene

    output = tmp_path / "unsettled.jsonl"
    cfg = _arguments(output, max_batches=2)
    cfg.settle.num_steps = 1
    cfg.settle.validators["physics_settled"]["lin_vel_thresh"] = 0.0001
    with patch(
        "isaaclab_arena.offline_placement.settled_placement.collect_settled_placements",
        wraps=collect_settled_placements,
    ) as collect:
        with pytest.raises(AssertionError, match="physics_settled"):
            generate_clutter_layouts(_make_primitive_clutter_scene(tmp_path), cfg)
        assert collect.call_count == 2
    assert not output.exists()
    return True


def test_generation_exhausts_settling_budget(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_generation_exhausts_settling_budget, tmp_path=tmp_path)
