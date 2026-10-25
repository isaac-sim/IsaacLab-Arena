# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from isaaclab_arena.evaluation import prepare_settled_placement as preparation
from isaaclab_arena.evaluation.arena_run import SettledPlacementPreparationCfg
from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams
from isaaclab_arena.tests.test_run_execution import _run


def test_preparation_closes_sampling_env_and_preserves_settings(monkeypatch, tmp_path):
    from isaaclab_arena.scripts import record_placement_layouts

    settings = SettledPlacementPreparationCfg()
    env = SimpleNamespace(unwrapped=SimpleNamespace())
    builder = SimpleNamespace(
        cfg=SimpleNamespace(num_envs=2),
        arena_env=SimpleNamespace(env_cfg_callback=None, get_placement_assets=lambda: [], task=None, embodiment=None),
        get_all_variations=lambda: {},
        hydra_overrides=None,
        build_env_cfg=lambda: (SimpleNamespace(recorders="enabled"), {}),
    )
    events = []

    def make_registered(env_cfg, env_kwargs):
        assert env_cfg.recorders is None
        events.append("build")
        return env

    builder.make_registered = make_registered
    monkeypatch.setattr(preparation, "close_environment", lambda env: events.append("close"))

    def record(env, path, **kwargs):
        assert kwargs["min_layouts"] == 3
        assert kwargs["params"].include_embodiment_links
        events.append("record")
        return SimpleNamespace(accepted=3, attempted=4, rejections={(1, 0): "moving"})

    monkeypatch.setattr(record_placement_layouts, "record_placements_to_jsonl", record)
    preparation.prepare_settled_placement(builder, settings, 3, tmp_path / "settled_placements_rebuild1.jsonl")
    assert not settings.settle.include_embodiment_links
    assert events == ["build", "record", "close"]
    summary = json.loads((tmp_path / "settled_placements_rebuild1_summary.json").read_text())
    assert summary["accepted"] == 3
    assert summary["rejections"] == [{"env_id": 1, "batch_index": 0, "reason": "moving"}]

    events.clear()
    monkeypatch.setattr(
        record_placement_layouts,
        "record_placements_to_jsonl",
        lambda *args, **kwargs: SimpleNamespace(accepted=0, attempted=4, rejections={(0, 0): "moving"}),
    )
    with pytest.raises(AssertionError, match="Only 0/3"):
        preparation.prepare_settled_placement(builder, settings, 3, tmp_path / "settled_placements_rebuild2.jsonl")
    assert events == ["build", "close"]
    assert (tmp_path / "settled_placements_rebuild2_summary.json").exists()

    from isaaclab_arena.variations import variations_hydra

    variation = SimpleNamespace(enabled=False)
    builder.get_all_variations = lambda: {"robot": [variation]}
    builder.hydra_overrides = ["enable_variation"]
    monkeypatch.setattr(variations_hydra, "apply_overrides", lambda *args: setattr(variation, "enabled", True))
    with pytest.raises(AssertionError, match="variations to be disabled"):
        preparation.prepare_settled_placement(builder, settings, 3, tmp_path / "settled_placements_rebuild3.jsonl")


def test_preparation_requires_episode_budget_and_compatible_replay():
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.evaluation.arena_run import RolloutLimitCfg

    with pytest.raises(AssertionError, match="episode rollout limit"):
        _run(settled_placement=SettledPlacementPreparationCfg(), rollout_limit=RolloutLimitCfg(num_steps=5))
    with pytest.raises(AssertionError, match="cannot replace recorded replay"):
        _run(
            settled_placement=SettledPlacementPreparationCfg(),
            environment_builder=ArenaEnvBuilderCfg(recorded_variation_samples_path="existing.jsonl"),
        )
    from isaaclab_arena.evaluation.run_execution import _resolve_rollout_limit

    with pytest.raises(AssertionError, match="finite-action policy"):
        _resolve_rollout_limit(
            _run(settled_placement=SettledPlacementPreparationCfg()), SimpleNamespace(has_length=lambda: True), 3
        )
    with pytest.raises(AssertionError, match="required_settled_steps"):
        SettledPlacementParams(num_steps=2, required_settled_steps=3)


def test_velocity_validator_requires_sustained_rest():
    import torch

    from isaaclab_arena.offline_placement.post_physics_validation import VelocityValidator

    history = torch.zeros(3, 2, 6)
    history[0, 0, 0] = 0.2
    data = SimpleNamespace(
        env_ids=[0, 1], final_root_velocities={"cube": history[-1]}, root_velocity_history={"cube": history}
    )
    assert [report.passed for report in VelocityValidator().validate(data)] == [False, True]
    history[0, 0, 0] = 0.0
    assert all(report.passed for report in VelocityValidator().validate(data))


def _test_prepare_and_replay(simulation_app, tmp_path, with_robot=False):
    import torch
    import yaml
    from copy import deepcopy
    from unittest.mock import patch

    from isaaclab_arena.assets.registries import AssetRegistry, ensure_assets_registered
    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.evaluation import run_execution
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, RolloutLimitCfg, RunStatus
    from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg
    from isaaclab_arena.tests.test_settled_placement import _write_scene

    ensure_assets_registered()
    source = tmp_path / "scene.yaml"
    _write_scene(source)
    scene = yaml.safe_load(source.read_text())
    if with_robot:
        scene["embodiment"] = {"id": "robot", "registry_name": "franka_ik"}
        scene["relations"].append(
            {"kind": "at_position", "subject": "robot", "params": {"x": -1.0, "y": 0.0, "z": 0.0}}
        )
    source.write_text(yaml.safe_dump(scene))
    cfg = ArenaRunCfg(
        name="settled",
        environment=LegacyGraphEnvironmentCfg(env_spec_path=str(source)),
        policy=ZeroActionPolicyCfg(),
        environment_builder=ArenaEnvBuilderCfg(num_envs=2, env_spacing=2.0, device="cpu"),
        rollout_limit=RolloutLimitCfg(num_episodes=3),
        settled_placement=SettledPlacementPreparationCfg(max_batches=3),
    )
    original_cfg = deepcopy(cfg)
    if with_robot:
        from isaaclab_arena.embodiments.franka.franka import FrankaEventCfg

        source_events = FrankaEventCfg()
        compiled_cfg = SimpleNamespace(events=deepcopy(source_events))
        builder = SimpleNamespace(
            arena_env=SimpleNamespace(
                task=None, embodiment=SimpleNamespace(get_events_cfg=lambda: deepcopy(source_events))
            )
        )
        preparation.configure_preparation_resets(builder, compiled_cfg)
        assert source_events.randomize_franka_joint_state.params["std"] == 0.02
        assert compiled_cfg.events.randomize_franka_joint_state.params["std"] == 0.0
    observations = []

    class InspectFirstObservation(ZeroActionPolicy):
        def reset(self, env_ids=None):
            self.inspect_ids = env_ids
            self.inspect_next = True

        def get_action(self, env, observation):
            if self.inspect_next:
                body = env.unwrapped.scene["cube_body"]
                ids = slice(None) if self.inspect_ids is None else self.inspect_ids
                torch.testing.assert_close(
                    body.data.root_lin_vel_w.torch[ids],
                    torch.zeros_like(body.data.root_lin_vel_w.torch[ids]),
                    atol=1e-5,
                    rtol=0,
                )
                if with_robot:
                    robot = env.unwrapped.scene["robot"]
                    torch.testing.assert_close(
                        robot.data.joint_pos.torch[ids], robot.data.default_joint_pos.torch[ids], atol=1e-5, rtol=0
                    )
                observations.append(env.unwrapped.arena_world.get_pose_e("cube_body")[ids, :3].clone())
                self.inspect_next = False
            return super().get_action(env, observation)

    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    with (
        patch.dict(AssetRegistry()._components, {"recording_no_embodiment": NoEmbodiment}),
        patch.object(run_execution, "_build_policy_from_cfg", lambda cfg: InspectFirstObservation(cfg.policy)),
        patch.object(NoTask, "get_termination_cfg", lambda self: TaskTerminationCfg(timeout_s=0.1)),
    ):
        result = run_execution.build_and_run(cfg, tmp_path / "run")
    assert cfg == original_cfg
    assert result.status == RunStatus.COMPLETED
    assert len(observations) == 2  # Initial batch plus replacement episode in just one slot.
    output = tmp_path / "run"
    layouts = [json.loads(line) for line in (output / "settled_placements_rebuild0.jsonl").read_text().splitlines()]
    assert len(layouts) == 3
    expected = torch.tensor([row["placement"]["poses"]["cube_body"]["position_xyz"] for row in layouts])
    torch.testing.assert_close(torch.cat(observations), expected, atol=1e-5, rtol=0)
    assert all(row["placement"]["validation"]["sampling"]["required_settled_steps"] == 10 for row in layouts)
    assert len((output / "episode_results_rebuild0.jsonl").read_text().splitlines()) == 3
    assert json.loads((output / "settled_placements_rebuild0_summary.json").read_text())["accepted"] == 3
    return True


@pytest.mark.parametrize("with_robot", [False, True])
def test_prepare_and_replay(tmp_path, with_robot):
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(
        _test_prepare_and_replay, tmp_path=tmp_path, with_robot=with_robot
    )
