# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bind quotas to spawned members and compare them with native separate Runs."""

import json
from collections import Counter
from time import perf_counter
from unittest.mock import patch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _member_finished(env, members, member, length):
    import torch

    selected = torch.tensor([name == member for name in members], device=env.device)
    return selected & (env.episode_length_buf >= length)


def _builder_for_members(run_cfg, output_dir, member_names, random_choice=False, change_spawn=False):
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.sim import MassPropertiesCfg, RigidBodyPropertiesCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tests.test_object_set import _write_rigid_cube_usd

    output_dir.mkdir(parents=True, exist_ok=True)
    members = [
        Object(
            name=name,
            object_type=ObjectType.RIGID,
            usd_path=_write_rigid_cube_usd(output_dir / f"member_{index}.usd"),
            spawn_cfg_addon={
                "mass_props": MassPropertiesCfg(mass=1.25 if name == "A" else 2.5),
                "rigid_props": RigidBodyPropertiesCfg(disable_gravity=True),
            },
        )
        for index, name in enumerate(member_names)
    ]
    object_set = RigidObjectSet("fixture", objects=members, random_choice=random_choice)
    members_by_env = [member_names[i % len(member_names)] for i in range(run_cfg.environment_builder.num_envs)]

    def configure(env_cfg):
        env_cfg.decimation = 2
        env_cfg.sim.render_interval = 2
        env_cfg.terminations = {
            "success": TerminationTermCfg(
                func=_member_finished, params={"members": members_by_env, "member": "A", "length": 2}
            ),
            "time_out": TerminationTermCfg(
                func=_member_finished,
                params={"members": members_by_env, "member": "B", "length": 3},
                time_out=True,
            ),
        }
        if change_spawn:
            env_cfg.scene.fixture.spawn.assets_cfg[0].mass_props.mass = 100.0
        return env_cfg

    environment = IsaacLabArenaEnvironment(
        name="condition_object_set",
        scene=Scene(assets=[object_set]),
        embodiment=NoEmbodiment(),
        env_cfg_callback=configure,
    )
    return ArenaEnvBuilder(environment, run_cfg.environment_builder)


def _run_members(output_dir, member_names, quotas=None, num_episodes=None, max_steps=None, interrupt=False):
    import torch

    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
    from isaaclab_arena.evaluation import run_execution
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, RolloutLimitCfg
    from isaaclab_arena.evaluation.condition_episode_budget import ObjectSetEpisodeBudgetCfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg
    from isaaclab_arena.recording.episode_results import read_episode_records

    class ObservedZeroActionPolicy(ZeroActionPolicy):
        def __init__(self):
            super().__init__(ZeroActionPolicyCfg())
            self.steps = 0
            self.idle_slot_steps = 0
            self.reset_batches = []

        def reset(self, env_ids=None):
            self.reset_batches.append(None if env_ids is None else env_ids.tolist())
            super().reset(env_ids)

        def get_action(self, env, observation):
            if interrupt and self.steps == 2:
                raise KeyboardInterrupt()
            if self.steps == 0:
                masses = env.unwrapped.scene["fixture"].root_physx_view.get_masses().flatten().cpu()
                expected = [1.25 if member_names[i % len(member_names)] == "A" else 2.5 for i in range(3)]
                torch.testing.assert_close(masses, torch.tensor(expected))
            self.steps += 1
            self.idle_slot_steps += 3 - int(env.unwrapped.active_episode_mask.sum().item())
            return super().get_action(env, observation)

    policy = ObservedZeroActionPolicy()
    run = ArenaRunCfg(
        name="fixture",
        environment=ArenaEnvironmentCfg(),
        policy=ZeroActionPolicyCfg(),
        environment_builder=ArenaEnvBuilderCfg(num_envs=3, solve_relations=False),
        rollout_limit=RolloutLimitCfg(
            num_episodes=num_episodes,
            object_set_episodes=ObjectSetEpisodeBudgetCfg("fixture", quotas, max_steps) if quotas else None,
        ),
    )
    with (
        patch.object(
            run_execution,
            "build_arena_builder_from_run_cfg",
            side_effect=lambda cfg: _builder_for_members(cfg, output_dir, member_names),
        ),
        patch.object(run_execution, "_build_policy_from_cfg", return_value=policy),
    ):
        start = perf_counter()
        if interrupt:
            with pytest.raises(KeyboardInterrupt):
                run_execution.build_and_run(run, output_dir)
            result = None
        else:
            result = run_execution.build_and_run(run, output_dir)
        elapsed = perf_counter() - start
    records = read_episode_records(output_dir / "episode_results_rebuild0.jsonl")
    costs = {
        "build_rollout_cleanup_seconds": elapsed,
        "control_steps": policy.steps,
        "policy_slot_steps": 3 * policy.steps,
        "idle_slot_steps": policy.idle_slot_steps,
        "policy_reset_batches": policy.reset_batches,
    }
    return result, records, costs


def _test_shared_and_separate_runs(simulation_app, output_dir):
    from isaaclab_arena.evaluation.arena_run import RunStatus

    shared, shared_records, shared_costs = _run_members(output_dir / "shared", ["A", "B"], quotas={"A": 2, "B": 3})
    a, a_records, a_costs = _run_members(output_dir / "separate_A", ["A"], num_episodes=2)
    b, b_records, b_costs = _run_members(output_dir / "separate_B", ["B"], num_episodes=3)
    assert all(result.status is RunStatus.COMPLETED for result in (shared, a, b))
    assert shared.condition_episode_budget["members_by_env"] == ["A", "B", "A"]
    assert Counter(row["condition_member"] for row in shared_records) == {"A": 2, "B": 3}
    assert Counter(row["success"] for row in shared_records) == Counter(row["success"] for row in a_records + b_records)
    for member, separate_records in (("A", a_records), ("B", b_records)):
        shared_outcomes = Counter(
            (row["episode_length"], row["success"]) for row in shared_records if row["condition_member"] == member
        )
        assert shared_outcomes == Counter((row["episode_length"], row["success"]) for row in separate_records)
    assert shared_costs["control_steps"] == 9
    assert a_costs["control_steps"] + b_costs["control_steps"] == 5
    assert shared_costs["policy_reset_batches"] == [None, [1], [1]]
    assert all(row["terminated"] and not row["truncated"] for row in shared_records if row["condition_member"] == "A")
    assert all(row["truncated"] and not row["terminated"] for row in shared_records if row["condition_member"] == "B")
    (output_dir / "comparison.json").write_text(
        json.dumps(
            {
                "scope": "Deterministic native lifecycle fixture; excludes process and environment installation.",
                "shared": shared_costs,
                "separate_A": a_costs,
                "separate_B": b_costs,
            },
            indent=2,
        )
        + "\n"
    )
    return True


def test_shared_conditions_match_native_separate_runs(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_shared_and_separate_runs, output_dir=tmp_path)


def _test_stopped_condition_run(simulation_app, output_dir, interrupt):
    from isaaclab_arena.evaluation.arena_run import RunStatus

    result, records, _ = _run_members(
        output_dir,
        ["A", "B"],
        quotas={"A": 2, "B": 3},
        max_steps=None if interrupt else 2,
        interrupt=interrupt,
    )
    if not interrupt:
        assert result.status is RunStatus.INCOMPLETE
    state = json.loads((output_dir / "condition_episode_budget.json").read_text())
    assert state["status"] == "incomplete"
    assert state["sampling_complete"] is False
    assert state["conditions"]["A"]["completed"] == 2
    assert state["conditions"]["B"] == {"requested": 3, "started": 1, "completed": 0, "in_flight": 1, "unstarted": 2}
    assert state["unfinished_assignments"] == [{
        "env_id": 1,
        "condition_object_set": "fixture",
        "condition_member": "B",
        "episode_in_condition": 0,
    }]
    assert len(records) == 2 and all(row["condition_member"] == "A" for row in records)
    return True


@pytest.mark.parametrize("interrupt", [False, True])
def test_stopped_condition_run_preserves_completed_prefix(tmp_path, interrupt):
    assert run_function_with_persistent_simulation_app(
        _test_stopped_condition_run, output_dir=tmp_path, interrupt=interrupt
    )


def _test_invalid_condition_binding(simulation_app, output_dir, invalid):
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
    from isaaclab_arena.evaluation import run_execution
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, RolloutLimitCfg
    from isaaclab_arena.evaluation.condition_episode_budget import ObjectSetEpisodeBudgetCfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg

    quotas = {"A": 2} if invalid == "missing_quota" else {"A": 2, "B": 3}
    run = ArenaRunCfg(
        name="fixture",
        environment=ArenaEnvironmentCfg(),
        policy=ZeroActionPolicyCfg(),
        environment_builder=ArenaEnvBuilderCfg(num_envs=1 if invalid == "unreachable" else 3, solve_relations=False),
        rollout_limit=RolloutLimitCfg(object_set_episodes=ObjectSetEpisodeBudgetCfg("fixture", quotas)),
    )
    builder = _builder_for_members(
        run,
        output_dir,
        ["A", "A"] if invalid == "duplicate_name" else ["A", "B"],
        random_choice=invalid == "random",
        change_spawn=invalid == "changed_spawn",
    )
    with (
        patch.object(run_execution, "build_arena_builder_from_run_cfg", return_value=builder),
        patch.object(builder, "make_registered") as make,
        pytest.raises(AssertionError),
    ):
        run_execution._build_environment_from_cfg(run, None, output_dir)
    make.assert_not_called()
    return True


@pytest.mark.parametrize("invalid", ["missing_quota", "unreachable", "duplicate_name", "random", "changed_spawn"])
def test_invalid_condition_binding_rejected_before_environment_spawn(tmp_path, invalid):
    assert run_function_with_persistent_simulation_app(
        _test_invalid_condition_binding, output_dir=tmp_path, invalid=invalid
    )
