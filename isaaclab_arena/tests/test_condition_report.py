# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep incomplete sampling visible without treating its prefix as a finished Run."""

import json

import pytest

from isaaclab_arena.evaluation.condition_episode_budget import ConditionEpisodeBudget, ObjectSetEpisodeBudgetCfg
from isaaclab_arena.visualization.report_data import RunExecutionReport, build_experiment_summary
from isaaclab_arena.visualization.report_render import render_index


def _condition_run(root, status, identity=0):
    budget = ConditionEpisodeBudget(ObjectSetEpisodeBudgetCfg("fruit", {"A": 2}), ["A"])
    budget.set_output_path(root / "condition" / "condition_episode_budget.json")
    budget.assign([0])
    budget.complete([0])
    if status == "completed":
        budget.assign([0])
        budget.complete([0])
    budget.write(status, "test")
    records = [
        {
            "env_id": 0,
            "episode_in_env": index,
            "condition_member": "A",
            "episode_in_condition": index,
            "success": True,
            "terminated": True,
            "truncated": False,
        }
        for index in range(budget.completed["A"])
    ]
    records[0]["episode_in_condition"] = identity
    (root / "condition" / "episode_results_rebuild0.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    complete = root / "separate"
    complete.mkdir()
    (complete / "episode_results_rebuild0.jsonl").write_text(
        json.dumps({"env_id": 0, "episode_in_env": 0, "success": False}) + "\n"
    )


@pytest.mark.parametrize(
    "budget_status,execution_status",
    [
        ("incomplete", "failed"),
        ("incomplete", "incomplete"),
        ("running", None),
        ("completed", "failed"),
    ],
)
def test_condition_prefix_does_not_change_pooled_results(tmp_path, budget_status, execution_status):
    _condition_run(tmp_path, budget_status)
    executions = [] if execution_status is None else [RunExecutionReport("condition", execution_status, 1)]
    summary = build_experiment_summary(tmp_path, "prefix", executions)
    assert summary.num_episodes == 1
    assert summary.overall_success_rate == 0.0
    assert [job.name for job in summary.jobs] == ["separate"]
    assert summary.condition_episodes["condition"][0].success is True
    page = render_index(summary, {task.name: "task.html" for task in summary.tasks})
    assert "condition: condition episodes" in page
    assert "excluded from pooled" in page


def test_completed_condition_run_still_contributes_to_pooled_results(tmp_path):
    _condition_run(tmp_path, "completed")
    summary = build_experiment_summary(tmp_path, "complete", [RunExecutionReport("condition", "completed", 0)])
    assert summary.num_episodes == 3
    assert summary.overall_success_rate == pytest.approx(2 / 3)


@pytest.mark.parametrize("identity", [[], {}, None, "0", -1, True, 0.0])
def test_malformed_condition_identity_is_reported_without_crashing(tmp_path, identity):
    _condition_run(tmp_path, "completed", identity)
    summary = build_experiment_summary(tmp_path, "malformed")
    page = render_index(summary, {task.name: "task.html" for task in summary.tasks})
    assert "Completion unverified" in page
