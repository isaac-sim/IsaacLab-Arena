# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the OSMO workflow for the time-to-resolved-layout benchmark and its result collector."""

import json

import pytest

from isaaclab_arena_examples.agentic_environment_generation.benchmark.run_time_to_resolved_layout_benchmarks import (
    DEFAULT_ENV_SPECS_DIR,
    find_benchmark_cases,
)
from osmo.scripts.combine_time_to_resolved_layout_results import combine_results
from osmo.tasks.time_to_resolved_layout_benchmark_task import TimeToResolvedLayoutBenchmarkCfg
from osmo.workflows.time_to_resolved_layout_benchmark_workflow import (
    TimeToResolvedLayoutBenchmarkWorkflow,
    TimeToResolvedLayoutBenchmarkWorkflowCfg,
)

_CASE = "tabletop_banana_plate_distractors_14"


def _render_groups(benchmark_cfg: TimeToResolvedLayoutBenchmarkCfg, exec_timeout: str = "2h") -> list[dict]:
    workflow = TimeToResolvedLayoutBenchmarkWorkflow(
        workflow_cfg=TimeToResolvedLayoutBenchmarkWorkflowCfg(exec_timeout=exec_timeout),
        task_cfg=benchmark_cfg,
    )
    return workflow.generate_workflow()["workflow"]["groups"]


def test_runs_each_case_and_environment_count_in_its_own_group_before_collecting_results():
    groups = _render_groups(TimeToResolvedLayoutBenchmarkCfg(cases=[_CASE], num_envs=[1, 16], num_runs=3))

    assert [group["name"] for group in groups] == [
        "layout-benchmark-0",
        "layout-benchmark-1",
        "layout-benchmark-results",
    ]
    # OSMO rejects a workflow whose groups and tasks share a name.
    names = [group["name"] for group in groups]
    for group in groups:
        names += [task["name"] for task in group["tasks"]]
    assert len(set(names)) == len(names)
    run_script = groups[1]["tasks"][0]["files"][0]["contents"]
    assert f"--env_spec isaaclab_arena_environments/agentic_env_gen_benchmark/{_CASE}.yaml" in run_script
    assert "--num_envs 16 --num_runs 3 --warmup_runs 1 --placement_seed 42" in run_script
    # A 2 h group leaves the benchmark 1.5 h, keeping 30 min to stop it and upload its output.
    assert "timeout --kill-after=60 5400 " in run_script
    assert groups[1]["tasks"][0]["outputs"] == []

    collect_task = groups[2]["tasks"][0]
    assert collect_task["inputs"] == [{"task": "layout-benchmark-runner-0"}, {"task": "layout-benchmark-runner-1"}]
    benchmark = json.loads(collect_task["files"][2]["contents"])
    assert benchmark["configurations"][1] == {
        "name": f"{_CASE}_envs_16",
        "case": _CASE,
        "num_envs": 16,
        "task_output_directory": "{{input:layout-benchmark-runner-1}}",
    }


def test_empty_case_list_selects_every_benchmark_spec():
    groups = _render_groups(TimeToResolvedLayoutBenchmarkCfg(num_envs=[1]))

    assert len(groups) == len(find_benchmark_cases(DEFAULT_ENV_SPECS_DIR)) + 1


def test_rejects_unknown_case():
    with pytest.raises(AssertionError, match="unknown benchmark cases"):
        _render_groups(TimeToResolvedLayoutBenchmarkCfg(cases=["not_a_case"]))


def test_combined_result_marks_failed_and_missing_configurations(tmp_path):
    configurations = []
    for name, num_envs, process_exit_code in (
        ("case_envs_1", 1, 0),
        ("case_envs_16", 16, 1),
        ("case_envs_64", 64, None),
    ):
        task_output_directory = tmp_path / name
        task_output_directory.mkdir()
        if process_exit_code is not None:
            (task_output_directory / f"{name}.json").write_text(
                json.dumps({"results_by_num_envs": {str(num_envs): {"summary": {}, "samples": []}}}),
                encoding="utf-8",
            )
            (task_output_directory / f"{name}_task_result.json").write_text(
                json.dumps({"process_exit_code": process_exit_code}), encoding="utf-8"
            )
        (task_output_directory / f"{name}.log").write_text("log\n", encoding="utf-8")
        configurations.append({
            "name": name,
            "case": "case",
            "num_envs": num_envs,
            "task_output_directory": str(task_output_directory),
        })
    benchmark = {
        "num_runs": 1,
        "warmup_runs": 0,
        "placement_seed": 42,
        "env_specs_dir": "specs",
        "cases": ["case"],
        "num_envs": [1, 16, 64],
        "configurations": configurations,
    }
    output_directory = tmp_path / "combined"

    combined = combine_results(benchmark, output_directory)

    assert combined["failed_configurations"] == ["case_envs_16", "case_envs_64"]
    assert [result["child_returncode"] for result in combined["results"]] == [0, 1]
    assert json.loads((output_directory / "all_results.json").read_text(encoding="utf-8")) == combined
    assert (output_directory / "case_envs_1.json").is_file()
    assert (output_directory / "case_envs_64.log").is_file()
