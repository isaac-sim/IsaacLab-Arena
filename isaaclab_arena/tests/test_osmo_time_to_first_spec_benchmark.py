# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the OSMO workflow for the time-to-first-spec benchmark."""

from osmo.tasks.time_to_first_spec_benchmark_task import TimeToFirstSpecBenchmarkCfg
from osmo.workflows.time_to_first_spec_benchmark_workflow import (
    TimeToFirstSpecBenchmarkWorkflow,
    TimeToFirstSpecBenchmarkWorkflowCfg,
)


def _render_workflow(benchmark_cfg: TimeToFirstSpecBenchmarkCfg, exec_timeout: str = "2h") -> dict:
    workflow = TimeToFirstSpecBenchmarkWorkflow(
        workflow_cfg=TimeToFirstSpecBenchmarkWorkflowCfg(exec_timeout=exec_timeout),
        task_cfg=benchmark_cfg,
    )
    return workflow.generate_workflow()["workflow"]


def test_runs_every_case_in_one_cpu_task_with_the_inference_credential():
    workflow = _render_workflow(TimeToFirstSpecBenchmarkCfg(num_runs=3, inference_endpoint="internal"))

    (group,) = workflow["groups"]
    (task,) = group["tasks"]
    # OSMO rejects a workflow whose groups and tasks share a name.
    assert group["name"] != task["name"]
    assert workflow["resources"]["default"]["gpu"] == 0
    assert task["credentials"]["nvidia-inference-api"] == {
        "NVIDIA_API_KEY": "NVIDIA_API_KEY",
        "NV_API_KEY": "NV_API_KEY",
    }
    assert task["outputs"] == [{"url": TimeToFirstSpecBenchmarkWorkflowCfg().output_url}]
    run_script = task["files"][0]["contents"]
    assert "--num_runs 3 --output_dir '{{output}}' --inference_endpoint internal" in run_script
    assert "--model" not in run_script
    # A 2 h workflow leaves the benchmark 1.5 h, keeping 30 min to stop it and upload its output.
    assert "timeout --kill-after=60 5400 " in run_script
