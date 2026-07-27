# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify typed OSMO workflow construction and its compatibility CLI."""

import pytest

from osmo.submit_evaluation_workflow import main
from osmo.tasks.dreamzero_policy_runner_task import DreamZeroPolicyRunnerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.tasks.policy_runner_task import PolicyRunnerTaskCfg
from osmo.workflows.dreamzero_split_workflows import DreamZeroPolicyRunnerWorkflow
from osmo.workflows.server_plus_policy_runner_workflow import Pi0PlusPolicyRunnerWorkflow
from osmo.workflows.workflow import WorkflowCfg


def test_task_name_is_a_required_keyword_argument():
    """Reject construction when a workflow does not name its task instance."""
    with pytest.raises(TypeError, match="task_name"):
        Pi0ServerTask(Pi0ServerTaskCfg())


def test_typed_workflow_config_renders_policy_runner_and_server():
    """Construct a multi-task workflow without passing through argparse."""
    workflow = Pi0PlusPolicyRunnerWorkflow(
        workflow_cfg=WorkflowCfg(workflow_name="typed-evaluation"),
        task_cfg=PolicyRunnerTaskCfg(
            arena_env="example_environment",
            policy_runner_args=["--num_envs", "2"],
            variation_args=["light.hdr_image.enabled=true"],
        ),
    )

    workflow_dict = workflow.generate_workflow()
    tasks = workflow_dict["workflow"]["groups"][0]["tasks"]
    policy_runner_command = tasks[0]["files"][0]["contents"]

    assert workflow_dict["workflow"]["name"] == "typed-evaluation"
    assert [task["name"] for task in tasks] == ["policy_runner", "policy_server"]
    assert "--remote_host {{host:policy_server}}" in policy_runner_command
    assert "--ping_timeout 300" in policy_runner_command
    assert "--num_envs 2" in policy_runner_command
    assert "example_environment light.hdr_image.enabled=true" in policy_runner_command


def test_static_workflow_threads_declared_task_names_into_host_token():
    """Use the same explicit server name for its task and the runner's host token."""

    class CustomNamedPi0Workflow(Pi0PlusPolicyRunnerWorkflow):
        task_names = ["custom-runner", "custom-server"]

    workflow = CustomNamedPi0Workflow(
        workflow_cfg=WorkflowCfg(),
        task_cfg=PolicyRunnerTaskCfg(arena_env="example_environment"),
    )

    tasks = workflow.generate_workflow()["workflow"]["groups"][0]["tasks"]
    assert [task["name"] for task in tasks] == ["custom-runner", "custom-server"]
    assert "--remote_host {{host:custom-server}}" in tasks[0]["files"][0]["contents"]


def test_dreamzero_runner_quotes_explicit_server_task_name():
    """Quote the submitted server task name in DreamZero's port-forward command."""
    workflow = DreamZeroPolicyRunnerWorkflow(
        workflow_cfg=WorkflowCfg(),
        task_cfg=DreamZeroPolicyRunnerTaskCfg(arena_env="example_environment"),
        server_workflow_id="server-workflow-id",
        server_task_name="custom-server; false",
    )

    task = workflow.generate_workflow()["workflow"]["groups"][0]["tasks"][0]
    assert task["name"] == "policy_runner"
    assert "port-forward server-workflow-id 'custom-server; false'" in task["files"][0]["contents"]


def test_compatibility_cli_builds_typed_config(capsys):
    """Keep the submission CLI as a thin adapter around typed workflow configs."""
    return_code = main([
        "--policy",
        "zero_action",
        "--arena_env",
        "example_environment",
        "--priority",
        "HIGH",
        "--dry_run",
    ])

    assert return_code == 0
    rendered = capsys.readouterr().out
    assert "[dry-run] Rendered workflow YAML" in rendered
    assert "name: policy_runner" in rendered
    assert "example_environment" in rendered
