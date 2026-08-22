# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify typed OSMO workflow construction and its compatibility CLI."""

import subprocess

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from osmo.submit_evaluation_workflow import main
from osmo.tasks.dreamzero_policy_runner_task import DreamZeroPolicyRunnerTaskCfg
from osmo.tasks.gr00t_policy_runner_task import DEFAULT_POLICY_CONFIG, Gr00tPolicyRunnerTaskCfg
from osmo.tasks.gr00t_server_task import Gr00tServerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.tasks.policy_runner_task import PolicyRunnerTaskCfg
from osmo.workflows.dreamzero_split_workflows import DreamZeroPolicyRunnerWorkflow
from osmo.workflows.server_plus_policy_runner_workflow import (
    CosmosPolicyRunnerWorkflow,
    Gr00tPolicyRunnerWorkflow,
    Pi0PlusPolicyRunnerWorkflow,
)
from osmo.workflows.workflow import WorkflowCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT


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


def test_cosmos_workflow_renders_policy_runner_and_server():
    """Pair the Cosmos policy-runner with the Cosmos server it connects to."""
    workflow = CosmosPolicyRunnerWorkflow(
        workflow_cfg=WorkflowCfg(),
        task_cfg=PolicyRunnerTaskCfg(arena_env="example_environment"),
    )

    tasks = workflow.generate_workflow()["workflow"]["groups"][0]["tasks"]
    policy_runner_command = tasks[0]["files"][0]["contents"]

    assert [task["name"] for task in tasks] == ["policy_runner", "cosmos_server"]
    assert "isaaclab_arena_cosmos.policy.cosmos_remote_policy.CosmosRemotePolicy" in policy_runner_command
    assert "--remote_host {{host:cosmos_server}}" in policy_runner_command
    assert f"--remote_port {POLICY_SERVER_PORT}" in policy_runner_command
    assert "action_policy_server_robolab" in tasks[1]["files"][0]["contents"]


def test_gr00t_workflow_renders_policy_runner_and_server():
    """Pair the GR00T policy-runner with the GR00T server it connects to."""
    workflow = Gr00tPolicyRunnerWorkflow(
        workflow_cfg=WorkflowCfg(),
        task_cfg=Gr00tPolicyRunnerTaskCfg(arena_env="example_environment"),
    )

    tasks = workflow.generate_workflow()["workflow"]["groups"][0]["tasks"]
    policy_runner_command = tasks[0]["files"][0]["contents"]
    server_command = tasks[1]["files"][0]["contents"]

    assert [task["name"] for task in tasks] == ["policy_runner", "gr00t_server"]
    assert tasks[1]["image"] == Gr00tServerTaskCfg().image
    assert (
        "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy"
        in policy_runner_command
    )
    assert f"--policy_config_yaml_path {DEFAULT_POLICY_CONFIG}" in policy_runner_command
    assert "--remote_host {{host:gr00t_server}}" in policy_runner_command
    assert f"--remote_port {POLICY_SERVER_PORT}" in policy_runner_command
    assert "gr00t/eval/run_gr00t_server.py" in server_command


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


def test_zero_action_workflow_does_not_require_gr00t():
    """Prevent the native-uv regression that broke zero-action workflows without GR00T."""
    child_script = (
        'import sys; sys.modules["gr00t"] = None; '
        "from osmo.submit_evaluation_workflow import main; "
        'raise SystemExit(main(["--policy", "zero_action", '
        '"--arena_env", "example_environment", "--dry_run"]))'
    )
    subprocess.run(
        [TestConstants.python_path, "-c", child_script],
        check=True,
        timeout=60,
    )
