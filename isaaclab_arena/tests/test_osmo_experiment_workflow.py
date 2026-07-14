# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify OSMO workflows for complete Arena Experiments."""

import yaml
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf.errors import MissingMandatoryValue

from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg
from isaaclab_arena_openpi.policy import pi0_remote_policy  # noqa: F401
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from osmo.submit_arena_experiment_workflow import (
    POLICY_SERVER_WORKFLOWS,
    ArenaExperimentSubmissionCfg,
    compose_arena_experiment_submission,
    main,
    submit_arena_experiment_workflow,
)
from osmo.tasks.base_task import TaskCfg
from osmo.tasks.eval_runner_task import DEFAULT_EVAL_RUNNER_IMAGE, REMOTE_EXPERIMENT_PATH, EvalRunnerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.workflows.arena_experiment_workflow import ArenaExperimentWorkflow, Pi0ArenaExperimentWorkflow
from osmo.workflows.workflow import WorkflowCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT


def _pi0_experiment(first_variant: str = "pi05") -> dict:
    return {
        "runs": {
            "first": ArenaRunCfg(
                name="first",
                environment=PickAndPlaceMapleTableEnvironmentCfg(),
                policy=Pi0RemotePolicyCfg(
                    policy_variant=first_variant,
                    remote_host="user-host",
                    remote_port=9999,
                    ping_timeout=10,
                ),
            ),
            "second": ArenaRunCfg(
                name="second",
                environment=PickAndPlaceMapleTableEnvironmentCfg(),
                policy=Pi0RemotePolicyCfg(),
            ),
            "local": ArenaRunCfg(
                name="local",
                environment=PickAndPlaceMapleTableEnvironmentCfg(),
                policy=ZeroActionPolicyCfg(),
            ),
        }
    }


def _zero_action_experiment() -> dict:
    return {
        "runs": {
            "baseline": ArenaRunCfg(
                name="baseline",
                environment=PickAndPlaceMapleTableEnvironmentCfg(),
                policy=ZeroActionPolicyCfg(),
            )
        }
    }


def _task_file(task: dict, remote_path: str) -> dict:
    return next(file for file in task["files"] if file["path"] == remote_path)


def _embedded_experiment(task: dict) -> dict:
    experiment_file = _task_file(task, REMOTE_EXPERIMENT_PATH)
    assert "localpath" not in experiment_file
    return yaml.safe_load(experiment_file["contents"])


def _rendered_workflow(output: str) -> dict:
    return yaml.safe_load(output[output.index("version: 2\n") :])


def _workflow_tasks(workflow: dict) -> list[dict]:
    return workflow["workflow"]["groups"][0]["tasks"]


def test_declares_server_workflow():
    """Keep server dispatch explicit."""
    assert POLICY_SERVER_WORKFLOWS == {"pi0": Pi0ArenaExperimentWorkflow}
    assert ArenaExperimentWorkflow.task_cfg_type is EvalRunnerTaskCfg
    assert Pi0ArenaExperimentWorkflow.server_task_cfg_type is Pi0ServerTaskCfg


def test_server_config_group_composes_typed_defaults():
    """Compose the selected server directly from its typed configuration defaults."""
    submission_cfg = compose_arena_experiment_submission([
        "experiment_config=openpi_experiment",
        "server_config=pi0",
    ])

    assert submission_cfg.osmo_config == WorkflowCfg()
    assert submission_cfg.eval_runner_config == EvalRunnerTaskCfg()
    assert submission_cfg.server_config == Pi0ServerTaskCfg()

    with pytest.raises(ConfigCompositionException, match="unknown"):
        compose_arena_experiment_submission([
            "experiment_config=openpi_experiment",
            "server_config=unknown",
        ])

    with pytest.raises(AssertionError, match="policy_variant must be one of"):
        compose_arena_experiment_submission([
            "experiment_config=openpi_experiment",
            "server_config=pi0",
            "server_config.policy_variant=unknown",
        ])


def test_submitter_rejects_unregistered_server_config_type():
    """Reject a typed server config without a registered workflow implementation."""
    submission_cfg = ArenaExperimentSubmissionCfg(
        experiment_config=_pi0_experiment(),
        osmo_config=WorkflowCfg(dry_run=True),
        server_config=TaskCfg(),
    )

    with pytest.raises(AssertionError, match="No policy-server workflow.*TaskCfg"):
        submit_arena_experiment_workflow(submission_cfg)


@pytest.mark.parametrize("config_path", ["osmo_config.not_a_field", "eval_runner_config.not_a_field"])
def test_hydra_rejects_unknown_typed_config_fields(config_path):
    """Let the structured Hydra root reject fields outside their owning config."""
    with pytest.raises(ConfigCompositionException, match="not_a_field"):
        compose_arena_experiment_submission([
            "experiment_config=getting_started_experiment",
            f"{config_path}=true",
        ])


def test_server_config_rejects_workflow_fields():
    """Validate the selected server definition against its task config type."""
    with pytest.raises(ConfigCompositionException, match="workflow_name"):
        main([
            "experiment_config=openpi_experiment",
            "server_config=pi0",
            "server_config.workflow_name=experiment",
            "osmo_config.dry_run=true",
        ])


def test_renders_eval_runner_and_shared_pi0_server_with_effective_endpoints():
    """Wire every matching pi0 Run in the effective Experiment to one server."""
    source_experiment = _pi0_experiment()
    workflow = Pi0ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(workflow_name="pi0-experiment"),
        experiment_config=source_experiment,
        server_task_cfg=Pi0ServerTaskCfg(),
    )

    tasks = _workflow_tasks(workflow.generate_workflow())
    assert [task["name"] for task in tasks] == ["eval_runner", "policy_server"]
    assert [task["lead"] for task in tasks] == [True, False]

    eval_task = tasks[0]
    experiment = _embedded_experiment(eval_task)
    server_host = Pi0ServerTask.host_token()
    assert experiment["runs"]["first"]["policy"]["remote_host"] == server_host
    assert experiment["runs"]["first"]["policy"]["remote_port"] == POLICY_SERVER_PORT
    assert experiment["runs"]["second"]["policy"]["remote_host"] == server_host
    assert experiment["runs"]["second"]["policy"]["remote_port"] == POLICY_SERVER_PORT
    assert "remote_host" not in experiment["runs"]["local"]["policy"]
    assert "remote_port" not in experiment["runs"]["local"]["policy"]
    assert source_experiment["runs"]["first"].policy.remote_host == "user-host"

    command = _task_file(eval_task, "/tmp/entry.sh")["contents"]
    assert "eval_runner.py" in command
    assert "--enable_cameras" in command
    assert "policy_runner.py" not in command
    assert "runs." not in command

    server_command = _task_file(tasks[1], "/tmp/entry.sh")["contents"]
    assert f"scripts/serve_policy.py --port={POLICY_SERVER_PORT} policy:checkpoint" in server_command
    assert "--policy.config=pi05_droid_jointpos_polaris" in server_command


def test_embeds_effective_experiment_yaml():
    """Embed the composed Experiment instead of staging its source file."""
    experiment = _zero_action_experiment()
    workflow = ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(),
        experiment_config=experiment,
        task_cfg=EvalRunnerTaskCfg(image="registry.example.com/evaluator:typed-api"),
    )

    eval_task = _workflow_tasks(workflow.generate_workflow())[0]
    assert eval_task["image"] == "registry.example.com/evaluator:typed-api"
    embedded_experiment = _embedded_experiment(eval_task)
    assert embedded_experiment["runs"]["baseline"]["environment"]["type"] == "pick_and_place_maple_table"
    assert embedded_experiment["runs"]["baseline"]["policy"]["type"] == "zero_action"
    assert embedded_experiment["runs"]["baseline"]["environment_builder"]["num_envs"] == 1


def test_submission_removes_temporary_workflow(monkeypatch):
    """Submit one temporary workflow and remove it afterwards."""
    experiment = _zero_action_experiment()
    workflow = ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(),
        experiment_config=experiment,
    )
    captured_workflow_path = None

    def capture_submission(command):
        nonlocal captured_workflow_path
        assert command[:3] == ["osmo", "workflow", "submit"]
        captured_workflow_path = Path(command[3])
        assert captured_workflow_path.is_file()
        submitted_workflow = yaml.safe_load(captured_workflow_path.read_text(encoding="utf-8"))
        embedded_experiment = _embedded_experiment(_workflow_tasks(submitted_workflow)[0])
        assert embedded_experiment["runs"]["baseline"]["policy"]["type"] == "zero_action"
        return SimpleNamespace(returncode=23)

    monkeypatch.setattr("osmo.workflows.workflow.subprocess.run", capture_submission)

    assert workflow.submit_workflow() == 23
    assert captured_workflow_path is not None
    assert not captured_workflow_path.exists()


def test_cli_composes_named_groups_and_overrides(capsys):
    """Compose typed defaults and selected groups before applying CLI overrides."""
    return_code = main([
        "experiment_config=openpi_experiment",
        "server_config=pi0",
        "osmo_config.dry_run=true",
        "osmo_config.workflow_name=overridden-experiment",
        "eval_runner_config.image=registry.example.com/evaluator:branch",
        "server_config.image=registry.example.com/openpi:overridden",
        "server_config.policy_config=overridden-pi0-config",
        "server_config.policy_dir=gs://overridden/checkpoint",
        "experiment_config.runs.openpi_maple_table.rollout_limit.num_episodes=4",
        "experiment_config.runs.openpi_maple_table.environment_builder.num_envs=2",
        "experiment_config.runs.openpi_maple_table.policy.ping_interval=33.0",
        "experiment_config.runs.openpi_maple_table.policy.ping_timeout=450.0",
    ])

    assert return_code == 0
    rendered = capsys.readouterr().out
    assert "[dry-run] Rendered workflow YAML" in rendered
    workflow = _rendered_workflow(rendered)
    assert workflow["workflow"]["name"] == "overridden-experiment"
    tasks = _workflow_tasks(workflow)
    assert [task["name"] for task in tasks] == ["eval_runner", "policy_server"]
    assert tasks[0]["image"] == "registry.example.com/evaluator:branch"
    assert tasks[1]["image"] == "registry.example.com/openpi:overridden"

    experiment = _embedded_experiment(tasks[0])
    policy = experiment["runs"]["openpi_maple_table"]["policy"]
    assert experiment["runs"]["openpi_maple_table"]["rollout_limit"]["num_episodes"] == 4
    assert experiment["runs"]["openpi_maple_table"]["environment_builder"]["num_envs"] == 2
    assert policy["ping_interval"] == 33.0
    assert policy["ping_timeout"] == 450.0
    assert policy["remote_host"] == Pi0ServerTask.host_token()
    assert policy["remote_port"] == POLICY_SERVER_PORT
    assert "experiment_config.runs" not in _task_file(tasks[0], "/tmp/entry.sh")["contents"]

    server_command = _task_file(tasks[1], "/tmp/entry.sh")["contents"]
    assert "--policy.config=overridden-pi0-config" in server_command
    assert "--policy.dir=gs://overridden/checkpoint" in server_command


def test_embedded_openpi_experiment_composes_through_eval_runner_loader(tmp_path):
    """Keep the rendered OSMO handoff compatible with eval_runner's typed loader."""
    submission_cfg = compose_arena_experiment_submission([
        "experiment_config=openpi_experiment",
        "server_config=pi0",
    ])
    assert isinstance(submission_cfg.server_config, Pi0ServerTaskCfg)
    workflow = Pi0ArenaExperimentWorkflow(
        workflow_cfg=submission_cfg.osmo_config,
        experiment_config=submission_cfg.experiment_config,
        server_task_cfg=submission_cfg.server_config,
        task_cfg=submission_cfg.eval_runner_config,
    )
    experiment_path = tmp_path / "effective_experiment.yaml"
    experiment_file = _task_file(_workflow_tasks(workflow.generate_workflow())[0], REMOTE_EXPERIMENT_PATH)
    experiment_path.write_text(experiment_file["contents"], encoding="utf-8")

    experiment = load_arena_experiment_from_config_file(experiment_path, device="cuda:0")

    assert len(experiment) == 1
    assert isinstance(experiment[0].policy, Pi0RemotePolicyCfg)
    assert experiment[0].policy.remote_host == Pi0ServerTask.host_token()
    assert experiment[0].policy.remote_port == POLICY_SERVER_PORT


def test_cli_overrides_osmo_submission_resources(monkeypatch):
    """Apply scheduler overrides after the typed workflow defaults."""
    submitted_command = None
    submitted_resources = None

    def capture_submission(command):
        nonlocal submitted_command, submitted_resources
        submitted_command = command
        submitted_workflow = yaml.safe_load(Path(command[3]).read_text(encoding="utf-8"))
        submitted_resources = submitted_workflow["workflow"]["resources"]["default"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("osmo.workflows.workflow.subprocess.run", capture_submission)

    return_code = main([
        "experiment_config=getting_started_experiment",
        "osmo_config.pool=isaac-dev-l40-03",
        "osmo_config.platform=ovx-l40",
        "osmo_config.memory=120Gi",
    ])

    assert return_code == 0
    assert submitted_command is not None
    pool_flag_index = submitted_command.index("--pool")
    assert submitted_command[pool_flag_index + 1] == "isaac-dev-l40-03"
    assert submitted_resources["platform"] == "ovx-l40"
    assert submitted_resources["memory"] == "120Gi"


def test_cli_requires_experiment_group():
    """Require one named Experiment selection at the Hydra root."""
    with pytest.raises(MissingMandatoryValue, match="experiment_config"):
        compose_arena_experiment_submission([])


def test_structural_policy_override_is_checked_against_server_variant():
    """Check server compatibility against the effective structurally overridden policy."""
    with pytest.raises(AssertionError, match="pi0 server is configured for 'pi05'"):
        main([
            "experiment_config=openpi_experiment",
            "server_config=pi0",
            "osmo_config.dry_run=true",
            "experiment_config.runs.openpi_maple_table.policy={policy_variant:pi0}",
        ])


def test_server_variant_cannot_relabel_known_pi05_model():
    """Reject a client-compatible label when the selected server model remains pi05."""
    with pytest.raises(AssertionError, match="policy_config.*serves variant 'pi05'.*policy_variant 'pi0'"):
        main([
            "experiment_config=openpi_experiment",
            "server_config=pi0",
            "osmo_config.dry_run=true",
            "experiment_config.runs.openpi_maple_table.policy.policy_variant=pi0",
            "server_config.policy_variant=pi0",
        ])


def test_submitter_runs_zero_action_experiment_without_server(capsys):
    """Render exactly one eval-runner task when no policy server is selected."""
    submission_cfg = ArenaExperimentSubmissionCfg(
        experiment_config=_zero_action_experiment(),
        osmo_config=WorkflowCfg(dry_run=True),
    )

    assert submit_arena_experiment_workflow(submission_cfg) == 0

    tasks = _workflow_tasks(_rendered_workflow(capsys.readouterr().out))
    assert [task["name"] for task in tasks] == ["eval_runner"]
    assert tasks[0]["image"] == DEFAULT_EVAL_RUNNER_IMAGE


def test_submitter_rejects_server_without_matching_run():
    """Reject an explicitly selected server that cannot serve any Experiment Run."""
    submission_cfg = ArenaExperimentSubmissionCfg(
        experiment_config=_zero_action_experiment(),
        osmo_config=WorkflowCfg(dry_run=True),
        server_config=Pi0ServerTaskCfg(),
    )

    with pytest.raises(AssertionError, match="requires at least one Run with policy.type 'pi0_remote'"):
        submit_arena_experiment_workflow(submission_cfg)


def test_submitter_preserves_external_endpoint_without_server(capsys):
    """Preserve an externally hosted remote policy when no server is selected."""
    experiment = {
        "runs": {
            "externally_hosted": ArenaRunCfg(
                name="externally_hosted",
                environment=PickAndPlaceMapleTableEnvironmentCfg(),
                policy=Pi0RemotePolicyCfg(
                    remote_host="external.example.com",
                    remote_port=8123,
                ),
            )
        }
    }
    submission_cfg = ArenaExperimentSubmissionCfg(
        experiment_config=experiment,
        osmo_config=WorkflowCfg(dry_run=True),
    )

    assert submit_arena_experiment_workflow(submission_cfg) == 0

    task = _workflow_tasks(_rendered_workflow(capsys.readouterr().out))[0]
    embedded_policy = _embedded_experiment(task)["runs"]["externally_hosted"]["policy"]
    assert embedded_policy["type"] == "isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy"
    assert embedded_policy["remote_host"] == "external.example.com"
    assert embedded_policy["remote_port"] == 8123
    assert "{{host:" not in _task_file(task, "/tmp/entry.sh")["contents"]


def test_pi0_server_quotes_configurable_shell_values():
    """Keep configured checkpoint values within one shell argument."""
    task = Pi0ServerTask(
        Pi0ServerTaskCfg(
            policy_config="config with spaces",
            policy_dir="gs://bucket/checkpoint; false",
        )
    )

    command = task._get_run_script()

    assert "'--policy.config=config with spaces'" in command
    assert "'--policy.dir=gs://bucket/checkpoint; false'" in command
