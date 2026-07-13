# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify OSMO workflows for complete Arena Experiments."""

import yaml
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf.errors import ConfigKeyError

from osmo.submit_arena_experiment_workflow import POLICY_SERVER_WORKFLOWS, main, submit_arena_experiment_workflow
from osmo.tasks.eval_runner_task import DEFAULT_EVAL_RUNNER_IMAGE, REMOTE_EXPERIMENT_PATH, EvalRunnerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.workflows.arena_experiment_workflow import ArenaExperimentWorkflow, Pi0ArenaExperimentWorkflow
from osmo.workflows.workflow import WorkflowCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT


def _write_pi0_experiment(path: Path, first_variant: str = "pi05") -> bytes:
    source = f"""# Preserve this source exactly.
runs:
  first:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: pi0_remote
      policy_variant: {first_variant}
  second:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: pi0_remote
  local:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action
""".encode()
    path.write_bytes(source)
    return source


def _task_file(task: dict, remote_path: str) -> dict:
    return next(file for file in task["files"] if file["path"] == remote_path)


def _rendered_workflow(output: str) -> dict:
    return yaml.safe_load(output[output.index("version: 2\n") :])


def _workflow_tasks(workflow: dict) -> list[dict]:
    return workflow["workflow"]["groups"][0]["tasks"]


def test_declares_server_workflow():
    """Keep server dispatch explicit."""
    assert POLICY_SERVER_WORKFLOWS == {"pi0": Pi0ArenaExperimentWorkflow}
    assert ArenaExperimentWorkflow.task_cfg_type is EvalRunnerTaskCfg
    assert Pi0ArenaExperimentWorkflow.server_task_cfg_type is Pi0ServerTaskCfg


def test_policy_server_config_requires_known_type(tmp_path):
    """Require an explicit supported selector rather than guessing from config fields or filenames."""
    experiment_path = tmp_path / "experiment.yaml"
    _write_pi0_experiment(experiment_path)
    server_path = tmp_path / "server.yaml"
    server_path.write_text("image: registry.example.com/server:test\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="must define a non-empty string 'type'"):
        submit_arena_experiment_workflow(experiment_path, server_config=server_path, dry_run=True)

    server_path.write_text("type: unknown\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="Unknown policy server type 'unknown'"):
        submit_arena_experiment_workflow(experiment_path, server_config=server_path, dry_run=True)

    server_path.write_text("type: pi0\npolicy_variant: unknown\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="policy_variant must be one of"):
        submit_arena_experiment_workflow(experiment_path, server_config=server_path, dry_run=True)


def test_workflow_and_server_configs_reject_each_others_fields(tmp_path):
    """Keep scheduler, server, and Experiment-policy fields within their owning configs."""
    experiment_path = tmp_path / "experiment.yaml"
    _write_pi0_experiment(experiment_path)
    workflow_path = tmp_path / "osmo.yaml"
    workflow_path.write_text("workflow_name: experiment\npolicy_variant: pi05\n", encoding="utf-8")
    with pytest.raises(ConfigKeyError, match="policy_variant"):
        submit_arena_experiment_workflow(experiment_path, osmo_config=workflow_path, dry_run=True)

    with pytest.raises(ConfigKeyError, match="policy_variant"):
        submit_arena_experiment_workflow(
            experiment_path,
            osmo_config_overrides=["policy_variant=pi05"],
            dry_run=True,
        )

    server_path = tmp_path / "server.yaml"
    server_path.write_text("type: pi0\nworkflow_name: experiment\n", encoding="utf-8")
    with pytest.raises(ConfigKeyError, match="workflow_name"):
        submit_arena_experiment_workflow(experiment_path, server_config=server_path, dry_run=True)

    server_path.write_text("type: pi0\nclient_ping_timeout: 450.0\n", encoding="utf-8")
    with pytest.raises(ConfigKeyError, match="client_ping_timeout"):
        submit_arena_experiment_workflow(experiment_path, server_config=server_path, dry_run=True)

    server_path.write_text("type: pi0\n", encoding="utf-8")
    with pytest.raises(ConfigKeyError, match="workflow_name"):
        submit_arena_experiment_workflow(
            experiment_path,
            server_config=server_path,
            server_config_overrides=["workflow_name=experiment"],
            dry_run=True,
        )


def test_renders_eval_runner_and_shared_pi0_server_with_trailing_endpoints(tmp_path):
    """Wire every matching pi0 Run to one server after all user Experiment overrides."""
    experiment_path = tmp_path / "experiment.yaml"
    _write_pi0_experiment(experiment_path)
    user_overrides = [
        "runs.first.policy.remote_host=user-host",
        "runs.first.policy.remote_port=9999",
        "runs.first.policy.ping_timeout=10",
    ]
    workflow = Pi0ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(workflow_name="pi0-experiment"),
        experiment_config_path=experiment_path,
        server_task_cfg=Pi0ServerTaskCfg(),
        experiment_overrides=user_overrides,
    )

    tasks = _workflow_tasks(workflow.generate_workflow())
    assert [task["name"] for task in tasks] == ["eval_runner", "policy_server"]
    assert [task["lead"] for task in tasks] == [True, False]

    eval_task = tasks[0]
    assert _task_file(eval_task, REMOTE_EXPERIMENT_PATH) == {
        "localpath": str(experiment_path.resolve()),
        "path": REMOTE_EXPERIMENT_PATH,
    }
    command = _task_file(eval_task, "/tmp/entry.sh")["contents"]
    assert "eval_runner.py" in command
    assert "--enable_cameras" in command
    assert "policy_runner.py" not in command
    assert command.index("runs.first.policy.remote_host=user-host") < command.index(
        'runs.first.policy.remote_host="{{host:policy_server}}"'
    )
    assert f"runs.second.policy.remote_port={POLICY_SERVER_PORT}" in command
    assert "runs.local.policy.remote_host" not in command
    assert "runs.local.policy.remote_port" not in command
    assert "runs.first.policy.ping_timeout=10" in command
    assert "runs.second.policy.ping_timeout" not in command

    server_command = _task_file(tasks[1], "/tmp/entry.sh")["contents"]
    assert f"scripts/serve_policy.py --port={POLICY_SERVER_PORT} policy:checkpoint" in server_command
    assert "--policy.config=pi05_droid_jointpos_polaris" in server_command


def test_references_experiment_source_by_absolute_path(tmp_path):
    """Reference the Experiment directly so OSMO can upload it during submission."""
    experiment_path = tmp_path / "experiment.yaml"
    _write_pi0_experiment(experiment_path)
    workflow = ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(),
        experiment_config_path=experiment_path,
        task_cfg=EvalRunnerTaskCfg(image="registry.example.com/evaluator:typed-api"),
    )

    eval_task = _workflow_tasks(workflow.generate_workflow())[0]
    assert eval_task["image"] == "registry.example.com/evaluator:typed-api"
    assert _task_file(eval_task, REMOTE_EXPERIMENT_PATH) == {
        "localpath": str(experiment_path.resolve()),
        "path": REMOTE_EXPERIMENT_PATH,
    }


def test_submission_removes_temporary_workflow(tmp_path, monkeypatch):
    """Submit one temporary workflow and remove it afterwards."""
    experiment_path = tmp_path / "experiment.yaml"
    _write_pi0_experiment(experiment_path)
    workflow = ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(),
        experiment_config_path=experiment_path,
    )
    captured_workflow_path = None

    def capture_submission(command):
        nonlocal captured_workflow_path
        assert command[:3] == ["osmo", "workflow", "submit"]
        captured_workflow_path = Path(command[3])
        assert captured_workflow_path.is_file()
        submitted_workflow = yaml.safe_load(captured_workflow_path.read_text(encoding="utf-8"))
        eval_task = _workflow_tasks(submitted_workflow)[0]
        assert _task_file(eval_task, REMOTE_EXPERIMENT_PATH)["localpath"] == str(experiment_path.resolve())
        return SimpleNamespace(returncode=23)

    monkeypatch.setattr("osmo.workflows.workflow.subprocess.run", capture_submission)

    assert workflow.submit_workflow() == 23
    assert captured_workflow_path is not None
    assert not captured_workflow_path.exists()


def test_cli_dry_run_composes_experiment_server_and_osmo_configs(tmp_path, capsys):
    """Compose independently selected Experiment, policy-server, and OSMO configs."""
    experiment_path = tmp_path / "experiment.yaml"
    _write_pi0_experiment(experiment_path)
    osmo_path = tmp_path / "osmo.yaml"
    osmo_path.write_text("workflow_name: cli-experiment\n", encoding="utf-8")
    server_path = tmp_path / "server.yaml"
    server_path.write_text(
        """type: pi0
image: registry.example.com/openpi:test
policy_variant: pi05
policy_config: custom-pi0-config
""",
        encoding="utf-8",
    )

    return_code = main([
        "--experiment_config",
        str(experiment_path),
        "--server_config",
        str(server_path),
        "--osmo_config",
        str(osmo_path),
        "--dry_run",
        "osmo_config.workflow_name=overridden-experiment",
        "eval_runner_config.image=registry.example.com/evaluator:branch",
        "server_config.image=registry.example.com/openpi:overridden",
        "server_config.policy_config=overridden-pi0-config",
        "server_config.policy_dir=gs://overridden/checkpoint",
        "runs.first.rollout_limit.num_steps=4",
        "runs.first.policy.ping_timeout=450.0",
    ])

    assert return_code == 0
    rendered = capsys.readouterr().out
    assert "[dry-run] Rendered workflow YAML" in rendered
    assert "name: overridden-experiment" in rendered
    assert "name: cli-experiment" not in rendered
    assert "name: eval_runner" in rendered
    assert "name: policy_server" in rendered
    tasks = _workflow_tasks(_rendered_workflow(rendered))
    assert tasks[0]["image"] == "registry.example.com/evaluator:branch"
    assert tasks[1]["image"] == "registry.example.com/openpi:overridden"
    assert "image: registry.example.com/openpi:overridden" in rendered
    assert "image: registry.example.com/openpi:test" not in rendered
    assert "--policy.config=overridden-pi0-config" in rendered
    assert "--policy.config=custom-pi0-config" not in rendered
    assert "--policy.dir=gs://overridden/checkpoint" in rendered
    assert "runs.first.policy.ping_timeout=450.0" in rendered
    assert "runs.first.rollout_limit.num_steps=4" in rendered


def test_cli_overrides_osmo_submission_pool(tmp_path, monkeypatch):
    """Apply a namespaced OSMO override after the workflow configuration YAML."""
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """runs:
  baseline:
    environment:
      type: test
    policy:
      type: zero_action
""",
        encoding="utf-8",
    )
    osmo_path = tmp_path / "osmo.yaml"
    osmo_path.write_text("pool: yaml-pool\n", encoding="utf-8")
    submitted_command = None

    def capture_submission(command):
        nonlocal submitted_command
        submitted_command = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("osmo.workflows.workflow.subprocess.run", capture_submission)

    return_code = main([
        "--experiment_config",
        str(experiment_path),
        "--osmo_config",
        str(osmo_path),
        "osmo_config.pool=isaac-dev-l40-03",
    ])

    assert return_code == 0
    assert submitted_command is not None
    pool_flag_index = submitted_command.index("--pool")
    assert submitted_command[pool_flag_index + 1] == "isaac-dev-l40-03"


def test_cli_rejects_server_override_without_server_definition(tmp_path, capsys):
    """Require a server definition before applying namespaced server overrides."""
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """runs:
  baseline:
    environment:
      type: test
    policy:
      type: zero_action
""",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        main([
            "--experiment_config",
            str(experiment_path),
            "server_config.policy_config=my-config",
        ])

    assert "server_config.* overrides require --server_config" in capsys.readouterr().err


def test_pi0_workflow_rejects_variant_mismatch_and_uses_last_exact_override(tmp_path):
    """Compare effective per-Run variants, including the last exact override."""
    experiment_path = tmp_path / "experiment.yaml"
    _write_pi0_experiment(experiment_path, first_variant="pi0")

    with pytest.raises(AssertionError, match="pi0 server is configured for 'pi05'"):
        Pi0ArenaExperimentWorkflow(
            workflow_cfg=WorkflowCfg(),
            experiment_config_path=experiment_path,
            server_task_cfg=Pi0ServerTaskCfg(policy_variant="pi05"),
        )

    Pi0ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(),
        experiment_config_path=experiment_path,
        server_task_cfg=Pi0ServerTaskCfg(policy_variant="pi05"),
        experiment_overrides=[
            "runs.first.policy.policy_variant=pi0",
            "runs.first.policy.policy_variant=pi05",
        ],
    )


def test_submitter_runs_zero_action_experiment_without_server(tmp_path, capsys):
    """Render exactly one eval-runner task when no policy-server definition is supplied."""
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """runs:
  baseline:
    environment:
      type: test
    policy:
      type: zero_action
""",
        encoding="utf-8",
    )

    assert submit_arena_experiment_workflow(experiment_path, dry_run=True) == 0

    tasks = _workflow_tasks(_rendered_workflow(capsys.readouterr().out))
    assert [task["name"] for task in tasks] == ["eval_runner"]
    assert tasks[0]["image"] == DEFAULT_EVAL_RUNNER_IMAGE


def test_submitter_rejects_server_without_matching_run(tmp_path):
    """Reject an explicitly selected server that cannot serve any Experiment Run."""
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """runs:
  baseline:
    environment:
      type: test
    policy:
      type: zero_action
""",
        encoding="utf-8",
    )
    server_path = tmp_path / "server.yaml"
    server_path.write_text("type: pi0\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="requires at least one Run with policy.type 'pi0_remote'"):
        submit_arena_experiment_workflow(
            experiment_path,
            server_config=server_path,
            dry_run=True,
        )


def test_submitter_does_not_inject_endpoint_without_server(tmp_path, capsys):
    """Preserve an externally hosted remote policy when no co-scheduled server is requested."""
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """runs:
  externally_hosted:
    environment:
      type: test
    policy:
      type: pi0_remote
      remote_host: external.example.com
      remote_port: 8123
""",
        encoding="utf-8",
    )

    assert submit_arena_experiment_workflow(experiment_path, dry_run=True) == 0

    tasks = _workflow_tasks(_rendered_workflow(capsys.readouterr().out))
    assert [task["name"] for task in tasks] == ["eval_runner"]
    command = _task_file(tasks[0], "/tmp/entry.sh")["contents"]
    assert "runs.externally_hosted.policy.remote_host=" not in command
    assert "runs.externally_hosted.policy.remote_port=" not in command
    assert "{{host:" not in command


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
