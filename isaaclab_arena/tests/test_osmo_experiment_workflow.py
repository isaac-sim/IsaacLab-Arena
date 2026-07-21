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

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg
from isaaclab_arena_openpi.policy import pi0_remote_policy  # noqa: F401
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from osmo.submit_arena_experiment import (
    POLICY_SERVER_TASK_CFG_BY_NAME,
    POLICY_SERVER_WORKFLOW_BY_CONFIG_TYPE,
    ArenaExperimentSubmissionCfg,
    build_arena_experiment_submission_cfg,
    main,
    submit_arena_experiment,
)
from osmo.tasks.base_task import TaskCfg
from osmo.tasks.experiment_runner_task import REMOTE_EXPERIMENT_PATH, ExperimentRunnerTask, ExperimentRunnerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.workflows.arena_experiment_workflow import Pi0ArenaExperimentWorkflow
from osmo.workflows.workflow import WorkflowCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT

# Composing complete Arena Experiments loads Isaac runtime modules, so these tests
# must not share a pytest process with the persistent SimulationApp tests.
pytestmark = pytest.mark.with_subprocess

REPOSITORY_ROOT = Path(__file__).parents[2]
OPENPI_EXPERIMENT_CFG_PATH = (
    REPOSITORY_ROOT / "isaaclab_arena_environments/experiment_configs/droid_pnp_srl_openpi_experiment.yaml"
)
OPENPI_RUN_NAME = "droid_pnp_srl_openpi_billiard_hall"


def _pi0_experiment_cfg(first_variant: str = "pi05") -> ArenaExperimentCfg:
    return ArenaExperimentCfg(
        runs={
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
    )


def _zero_action_experiment_cfg() -> ArenaExperimentCfg:
    return ArenaExperimentCfg(
        runs={
            "baseline": ArenaRunCfg(
                name="baseline",
                environment=PickAndPlaceMapleTableEnvironmentCfg(),
                policy=ZeroActionPolicyCfg(),
            )
        }
    )


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


def _compose_submission(
    overrides: list[str] | None = None,
    experiment_cfg_path: Path = OPENPI_EXPERIMENT_CFG_PATH,
    policy_server_name: str = "pi0",
) -> ArenaExperimentSubmissionCfg:
    return build_arena_experiment_submission_cfg(
        experiment_cfg_path,
        policy_server_name,
        overrides,
    )


def _compose_and_submit(
    overrides: list[str],
    experiment_cfg_path: Path = OPENPI_EXPERIMENT_CFG_PATH,
    policy_server_name: str = "pi0",
) -> int:
    return submit_arena_experiment(_compose_submission(overrides, experiment_cfg_path, policy_server_name))


def test_declares_policy_server_name_and_workflow_mappings():
    """Keep policy-server selection and runtime workflow dispatch explicit."""
    assert POLICY_SERVER_TASK_CFG_BY_NAME == {"pi0": Pi0ServerTaskCfg}
    assert POLICY_SERVER_WORKFLOW_BY_CONFIG_TYPE == {Pi0ServerTaskCfg: Pi0ArenaExperimentWorkflow}
    assert Pi0ArenaExperimentWorkflow.task_cfg_type is ExperimentRunnerTaskCfg
    assert Pi0ArenaExperimentWorkflow.server_task_cfg_type is Pi0ServerTaskCfg


def test_explicit_experiment_and_policy_server_selector_compose_typed_defaults():
    """Compose an explicit Experiment path and built-in policy-server defaults."""
    submission_cfg = _compose_submission()

    assert isinstance(submission_cfg.experiment_cfg, ArenaExperimentCfg)
    assert len(submission_cfg.experiment_cfg.runs) == 9
    assert isinstance(submission_cfg.experiment_cfg.runs[OPENPI_RUN_NAME].policy, Pi0RemotePolicyCfg)
    assert submission_cfg.osmo == WorkflowCfg()
    assert submission_cfg.osmo.pool == "isaac-dev-l40s-04"
    assert submission_cfg.osmo.platform == "ovx-l40s"
    assert submission_cfg.experiment_runner == ExperimentRunnerTaskCfg()
    assert submission_cfg.experiment_runner.image == "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
    assert submission_cfg.policy_server == Pi0ServerTaskCfg()
    assert submission_cfg.policy_server.client_ping_timeout_s == Pi0ServerTaskCfg.client_ping_timeout_s

    with pytest.raises(AssertionError, match="Unknown policy server 'unknown'.*pi0"):
        _compose_submission(policy_server_name="unknown")

    with pytest.raises(AssertionError, match="policy_variant must be one of"):
        _compose_submission(["policy_server.policy_variant=unknown"])


def test_submitter_rejects_unregistered_policy_server_type():
    """Reject a typed policy server without a registered workflow implementation."""
    submission_cfg = ArenaExperimentSubmissionCfg(
        experiment_cfg=_pi0_experiment_cfg(),
        osmo=WorkflowCfg(dry_run=True),
        policy_server=TaskCfg(),
    )

    with pytest.raises(AssertionError, match="No policy-server workflow.*TaskCfg"):
        submit_arena_experiment(submission_cfg)


@pytest.mark.parametrize("config_path", ["osmo.not_a_field", "experiment_runner.not_a_field"])
def test_hydra_rejects_unknown_typed_config_fields(config_path):
    """Let the structured Hydra root reject fields outside their owning config."""
    with pytest.raises(ConfigCompositionException, match="not_a_field"):
        _compose_submission([f"{config_path}=true"])


def test_policy_server_rejects_workflow_fields():
    """Validate server overrides against the selected task config type."""
    with pytest.raises(ConfigCompositionException, match="workflow_name"):
        _compose_and_submit([
            "policy_server.workflow_name=experiment",
            "osmo.dry_run=true",
        ])


def test_renders_experiment_runner_and_shared_pi0_server_with_effective_endpoints():
    """Wire every matching pi0 Run in the effective Experiment to one server."""
    source_experiment_cfg = _pi0_experiment_cfg()
    workflow = Pi0ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(workflow_name="pi0-experiment"),
        experiment_cfg=source_experiment_cfg,
        server_task_cfg=Pi0ServerTaskCfg(),
    )

    tasks = _workflow_tasks(workflow.generate_workflow())
    assert [task["name"] for task in tasks] == ["experiment_runner", "policy_server"]
    assert [task["lead"] for task in tasks] == [True, False]

    eval_task = tasks[0]
    experiment = _embedded_experiment(eval_task)
    server_host = Pi0ServerTask.host_token()
    assert experiment["runs"]["first"]["policy"]["remote_host"] == server_host
    assert experiment["runs"]["first"]["policy"]["remote_port"] == POLICY_SERVER_PORT
    assert experiment["runs"]["first"]["policy"]["ping_timeout"] == Pi0ServerTaskCfg.client_ping_timeout_s
    assert experiment["runs"]["second"]["policy"]["remote_host"] == server_host
    assert experiment["runs"]["second"]["policy"]["remote_port"] == POLICY_SERVER_PORT
    assert experiment["runs"]["second"]["policy"]["ping_timeout"] == Pi0ServerTaskCfg.client_ping_timeout_s
    assert "remote_host" not in experiment["runs"]["local"]["policy"]
    assert "remote_port" not in experiment["runs"]["local"]["policy"]
    assert source_experiment_cfg.runs["first"].policy.remote_host == "user-host"
    assert source_experiment_cfg.runs["first"].policy.ping_timeout == 10

    command = _task_file(eval_task, "/tmp/entry.sh")["contents"]
    assert "experiment_runner.py" in command
    assert f"--experiment_config {REMOTE_EXPERIMENT_PATH}" in command
    assert "--enable_cameras" in command
    assert "policy_runner.py" not in command
    assert "runs." not in command

    server_command = _task_file(tasks[1], "/tmp/entry.sh")["contents"]
    assert f"scripts/serve_policy.py --port={POLICY_SERVER_PORT} policy:checkpoint" in server_command
    assert "--policy.config=pi05_droid_jointpos_polaris" in server_command


def test_embeds_effective_experiment_yaml():
    """Embed the composed Experiment instead of staging its source file."""
    experiment_runner_task = ExperimentRunnerTask(
        task_cfg=ExperimentRunnerTaskCfg(image="registry.example.com/evaluator:typed-api"),
        experiment_cfg=_zero_action_experiment_cfg(),
        lead=True,
    )

    eval_task = experiment_runner_task.create_task_dict()
    assert eval_task["image"] == "registry.example.com/evaluator:typed-api"
    embedded_experiment = _embedded_experiment(eval_task)
    assert embedded_experiment["runs"]["baseline"]["environment"]["type"] == "pick_and_place_maple_table"
    assert embedded_experiment["runs"]["baseline"]["policy"]["type"] == "zero_action"
    assert embedded_experiment["runs"]["baseline"]["environment_builder"]["num_envs"] == 1


def test_submission_removes_temporary_workflow(monkeypatch):
    """Submit one temporary workflow and remove it afterwards."""
    experiment_cfg = _pi0_experiment_cfg()
    workflow = Pi0ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(),
        experiment_cfg=experiment_cfg,
        server_task_cfg=Pi0ServerTaskCfg(),
    )
    captured_workflow_path = None

    def capture_submission(command, **kwargs):
        nonlocal captured_workflow_path
        assert kwargs["text"] is True
        assert command[:3] == ["osmo", "workflow", "submit"]
        captured_workflow_path = Path(command[3])
        assert captured_workflow_path.is_file()
        submitted_workflow = yaml.safe_load(captured_workflow_path.read_text(encoding="utf-8"))
        embedded_experiment = _embedded_experiment(_workflow_tasks(submitted_workflow)[0])
        assert embedded_experiment["runs"]["first"]["policy"]["type"].endswith(".Pi0RemotePolicy")
        return SimpleNamespace(returncode=23, stdout="")

    monkeypatch.setattr("osmo.workflows.workflow.subprocess.run", capture_submission)

    assert workflow.submit_workflow().returncode == 23
    assert captured_workflow_path is not None
    assert not captured_workflow_path.exists()


def test_submission_composes_defaults_experiment_and_overrides(tmp_path, capsys):
    """Resolve typed defaults, Experiment values, then CLI overrides."""
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """runs:
  openpi_maple_table:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy
      ping_timeout: 300.0
    rollout_limit:
      num_episodes: 2
""",
        encoding="utf-8",
    )
    return_code = _compose_and_submit(
        [
            "osmo.dry_run=true",
            "osmo.workflow_name=overridden-experiment",
            "experiment_runner.image=registry.example.com/evaluator:branch",
            "policy_server.image=registry.example.com/openpi:overridden",
            "policy_server.policy_config=overridden-pi0-config",
            "policy_server.client_ping_timeout_s=600.0",
            "experiment_cfg.runs.openpi_maple_table.rollout_limit.num_episodes=4",
            "experiment_cfg.runs.openpi_maple_table.environment_builder.num_envs=2",
            "experiment_cfg.runs.openpi_maple_table.policy.ping_interval=33.0",
            "experiment_cfg.runs.openpi_maple_table.policy.ping_timeout=450.0",
        ],
        experiment_path,
        "pi0",
    )

    assert return_code == 0
    rendered = capsys.readouterr().out
    assert "[dry-run] Rendered workflow YAML" in rendered
    workflow = _rendered_workflow(rendered)
    assert workflow["workflow"]["name"] == "overridden-experiment"
    tasks = _workflow_tasks(workflow)
    assert [task["name"] for task in tasks] == ["experiment_runner", "policy_server"]
    assert tasks[0]["image"] == "registry.example.com/evaluator:branch"
    assert tasks[1]["image"] == "registry.example.com/openpi:overridden"

    experiment = _embedded_experiment(tasks[0])
    policy = experiment["runs"]["openpi_maple_table"]["policy"]
    assert experiment["runs"]["openpi_maple_table"]["rollout_limit"]["num_episodes"] == 4
    assert experiment["runs"]["openpi_maple_table"]["environment_builder"]["num_envs"] == 2
    assert policy["ping_interval"] == 33.0
    assert policy["remote_host"] == Pi0ServerTask.host_token()
    assert policy["remote_port"] == POLICY_SERVER_PORT
    assert policy["ping_timeout"] == 600.0
    assert "experiment_cfg.runs" not in _task_file(tasks[0], "/tmp/entry.sh")["contents"]

    server_command = _task_file(tasks[1], "/tmp/entry.sh")["contents"]
    assert "--policy.config=overridden-pi0-config" in server_command
    assert "--policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos" in server_command


def test_embedded_openpi_experiment_composes_through_experiment_runner_loader(tmp_path):
    """Keep the rendered OSMO handoff compatible with the Experiment Runner loader."""
    submission_cfg = _compose_submission()
    assert isinstance(submission_cfg.policy_server, Pi0ServerTaskCfg)
    workflow = Pi0ArenaExperimentWorkflow(
        workflow_cfg=submission_cfg.osmo,
        experiment_cfg=submission_cfg.experiment_cfg,
        server_task_cfg=submission_cfg.policy_server,
        task_cfg=submission_cfg.experiment_runner,
    )
    experiment_path = tmp_path / "effective_experiment.yaml"
    experiment_file = _task_file(_workflow_tasks(workflow.generate_workflow())[0], REMOTE_EXPERIMENT_PATH)
    experiment_path.write_text(experiment_file["contents"], encoding="utf-8")

    experiment_cfg = load_arena_experiment_from_config_file(experiment_path, device="cuda:0")
    run_cfg = experiment_cfg.runs[OPENPI_RUN_NAME]

    assert list(experiment_cfg.runs) == list(submission_cfg.experiment_cfg.runs)
    assert isinstance(run_cfg.policy, Pi0RemotePolicyCfg)
    assert run_cfg.policy.remote_host == Pi0ServerTask.host_token()
    assert run_cfg.policy.remote_port == POLICY_SERVER_PORT
    assert run_cfg.policy.ping_timeout == Pi0ServerTaskCfg.client_ping_timeout_s


def test_submission_overrides_osmo_resources(monkeypatch):
    """Apply scheduler overrides after the typed workflow defaults."""
    submitted_command = None
    submitted_resources = None

    def capture_submission(command, **kwargs):
        nonlocal submitted_command, submitted_resources
        assert kwargs["text"] is True
        submitted_command = command
        submitted_workflow = yaml.safe_load(Path(command[3]).read_text(encoding="utf-8"))
        submitted_resources = submitted_workflow["workflow"]["resources"]["default"]
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr("osmo.workflows.workflow.subprocess.run", capture_submission)

    return_code = _compose_and_submit([
        "osmo.pool=isaac-dev-l40-03",
        "osmo.platform=ovx-l40",
        "osmo.memory=120Gi",
    ])

    assert return_code == 0
    assert submitted_command is not None
    pool_flag_index = submitted_command.index("--pool")
    assert submitted_command[pool_flag_index + 1] == "isaac-dev-l40-03"
    assert submitted_resources["platform"] == "ovx-l40"
    assert submitted_resources["memory"] == "120Gi"


def test_cli_requires_experiment_cfg_path_and_policy_server(capsys):
    """Require the Experiment config path and policy server at the CLI boundary."""
    with pytest.raises(SystemExit, match="2"):
        main([])
    assert "--experiment_cfg" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        main(["--experiment_cfg", str(OPENPI_EXPERIMENT_CFG_PATH)])
    assert "--policy_server" in capsys.readouterr().err


def test_cli_help_explains_paths_and_override_names(capsys):
    """Describe the Experiment path, server selector, and typed override syntax."""
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    help_text = capsys.readouterr().out
    normalized_help_text = " ".join(help_text.split())
    assert "--experiment_cfg PATH" in help_text
    assert "path to a typed Arena Experiment YAML configuration" in normalized_help_text
    assert "droid_pnp_srl_openpi_experiment.yaml" in help_text
    assert "--policy_server {pi0}" in help_text
    assert "typed defaults < Experiment YAML < CLI overrides" in help_text
    assert "osmo.workflow_name=my-evaluation" in help_text
    assert "experiment_cfg.runs.droid_pnp_srl_openpi_billiard_hall.rollout_limit.num_episodes=4" in help_text
    assert "Common overrides:" not in help_text
    assert "Current defaults:" not in help_text
    assert "experiment_cfg=<name>" not in help_text


def test_submission_rejects_legacy_json_experiment(tmp_path):
    """Limit OSMO submission to typed YAML that can be embedded for the remote runner."""
    experiment_path = tmp_path / "legacy.json"
    experiment_path.write_text("{}", encoding="utf-8")

    with pytest.raises(AssertionError, match="requires a typed YAML Experiment Definition"):
        build_arena_experiment_submission_cfg(experiment_path, "pi0")


def test_cli_accepts_arbitrary_paths_and_trailing_overrides(tmp_path, capsys):
    """Submit an arbitrary Experiment path through the real CLI parser."""
    experiment_path = tmp_path / "my_experiment.yaml"
    experiment_path.write_text(
        """runs:
  openpi:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy
""",
        encoding="utf-8",
    )
    assert (
        main([
            "--experiment_cfg",
            str(experiment_path),
            "--policy_server",
            "pi0",
            "osmo.dry_run=true",
            "osmo.workflow_name=path-based-submission",
            "experiment_runner.image=registry.example.com/evaluator:cli",
        ])
        == 0
    )

    workflow = _rendered_workflow(capsys.readouterr().out)
    assert workflow["workflow"]["name"] == "path-based-submission"
    tasks = _workflow_tasks(workflow)
    assert [task["name"] for task in tasks] == ["experiment_runner", "policy_server"]
    assert tasks[0]["image"] == "registry.example.com/evaluator:cli"


def test_experiment_path_is_relative_to_the_invocation_directory(tmp_path, monkeypatch):
    """Resolve a relative Experiment path from the caller's working directory."""
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(OPENPI_EXPERIMENT_CFG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    submission_cfg = build_arena_experiment_submission_cfg("experiment.yaml", "pi0")

    assert isinstance(submission_cfg.experiment_cfg, ArenaExperimentCfg)
    assert isinstance(submission_cfg.policy_server, Pi0ServerTaskCfg)


def test_structural_policy_override_is_checked_against_server_variant():
    """Check server compatibility against the effective structurally overridden policy."""
    with pytest.raises(AssertionError, match="pi0 server is configured for 'pi05'"):
        _compose_and_submit([
            "osmo.dry_run=true",
            f"experiment_cfg.runs.{OPENPI_RUN_NAME}.policy={{policy_variant:pi0}}",
        ])


def test_server_variant_cannot_relabel_known_pi05_model():
    """Reject a client-compatible label when the selected server model remains pi05."""
    with pytest.raises(AssertionError, match="policy_config.*serves variant 'pi05'.*policy_variant 'pi0'"):
        _compose_and_submit([
            "osmo.dry_run=true",
            f"experiment_cfg.runs.{OPENPI_RUN_NAME}.policy.policy_variant=pi0",
            "policy_server.policy_variant=pi0",
        ])


def test_submitter_rejects_server_without_matching_run():
    """Reject an explicitly selected server that cannot serve any Experiment Run."""
    submission_cfg = ArenaExperimentSubmissionCfg(
        experiment_cfg=_zero_action_experiment_cfg(),
        osmo=WorkflowCfg(dry_run=True),
        policy_server=Pi0ServerTaskCfg(),
    )

    with pytest.raises(AssertionError, match="requires at least one Run using Pi0RemotePolicy"):
        submit_arena_experiment(submission_cfg)


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
