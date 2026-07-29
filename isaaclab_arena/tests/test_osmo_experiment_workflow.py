# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify distributed OSMO workflows for Arena Experiments."""

import json
import yaml
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra.errors import ConfigCompositionException

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena_cosmos.policy.cosmos_remote_config import CosmosRemotePolicyCfg
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg
from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import Gr00tRemoteClosedloopPolicyCfg
from isaaclab_arena_openpi.policy import pi0_remote_policy  # noqa: F401
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from osmo.submit_arena_experiment import (
    ArenaExperimentSubmissionCfg,
    build_arena_experiment_submission_cfg,
    format_submission_config,
    main,
    submit_arena_experiment,
)
from osmo.tasks.collect_experiment_outputs_task import (
    _REMOTE_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH,
    _REMOTE_EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_PATH,
    experiment_runner_output_directory_input_token,
)
from osmo.tasks.cosmos_server_task import CosmosServerTask
from osmo.tasks.experiment_runner_task import REMOTE_EXPERIMENT_PATH, ExperimentRunnerTask, ExperimentRunnerTaskCfg
from osmo.tasks.gr00t_server_task import Gr00tServerTask, Gr00tServerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.workflows.arena_experiment_workflow import ArenaExperimentWorkflow
from osmo.workflows.server_bindings import GR00T_SERVER_BINDING, REMOTE_POLICY_SERVERS, ServersCfg
from osmo.workflows.workflow import WorkflowCfg
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR, POLICY_SERVER_PORT

# Composing complete Arena Experiments loads Isaac runtime modules, so these tests
# must not share a pytest process with the persistent SimulationApp tests.
pytestmark = pytest.mark.with_subprocess

REPOSITORY_ROOT = Path(__file__).parents[2]
OPENPI_EXPERIMENT_CFG_PATH = (
    REPOSITORY_ROOT / "isaaclab_arena_environments/experiment_configs/droid_pnp_srl_openpi_experiment.yaml"
)
OPENPI_RUN_NAME = "droid_pnp_srl_openpi_billiard_hall"
GR00T_CONFIG_YAML_PATH = "isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml"


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


def _mixed_experiment_cfg() -> ArenaExperimentCfg:
    return ArenaExperimentCfg(
        runs={
            "pi0_run": ArenaRunCfg(
                name="pi0_run",
                environment=PickAndPlaceMapleTableEnvironmentCfg(),
                policy=Pi0RemotePolicyCfg(),
            ),
            "gr00t_run": ArenaRunCfg(
                name="gr00t_run",
                environment=PickAndPlaceMapleTableEnvironmentCfg(),
                policy=Gr00tRemoteClosedloopPolicyCfg(policy_config_yaml_path=GR00T_CONFIG_YAML_PATH),
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


def _workflow_groups(workflow: dict) -> list[dict]:
    return workflow["workflow"]["groups"]


def _workflow_tasks(workflow: dict, group_index: int = 0) -> list[dict]:
    return _workflow_groups(workflow)[group_index]["tasks"]


def _compose_submission(
    overrides: list[str] | None = None,
    experiment_cfg_path: Path = OPENPI_EXPERIMENT_CFG_PATH,
) -> ArenaExperimentSubmissionCfg:
    return build_arena_experiment_submission_cfg(experiment_cfg_path, overrides)


def _compose_and_submit(
    overrides: list[str],
    experiment_cfg_path: Path = OPENPI_EXPERIMENT_CFG_PATH,
) -> int:
    return submit_arena_experiment(_compose_submission(overrides, experiment_cfg_path))


def test_declares_server_registry():
    """Keep the client-policy-to-server binding registry explicit."""
    assert set(REMOTE_POLICY_SERVERS) == {
        Pi0RemotePolicyCfg,
        Gr00tRemoteClosedloopPolicyCfg,
        CosmosRemotePolicyCfg,
    }
    assert REMOTE_POLICY_SERVERS[Pi0RemotePolicyCfg].name == "pi0"
    assert REMOTE_POLICY_SERVERS[Pi0RemotePolicyCfg].server_task_cls is Pi0ServerTask
    assert REMOTE_POLICY_SERVERS[Gr00tRemoteClosedloopPolicyCfg].name == "gr00t"
    assert REMOTE_POLICY_SERVERS[Gr00tRemoteClosedloopPolicyCfg].server_task_cls is Gr00tServerTask
    assert REMOTE_POLICY_SERVERS[CosmosRemotePolicyCfg].name == "cosmos"
    assert REMOTE_POLICY_SERVERS[CosmosRemotePolicyCfg].server_task_cls is CosmosServerTask
    assert ArenaExperimentWorkflow.task_cfg_type is ExperimentRunnerTaskCfg


def test_explicit_experiment_composes_typed_defaults():
    """Compose an explicit Experiment path with derived server and OSMO defaults."""
    submission_cfg = _compose_submission()

    assert isinstance(submission_cfg.experiment_cfg, ArenaExperimentCfg)
    assert len(submission_cfg.experiment_cfg.runs) == 9
    assert isinstance(submission_cfg.experiment_cfg.runs[OPENPI_RUN_NAME].policy, Pi0RemotePolicyCfg)
    assert submission_cfg.osmo == WorkflowCfg()
    assert submission_cfg.osmo.pool == "isaac-dev-l40s-04"
    assert submission_cfg.osmo.platform == "ovx-l40s"
    assert submission_cfg.experiment_runner == ExperimentRunnerTaskCfg()
    assert submission_cfg.experiment_runner.image == "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
    assert submission_cfg.servers.pi0 == Pi0ServerTaskCfg()
    assert submission_cfg.servers.pi0.client_ping_timeout_s == Pi0ServerTaskCfg.client_ping_timeout_s

    with pytest.raises(AssertionError, match="policy_variant must be one of"):
        _compose_submission(["servers.pi0.policy_variant=unknown"])


@pytest.mark.parametrize("config_path", ["osmo.not_a_field", "experiment_runner.not_a_field"])
def test_hydra_rejects_unknown_typed_config_fields(config_path):
    """Let the structured Hydra root reject fields outside their owning config."""
    with pytest.raises(ConfigCompositionException, match="not_a_field"):
        _compose_submission([f"{config_path}=true"])


def test_server_config_rejects_workflow_fields():
    """Validate server overrides against the concrete server task config type."""
    with pytest.raises(ConfigCompositionException, match="workflow_name"):
        _compose_and_submit([
            "servers.pi0.workflow_name=experiment",
            "osmo.dry_run=true",
        ])


def test_fans_out_single_run_experiments_with_dedicated_pi0_servers_and_one_experiment_output():
    """Render one independent Run group per Run and collect their outputs into one Experiment output."""
    source_experiment_cfg = _pi0_experiment_cfg()
    workflow = ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(workflow_name="pi0-experiment"),
        experiment_cfg=source_experiment_cfg,
        servers=ServersCfg(),
    )

    rendered_workflow = workflow.generate_workflow()
    groups = _workflow_groups(rendered_workflow)
    assert rendered_workflow == workflow.generate_workflow()
    assert [group["name"] for group in groups] == [
        "arena-run-0",
        "arena-run-1",
        "arena-run-2",
        "arena-experiment-output",
    ]
    task_names = [task["name"] for group in groups for task in group["tasks"]]
    assert len(task_names) == len(set(task_names))
    workflow_names = [group["name"] for group in groups] + task_names
    normalized_workflow_names = [name.lower().replace("_", "-") for name in workflow_names]
    assert len(normalized_workflow_names) == len(set(normalized_workflow_names))

    first_tasks = groups[0]["tasks"]
    second_tasks = groups[1]["tasks"]
    local_tasks = groups[2]["tasks"]
    assert [task["name"] for task in first_tasks] == ["experiment-runner-0", "policy-server-0"]
    assert [task["name"] for task in second_tasks] == ["experiment-runner-1", "policy-server-1"]
    assert [task["name"] for task in local_tasks] == ["experiment-runner-2"]
    assert [[task["lead"] for task in group["tasks"]] for group in groups] == [
        [True, False],
        [True, False],
        [True],
        [True],
    ]

    first_experiment = _embedded_experiment(first_tasks[0])
    second_experiment = _embedded_experiment(second_tasks[0])
    local_experiment = _embedded_experiment(local_tasks[0])
    assert list(first_experiment["runs"]) == ["first"]
    assert list(second_experiment["runs"]) == ["second"]
    assert list(local_experiment["runs"]) == ["local"]
    assert first_experiment["runs"]["first"]["policy"]["remote_host"] == Pi0ServerTask.host_token("policy-server-0")
    assert second_experiment["runs"]["second"]["policy"]["remote_host"] == Pi0ServerTask.host_token("policy-server-1")
    for run_name, experiment in (("first", first_experiment), ("second", second_experiment)):
        policy = experiment["runs"][run_name]["policy"]
        assert policy["remote_port"] == POLICY_SERVER_PORT
        assert policy["ping_timeout"] == Pi0ServerTaskCfg.client_ping_timeout_s
    assert "remote_host" not in local_experiment["runs"]["local"]["policy"]
    assert "remote_port" not in local_experiment["runs"]["local"]["policy"]
    assert source_experiment_cfg.runs["first"].policy.remote_host == "user-host"
    assert source_experiment_cfg.runs["first"].policy.ping_timeout == 10

    experiment_runner_command = _task_file(first_tasks[0], "/tmp/entry.sh")["contents"]
    assert "experiment_runner.py" in experiment_runner_command
    assert f"--experiment_config {REMOTE_EXPERIMENT_PATH}" in experiment_runner_command
    assert f"--experiment_output_directory '{OSMO_TASK_OUTPUT_DIR}'" in experiment_runner_command
    assert "--output_base_dir" not in experiment_runner_command
    assert "--enable_cameras" in experiment_runner_command
    assert "policy_runner.py" not in experiment_runner_command
    assert "runs." not in experiment_runner_command

    assert first_tasks[0]["outputs"] == []
    assert second_tasks[0]["outputs"] == []
    assert local_tasks[0]["outputs"] == []

    server_command = _task_file(first_tasks[1], "/tmp/entry.sh")["contents"]
    assert f"scripts/serve_policy.py --port={POLICY_SERVER_PORT} policy:checkpoint" in server_command
    assert "--policy.config=pi05_droid_jointpos_polaris" in server_command

    experiment_output_task = groups[3]["tasks"][0]
    assert experiment_output_task["name"] == "collect-experiment-outputs"
    assert experiment_output_task["resource"] == "experiment-output"
    assert experiment_output_task["inputs"] == [
        {"task": "experiment-runner-0"},
        {"task": "experiment-runner-1"},
        {"task": "experiment-runner-2"},
    ]
    assert experiment_output_task["outputs"] == [{"url": DATASET_SWIFT_URL}]
    experiment_runner_output_directories_by_run_name = json.loads(
        _task_file(experiment_output_task, _REMOTE_EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_PATH)["contents"]
    )
    assert experiment_runner_output_directories_by_run_name == {
        "first": experiment_runner_output_directory_input_token("experiment-runner-0"),
        "second": experiment_runner_output_directory_input_token("experiment-runner-1"),
        "local": experiment_runner_output_directory_input_token("experiment-runner-2"),
    }
    experiment_output_script_file = _task_file(experiment_output_task, _REMOTE_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH)
    assert "localpath" not in experiment_output_script_file
    assert "def build_experiment_output" in experiment_output_script_file["contents"]
    experiment_output_command = _task_file(experiment_output_task, "/tmp/entry.sh")["contents"]
    assert experiment_output_command.startswith("set -euo pipefail")
    assert _REMOTE_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH in experiment_output_command
    assert (
        f"--experiment-runner-output-directories-file {_REMOTE_EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_PATH}"
        in experiment_output_command
    )
    assert "--experiment-output-directory" in experiment_output_command
    assert OSMO_TASK_OUTPUT_DIR in experiment_output_command
    assert rendered_workflow["workflow"]["resources"]["experiment-output"]["gpu"] == 0


def test_mixed_pi0_and_gr00t_experiment_fans_out_per_run_servers():
    """Derive one pi0 server and one GR00T server from a mixed Experiment; leave the local Run server-free."""
    workflow = ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(workflow_name="mixed-experiment"),
        experiment_cfg=_mixed_experiment_cfg(),
        servers=ServersCfg(),
    )

    groups = _workflow_groups(workflow.generate_workflow())
    assert [group["name"] for group in groups] == [
        "arena-run-0",
        "arena-run-1",
        "arena-run-2",
        "arena-experiment-output",
    ]
    pi0_tasks = groups[0]["tasks"]
    gr00t_tasks = groups[1]["tasks"]
    local_tasks = groups[2]["tasks"]
    assert [task["name"] for task in pi0_tasks] == ["experiment-runner-0", "policy-server-0"]
    assert [task["name"] for task in gr00t_tasks] == ["experiment-runner-1", "policy-server-1"]
    assert [task["name"] for task in local_tasks] == ["experiment-runner-2"]
    assert pi0_tasks[1]["image"] == Pi0ServerTaskCfg().image
    assert gr00t_tasks[1]["image"] == Gr00tServerTaskCfg().image

    pi0_policy = _embedded_experiment(pi0_tasks[0])["runs"]["pi0_run"]["policy"]
    assert pi0_policy["remote_host"] == Pi0ServerTask.host_token("policy-server-0")
    assert pi0_policy["remote_port"] == POLICY_SERVER_PORT
    gr00t_policy = _embedded_experiment(gr00t_tasks[0])["runs"]["gr00t_run"]["policy"]
    assert gr00t_policy["remote_host"] == Gr00tServerTask.host_token("policy-server-1")
    assert gr00t_policy["remote_port"] == POLICY_SERVER_PORT
    assert "remote_host" not in _embedded_experiment(local_tasks[0])["runs"]["local"]["policy"]


def test_rejects_servers_requiring_different_pools(monkeypatch):
    """Reject an Experiment whose derived servers need different pools (one submission, one pool)."""
    monkeypatch.setitem(
        REMOTE_POLICY_SERVERS,
        Gr00tRemoteClosedloopPolicyCfg,
        replace(GR00T_SERVER_BINDING, pool="isaac-other-pool"),
    )

    with pytest.raises(AssertionError, match="different resources"):
        ArenaExperimentWorkflow(
            workflow_cfg=WorkflowCfg(),
            experiment_cfg=_mixed_experiment_cfg(),
            servers=ServersCfg(),
        )


def test_gr00t_cli_dry_run_renders_workflow(tmp_path, capsys):
    """Compose and render a derived GR00T Experiment submission through the real CLI parser."""
    experiment_path = tmp_path / "gr00t_experiment.yaml"
    experiment_path.write_text(
        f"""runs:
  gr00t_run:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy
      policy_config_yaml_path: {GR00T_CONFIG_YAML_PATH}
""",
        encoding="utf-8",
    )

    assert main(["--experiment_cfg", str(experiment_path), "--dry_run"]) == 0
    workflow = _rendered_workflow(capsys.readouterr().out)
    tasks = _workflow_tasks(workflow)
    assert [task["name"] for task in tasks] == ["experiment-runner-0", "policy-server-0"]
    assert tasks[1]["image"] == Gr00tServerTaskCfg().image
    served_policy = _embedded_experiment(tasks[0])["runs"]["gr00t_run"]["policy"]
    assert served_policy["remote_host"] == Gr00tServerTask.host_token("policy-server-0")
    assert served_policy["remote_port"] == POLICY_SERVER_PORT


def test_embeds_effective_experiment_yaml():
    """Embed the composed Experiment instead of staging its source file."""
    experiment_runner_task = ExperimentRunnerTask(
        task_cfg=ExperimentRunnerTaskCfg(image="registry.example.com/evaluator:typed-api"),
        experiment_cfg=_zero_action_experiment_cfg(),
        lead=True,
        task_name="experiment-runner",
    )

    eval_task = experiment_runner_task.create_task_dict()
    assert eval_task["image"] == "registry.example.com/evaluator:typed-api"
    embedded_experiment = _embedded_experiment(eval_task)
    assert embedded_experiment["runs"]["baseline"]["environment"]["type"] == "pick_and_place_maple_table"
    assert embedded_experiment["runs"]["baseline"]["policy"]["type"] == "zero_action"
    assert embedded_experiment["runs"]["baseline"]["environment_builder"]["num_envs"] == 1


def test_all_local_experiment_runs_standalone_without_servers():
    """An Experiment with no remote-policy Runs derives no servers and runs every Run standalone."""
    workflow = ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(),
        experiment_cfg=_zero_action_experiment_cfg(),
        servers=ServersCfg(),
    )

    groups = _workflow_groups(workflow.generate_workflow())
    assert [group["name"] for group in groups] == ["arena-run-0", "arena-experiment-output"]
    assert [task["name"] for task in groups[0]["tasks"]] == ["experiment-runner-0"]


def test_submission_removes_temporary_workflow(monkeypatch):
    """Submit one temporary workflow and remove it afterwards."""
    experiment_cfg = _pi0_experiment_cfg()
    workflow = ArenaExperimentWorkflow(
        workflow_cfg=WorkflowCfg(),
        experiment_cfg=experiment_cfg,
        servers=ServersCfg(),
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
            "servers.pi0.image=registry.example.com/openpi:overridden",
            "servers.pi0.policy_config=overridden-pi0-config",
            "servers.pi0.client_ping_timeout_s=600.0",
            "experiment_cfg.runs.openpi_maple_table.rollout_limit.num_episodes=4",
            "experiment_cfg.runs.openpi_maple_table.environment_builder.num_envs=2",
            "experiment_cfg.runs.openpi_maple_table.policy.ping_interval=33.0",
            "experiment_cfg.runs.openpi_maple_table.policy.ping_timeout=450.0",
        ],
        experiment_path,
    )

    assert return_code == 0
    rendered = capsys.readouterr().out
    assert "[dry-run] Rendered workflow YAML" in rendered
    workflow = _rendered_workflow(rendered)
    assert workflow["workflow"]["name"] == "overridden-experiment"
    tasks = _workflow_tasks(workflow)
    assert [task["name"] for task in tasks] == ["experiment-runner-0", "policy-server-0"]
    assert tasks[0]["image"] == "registry.example.com/evaluator:branch"
    assert tasks[1]["image"] == "registry.example.com/openpi:overridden"

    experiment = _embedded_experiment(tasks[0])
    policy = experiment["runs"]["openpi_maple_table"]["policy"]
    assert experiment["runs"]["openpi_maple_table"]["rollout_limit"]["num_episodes"] == 4
    assert experiment["runs"]["openpi_maple_table"]["environment_builder"]["num_envs"] == 2
    assert policy["ping_interval"] == 33.0
    assert policy["remote_host"] == Pi0ServerTask.host_token("policy-server-0")
    assert policy["remote_port"] == POLICY_SERVER_PORT
    assert policy["ping_timeout"] == 600.0
    assert "experiment_cfg.runs" not in _task_file(tasks[0], "/tmp/entry.sh")["contents"]

    server_command = _task_file(tasks[1], "/tmp/entry.sh")["contents"]
    assert "--policy.config=overridden-pi0-config" in server_command
    assert "--policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos" in server_command


def test_embedded_openpi_experiment_composes_through_experiment_runner_loader(tmp_path):
    """Keep every single-Run OSMO handoff compatible with the Experiment Runner loader."""
    submission_cfg = _compose_submission()
    assert isinstance(submission_cfg.servers.pi0, Pi0ServerTaskCfg)
    workflow = ArenaExperimentWorkflow(
        workflow_cfg=submission_cfg.osmo,
        experiment_cfg=submission_cfg.experiment_cfg,
        servers=submission_cfg.servers,
        task_cfg=submission_cfg.experiment_runner,
    )
    rendered_workflow = workflow.generate_workflow()
    for index, run_name in enumerate(submission_cfg.experiment_cfg.runs):
        experiment_path = tmp_path / f"effective_experiment_{index}.yaml"
        experiment_file = _task_file(_workflow_tasks(rendered_workflow, index)[0], REMOTE_EXPERIMENT_PATH)
        experiment_path.write_text(experiment_file["contents"], encoding="utf-8")

        experiment_cfg = load_arena_experiment_from_config_file(experiment_path, device="cuda:0")
        assert list(experiment_cfg.runs) == [run_name]
        run_cfg = experiment_cfg.runs[run_name]
        assert isinstance(run_cfg.policy, Pi0RemotePolicyCfg)
        assert run_cfg.policy.remote_host == Pi0ServerTask.host_token(f"policy-server-{index}")
        assert run_cfg.policy.remote_port == POLICY_SERVER_PORT
        assert run_cfg.policy.ping_timeout == Pi0ServerTaskCfg.client_ping_timeout_s


def test_embedded_graph_environment_experiment_composes_through_experiment_runner_loader(tmp_path):
    """Embed graph-YAML environment Runs and keep the handoff loadable by the runner."""
    experiment_path = tmp_path / "graph_experiment.yaml"
    experiment_path.write_text(
        """runs:
  graph_run:
    environment:
      type: isaaclab_arena/tests/test_data/pick_and_place_maple_table_env_graph.yaml
      enable_cameras: true
    policy:
      type: isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy
    environment_builder:
      num_envs: 2
    rollout_limit:
      num_episodes: 3
""",
        encoding="utf-8",
    )
    submission_cfg = _compose_submission(
        ["experiment_cfg.runs.graph_run.environment_builder.num_envs=4"],
        experiment_path,
    )
    workflow = ArenaExperimentWorkflow(
        workflow_cfg=submission_cfg.osmo,
        experiment_cfg=submission_cfg.experiment_cfg,
        servers=submission_cfg.servers,
        task_cfg=submission_cfg.experiment_runner,
    )
    rendered_workflow = workflow.generate_workflow()
    experiment_runner_task = _workflow_tasks(rendered_workflow)[0]
    embedded_run = _embedded_experiment(experiment_runner_task)["runs"]["graph_run"]
    assert embedded_run["environment"] == {
        "type": "isaaclab_arena/tests/test_data/pick_and_place_maple_table_env_graph.yaml",
        "enable_cameras": True,
    }
    # The post-load Hydra override must land in the typed builder the runner executes with.
    assert embedded_run["environment_builder"]["num_envs"] == 4

    embedded_path = tmp_path / "embedded_experiment.yaml"
    embedded_path.write_text(_task_file(experiment_runner_task, REMOTE_EXPERIMENT_PATH)["contents"], encoding="utf-8")
    experiment_cfg = load_arena_experiment_from_config_file(embedded_path, device="cuda:0")
    run_cfg = experiment_cfg.runs["graph_run"]
    assert isinstance(run_cfg.environment, LegacyGraphEnvironmentCfg)
    assert run_cfg.environment.arena_env_args == [
        "--enable_cameras",
        "--env_graph_spec_yaml",
        "isaaclab_arena/tests/test_data/pick_and_place_maple_table_env_graph.yaml",
    ]
    assert run_cfg.environment_builder.num_envs == 4
    assert run_cfg.policy.remote_host == Pi0ServerTask.host_token("policy-server-0")


def test_submission_overrides_osmo_resources(monkeypatch):
    """Apply scheduler overrides after the derived workflow resource."""
    submitted_command = None
    submitted_resources = None

    def capture_submission(command, **kwargs):
        nonlocal submitted_command, submitted_resources
        assert kwargs["text"] is True
        submitted_command = command
        submitted_workflow = yaml.safe_load(Path(command[3]).read_text(encoding="utf-8"))
        submitted_resources = submitted_workflow["workflow"]["resources"]
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
    assert submitted_resources["default"]["platform"] == "ovx-l40"
    assert submitted_resources["default"]["memory"] == "120Gi"
    assert submitted_resources["experiment-output"]["platform"] == "ovx-l40"
    assert submitted_resources["experiment-output"]["memory"] == "120Gi"
    assert submitted_resources["experiment-output"]["gpu"] == 0


def test_cli_requires_experiment_cfg_path(capsys):
    """Require the Experiment config path at the CLI boundary."""
    with pytest.raises(SystemExit, match="2"):
        main([])
    assert "--experiment_cfg" in capsys.readouterr().err


def test_cli_help_explains_paths_and_override_names(capsys):
    """Describe the Experiment path, derived servers, and typed override syntax."""
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    help_text = capsys.readouterr().out
    normalized_help_text = " ".join(help_text.split())
    assert "--experiment_cfg PATH" in help_text
    assert "path to a typed Arena Experiment YAML configuration" in normalized_help_text
    assert "droid_pnp_srl_openpi_experiment.yaml" in help_text
    assert "servers.<name>" in help_text
    assert "derived from the Run's client policy" in normalized_help_text
    assert "typed defaults < Experiment YAML < CLI overrides" in help_text
    assert "osmo.workflow_name=my-evaluation" in help_text
    assert "experiment_cfg.runs.droid_pnp_srl_openpi_billiard_hall.rollout_limit.num_episodes=4" in help_text


def test_submission_rejects_legacy_json_experiment(tmp_path):
    """Limit OSMO submission to typed YAML that can be embedded for the remote runner."""
    experiment_path = tmp_path / "legacy.json"
    experiment_path.write_text("{}", encoding="utf-8")

    with pytest.raises(AssertionError, match="requires a typed YAML Experiment Definition"):
        build_arena_experiment_submission_cfg(experiment_path)


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
            "osmo.dry_run=true",
            "osmo.workflow_name=path-based-submission",
            "experiment_runner.image=registry.example.com/evaluator:cli",
        ])
        == 0
    )

    workflow = _rendered_workflow(capsys.readouterr().out)
    assert workflow["workflow"]["name"] == "path-based-submission"
    tasks = _workflow_tasks(workflow)
    assert [task["name"] for task in tasks] == ["experiment-runner-0", "policy-server-0"]
    assert tasks[0]["image"] == "registry.example.com/evaluator:cli"


def test_dry_run_flag_renders_workflow_without_submitting(tmp_path, capsys):
    """Render the workflow via the --dry_run flag instead of the osmo.dry_run override."""
    experiment_path = tmp_path / "experiment.yaml"
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

    assert main(["--experiment_cfg", str(experiment_path), "--dry_run"]) == 0
    rendered = capsys.readouterr().out
    assert "[dry-run] Rendered workflow YAML" in rendered
    assert _rendered_workflow(rendered)["workflow"]["name"] == WorkflowCfg().workflow_name

    # The flag coexists with an explicit override without producing a duplicate Hydra key.
    assert main(["--experiment_cfg", str(experiment_path), "--dry_run", "osmo.dry_run=true"]) == 0
    assert "[dry-run] Rendered workflow YAML" in capsys.readouterr().out


def test_format_submission_config_lists_every_override_section():
    """Render the submission sections whose leaves are the valid Hydra override keys."""
    submission_cfg = _compose_submission()

    values = yaml.safe_load(format_submission_config(submission_cfg))

    assert set(values) == {"osmo", "experiment_runner", "servers", "experiment_cfg"}
    assert values["osmo"]["pool"] == WorkflowCfg().pool
    assert values["experiment_runner"]["image"] == ExperimentRunnerTaskCfg().image
    # Only the server types the Experiment uses are shown, with their concrete fields.
    assert set(values["servers"]) == {"pi0"}
    assert values["servers"]["pi0"]["policy_variant"] == Pi0ServerTaskCfg().policy_variant
    assert OPENPI_RUN_NAME in values["experiment_cfg"]["runs"]


def test_list_overrides_flag_prints_config_without_submitting(capsys):
    """The --list_overrides flag composes and prints the config, then returns without submitting."""
    return_code = main([
        "--experiment_cfg",
        str(OPENPI_EXPERIMENT_CFG_PATH),
        "--list_overrides",
    ])

    assert return_code == 0
    output = capsys.readouterr().out
    assert "# Composed OSMO submission configuration" in output
    values = yaml.safe_load(output[output.index("osmo:") :])
    assert set(values) == {"osmo", "experiment_runner", "servers", "experiment_cfg"}
    assert set(values["servers"]) == {"pi0"}
    assert OPENPI_RUN_NAME in values["experiment_cfg"]["runs"]


def test_experiment_path_is_relative_to_the_invocation_directory(tmp_path, monkeypatch):
    """Resolve a relative Experiment path from the caller's working directory."""
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(OPENPI_EXPERIMENT_CFG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    submission_cfg = build_arena_experiment_submission_cfg("experiment.yaml")

    assert isinstance(submission_cfg.experiment_cfg, ArenaExperimentCfg)
    assert isinstance(submission_cfg.servers.pi0, Pi0ServerTaskCfg)


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
            "servers.pi0.policy_variant=pi0",
        ])


def test_pi0_server_quotes_configurable_shell_values():
    """Keep configured checkpoint values within one shell argument."""
    task = Pi0ServerTask(
        Pi0ServerTaskCfg(
            policy_config="config with spaces",
            policy_dir="gs://bucket/checkpoint; false",
        ),
        task_name="policy-server",
    )

    command = task._get_run_script()

    assert "'--policy.config=config with spaces'" in command
    assert "'--policy.dir=gs://bucket/checkpoint; false'" in command
