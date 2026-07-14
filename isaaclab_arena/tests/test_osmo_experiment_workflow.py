# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify OSMO workflow construction for a complete Arena Experiment."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_arena.evaluation.experiment_runner_cfg import ExperimentRunnerCfg
from isaaclab_arena.hydra.typed_yaml import load_typed_yaml_cfg
from osmo.tasks.experiment_runner_task import (
    REMOTE_EVALUATION_CONFIG_PATH,
    REMOTE_EXPERIMENT_PATH,
    ExperimentRunnerTaskCfg,
)
from osmo.tasks.openpi_server_task import OPENPI_SERVER_PORT, OpenPiServerTask, OpenPiServerTaskCfg
from osmo.workflows.arena_experiment_workflow import (
    STAGED_EXPERIMENT_FILENAME,
    ArenaExperimentWorkflow,
    ArenaExperimentWorkflowCfg,
    OpenPiArenaExperimentWorkflow,
    load_arena_experiment_workflow_cfg,
)
from osmo.workflows.workflow import WorkflowCfg, WorkflowPriority
from osmo.workflows.workflow_constants import OSMO_TASK_OUTPUT_DIR


def _write_zero_action_experiment(path: Path) -> bytes:
    source = b"""# Preserve this exact source.\nruns:\n- name: baseline\n  environment:\n    type: pick_and_place_maple_table\n  policy:\n    type: zero_action\n  rollout_limit:\n    num_steps: 2\n"""
    path.write_bytes(source)
    return source


def _write_openpi_experiment(path: Path) -> bytes:
    source = b"""runs:
- name: first
  environment:
    type: pick_and_place_maple_table
  policy:
    type: pi0_remote
    policy_variant: pi05
  rollout_limit:
    num_steps: 2
- name: second
  environment:
    type: pick_and_place_maple_table
  policy:
    type: pi0_remote
    policy_variant: pi05
  rollout_limit:
    num_steps: 2
"""
    path.write_bytes(source)
    return source


def _experiment_runner_cfg(experiment_config_path: Path, **values) -> ExperimentRunnerCfg:
    return ExperimentRunnerCfg(experiment_config=str(experiment_config_path), **values)


def _task_file(task: dict, remote_path: str) -> dict:
    return next(file for file in task["files"] if file["path"] == remote_path)


def _load_embedded_experiment_runner_cfg(config_contents: str, tmp_path: Path) -> ExperimentRunnerCfg:
    config_path = tmp_path / "embedded-evaluation.yaml"
    config_path.write_text(config_contents, encoding="utf-8")
    return load_typed_yaml_cfg(
        config_path,
        ExperimentRunnerCfg,
        config_name="embedded Experiment Runner",
    )


def test_loads_bundled_and_custom_workflow_configs(tmp_path):
    """Load typed OSMO config independently from the evaluation and Experiment configs."""
    default_cfg = load_arena_experiment_workflow_cfg()

    assert default_cfg.workflow.workflow_name == "arena-experiment"
    assert default_cfg.workflow.priority is WorkflowPriority.NORMAL
    assert default_cfg.experiment_runner_task.image.endswith("isaaclab_arena:latest")
    assert default_cfg.experiment_runner_task.output_url.endswith("/{{workflow_id}}")
    assert default_cfg.openpi_server.policy_variant == "pi05"
    assert default_cfg.openpi_server.client_ping_timeout == 300.0
    assert default_cfg.openpi_server.policy_config == "pi05_droid_jointpos_polaris"

    custom_config_path = tmp_path / "osmo.yaml"
    custom_config_path.write_text(
        """
workflow:
  workflow_name: custom-experiment
  priority: HIGH
  cpus: 8
experiment_runner_task:
  image: registry.example.com/arena:test
openpi_server:
  image: registry.example.com/openpi:test
""",
        encoding="utf-8",
    )
    custom_cfg = load_arena_experiment_workflow_cfg(custom_config_path)

    assert custom_cfg.workflow.workflow_name == "custom-experiment"
    assert custom_cfg.workflow.priority is WorkflowPriority.HIGH
    assert custom_cfg.workflow.cpus == 8
    assert custom_cfg.workflow.gpus == WorkflowCfg().gpus
    assert custom_cfg.experiment_runner_task.image == "registry.example.com/arena:test"
    assert custom_cfg.openpi_server.image == "registry.example.com/openpi:test"
    assert custom_cfg.openpi_server.client_ping_timeout == OpenPiServerTaskCfg().client_ping_timeout
    assert custom_cfg.openpi_server.policy_dir == OpenPiServerTaskCfg().policy_dir


def test_one_experiment_builds_one_lead_experiment_runner_task_from_typed_config(tmp_path):
    """Represent the full Experiment with one workflow and one typed evaluation task."""
    experiment_config_path = tmp_path / "source.yaml"
    _write_zero_action_experiment(experiment_config_path)
    source_cfg = _experiment_runner_cfg(
        experiment_config_path,
        experiment_overrides=["runs.baseline.rollout_limit.num_steps=4"],
        output_base_dir="/local/output",
        record_camera_video=True,
        continue_on_error=True,
        serve_evaluation_report=True,
    )
    workflow = ArenaExperimentWorkflow(
        cfg=ArenaExperimentWorkflowCfg(
            workflow=WorkflowCfg(workflow_name="zero-action-experiment"),
            experiment_runner_task=ExperimentRunnerTaskCfg(),
        ),
        experiment_runner_cfg=source_cfg,
    )

    workflow_dict = workflow.generate_workflow()
    tasks = workflow_dict["workflow"]["groups"][0]["tasks"]

    assert workflow_dict["workflow"]["name"] == "zero-action-experiment"
    assert len(tasks) == 1
    assert tasks[0]["name"] == "experiment_runner"
    assert tasks[0]["lead"] is True
    assert _task_file(tasks[0], REMOTE_EXPERIMENT_PATH) == {
        "localpath": STAGED_EXPERIMENT_FILENAME,
        "path": REMOTE_EXPERIMENT_PATH,
    }
    command = _task_file(tasks[0], "/tmp/entry.sh")["contents"]
    assert (
        f"isaaclab_arena/evaluation/experiment_runner.py --config {REMOTE_EVALUATION_CONFIG_PATH} "
        "--local --viz none --enable_cameras"
        in command
    )
    assert "--experiment_config" not in command
    assert "--output_base_dir" not in command
    assert "--record_camera_video" not in command
    assert "runs.baseline" not in command
    assert "policy_runner.py" not in command

    embedded_cfg = _load_embedded_experiment_runner_cfg(
        _task_file(tasks[0], REMOTE_EVALUATION_CONFIG_PATH)["contents"],
        tmp_path,
    )
    assert embedded_cfg == ExperimentRunnerCfg(
        experiment_config=REMOTE_EXPERIMENT_PATH,
        experiment_overrides=["runs.baseline.rollout_limit.num_steps=4"],
        output_base_dir=OSMO_TASK_OUTPUT_DIR,
        record_camera_video=True,
        continue_on_error=True,
        serve_evaluation_report=False,
    )
    assert source_cfg.output_base_dir == "/local/output"
    assert source_cfg.experiment_config == str(experiment_config_path)
    assert source_cfg.serve_evaluation_report


def test_stages_exact_experiment_source(tmp_path):
    """Copy the source Experiment YAML unchanged into the OSMO submission staging directory."""
    experiment_config_path = tmp_path / "source.yaml"
    source = _write_zero_action_experiment(experiment_config_path)
    workflow = ArenaExperimentWorkflow(
        cfg=ArenaExperimentWorkflowCfg(),
        experiment_runner_cfg=_experiment_runner_cfg(experiment_config_path),
    )
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    workflow._stage_submission_files(staging_dir)

    assert (staging_dir / STAGED_EXPERIMENT_FILENAME).read_bytes() == source


def test_submission_stages_workflow_and_experiment_together(tmp_path, monkeypatch):
    """Submit the embedded evaluation config and exact Experiment from one directory."""
    experiment_config_path = tmp_path / "source.yaml"
    source = _write_zero_action_experiment(experiment_config_path)
    workflow = ArenaExperimentWorkflow(
        cfg=ArenaExperimentWorkflowCfg(),
        experiment_runner_cfg=_experiment_runner_cfg(experiment_config_path),
    )
    captured_staging_dir = None

    def capture_submission(command, cwd):
        nonlocal captured_staging_dir
        captured_staging_dir = Path(cwd)
        assert command[:4] == ["osmo", "workflow", "submit", "workflow.yaml"]
        rendered_workflow = (captured_staging_dir / "workflow.yaml").read_text(encoding="utf-8")
        assert REMOTE_EVALUATION_CONFIG_PATH in rendered_workflow
        assert f"output_base_dir: '{OSMO_TASK_OUTPUT_DIR}'" in rendered_workflow
        assert (captured_staging_dir / STAGED_EXPERIMENT_FILENAME).read_bytes() == source
        return SimpleNamespace(returncode=23)

    monkeypatch.setattr("osmo.workflows.workflow.subprocess.run", capture_submission)

    assert workflow.submit_workflow() == 23
    assert captured_staging_dir is not None
    assert not captured_staging_dir.exists()


def test_openpi_workflow_embeds_trailing_server_endpoint_overrides(tmp_path):
    """Connect all OpenPI Runs to one co-scheduled server through trailing overrides."""
    experiment_config_path = tmp_path / "source.yaml"
    _write_openpi_experiment(experiment_config_path)
    user_overrides = [
        "runs.first.policy.remote_host=user-host",
        "runs.first.policy.remote_port=9999",
        "runs.first.policy.ping_timeout=10.0",
    ]
    source_cfg = _experiment_runner_cfg(
        experiment_config_path,
        experiment_overrides=user_overrides,
        output_base_dir="/local/output",
    )
    workflow = OpenPiArenaExperimentWorkflow(
        cfg=ArenaExperimentWorkflowCfg(),
        experiment_runner_cfg=source_cfg,
        openpi_run_names=["first", "second"],
    )

    workflow_dict = workflow.generate_workflow()
    tasks = workflow_dict["workflow"]["groups"][0]["tasks"]
    assert [task["name"] for task in tasks] == ["experiment_runner", "policy_server"]
    assert [task["lead"] for task in tasks] == [True, False]

    eval_command = _task_file(tasks[0], "/tmp/entry.sh")["contents"]
    assert "--local --viz none --enable_cameras" in eval_command
    embedded_contents = _task_file(tasks[0], REMOTE_EVALUATION_CONFIG_PATH)["contents"]
    host_token = OpenPiServerTask.host_token()
    assert host_token in embedded_contents
    assert host_token in workflow.render_yaml()

    resolved_contents = embedded_contents.replace(host_token, "10.0.0.42").replace(
        OSMO_TASK_OUTPUT_DIR,
        "/osmo/output",
    )
    embedded_cfg = _load_embedded_experiment_runner_cfg(resolved_contents, tmp_path)
    assert embedded_cfg.experiment_config == REMOTE_EXPERIMENT_PATH
    assert embedded_cfg.output_base_dir == "/osmo/output"
    assert embedded_cfg.experiment_overrides == [
        *user_overrides,
        'runs.first.policy.remote_host="10.0.0.42"',
        f"runs.first.policy.remote_port={OPENPI_SERVER_PORT}",
        "runs.first.policy.ping_timeout=300.0",
        'runs.second.policy.remote_host="10.0.0.42"',
        f"runs.second.policy.remote_port={OPENPI_SERVER_PORT}",
        "runs.second.policy.ping_timeout=300.0",
    ]
    assert source_cfg.experiment_overrides == user_overrides
    assert source_cfg.output_base_dir == "/local/output"

    server_command = _task_file(tasks[1], "/tmp/entry.sh")["contents"]
    assert "scripts/serve_policy.py policy:checkpoint" in server_command
    assert "--policy.config=pi05_droid_jointpos_polaris" in server_command
    assert "policy_runner.py" not in workflow.render_yaml()


def test_rejects_legacy_json_experiment(tmp_path):
    """Keep the OSMO workflow path restricted to typed YAML Experiments."""
    experiment_config_path = tmp_path / "legacy.json"
    experiment_config_path.write_text('{"jobs": []}', encoding="utf-8")

    with pytest.raises(AssertionError, match="typed YAML Experiments only"):
        ArenaExperimentWorkflow(
            cfg=ArenaExperimentWorkflowCfg(),
            experiment_runner_cfg=_experiment_runner_cfg(experiment_config_path),
        )
