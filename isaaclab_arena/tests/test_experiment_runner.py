# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import subprocess
import yaml
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_arena.evaluation.arena_experiment_result import ARENA_EXPERIMENT_RESULT_FILENAME, ArenaExperimentResult
from isaaclab_arena.evaluation.experiment_runner_cli import parse_experiment_runner_args
from isaaclab_arena.recording.episode_results import read_episode_records
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.tests.utils.subprocess import run_subprocess
from isaaclab_arena_benchmarking.tests.utils.influxdb_kpi_records_validation import (
    assert_influxdb_kpi_records_are_valid,
)

HEADLESS = True
NUM_STEPS = 2
DEFAULT_VISUALIZER = "kit"


def test_experiment_runner_parses_native_hydra_overrides():
    args_cli, experiment_overrides = parse_experiment_runner_args([
        "--experiment_config",
        "experiment.yaml",
        "runs.baseline.rollout_limit.num_steps=2",
        "runs.baseline.environment.enable_cameras=true",
        "runs.baseline.environment_builder.record_trajectories=true",
    ])

    assert args_cli.experiment_config == "experiment.yaml"
    assert experiment_overrides == [
        "runs.baseline.rollout_limit.num_steps=2",
        "runs.baseline.environment.enable_cameras=true",
        "runs.baseline.environment_builder.record_trajectories=true",
    ]


def test_experiment_runner_exports_the_kpis_selected_by_the_experiment(monkeypatch, tmp_path):
    from isaaclab_arena.evaluation import experiment_runner

    exported_selections = []
    monkeypatch.setattr(
        experiment_runner,
        "export_kpis",
        lambda output_dir, names, **kwargs: exported_selections.append((output_dir, list(names), kwargs)),
    )

    experiment_runner._export_selected_kpis(tmp_path, SimpleNamespace(kpi_exporters=["success_rate", "timing"]))

    assert exported_selections == [(tmp_path, ["success_rate", "timing"], {})]


def test_experiment_runner_skips_kpi_export_when_the_experiment_selects_none(monkeypatch, tmp_path, capsys):
    from isaaclab_arena.evaluation import experiment_runner

    exported_selections = []
    monkeypatch.setattr(experiment_runner, "export_kpis", lambda *args, **kwargs: exported_selections.append(args))

    experiment_runner._export_selected_kpis(tmp_path, SimpleNamespace(kpi_exporters=[]))

    assert exported_selections == []
    assert "KPI export is skipped" in capsys.readouterr().out


def test_experiment_runner_parses_timestamped_base_or_exact_output_directory(tmp_path):
    exact_experiment_output_directory = tmp_path / "exact-experiment-output"
    timestamped_experiment_output_base_directory = tmp_path / "timestamped-experiment-outputs"

    default_arguments, default_experiment_overrides = parse_experiment_runner_args([
        "--experiment_config",
        "experiment.yaml",
    ])
    assert default_arguments.output_base_dir == "outputs"
    assert default_arguments.experiment_output_directory is None
    assert default_experiment_overrides == []

    timestamped_output_arguments, timestamped_output_experiment_overrides = parse_experiment_runner_args([
        "--output_base_dir",
        str(timestamped_experiment_output_base_directory),
    ])
    assert timestamped_output_arguments.output_base_dir == str(timestamped_experiment_output_base_directory)
    assert timestamped_output_arguments.experiment_output_directory is None
    assert timestamped_output_experiment_overrides == []

    exact_output_arguments, exact_output_experiment_overrides = parse_experiment_runner_args([
        "--experiment_config",
        "experiment.yaml",
        "--experiment_output_directory",
        str(exact_experiment_output_directory),
    ])
    assert exact_output_arguments.experiment_output_directory == exact_experiment_output_directory
    assert exact_output_experiment_overrides == []


def test_experiment_runner_rejects_timestamped_base_with_exact_output_directory(tmp_path):
    """Reject mutually exclusive output directory flags in a fresh process."""
    result = subprocess.run(
        [
            TestConstants.python_path,
            f"{TestConstants.evaluation_dir}/experiment_runner.py",
            "--output_base_dir",
            str(tmp_path / "timestamped-experiment-outputs"),
            "--experiment_output_directory",
            str(tmp_path / "exact-experiment-output"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode != 0
    assert "argument --experiment_output_directory: not allowed with argument --output_base_dir" in result.stderr


@pytest.mark.with_subprocess
def test_experiment_runner_rejects_unknown_non_hydra_arguments():
    """Reject misspelled CLI flags in a fresh process."""
    result = subprocess.run(
        [TestConstants.python_path, f"{TestConstants.evaluation_dir}/experiment_runner.py", "--headles"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode != 0
    assert "Unrecognized arguments: --headles" in result.stderr


def run_experiment_runner(
    experiment_config_path: str,
    headless: bool = HEADLESS,
    extra_args: list[str] | None = None,
    capture_output: bool = False,
    timeout_sec: int | None = None,
) -> subprocess.CompletedProcess[str] | None:
    """Run the Experiment Runner as a subprocess with timeout.

    A failed Run makes the Experiment Runner exit non-zero, so run_subprocess() raises CalledProcessError,
    unless extra_args contains --continue_on_error.

    Args:
        experiment_config_path: Path to the Experiment configuration file.
        headless: Whether to run in headless mode.
        extra_args: Additional Experiment Runner arguments.
        capture_output: Whether to capture and return the subprocess output.
        timeout_sec: Override the default subprocess timeout in seconds.

    Returns:
        The completed subprocess when output is captured, otherwise None.
    """
    args = [TestConstants.python_path, f"{TestConstants.evaluation_dir}/experiment_runner.py"]
    args.append("--experiment_config")
    args.append(experiment_config_path)
    args.extend(extra_args or [])
    if not headless:
        args.append("--viz")
        args.append(DEFAULT_VISUALIZER)

    return run_subprocess(args, capture_output=capture_output, timeout_sec=timeout_sec)


def read_run_statuses(experiment_output_directory: Path) -> dict[str, str]:
    """Return each Run's status from an Experiment's result file."""
    experiment_result = ArenaExperimentResult.from_experiment_output(experiment_output_directory)
    return {run_name: run_data["status"] for run_name, run_data in experiment_result.runs.items()}


def test_written_experiment_result_has_integer_end_timestamp_for_every_run(monkeypatch, tmp_path):
    from isaaclab_arena.evaluation import arena_experiment_result
    from isaaclab_arena.evaluation.arena_run import ArenaRunResult
    from isaaclab_arena.evaluation.experiment_runner import _write_arena_experiment_result
    from isaaclab_arena.evaluation.run_status import RunStatus

    monkeypatch.setattr(
        arena_experiment_result,
        "build_arena_run_result_metadata",
        lambda run_cfg: {
            "environment": {"name": "env", "definition": "env"},
            "policy_variant": "zero_action",
        },
    )
    experiment_cfg = SimpleNamespace(
        name="pnp_benchmark", runs={"passing": SimpleNamespace(), "failing": SimpleNamespace()}
    )
    run_results = [
        ArenaRunResult(
            run_name="passing",
            status=RunStatus.COMPLETED,
            start_timestamp_ns=1792799000000000000,
            end_timestamp_ns=1792800000000000000,
        ),
        ArenaRunResult(
            run_name="failing",
            status=RunStatus.FAILED,
            start_timestamp_ns=1760313600000000000,
            end_timestamp_ns=1760313700000000000,
        ),
    ]
    (tmp_path / "passing").mkdir()

    experiment_result = _write_arena_experiment_result(experiment_cfg, run_results, tmp_path)

    assert experiment_result.to_dict()["experiment_name"] == "pnp_benchmark"
    runs = experiment_result.to_dict()["runs"]
    assert runs["passing"]["start_timestamp_ns"] == 1792799000000000000
    assert runs["passing"]["end_timestamp_ns"] == 1792800000000000000
    assert runs["failing"]["start_timestamp_ns"] == 1760313600000000000
    assert runs["failing"]["end_timestamp_ns"] == 1760313700000000000


@pytest.mark.with_subprocess
def test_experiment_runner_from_typed_yaml(tmp_path):
    """Execute a typed YAML Experiment through the Experiment Runner CLI."""
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text(
        """
kpi_exporters: [success_rate, timing]
runs:
  yaml_baseline:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action
    rollout_limit:
      num_steps: 10
""",
        encoding="utf-8",
    )

    result = run_experiment_runner(
        str(experiment_config_path),
        extra_args=[
            "--experiment_output_directory",
            str(tmp_path / "output"),
            "runs.yaml_baseline.rollout_limit.num_steps=2",
        ],
        capture_output=True,
    )
    assert result is not None
    run_row = next(line for line in result.stdout.splitlines() if "yaml_baseline" in line and "pending" in line)
    run_cells = [cell.strip() for cell in run_row.split("|")[1:-1]]
    assert run_cells[4] == "2"
    assert (tmp_path / "output/index.html").is_file()
    assert (tmp_path / "output/yaml_baseline/episode_results_rebuild0.jsonl").is_file()
    experiment_result_path = tmp_path / "output" / ARENA_EXPERIMENT_RESULT_FILENAME
    experiment_result = json.loads(experiment_result_path.read_text(encoding="utf-8"))
    assert list(experiment_result["runs"]) == ["yaml_baseline"]
    run_result = experiment_result["runs"]["yaml_baseline"]
    assert run_result["environment"] == {
        "name": "pick_and_place_maple_table",
        "definition": "pick_and_place_maple_table",
    }
    assert run_result["policy_variant"] == "zero_action"
    assert run_result["status"] == "completed"
    assert isinstance(run_result["start_timestamp_ns"], int)
    assert isinstance(run_result["end_timestamp_ns"], int)
    assert set(run_result) == {
        "environment",
        "policy_variant",
        "status",
        "start_timestamp_ns",
        "end_timestamp_ns",
        "rebuilds",
    }
    # The resource tracker always runs, so kpis_resource.json is expected whenever the default `resource` exporter is
    # selected; a file may be absent only if the experiment selects other exporters. Present files are validated.
    kpis_filenames = {path.name for path in (tmp_path / "output").glob("kpis_*.json")}
    assert "kpis_success_rates.json" in kpis_filenames
    assert kpis_filenames <= {"kpis_success_rates.json", "kpis_timing.json", "kpis_resource.json"}
    for kpis_filename in kpis_filenames:
        experiment_kpis = json.loads((tmp_path / "output" / kpis_filename).read_text(encoding="utf-8"))
        assert_influxdb_kpi_records_are_valid(experiment_kpis)
        assert {record["timestamp_ns"] for record in experiment_kpis} == {run_result["end_timestamp_ns"]}


@pytest.mark.with_subprocess
def test_experiment_runner_replays_recorded_variation_samples(tmp_path):
    """Record and replay variation samples through the typed Experiment Runner CLI."""
    source_config_path = (
        Path(TestConstants.arena_environments_dir)
        / "experiment_configs"
        / "placement_record_replay"
        / "variation_replay_e2e.yaml"
    )
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    source_run = source_config["runs"]["variation_e2e"]
    output_dir = tmp_path / "output"
    record_results_paths = [output_dir / "variation_record" / "episode_results_rebuild0.jsonl"]
    replay_run = copy.deepcopy(source_run)
    replay_run["environment_builder"]["seed"] = 123
    replay_run["recorded_variation_samples_paths"] = [str(path) for path in record_results_paths]
    replay_run.pop("num_rebuilds")
    experiment_config_path = tmp_path / "variation_replay_e2e.yaml"
    experiment_config_path.write_text(
        yaml.safe_dump({"runs": {"variation_record": source_run, "variation_replay": replay_run}}),
        encoding="utf-8",
    )

    run_experiment_runner(
        str(experiment_config_path),
        extra_args=["--experiment_output_directory", str(output_dir)],
    )

    expected_variation_keys = {
        "light.hdr_image",
        "pick_up_object.disappear",
        "droid_rel_joint_pos.camera_extrinsics_wrist_camera",
    }
    for rebuild_index, record_results_path in enumerate(record_results_paths):
        recorded = read_episode_records(record_results_path)
        assert len(recorded) == source_run["rollout_limit"]["num_episodes"]
        assert all(set(record["variations"]) == expected_variation_keys for record in recorded)

        replay_results_path = output_dir / "variation_replay" / f"episode_results_rebuild{rebuild_index}.jsonl"
        replayed = read_episode_records(replay_results_path)

        assert len(replayed) == len(recorded)
        assert sorted(record["replay_source_record_index"] for record in replayed) == list(range(len(recorded)))
        for replay_record in replayed:
            source_record = recorded[replay_record["replay_source_record_index"]]
            assert replay_record["variations"] == source_record["variations"]


@pytest.mark.with_subprocess
def test_experiment_runner_finishes_episode_budget_across_rebuilds(tmp_path):
    """Override the timeout through the CLI and match each rebuild's episode allocation to trajectories."""
    import h5py

    experiment_config_path = tmp_path / "finite_episodes.yaml"
    experiment_config_path.write_text(
        """
runs:
  finite_episodes:
    environment:
      type: pick_and_place_maple_table
    environment_builder:
      num_envs: 2
      record_trajectories: true
    policy:
      type: zero_action
    rollout_limit:
      num_episodes: 7
    num_rebuilds: 2
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    run_experiment_runner(
        str(experiment_config_path),
        extra_args=[
            "--experiment_output_directory",
            str(output_dir),
            "+runs.finite_episodes.environment_builder.env_cfg_override={episode_length_s:0.2}",
        ],
    )

    experiment_result = json.loads((output_dir / ARENA_EXPERIMENT_RESULT_FILENAME).read_text(encoding="utf-8"))
    run_result = experiment_result["runs"]["finite_episodes"]
    assert run_result["status"] == "completed"
    assert [len(rebuild["episodes"]) for rebuild in run_result["rebuilds"]] == [4, 3]
    run_dir = output_dir / "finite_episodes"
    for rebuild_index, expected_count in enumerate((4, 3)):
        records_path = run_dir / f"episode_results_rebuild{rebuild_index}.jsonl"
        records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == expected_count
        # At the default 15 Hz control rate, the CLI's 0.2 s timeout is three steps.
        assert all(record["episode_length"] == 3 for record in records)
        records_by_identity = {(record["env_id"], record["episode_in_env"]): record for record in records}
        assert len(records_by_identity) == expected_count
        dataset_path = run_dir / f"dataset_finite_episodes_rebuild{rebuild_index}.hdf5"
        with h5py.File(dataset_path, "r") as dataset:
            assert len(dataset["data"]) == expected_count
            exported_identities = set()
            for demo in dataset["data"].values():
                identity = (int(demo["episode_id/env_id"][0]), int(demo["episode_id/episode_in_env"][0]))
                assert identity not in exported_identities
                exported_identities.add(identity)
                record = records_by_identity[identity]
                assert demo["actions"].shape[0] == record["episode_length"] > 0
                assert bool(demo["success"][0]) is record["success"]
            assert exported_identities == set(records_by_identity)


@pytest.mark.with_subprocess
def test_experiment_runner_runs_default_experiment(tmp_path, monkeypatch):
    """Run experiment_runner.py without --experiment_config, as the installation check does."""
    # The default Experiment path is relative to the repository root, where the documented command runs.
    monkeypatch.chdir(TestConstants.repo_root)
    experiment_output_directory = tmp_path / "output"

    run_subprocess([
        TestConstants.python_path,
        f"{TestConstants.evaluation_dir}/experiment_runner.py",
        "--experiment_output_directory",
        str(experiment_output_directory),
    ])

    run_statuses = read_run_statuses(experiment_output_directory)
    assert run_statuses == {"smoke_test": "completed"}


@pytest.mark.with_subprocess
def test_experiment_runner_runs_with_different_env_counts(tmp_path):
    """Run two Runs with different num_envs in one Experiment Runner process."""
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text(
        f"""
runs:
  single_env_joint:
    environment:
      type: gr1_open_microwave
      embodiment: gr1_joint
    policy:
      type: zero_action
    rollout_limit:
      num_steps: {NUM_STEPS}
  multi_env_pink:
    environment:
      type: gr1_open_microwave
      embodiment: gr1_pink
    environment_builder:
      num_envs: 2
    policy:
      type: zero_action
    rollout_limit:
      num_steps: {NUM_STEPS}
""",
        encoding="utf-8",
    )
    experiment_output_directory = tmp_path / "output"

    run_experiment_runner(
        str(experiment_config_path),
        extra_args=["--experiment_output_directory", str(experiment_output_directory)],
    )

    run_statuses = read_run_statuses(experiment_output_directory)
    assert run_statuses == {"single_env_joint": "completed", "multi_env_pink": "completed"}


@pytest.mark.with_subprocess
def test_experiment_runner_graph_spec_camera_run_and_camera_less_run(tmp_path):
    """Run a graph-spec environment with a camera variation next to a camera-less Run.

    The graph-spec Run checks that enable_cameras reaches the embodiment, so the wrist camera variation resolves.
    The camera-less Run checks that a Run without cameras works when the app has camera support.
    """
    graph_spec_yaml = f"{TestConstants.test_data_dir}/pick_and_place_maple_table_env_graph.yaml"
    assert os.path.exists(graph_spec_yaml), f"Graph spec YAML not found: {graph_spec_yaml}"
    experiment_config = {
        "runs": {
            "kitchen_pick_and_place_no_cameras": {
                "environment": {
                    "type": "kitchen_pick_and_place",
                    "object": "cracker_box",
                    "embodiment": "franka_ik",
                },
                "policy": {"type": "zero_action"},
                "rollout_limit": {"num_steps": NUM_STEPS},
            },
            "maple_table_graph_spec_camera_variation": {
                "environment": {"type": graph_spec_yaml, "enable_cameras": True},
                "policy": {"type": "zero_action"},
                "rollout_limit": {"num_steps": NUM_STEPS},
                "variations": {
                    "light": {"hdr_image": {"enabled": True}},
                    "droid_abs_joint_pos": {"camera_extrinsics_wrist_camera": {"enabled": True}},
                },
            },
        }
    }
    experiment_config_path = tmp_path / "graph_spec_camera_and_camera_less_runs.yaml"
    experiment_config_path.write_text(yaml.safe_dump(experiment_config, sort_keys=False), encoding="utf-8")
    experiment_output_directory = tmp_path / "output"

    run_experiment_runner(
        str(experiment_config_path),
        extra_args=["--experiment_output_directory", str(experiment_output_directory)],
    )


def _test_run_variation_lands_in_events_cfg(simulation_app):
    """Enable a wrist camera extrinsics variation and check that it shows up as an event term in the cfg."""
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

    camera_name = "wrist_camera"
    event_name = f"{camera_name}_extrinsics_variation"

    run_cfg = ArenaRunCfg(
        name="maple_table_camera_extrinsics",
        environment=PickAndPlaceMapleTableEnvironmentCfg(enable_cameras=True, embodiment="droid_abs_joint_pos"),
        policy=ZeroActionPolicyCfg(),
        variations={"droid_abs_joint_pos": {f"camera_extrinsics_{camera_name}": {"enabled": True}}},
    )
    arena_builder = build_arena_builder_from_run_cfg(run_cfg)
    _, env_cfg, env_kwargs = arena_builder.build_registered()
    env = arena_builder.make_registered(env_cfg, env_kwargs)
    try:
        env_cfg = env.unwrapped.cfg
        assert hasattr(env_cfg.events, event_name), (
            f"Variation enabled via the run's variations block must add '{event_name}' to env_cfg.events; "
            f"got event fields: {sorted(vars(env_cfg.events))}."
        )
        event_cfg = getattr(env_cfg.events, event_name)
        assert event_cfg.func.__name__ == "apply_camera_extrinsics_from_sampler"
        assert event_cfg.mode == "reset"
        assert event_cfg.params["asset_cfg"].name == camera_name
    finally:
        env.close()
    return True


@pytest.mark.with_cameras
def test_run_variation_lands_in_events_cfg():
    assert run_function_with_persistent_simulation_app(
        _test_run_variation_lands_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=True,
    )
