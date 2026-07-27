# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess

import pytest

from isaaclab_arena.evaluation.experiment_runner_cli import parse_experiment_runner_args
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function, run_subprocess

HEADLESS = True
NUM_STEPS = 2
DEFAULT_VISUALIZER = "kit"


def test_experiment_runner_parses_native_hydra_overrides():
    args_cli, experiment_overrides = parse_experiment_runner_args([
        "--experiment_config",
        "experiment.yaml",
        "runs.baseline.rollout_limit.num_steps=2",
        "runs.baseline.environment.enable_cameras=true",
    ])

    assert args_cli.experiment_config == "experiment.yaml"
    assert experiment_overrides == [
        "runs.baseline.rollout_limit.num_steps=2",
        "runs.baseline.environment.enable_cameras=true",
    ]


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


def write_jobs_config_to_file(jobs: list[dict], tmp_file_path: str):
    jobs_config = {"jobs": jobs}

    with open(tmp_file_path, "w", encoding="utf-8") as f:
        json.dump(jobs_config, f, indent=4)


def run_experiment_runner(
    experiment_config_path: str,
    headless: bool = HEADLESS,
    config_option: str = "--eval_jobs_config",
    extra_args: list[str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    """Run the Experiment Runner as a subprocess with timeout.

    --continue_on_error is NOT passed, so the Experiment Runner re-raises on the
    first job failure, exiting non-zero.  run_subprocess() detects that and
    raises CalledProcessError, which surfaces as a test failure.

    Args:
        experiment_config_path: Path to the Experiment configuration file.
        headless: Whether to run in headless mode.
        config_option: CLI option used to pass the Experiment path.
        extra_args: Additional Experiment Runner arguments.
        capture_output: Whether to capture and return the subprocess output.

    Returns:
        The completed subprocess when output is captured, otherwise None.
    """
    args = [TestConstants.python_path, f"{TestConstants.evaluation_dir}/experiment_runner.py"]
    args.append(config_option)
    args.append(experiment_config_path)
    args.extend(extra_args or [])
    if headless:
        args.append("--headless")
    else:
        args.append("--viz")
        args.append(DEFAULT_VISUALIZER)

    return run_subprocess(args, capture_output=capture_output)


@pytest.mark.with_subprocess
def test_experiment_runner_from_typed_yaml(tmp_path):
    """Execute a typed YAML Experiment through the Experiment Runner CLI."""
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text(
        """
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
        config_option="--experiment_config",
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


@pytest.mark.with_subprocess
def test_experiment_runner_two_jobs_zero_action(tmp_path):
    """Test experiment_runner with 2 jobs using zero_action policy on different objects."""
    jobs = [
        {
            "name": "gr1_open_microwave_cracker_box",
            "arena_env_args": {
                "environment": "gr1_open_microwave",
                "object": "cracker_box",
                "embodiment": "gr1_joint",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
        {
            "name": "gr1_open_microwave_sugar_box",
            "arena_env_args": {
                "environment": "gr1_open_microwave",
                "object": "sugar_box",
                "embodiment": "gr1_joint",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
    ]

    temp_config_path = str(tmp_path / "test_experiment_runner_two_jobs_zero_action.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_experiment_runner(temp_config_path)


@pytest.mark.with_subprocess
def test_experiment_runner_multiple_environments(tmp_path):
    """Test experiment_runner with jobs across different environments."""
    jobs = [
        {
            "name": "kitchen_pick_cracker_box",
            "arena_env_args": {
                "environment": "kitchen_pick_and_place",
                "object": "cracker_box",
                "embodiment": "gr1_joint",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
        {
            "name": "kitchen_pick_power_drill",
            "arena_env_args": {
                "environment": "put_item_in_fridge_and_close_door",
                "object": "power_drill",
                "embodiment": "gr1_pink",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
    ]

    temp_config_path = str(tmp_path / "test_experiment_runner_multiple_environments.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_experiment_runner(temp_config_path)


@pytest.mark.with_subprocess
def test_experiment_runner_different_embodiments(tmp_path):
    """Test experiment_runner with jobs using different embodiments."""
    jobs = [
        {
            "name": "kitchen_pick_gr1_pink",
            "arena_env_args": {
                "environment": "kitchen_pick_and_place",
                "object": "tomato_soup_can",
                "embodiment": "gr1_pink",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
        {
            "name": "kitchen_pick_franka",
            "arena_env_args": {
                "environment": "kitchen_pick_and_place",
                "object": "tomato_soup_can",
                "embodiment": "franka_ik",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
    ]

    temp_config_path = str(tmp_path / "test_experiment_runner_different_embodiments.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_experiment_runner(temp_config_path)


@pytest.mark.with_subprocess
def test_experiment_runner_from_existing_config():
    """Test experiment_runner using the zero_action_jobs_config.json and verify no jobs failed."""
    config_path = f"{TestConstants.arena_environments_dir}/eval_jobs_configs/zero_action_jobs_config.json"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    run_experiment_runner(config_path)


@pytest.mark.with_subprocess
def test_experiment_runner_with_variations(tmp_path):
    """Test experiment_runner applies a per-job variations block via Hydra overrides."""
    jobs = [
        {
            "name": "maple_table_hdr_variation",
            "arena_env_args": {
                "environment": "pick_and_place_maple_table",
                "embodiment": "droid_abs_joint_pos",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_config_dict": {},
            "variations": {"light": {"hdr_image": {"enabled": True}}},
        },
    ]

    temp_config_path = str(tmp_path / "test_experiment_runner_with_variations.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_experiment_runner(temp_config_path)


@pytest.mark.with_subprocess
def test_experiment_runner_enable_cameras(tmp_path):
    """Test experiment_runner with enable_cameras set to true."""
    jobs = [
        {
            "name": "kitchen_pick_and_place_no_cameras",
            "arena_env_args": {
                "environment": "kitchen_pick_and_place",
                "object": "cracker_box",
                "embodiment": "franka_ik",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
        {
            "name": "kitchen_pick_and_place",
            "arena_env_args": {
                "enable_cameras": True,
                "environment": "kitchen_pick_and_place",
                "object": "cracker_box",
                "embodiment": "franka_ik",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
    ]

    temp_config_path = str(tmp_path / "test_experiment_runner_enable_cameras.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_experiment_runner(temp_config_path)


@pytest.mark.with_subprocess
def test_experiment_runner_graph_spec_with_variation(tmp_path):
    """Eval a graph-spec env (built from YAML) with --enable_cameras and a camera variation.

    Mirrors the example-environment camera-variation job but sources the env from a graph spec
    YAML, exercising that the Experiment Runner builds graph-spec envs and that --enable_cameras reaches
    the embodiment so the wrist camera (and its variation) resolve.
    """
    graph_spec_yaml = f"{TestConstants.test_data_dir}/pick_and_place_maple_table_env_graph.yaml"
    assert os.path.exists(graph_spec_yaml), f"Graph spec YAML not found: {graph_spec_yaml}"
    jobs = [
        {
            "name": "maple_table_graph_spec_camera_variation",
            "arena_env_args": {
                "enable_cameras": True,
                "environment": graph_spec_yaml,
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
            "variations": {
                "light": {"hdr_image": {"enabled": True}},
                "droid_abs_joint_pos": {"camera_extrinsics_wrist_camera": {"enabled": True}},
            },
        },
    ]

    temp_config_path = str(tmp_path / "test_experiment_runner_graph_spec_with_variation.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_experiment_runner(temp_config_path)


def _test_eval_config_variation_lands_in_events_cfg(simulation_app):
    """Enable a wrist camera extrinsics variation and check that it shows up as an event term in the cfg."""
    from isaaclab_arena.evaluation.legacy_eval_config import run_cfgs_from_legacy_eval_config
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg

    camera_name = "wrist_camera"
    event_name = f"{camera_name}_extrinsics_variation"

    experiment_config = {
        "jobs": [{
            "name": "maple_table_camera_extrinsics",
            "arena_env_args": {
                "num_envs": 1,
                "enable_cameras": True,
                "environment": "pick_and_place_maple_table",
                "embodiment": "droid_abs_joint_pos",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_config_dict": {},
            # Enabling wrist camera extrinsics variation.
            "variations": {"droid_abs_joint_pos": {f"camera_extrinsics_{camera_name}": {"enabled": True}}},
        }]
    }
    (run_cfg,) = run_cfgs_from_legacy_eval_config(experiment_config, device="cuda:0")
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
def test_eval_config_variation_lands_in_events_cfg():
    assert run_simulation_app_function(
        _test_eval_config_variation_lands_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=True,
    )
