# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function, run_subprocess

HEADLESS = True
NUM_STEPS = 2
DEFAULT_VISUALIZER = "kit"


def write_legacy_json_experiment(jobs: list[dict], config_path: str | Path) -> None:
    """Write a legacy JSON Experiment for compatibility coverage."""
    legacy_experiment = {"jobs": jobs}

    with Path(config_path).open("w", encoding="utf-8") as config_file:
        json.dump(legacy_experiment, config_file, indent=4)


def write_yaml_experiment(runs: list[dict], config_path: str | Path) -> None:
    """Write a typed YAML Experiment for eval-runner coverage."""
    OmegaConf.save(config={"runs": runs}, f=config_path)


def run_eval_runner(
    experiment_config_path: str | None,
    headless: bool = HEADLESS,
    config_option: str = "--experiment_config",
    extra_args: list[str] | None = None,
):
    """Run the eval_runner as a subprocess with timeout.

    --continue_on_error is NOT passed, so the eval_runner re-raises on the
    first Run failure, exiting non-zero. run_subprocess() detects that and
    raises CalledProcessError, which surfaces as a test failure.

    Args:
        experiment_config_path: Path to the Experiment configuration file, or None to use the default.
        headless: Whether to run in headless mode.
        config_option: CLI option used to pass the Experiment path.
        extra_args: Additional eval_runner arguments.
    """
    args = [TestConstants.python_path, f"{TestConstants.evaluation_dir}/eval_runner.py"]
    if experiment_config_path is not None:
        args.append(config_option)
        args.append(experiment_config_path)
    args.extend(extra_args or [])
    if headless:
        args.append("--headless")
    else:
        args.append("--viz")
        args.append(DEFAULT_VISUALIZER)

    run_subprocess(args)


@pytest.mark.with_subprocess
def test_eval_runner_from_typed_yaml(tmp_path):
    """Execute a typed YAML Experiment through the neutral eval_runner CLI."""
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text(
        """
runs:
- name: yaml_baseline
  environment:
    type: pick_and_place_maple_table
  policy:
    type: zero_action
  rollout_limit:
    num_steps: 10
""",
        encoding="utf-8",
    )

    run_eval_runner(
        str(experiment_config_path),
        config_option="--experiment_config",
        extra_args=[
            "--experiment_override",
            "runs.yaml_baseline.rollout_limit.num_steps=2",
            "--output_base_dir",
            str(tmp_path / "output"),
        ],
    )


@pytest.mark.with_subprocess
def test_eval_runner_from_legacy_json_experiment(tmp_path):
    """Keep one registered-environment JSON Experiment covered during migration."""
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

    config_path = tmp_path / "legacy_experiment.json"
    write_legacy_json_experiment(jobs, config_path)
    run_eval_runner(str(config_path), config_option="--eval_jobs_config")


@pytest.mark.with_subprocess
def test_eval_runner_multiple_environments(tmp_path):
    """Test a typed YAML Experiment with Runs across different environments."""
    runs = [
        {
            "name": "kitchen_pick_cracker_box",
            "environment": {
                "type": "kitchen_pick_and_place",
                "object": "cracker_box",
                "embodiment": "gr1_joint",
            },
            "policy": {"type": "zero_action"},
            "rollout_limit": {"num_steps": NUM_STEPS},
        },
        {
            "name": "kitchen_pick_power_drill",
            "environment": {
                "type": "put_item_in_fridge_and_close_door",
                "object": "power_drill",
                "embodiment": "gr1_pink",
            },
            "policy": {"type": "zero_action"},
            "rollout_limit": {"num_steps": NUM_STEPS},
        },
    ]

    config_path = tmp_path / "multiple_environments_experiment.yaml"
    write_yaml_experiment(runs, config_path)
    run_eval_runner(str(config_path))


@pytest.mark.with_subprocess
def test_eval_runner_different_embodiments(tmp_path):
    """Test a typed YAML Experiment with different embodiments."""
    runs = [
        {
            "name": "kitchen_pick_gr1_pink",
            "environment": {
                "type": "kitchen_pick_and_place",
                "object": "tomato_soup_can",
                "embodiment": "gr1_pink",
            },
            "policy": {"type": "zero_action"},
            "rollout_limit": {"num_steps": NUM_STEPS},
        },
        {
            "name": "kitchen_pick_franka",
            "environment": {
                "type": "kitchen_pick_and_place",
                "object": "tomato_soup_can",
                "embodiment": "franka_ik",
            },
            "policy": {"type": "zero_action"},
            "rollout_limit": {"num_steps": NUM_STEPS},
        },
    ]

    config_path = tmp_path / "different_embodiments_experiment.yaml"
    write_yaml_experiment(runs, config_path)
    run_eval_runner(str(config_path))


@pytest.mark.with_subprocess
def test_eval_runner_uses_default_typed_yaml_experiment(tmp_path):
    """Run the default typed YAML Experiment without selecting a config file."""
    run_eval_runner(
        None,
        extra_args=[
            "--experiment_override",
            "runs.gr1_open_microwave_cracker_box.rollout_limit.num_steps=2",
            "--experiment_override",
            "runs.gr1_open_microwave_sugar_box.rollout_limit.num_steps=2",
            "--output_base_dir",
            str(tmp_path / "output"),
        ],
    )


@pytest.mark.with_subprocess
def test_eval_runner_with_variations(tmp_path):
    """Test eval_runner applies variations declared by a typed YAML Run."""
    runs = [
        {
            "name": "maple_table_hdr_variation",
            "environment": {
                "type": "pick_and_place_maple_table",
                "embodiment": "droid_abs_joint_pos",
            },
            "policy": {"type": "zero_action"},
            "rollout_limit": {"num_steps": NUM_STEPS},
            "variations": {"light": {"hdr_image": {"enabled": True}}},
        },
    ]

    config_path = tmp_path / "variations_experiment.yaml"
    write_yaml_experiment(runs, config_path)
    run_eval_runner(str(config_path))


@pytest.mark.with_subprocess
def test_eval_runner_enable_cameras(tmp_path):
    """Test a camera-enabled typed YAML Run with process camera support."""
    runs = [
        {
            "name": "kitchen_pick_and_place_no_cameras",
            "environment": {
                "type": "kitchen_pick_and_place",
                "object": "cracker_box",
                "embodiment": "franka_ik",
            },
            "policy": {"type": "zero_action"},
            "rollout_limit": {"num_steps": NUM_STEPS},
        },
        {
            "name": "kitchen_pick_and_place",
            "environment": {
                "type": "kitchen_pick_and_place",
                "enable_cameras": True,
                "object": "cracker_box",
                "embodiment": "franka_ik",
            },
            "policy": {"type": "zero_action"},
            "rollout_limit": {"num_steps": NUM_STEPS},
        },
    ]

    config_path = tmp_path / "camera_experiment.yaml"
    write_yaml_experiment(runs, config_path)
    run_eval_runner(str(config_path), extra_args=["--enable_cameras"])


@pytest.mark.with_subprocess
def test_eval_runner_graph_spec_with_variation(tmp_path):
    """Eval a graph-spec env (built from YAML) with --enable_cameras and a camera variation.

    Mirrors the example-environment camera-variation job but sources the env from a graph spec
    YAML, exercising that the eval runner builds graph-spec envs and that --enable_cameras reaches
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

    temp_config_path = str(tmp_path / "test_eval_runner_graph_spec_with_variation.json")
    write_legacy_json_experiment(jobs, temp_config_path)
    run_eval_runner(temp_config_path, config_option="--eval_jobs_config")


def _test_eval_config_variation_lands_in_events_cfg(simulation_app, experiment_config_path: Path):
    """Enable a wrist camera extrinsics variation and check that it shows up as an event term in the cfg."""
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg

    camera_name = "wrist_camera"
    event_name = f"{camera_name}_extrinsics_variation"

    (run_cfg,) = load_arena_experiment_from_config_file(experiment_config_path, device="cuda:0")
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
def test_eval_config_variation_lands_in_events_cfg(tmp_path):
    experiment_config_path = tmp_path / "camera_variation_experiment.yaml"
    write_yaml_experiment(
        [{
            "name": "maple_table_camera_extrinsics",
            "environment": {
                "type": "pick_and_place_maple_table",
                "enable_cameras": True,
                "embodiment": "droid_abs_joint_pos",
            },
            "policy": {"type": "zero_action"},
            "rollout_limit": {"num_steps": NUM_STEPS},
            "variations": {"droid_abs_joint_pos": {"camera_extrinsics_wrist_camera": {"enabled": True}}},
        }],
        experiment_config_path,
    )

    assert run_simulation_app_function(
        _test_eval_config_variation_lands_in_events_cfg,
        headless=HEADLESS,
        enable_cameras=True,
        experiment_config_path=experiment_config_path,
    )
