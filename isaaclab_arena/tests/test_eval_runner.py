# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function, run_subprocess

HEADLESS = True
NUM_STEPS = 2
DEFAULT_VISUALIZER = "kit"


def write_jobs_config_to_file(jobs: list[dict], tmp_file_path: str):
    jobs_config = {"jobs": jobs}

    with open(tmp_file_path, "w", encoding="utf-8") as f:
        json.dump(jobs_config, f, indent=4)


def run_eval_runner(jobs_config_path: str, headless: bool = HEADLESS):
    """Run the eval_runner as a subprocess with timeout.

    --continue_on_error is NOT passed, so the eval_runner re-raises on the
    first job failure, exiting non-zero.  run_subprocess() detects that and
    raises CalledProcessError, which surfaces as a test failure.

    Args:
        jobs_config_path: Path to the jobs config JSON file.
        headless: Whether to run in headless mode.
    """
    args = [TestConstants.python_path, f"{TestConstants.evaluation_dir}/eval_runner.py"]
    args.append("--eval_jobs_config")
    args.append(jobs_config_path)
    if headless:
        args.append("--headless")
    else:
        args.append("--viz")
        args.append(DEFAULT_VISUALIZER)

    run_subprocess(args)


@pytest.mark.with_subprocess
def test_eval_runner_two_jobs_zero_action(tmp_path):
    """Test eval_runner with 2 jobs using zero_action policy on different objects."""
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

    temp_config_path = str(tmp_path / "test_eval_runner_two_jobs_zero_action.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner(temp_config_path)


@pytest.mark.with_subprocess
def test_eval_runner_multiple_environments(tmp_path):
    """Test eval_runner with jobs across different environments."""
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

    temp_config_path = str(tmp_path / "test_eval_runner_multiple_environments.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner(temp_config_path)


@pytest.mark.with_subprocess
def test_eval_runner_different_embodiments(tmp_path):
    """Test eval_runner with jobs using different embodiments."""
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

    temp_config_path = str(tmp_path / "test_eval_runner_different_embodiments.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner(temp_config_path)


@pytest.mark.with_subprocess
def test_eval_runner_from_existing_config():
    """Test eval_runner using the zero_action_jobs_config.json and verify no jobs failed."""
    config_path = f"{TestConstants.arena_environments_dir}/eval_jobs_configs/zero_action_jobs_config.json"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    run_eval_runner(config_path)


@pytest.mark.with_subprocess
def test_eval_runner_with_variations(tmp_path):
    """Test eval_runner applies a per-job variations block via Hydra overrides."""
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

    temp_config_path = str(tmp_path / "test_eval_runner_with_variations.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner(temp_config_path)


@pytest.mark.with_subprocess
def test_eval_runner_enable_cameras(tmp_path):
    """Test eval_runner with enable_cameras set to true."""
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

    temp_config_path = str(tmp_path / "test_eval_runner_enable_cameras.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner(temp_config_path)


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
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner(temp_config_path)


def _test_eval_config_variation_lands_in_events_cfg(simulation_app):
    """Enable a wrist camera extrinsics variation and check that it shows up as an event term in the cfg."""
    from isaaclab_arena.evaluation.eval_runner import load_env
    from isaaclab_arena.evaluation.job_manager import Job

    camera_name = "wrist_camera"
    event_name = f"{camera_name}_extrinsics_variation"

    job = Job.from_dict({
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
    })

    env = load_env(job.arena_env_args, job.name, variations=job.variations)
    try:
        env_cfg = env.unwrapped.cfg
        assert hasattr(env_cfg.events, event_name), (
            f"Variation enabled via the job's variations block must add '{event_name}' to env_cfg.events; "
            f"got event fields: {sorted(vars(env_cfg.events))}."
        )
        event_cfg = getattr(env_cfg.events, event_name)
        # load_env() reloads arena modules, so compare by name rather than class identity.
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
