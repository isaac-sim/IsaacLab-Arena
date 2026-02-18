# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import yaml

import pytest

from isaaclab_arena.tests.test_eval_runner import run_eval_runner_and_check_no_failures, write_jobs_config_to_file
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function, run_subprocess
from isaaclab_arena_gr00t.tests.utils.constants import TestConstants as Gr00tTestConstants

HEADLESS = True
ENABLE_CAMERAS = True
NUM_STEPS = 17
NUM_ENVS = 3


@pytest.fixture(scope="module")
def gr00t_finetuned_model_path(tmp_path_factory):
    # This function creates a finetuned model for the G1 locomanipulation task.
    # This model is then used by the other tests in the file.

    # Create a temporary directory to store the finetuned model.
    model_dir = tmp_path_factory.mktemp("shared")

    # Run the finetuning script.
    args = [
        TestConstants.python_path,
        f"{TestConstants.submodules_dir}/Isaac-GR00T/gr00t/experiment/launch_finetune.py",
    ]
    args.append("--dataset_path")
    args.append(Gr00tTestConstants.test_data_dir + "/test_g1_locomanip_lerobot")
    args.append("--output_dir")
    args.append(model_dir)
    args.append("--save_total_limit")
    args.append("2")
    args.append("--global_batch_size")
    args.append("1")  # Small batch size for testing
    args.append("--max_steps")
    args.append("10")  # Small number of steps for testing
    args.append("--num_gpus")
    args.append("1")  # Single GPU for testing
    args.append("--save_steps")
    args.append("10")
    args.append("--base_model_path")
    args.append("nvidia/GR00T-N1.6-3B")
    # Disable tuning of the LLM, visual, projector, and diffusion model.
    # This is done to save GPU memory in CI.
    args.append("--no_tune_llm")
    args.append("--no_tune_visual")
    args.append("--no_tune_projector")
    args.append("--no_tune_diffusion_model")
    args.append("--dataloader_num_workers")
    args.append("1")  # Small number of workers for testing
    args.append("--no_use_wandb")
    args.append("--embodiment_tag")
    args.append("NEW_EMBODIMENT")
    args.append("--modality_config_path")
    args.append("isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py")
    args.append("--color_jitter_params")
    # Tyro expects key-value pairs as separate arguments
    args.extend(["brightness", "0.3", "contrast", "0.4", "saturation", "0.5", "hue", "0.08"])
    run_subprocess(args)

    return model_dir / "checkpoint-10"


def get_tmp_config_file(input_config_file, tmp_path, model_path):
    """This function takes a gr00t config file on disk and saves a
    modified version of the file with the model path replaced.
    """
    # TODO(alexmillane. 2025-11-28): The model path should be passed in as a parameter,
    # not read from the file. This would save us the ugly step. Fix this.
    # We open the original config file.
    output_config_file = tmp_path / "test_g1_locomanip_gr00t_closedloop_config.yaml"
    with open(input_config_file) as f:
        config = yaml.safe_load(f)
    # Modify the model path.
    config["model_path"] = str(model_path)
    # Write out to another temporary config file.
    with open(output_config_file, "w") as f:
        yaml.dump(config, f)
    return output_config_file


def _run_gr00t_closedloop_policy(
    simulation_app,
    policy_config_yaml_path: str,
    environment: str,
    object_name: str,
    embodiment: str,
    num_steps: int,
    num_envs: int = 1,
) -> bool:
    """Run the GR00T closedloop policy within a shared simulation app."""
    import sys

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.evaluation.policy_runner import get_policy_cls, rollout_policy
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    try:
        # Get the policy class
        policy_type = "isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy"
        policy_cls = get_policy_cls(policy_type)
        print(f"Requested policy type: {policy_type} -> Policy class: {policy_cls}")

        # Build argument list FIRST - needed before parser setup because
        # get_isaaclab_arena_environments_cli_parser internally calls parse_known_args()
        # which parses sys.argv
        arg_list = [
            "--policy_type",
            policy_type,
            "--policy_config_yaml_path",
            str(policy_config_yaml_path),
            "--num_steps",
            str(num_steps),
            "--num_envs",
            str(num_envs),
            "--headless",
            "--enable_cameras",
            # Environment subparser command and its specific args
            environment,
            "--object",
            object_name,
            "--embodiment",
            embodiment,
        ]

        # Temporarily set sys.argv so that internal parse_known_args() calls work correctly
        old_argv = sys.argv
        sys.argv = [sys.argv[0]] + arg_list

        try:
            # Build the arguments for the policy runner
            args_parser = get_isaaclab_arena_cli_parser()

            # Add environment and policy arguments
            from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments

            add_policy_runner_arguments(args_parser)
            args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
            args_parser = policy_cls.add_args_to_parser(args_parser)

            args_cli = args_parser.parse_args(arg_list)
        finally:
            sys.argv = old_argv

        # Build the environment
        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()

        # Create the policy
        policy = policy_cls.from_args(args_cli)

        # Run the rollout
        rollout_policy(env, policy, num_steps, num_episodes=None)

        # Close the environment
        env.close()
        return True

    except Exception as e:
        print(f"Error running GR00T closedloop policy: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_g1_locomanip_gr00t_closedloop_policy_runner_single_env(gr00t_finetuned_model_path, tmp_path):
    # Write a new temporary config file with the finetuned model path.
    default_config_file = (
        Gr00tTestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_gr00t_closedloop_config.yaml"
    )
    config_file = get_tmp_config_file(default_config_file, tmp_path, gr00t_finetuned_model_path)

    result = run_simulation_app_function(
        _run_gr00t_closedloop_policy,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
        policy_config_yaml_path=str(config_file),
        environment="galileo_g1_locomanip_pick_and_place",
        object_name="brown_box",
        embodiment="g1_wbc_joint",
        num_steps=NUM_STEPS,
        num_envs=1,
    )
    assert result, "Test test_g1_locomanip_gr00t_closedloop_policy_runner_single_env failed"


def test_g1_locomanip_gr00t_closedloop_policy_runner_multi_envs(gr00t_finetuned_model_path, tmp_path):
    # Write a new temporary config file with the finetuned model path.
    default_config_file = (
        Gr00tTestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_gr00t_closedloop_config.yaml"
    )
    config_file = get_tmp_config_file(default_config_file, tmp_path, gr00t_finetuned_model_path)

    result = run_simulation_app_function(
        _run_gr00t_closedloop_policy,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
        policy_config_yaml_path=str(config_file),
        environment="galileo_g1_locomanip_pick_and_place",
        object_name="brown_box",
        embodiment="g1_wbc_joint",
        num_steps=NUM_STEPS,
        num_envs=NUM_ENVS,
    )
    assert result, "Test test_g1_locomanip_gr00t_closedloop_policy_runner_multi_envs failed"


def test_g1_locomanip_gr00t_closedloop_policy_runner_eval_runner(gr00t_finetuned_model_path, tmp_path):
    """Test eval_runner including a G00T closedloop policy and a zero action policy."""

    # Write a new temporary config file with the finetuned model path.
    default_config_file = (
        Gr00tTestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_gr00t_closedloop_config.yaml"
    )
    policy_config_file = get_tmp_config_file(default_config_file, tmp_path, gr00t_finetuned_model_path)

    # create a temporary config file only has two jobs for g1_locomanipulation task
    jobs = [
        {
            "name": "gr1_open_microwave_cracker_box",
            "arena_env_args": {
                "environment": "gr1_open_microwave",
                "num_envs": 10,
                "object": "cracker_box",
                "embodiment": "gr1_joint",
            },
            "num_steps": 2,
            "policy_type": "zero_action",
            "policy_config_dict": {},
        },
        {
            "name": "g1_locomanip_pick_and_place_brown_box",
            "arena_env_args": {
                "enable_cameras": ENABLE_CAMERAS,
                "num_envs": 1,
                "environment": "galileo_g1_locomanip_pick_and_place",
                "object": "brown_box",
                "embodiment": "g1_wbc_joint",
            },
            "num_steps": 2,
            "policy_type": "isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy",
            "policy_config_dict": {"policy_config_yaml_path": str(policy_config_file), "policy_device": "cuda:0"},
        },
    ]
    temp_config_path = str(tmp_path / "test_g1_locomanip_gr00t_closedloop_policy_runner_eval_runner.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner_and_check_no_failures(temp_config_path, headless=HEADLESS)


if __name__ == "__main__":
    # These tests require pytest fixtures, run with: pytest -sv isaaclab_arena_gr00t/tests/test_gr00t_closedloop_policy.py
    import sys

    sys.exit("Run these tests with pytest, not directly.")
