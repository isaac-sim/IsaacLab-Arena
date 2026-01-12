# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import yaml

import pytest

from isaaclab_arena.tests.test_eval_runner import run_eval_runner, write_jobs_config_to_file
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess
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
    args = [TestConstants.python_path, f"{TestConstants.submodules_dir}/Isaac-GR00T/scripts/gr00t_finetune.py"]
    args.append("--dataset_path")
    args.append(Gr00tTestConstants.test_data_dir + "/test_g1_locomanip_lerobot")
    args.append("--output_dir")
    args.append(model_dir)
    args.append("--data_config")
    args.append("isaaclab_arena_gr00t.embodiments.g1.g1_sim_wbc_data_config:UnitreeG1SimWBCDataConfig")
    args.append("--batch_size")
    args.append("1")  # Small batch size for testing
    args.append("--max_steps")
    args.append("10")  # Small number of steps for testing
    args.append("--num_gpus")
    args.append("1")  # Single GPU for testing
    args.append("--save_steps")
    args.append("10")
    args.append("--base_model_path")
    args.append("nvidia/GR00T-N1.5-3B")
    # Disable tuning of the LLM, visual, projector, and diffusion model.
    # This is done to save GPU memory in CI.
    args.append("--no_tune_llm")
    args.append("--no_tune_visual")
    args.append("--no_tune_projector")
    args.append("--no_tune_diffusion_model")
    args.append("--no-resume")
    args.append("--dataloader_num_workers")
    args.append("1")  # Small number of workers for testing
    args.append("--report_to")
    args.append("tensorboard")
    args.append("--embodiment_tag")
    args.append("new_embodiment")
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


def test_g1_locomanip_gr00t_closedloop_policy_runner_single_env(gr00t_finetuned_model_path, tmp_path):
    # Write a new temporary config file with the finetuned model path.
    default_config_file = (
        Gr00tTestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_gr00t_closedloop_config.yaml"
    )
    config_file = get_tmp_config_file(default_config_file, tmp_path, gr00t_finetuned_model_path)

    # Run the model
    args = [TestConstants.python_path, f"{TestConstants.evaluation_dir}/policy_runner.py"]
    args.append("--policy_type")
    args.append("isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy")
    args.append("--policy_config_yaml_path")
    args.append(config_file)
    args.append("--num_steps")
    args.append(str(NUM_STEPS))
    if HEADLESS:
        args.append("--headless")
    if ENABLE_CAMERAS:
        args.append("--enable_cameras")
    # example env
    args.append("galileo_g1_locomanip_pick_and_place")
    args.append("--object")
    args.append("brown_box")
    args.append("--embodiment")
    args.append("g1_wbc_joint")
    run_subprocess(args)


def test_g1_locomanip_gr00t_closedloop_policy_runner_multi_envs(gr00t_finetuned_model_path, tmp_path):
    # Write a new temporary config file with the finetuned model path.
    default_config_file = (
        Gr00tTestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_gr00t_closedloop_config.yaml"
    )
    config_file = get_tmp_config_file(default_config_file, tmp_path, gr00t_finetuned_model_path)

    # Run the model
    args = [TestConstants.python_path, f"{TestConstants.evaluation_dir}/policy_runner.py"]
    args.append("--policy_type")
    args.append("isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy")
    args.append("--policy_config_yaml_path")
    args.append(config_file)
    args.append("--num_steps")
    args.append(str(NUM_STEPS))
    args.append("--num_envs")
    args.append(str(NUM_ENVS))
    if HEADLESS:
        args.append("--headless")
    if ENABLE_CAMERAS:
        args.append("--enable_cameras")
    # example env
    args.append("galileo_g1_locomanip_pick_and_place")
    args.append("--object")
    args.append("brown_box")
    args.append("--embodiment")
    args.append("g1_wbc_joint")
    run_subprocess(args)


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
            "name": "g1_locomanip_pick_and_place_brown_box",
            "arena_env_args": {
                "environment": "galileo_g1_locomanip_pick_and_place",
                "object": "brown_box",
                "embodiment": "g1_wbc_joint",
            },
            "num_steps": 2,
            "policy_type": "isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy",
            "policy_args": {"policy_config_yaml_path": str(policy_config_file), "policy_device": "cuda:0"},
        },
        {
            "name": "gr1_open_microwave_cracker_box",
            "arena_env_args": {
                "environment": "gr1_open_microwave",
                "object": "cracker_box",
                "embodiment": "gr1_joint",
            },
            "num_steps": 2,
            "policy_type": "zero_action",
            "policy_args": {},
        },
    ]
    temp_config_path = str(tmp_path / "test_g1_locomanip_gr00t_closedloop_policy_runner_eval_runner.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner(temp_config_path)


if __name__ == "__main__":
    test_g1_locomanip_gr00t_closedloop_policy_runner_single_env()
    test_g1_locomanip_gr00t_closedloop_policy_runner_multi_envs()
    test_g1_locomanip_gr00t_closedloop_policy_runner_eval_runner()
