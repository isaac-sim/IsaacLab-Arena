# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the typed OpenPI Experiment configuration."""

from pathlib import Path

from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg

OPENPI_EXPERIMENT_PATH = (
    Path(__file__).parents[2]
    / "isaaclab_arena_environments"
    / "experiment_configs"
    / "droid_pnp_srl_openpi_experiment.yaml"
)


def test_openpi_experiment_loads_all_typed_runs():
    experiment_cfg = load_arena_experiment_from_config_file(OPENPI_EXPERIMENT_PATH, device="cuda:1")
    expected_run_names = [
        "droid_pnp_srl_openpi_billiard_hall",
        "droid_pnp_srl_openpi_rubiks_cube_home_office",
        "droid_pnp_srl_openpi_alphabet_soup_can",
        "droid_pnp_srl_openpi_orange",
        "droid_pnp_srl_openpi_lemon",
        "droid_pnp_srl_openpi_tomato_sauce_can",
        "droid_pnp_srl_openpi_mustard_bottle",
        "droid_pnp_srl_openpi_sugar_box",
        "droid_pnp_srl_openpi_mug",
    ]

    assert list(experiment_cfg.runs) == expected_run_names
    for run_name, run_cfg in experiment_cfg.runs.items():
        assert run_cfg.name == run_name
        assert isinstance(run_cfg.environment, PickAndPlaceMapleTableEnvironmentCfg)
        assert run_cfg.environment.enable_cameras is True
        assert run_cfg.environment.embodiment == "droid_abs_joint_pos"
        assert isinstance(run_cfg.policy, Pi0RemotePolicyCfg)
        assert run_cfg.policy.policy_variant == "pi05"
        assert run_cfg.policy.policy_device == "cuda:0"
        assert run_cfg.policy.remote_host == "127.0.0.1"
        assert run_cfg.policy.remote_port == 8000
        assert run_cfg.policy.openpi_embodiment_adapter == "droid"
        assert run_cfg.rollout_limit.num_episodes == 3
        assert run_cfg.environment_builder.device == "cuda:1"

    home_office_run = experiment_cfg.runs["droid_pnp_srl_openpi_rubiks_cube_home_office"]
    assert home_office_run.environment.pick_up_object == "rubiks_cube_hot3d_robolab"
    assert home_office_run.environment.destination_location == "wooden_bowl_hot3d_robolab"
    assert home_office_run.environment.hdr == "home_office_robolab"
    assert (
        home_office_run.environment_builder.language_instruction == "Pick up the Rubik's cube and place it in the bowl."
    )

    soup_run = experiment_cfg.runs["droid_pnp_srl_openpi_alphabet_soup_can"]
    assert soup_run.environment.pick_up_object == "alphabet_soup_can_hope_robolab"
    assert soup_run.environment.destination_location == "bowl_ycb_robolab"
    assert soup_run.environment.hdr == "empty_warehouse_robolab"
    assert soup_run.environment_builder.language_instruction == "Pick up the soup can and place it in the bowl."
