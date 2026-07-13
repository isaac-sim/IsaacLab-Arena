# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test OpenPI policy composition through typed Arena Experiment YAML."""

from pathlib import Path

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation import arena_experiment_config_loader
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
    expected_run_values = {
        "droid_pnp_srl_openpi_billiard_hall": (
            "rubiks_cube_hot3d_robolab",
            "bowl_ycb_robolab",
            "billiard_hall_robolab",
            "Pick up the Rubik's cube and place it in the bowl.",
        ),
        "droid_pnp_srl_openpi_rubiks_cube_home_office": (
            "rubiks_cube_hot3d_robolab",
            "wooden_bowl_hot3d_robolab",
            "home_office_robolab",
            "Pick up the Rubik's cube and place it in the bowl.",
        ),
        "droid_pnp_srl_openpi_alphabet_soup_can": (
            "alphabet_soup_can_hope_robolab",
            "bowl_ycb_robolab",
            "empty_warehouse_robolab",
            "Pick up the soup can and place it in the bowl.",
        ),
        "droid_pnp_srl_openpi_orange": (
            "orange_01_fruits_veggies_robolab",
            "wooden_bowl_hot3d_robolab",
            "aerodynamics_workshop_robolab",
            "Pick up the orange and place it in the bowl.",
        ),
        "droid_pnp_srl_openpi_lemon": (
            "lemon_01_fruits_veggies_robolab",
            "bowl_ycb_robolab",
            "wooden_lounge_robolab",
            "Pick up the lemon and place it in the bowl.",
        ),
        "droid_pnp_srl_openpi_tomato_sauce_can": (
            "tomato_sauce_can_hot3d_robolab",
            "wooden_bowl_hot3d_robolab",
            "garage_robolab",
            "Pick up the tomato sauce can and place it in the bowl.",
        ),
        "droid_pnp_srl_openpi_mustard_bottle": (
            "mustard_bottle_hot3d_robolab",
            "wooden_bowl_hot3d_robolab",
            "kiara_interior_robolab",
            "Pick up the mustard bottle and place it in the bowl.",
        ),
        "droid_pnp_srl_openpi_sugar_box": (
            "sugar_box_ycb_robolab",
            "bowl_ycb_robolab",
            "brown_photostudio_robolab",
            "Pick up the sugar box and place it in the bowl.",
        ),
        "droid_pnp_srl_openpi_mug": (
            "mug_ycb_robolab",
            "wooden_bowl_hot3d_robolab",
            "carpentry_shop_robolab",
            "Pick up the mug and place it in the bowl.",
        ),
    }

    assert list(experiment_cfg.runs) == list(expected_run_values)
    for run_name, expected_values in expected_run_values.items():
        run_cfg = experiment_cfg.runs[run_name]
        pick_up_object, destination_location, hdr, language_instruction = expected_values
        assert run_cfg.name == run_name
        assert isinstance(run_cfg.environment, PickAndPlaceMapleTableEnvironmentCfg)
        assert run_cfg.environment.enable_cameras is True
        assert run_cfg.environment.embodiment == "droid_abs_joint_pos"
        assert run_cfg.environment.pick_up_object == pick_up_object
        assert run_cfg.environment.destination_location == destination_location
        assert run_cfg.environment.hdr == hdr
        assert run_cfg.environment_builder.language_instruction == language_instruction
        assert isinstance(run_cfg.policy, Pi0RemotePolicyCfg)
        assert run_cfg.policy.policy_variant == "pi05"
        assert run_cfg.policy.policy_device == "cuda:0"
        assert run_cfg.policy.remote_host == "127.0.0.1"
        assert run_cfg.policy.remote_port == 8000
        assert run_cfg.policy.openpi_embodiment_adapter == "droid"
        assert run_cfg.rollout_limit.num_episodes == 3
        assert run_cfg.environment_builder.device == "cuda:1"


def _write_openpi_experiment(path: Path) -> Path:
    path.write_text(
        """
runs:
  remote:
    environment:
      type: test_environment
    policy:
      type: pi0_remote
      openpi_embodiment_adapter: droid
      remote_host: localhost
      remote_port: 8000
    rollout_limit:
      num_steps: 1
""",
        encoding="utf-8",
    )
    return path


def test_typed_openpi_experiment_composes_runtime_endpoint_overrides(tmp_path, monkeypatch):
    """Compose OpenPI and apply the runtime server endpoint through Hydra overrides."""
    experiment_path = _write_openpi_experiment(tmp_path / "openpi.yaml")
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "_registered_environment_cfg_types",
        lambda: {"test_environment": ArenaEnvironmentCfg},
    )

    experiment = load_arena_experiment_from_config_file(
        experiment_path,
        device="cuda:1",
        overrides=[
            "runs.remote.policy.remote_host='{{host:policy_server}}'",
            "runs.remote.policy.remote_port=8123",
        ],
    )

    assert len(experiment) == 1
    assert isinstance(experiment[0].policy, Pi0RemotePolicyCfg)
    assert experiment[0].policy.remote_host == "{{host:policy_server}}"
    assert experiment[0].policy.remote_port == 8123
    assert experiment[0].environment_builder.device == "cuda:1"
