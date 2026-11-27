# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load the Experiments shown on the concept pages with the overrides the pages document."""

from pathlib import Path

import pytest

from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg
from isaaclab_arena.tests.utils.constants import TestConstants

_EXPERIMENT_CONFIG_DIRECTORY = Path(TestConstants.arena_environments_dir) / "experiment_configs"
_CONCEPT_EXPERIMENT_CONFIG_DIRECTORY = _EXPERIMENT_CONFIG_DIRECTORY / "concepts"


def _load_run(config_path: Path, run_name: str, overrides: list[str] | None = None) -> ArenaRunCfg:
    """Load an Experiment that declares only ``run_name`` and return that Run."""
    experiment_cfg = load_arena_experiment_from_config_file(config_path, device="cpu", overrides=overrides)
    assert list(experiment_cfg.runs) == [run_name]
    return experiment_cfg.runs[run_name]


@pytest.mark.parametrize(
    "config_path", sorted(_CONCEPT_EXPERIMENT_CONFIG_DIRECTORY.glob("*.yaml")), ids=lambda path: path.name
)
def test_concept_experiment_loads(config_path):
    experiment_cfg = load_arena_experiment_from_config_file(config_path, device="cpu")

    assert experiment_cfg.runs


def test_additional_objects_override_replaces_the_table_objects():
    config_path = _CONCEPT_EXPERIMENT_CONFIG_DIRECTORY / "maple_table_additional_objects_experiment.yaml"

    run_cfg = _load_run(
        config_path,
        "additional_objects",
        ["runs.additional_objects.environment.additional_table_objects=[sugar_box,mustard_bottle,power_drill]"],
    )

    assert run_cfg.environment.additional_table_objects == ["sugar_box", "mustard_bottle", "power_drill"]


def test_placement_seed_override_unlocks_placement():
    config_path = _CONCEPT_EXPERIMENT_CONFIG_DIRECTORY / "maple_table_seeds_experiment.yaml"

    locked_run_cfg = _load_run(config_path, "locked_seeds")
    unlocked_run_cfg = _load_run(
        config_path, "locked_seeds", ["runs.locked_seeds.environment_builder.placement_seed=null"]
    )

    assert locked_run_cfg.environment_builder.seed == 42
    assert locked_run_cfg.environment_builder.placement_seed == 7
    assert unlocked_run_cfg.environment_builder.placement_seed is None


def test_mode_override_switches_to_heterogeneous_placement():
    config_path = _CONCEPT_EXPERIMENT_CONFIG_DIRECTORY / "droid_table_multi_object_placement_experiment.yaml"

    homogeneous_run_cfg = _load_run(config_path, "multi_object_placement")
    heterogeneous_run_cfg = _load_run(
        config_path, "multi_object_placement", ["runs.multi_object_placement.environment.mode=heterogeneous"]
    )

    assert homogeneous_run_cfg.environment.mode == "homogeneous"
    assert heterogeneous_run_cfg.environment.mode == "heterogeneous"
    assert heterogeneous_run_cfg.environment_builder.num_envs > 1


def test_robot_placement_experiment_selects_a_graph_spec_file():
    config_path = _CONCEPT_EXPERIMENT_CONFIG_DIRECTORY / "kitchen_bench_robot_placement_experiment.yaml"

    run_cfg = _load_run(config_path, "robot_placement")

    assert isinstance(run_cfg.environment, LegacyGraphEnvironmentCfg)
    assert (Path(TestConstants.repo_root) / run_cfg.environment.env_spec_path).is_file()


def test_variation_overrides_add_tuned_control_parameters():
    config_path = _EXPERIMENT_CONFIG_DIRECTORY / "droid_pnp_variations_experiment.yaml"

    run_cfg = _load_run(
        config_path,
        "variations_demo",
        [
            "runs.variations_demo.variations.light.hdr_image.hdr_names=[home_office_robolab,billiard_hall_robolab,garage_robolab]",
            "runs.variations_demo.variations.droid_rel_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.low=[-0.01,-0.01,-0.01]",
            "runs.variations_demo.variations.droid_rel_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.high=[0.01,0.01,0.01]",
        ],
    )

    assert run_cfg.variations["light.hdr_image.enabled"] is True
    assert run_cfg.variations["droid_rel_joint_pos.camera_extrinsics_wrist_camera.enabled"] is True
    assert run_cfg.variations["light"]["hdr_image"]["hdr_names"] == [
        "home_office_robolab",
        "billiard_hall_robolab",
        "garage_robolab",
    ]
    wrist_camera_sampler_cfg = run_cfg.variations["droid_rel_joint_pos"]["camera_extrinsics_wrist_camera"][
        "sampler_cfg"
    ]
    assert wrist_camera_sampler_cfg == {"low": [-0.01, -0.01, -0.01], "high": [0.01, 0.01, 0.01]}
