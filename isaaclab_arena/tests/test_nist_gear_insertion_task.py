# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import torch
from dataclasses import dataclass

import pytest

_PEG_BASE_OFFSET = (0.02025, 0.0, 0.0)
_PEG_TIP_OFFSET = (0.02025, 0.0, 0.025)


@dataclass
class _DummyAsset:
    name: str
    object_min_z: float | None = None
    reset_pose_disabled: bool = False

    def disable_reset_pose(self) -> None:
        self.reset_pose_disabled = True


def _make_nist_task(**kwargs):
    from isaaclab_arena.tasks.nist_gear_insertion.task import GearInsertionGeometryCfg, NistGearInsertionTask

    return NistGearInsertionTask(
        held_gear=_DummyAsset("medium_nist_gear"),
        background_scene=_DummyAsset("table", object_min_z=-0.05),
        gear_base_asset=_DummyAsset("gears_and_base"),
        geometry_cfg=GearInsertionGeometryCfg(
            peg_base_offset=_PEG_BASE_OFFSET,
            peg_tip_offset=_PEG_TIP_OFFSET,
            held_gear_base_offset=_PEG_BASE_OFFSET,
            success_z_fraction=0.20,
            xy_threshold=0.0025,
        ),
        **kwargs,
    )


def test_task_is_evaluation_only_and_uses_geometry_for_success():
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.tasks.nist_gear_insertion import geometry
    from isaaclab_arena.tasks.nist_gear_insertion.terminations import gear_mesh_insertion_success

    task = _make_nist_task()

    obs_cfg = task.get_observation_cfg()
    termination_cfg = task.get_termination_cfg()

    assert not task.held_gear.reset_pose_disabled
    assert task.get_events_cfg() is None
    assert task.get_rewards_cfg() is None
    assert task.get_commands_cfg() is None
    assert task.get_curriculum_cfg() is None
    assert task.get_scene_cfg() is None
    assert task.get_task_description() == "Insert the medium NIST gear onto the NIST gear base."

    metrics = task.get_metrics()
    assert len(metrics) == 1
    assert isinstance(metrics[0], SuccessRateMetric)

    assert obs_cfg.task_obs.gear_pos.params["asset_cfg"].name == "medium_nist_gear"
    assert obs_cfg.task_obs.fixed_base_pos.params["asset_cfg"].name == "gears_and_base"
    assert obs_cfg.task_obs.peg_pos.func is geometry.peg_pos_in_env_frame
    assert obs_cfg.task_obs.peg_pos.params["peg_offset"] == _PEG_TIP_OFFSET
    assert obs_cfg.task_obs.peg_delta.func is geometry.peg_delta_from_held_gear_base
    assert obs_cfg.task_obs.peg_delta.params["held_gear_base_offset"] == _PEG_BASE_OFFSET
    assert obs_cfg.task_obs.concatenate_terms

    assert termination_cfg.success.func is gear_mesh_insertion_success
    assert termination_cfg.success.params["gear_base_offset"] == _PEG_BASE_OFFSET
    assert termination_cfg.success.params["held_gear_base_offset"] == _PEG_BASE_OFFSET
    assert termination_cfg.success.params["success_z_fraction"] == 0.20
    assert "disable_success_termination" not in termination_cfg.success.params
    assert termination_cfg.object_dropped.params["asset_cfg"].name == "medium_nist_gear"
    assert termination_cfg.object_dropped.params["minimum_height"] == -0.05


def test_gear_insertion_geometry_success_thresholds():
    from isaaclab_arena.tasks.nist_gear_insertion.geometry import check_gear_insertion_geometry

    held_base_pos = torch.tensor([
        [0.0, 0.0, 0.001],
        [0.01, 0.0, 0.001],
        [0.0, 0.0, 0.01],
    ])
    peg_pos = torch.zeros(3, 3)

    success = check_gear_insertion_geometry(
        held_base_pos=held_base_pos,
        peg_pos=peg_pos,
        gear_peg_height=0.02,
        z_fraction=0.2,
        xy_threshold=0.0025,
    )

    assert success.tolist() == [True, False, False]


def test_nist_environment_is_registered_with_default_droid_args():
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena_environments.cli import add_environment_cli_args, ensure_environments_registered
    from isaaclab_arena_environments.nist_assembled_gear_mesh_environment import (
        DROID_EMBODIMENTS,
        NistAssembledGearMeshEnvironmentCfg,
    )

    ensure_environments_registered()
    env_registry = EnvironmentRegistry()
    assert env_registry.is_registered("nist_assembled_gear_mesh")

    environment_factory_type = env_registry.get_component_by_name("nist_assembled_gear_mesh")
    assert environment_factory_type.name == "nist_assembled_gear_mesh"
    assert environment_factory_type._legacy_argparse_cfg_type is NistAssembledGearMeshEnvironmentCfg

    cfg = NistAssembledGearMeshEnvironmentCfg()
    assert cfg.embodiment == "droid_abs_joint_pos"
    assert cfg.embodiment in DROID_EMBODIMENTS
    assert cfg.episode_length_s == 15.0

    parser = argparse.ArgumentParser(exit_on_error=False)
    add_environment_cli_args(parser, environment_factory_type)
    args = parser.parse_args([])
    assert args.embodiment == "droid_abs_joint_pos"
    assert args.episode_length_s == 15.0


def test_nist_environment_rejects_non_droid_embodiment():
    from isaaclab_arena_environments.nist_assembled_gear_mesh_environment import (
        NistAssembledGearMeshEnvironment,
        NistAssembledGearMeshEnvironmentCfg,
    )

    environment = NistAssembledGearMeshEnvironment()
    with pytest.raises(AssertionError, match="only supports existing DROID embodiments"):
        environment.build(NistAssembledGearMeshEnvironmentCfg(embodiment="unsupported_robot"))
