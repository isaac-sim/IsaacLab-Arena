# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Dexsuite Kuka Allegro lift Arena example (no simulation)."""

import pytest

pytest.importorskip(
    "isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg"
)


def test_dexsuite_lift_example_in_cli_registry() -> None:
    from isaaclab_arena_environments.cli import ExampleEnvironments

    assert "dexsuite_lift" in ExampleEnvironments
    assert ExampleEnvironments["dexsuite_lift"].name == "dexsuite_lift"


def test_procedural_assets_registered() -> None:
    from isaaclab_arena.assets.asset_registry import AssetRegistry

    reg = AssetRegistry()
    assert reg.is_registered("procedural_table")
    assert reg.is_registered("procedural_cube")


def test_dexsuite_kuka_lift_task_matches_lift_mdp_flags() -> None:
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.tasks.lift_object_task import DexsuiteLiftTask, DexsuiteLiftTerminationsCfg, LiftObjectTask
    from isaaclab_arena.utils.pose import Pose, PoseRange

    reg = AssetRegistry()
    lift = reg.get_asset_by_name("procedural_cube")()
    lift.set_initial_pose(PoseRange(position_xyz_min=(-0.75, -0.1, 0.35), position_xyz_max=(-0.35, 0.3, 0.75)))
    table = reg.get_asset_by_name("procedural_table")()
    table.set_initial_pose(Pose(position_xyz=(-0.55, 0.0, 0.235)))
    task = DexsuiteLiftTask(lift_object=lift, background_scene=table)
    assert isinstance(task, LiftObjectTask)
    assert task.lift_object is lift
    assert task.get_scene_cfg() is None
    assert task.get_rewards_cfg() is None
    assert task.commands_cfg.object_pose.position_only is True
    metrics = task.get_metrics()
    assert len(metrics) == 1
    assert isinstance(metrics[0], SuccessRateMetric)
    assert metrics[0].recorder_term_name == "success"
    assert isinstance(task.termination_cfg, DexsuiteLiftTerminationsCfg)
    assert hasattr(task.termination_cfg, "success")
