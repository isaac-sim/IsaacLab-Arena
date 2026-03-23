# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Dexsuite Kuka Allegro lift Arena example (no simulation)."""

import pytest

pytest.importorskip("isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg")


def test_kuka_allegro_dexsuite_lift_example_in_cli_registry() -> None:
    from isaaclab_arena_environments.cli import ExampleEnvironments

    assert "kuka_allegro_dexsuite_lift" in ExampleEnvironments
    assert ExampleEnvironments["kuka_allegro_dexsuite_lift"].name == "kuka_allegro_dexsuite_lift"


def test_dexsuite_procedural_assets_registered() -> None:
    from isaaclab_arena.assets.asset_registry import AssetRegistry

    reg = AssetRegistry()
    assert reg.is_registered("dexsuite_manip_table")
    assert reg.is_registered("dexsuite_lift_object")


def test_dexsuite_kuka_lift_task_matches_lift_mdp_flags() -> None:
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.tasks.dexsuite_kuka_allegro_lift_task import DexsuiteKukaAllegroLiftTask

    from isaaclab_arena.tasks.lift_object_task import LiftObjectTask

    reg = AssetRegistry()
    lift = reg.get_asset_by_name("dexsuite_lift_object")()
    table = reg.get_asset_by_name("dexsuite_manip_table")()
    task = DexsuiteKukaAllegroLiftTask(lift_object=lift, background_scene=table)
    assert isinstance(task, LiftObjectTask)
    assert task.lift_object is lift
    assert task.get_scene_cfg() is None
    assert task._rewards_cfg.orientation_tracking is None
    assert task._commands_cfg.object_pose.position_only is True
    assert task._rewards_cfg.success.params.get("rot_std") is None
    assert task.get_metrics() == []
