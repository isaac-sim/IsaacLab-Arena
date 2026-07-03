# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed Maple-table environment configuration."""

import argparse

from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
    PickAndPlaceMapleTableEnvironment,
    PickAndPlaceMapleTableEnvironmentCfg,
)


def test_maple_table_environment_cfg_defaults():
    cfg = PickAndPlaceMapleTableEnvironmentCfg()

    assert cfg.name == "pick_and_place_maple_table"
    assert not cfg.enable_cameras
    assert cfg.embodiment_asset_name == "droid_abs_joint_pos"
    assert cfg.teleoperation_device_name is None
    assert cfg.high_dynamic_range_image_name is None
    assert cfg.light_intensity == 500.0
    assert cfg.pick_up_object_asset_name == "rubiks_cube_hot3d_robolab"
    assert cfg.destination_location_asset_name == "bowl_ycb_robolab"
    assert cfg.additional_table_object_asset_names == []


def test_maple_table_legacy_namespace_maps_to_typed_cfg(monkeypatch):
    provider = object.__new__(PickAndPlaceMapleTableEnvironment)
    captured = {}
    expected_environment = object()

    def fake_build(self, cfg):
        captured["cfg"] = cfg
        return expected_environment

    monkeypatch.setattr(PickAndPlaceMapleTableEnvironment, "build", fake_build)
    legacy_arguments = argparse.Namespace(
        enable_cameras=True,
        embodiment="droid_rel_joint_pos",
        teleop_device="spacemouse",
        hdr="home_office_robolab",
        light_intensity=725.0,
        pick_up_object="mustard_bottle_ycb_robolab",
        destination_location="bowl_ycb_robolab",
        additional_table_objects=["cracker_box_ycb_robolab"],
    )

    environment = provider.get_env(legacy_arguments)

    assert environment is expected_environment
    assert captured["cfg"] == PickAndPlaceMapleTableEnvironmentCfg(
        enable_cameras=True,
        embodiment_asset_name="droid_rel_joint_pos",
        teleoperation_device_name="spacemouse",
        high_dynamic_range_image_name="home_office_robolab",
        light_intensity=725.0,
        pick_up_object_asset_name="mustard_bottle_ycb_robolab",
        destination_location_asset_name="bowl_ycb_robolab",
        additional_table_object_asset_names=["cracker_box_ycb_robolab"],
    )
