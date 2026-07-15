# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

_DEFAULT_STAND_SCALE = (1.2, 1.2, 1.7)
_CUSTOM_STAND_HEIGHT = 2.0
_EXPECTED_CUSTOM_SCALE = (1.2, 1.2, 2.0)


def _test_droid_stand_scale(simulation_app) -> bool:
    """Check ``stand_height`` sets only the stand z-scale and lifts the robot base to match."""

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.droid.droid import (
        _DEFAULT_STAND_HEIGHT_SCALE,
        DroidAbsoluteJointPositionEmbodiment,
        _stand_unit_height_m,
    )
    from isaaclab_arena.utils.pose import Pose

    try:
        # The default leaves the stand at its nominal scale and the robot base at z=0.
        default_emb = DroidAbsoluteJointPositionEmbodiment()
        assert tuple(default_emb.scene_config.stand.spawn.scale) == _DEFAULT_STAND_SCALE
        assert default_emb.scene_config.robot.init_state.pos[2] == 0.0

        # The lift is driven by the stand's native height (read from its USD, with a fallback).
        unit_height = _stand_unit_height_m(default_emb.scene_config.stand.spawn.usd_path)
        expected_offset = unit_height * (_CUSTOM_STAND_HEIGHT - _DEFAULT_STAND_HEIGHT_SCALE)

        # An override changes only the z-scale (x/y footprint and robot mesh untouched)...
        custom_emb = DroidAbsoluteJointPositionEmbodiment(stand_height=_CUSTOM_STAND_HEIGHT)
        assert tuple(custom_emb.scene_config.stand.spawn.scale) == _EXPECTED_CUSTOM_SCALE
        assert custom_emb.scene_config.robot.spawn.scale in (None, (1.0, 1.0, 1.0))

        # ...and lifts the robot base and stand together so the stand's floor contact is preserved.
        assert abs(custom_emb.scene_config.robot.init_state.pos[2] - expected_offset) < 1e-6
        assert abs(custom_emb.scene_config.stand.init_state.pos[2] - expected_offset) < 1e-6

        # The lift is re-applied on top of an explicit initial_pose (which the base class would
        # otherwise overwrite): the requested z plus the stand-height offset.
        posed_emb = DroidAbsoluteJointPositionEmbodiment(stand_height=_CUSTOM_STAND_HEIGHT)
        posed_emb.set_initial_pose(Pose(position_xyz=(0.3, 0.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        scene_cfg = posed_emb.get_scene_cfg()
        assert abs(scene_cfg.robot.init_state.pos[2] - (0.5 + expected_offset)) < 1e-6
        assert abs(scene_cfg.stand.init_state.pos[2] - (0.5 + expected_offset)) < 1e-6

        # The YAML-spec path instantiates the embodiment via asset_class(**params); a scalar
        # stand_height from YAML applies just the same.
        registry_emb = AssetRegistry().get_asset_by_name("droid_abs_joint_pos")(stand_height=_CUSTOM_STAND_HEIGHT)
        assert tuple(registry_emb.scene_config.stand.spawn.scale) == _EXPECTED_CUSTOM_SCALE

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def test_droid_stand_scale():
    """Pytest entry point for the Droid stand-scale configuration test."""
    result = run_simulation_app_function(_test_droid_stand_scale, headless=True)
    assert result, f"Test {test_droid_stand_scale.__name__} failed"


if __name__ == "__main__":
    test_droid_stand_scale()
