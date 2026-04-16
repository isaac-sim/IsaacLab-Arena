# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for XR anchor configuration on pelvis-relative humanoid embodiments."""

import numpy as np

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True

_EXPECTED_ANCHOR_POS = (0.0, 0.0, -1.0)
_EXPECTED_ANCHOR_ROT = (0.0, 0.0, -0.70711, 0.70711)
_EXPECTED_ANCHOR_PRIM = "/World/envs/env_0/Robot/pelvis"


def _assert_pelvis_relative_xr_cfg(embodiment_name: str, simulation_app) -> bool:
    """GR1T2 (gr1_pink) and G1 WBC share the same pelvis-anchored XrCfg semantics."""
    from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    embodiment = asset_registry.get_asset_by_name(embodiment_name)()
    xr_cfg = embodiment.get_xr_cfg()

    np.testing.assert_allclose(
        xr_cfg.anchor_pos,
        _EXPECTED_ANCHOR_POS,
        rtol=1e-5,
        err_msg=f"{embodiment_name}: anchor_pos expected {_EXPECTED_ANCHOR_POS}, got {xr_cfg.anchor_pos}",
    )
    np.testing.assert_allclose(
        xr_cfg.anchor_rot,
        _EXPECTED_ANCHOR_ROT,
        rtol=1e-5,
        err_msg=f"{embodiment_name}: anchor_rot expected {_EXPECTED_ANCHOR_ROT}, got {xr_cfg.anchor_rot}",
    )
    assert xr_cfg.anchor_prim_path == _EXPECTED_ANCHOR_PRIM, (
        f"{embodiment_name}: anchor_prim_path expected {_EXPECTED_ANCHOR_PRIM}, got {xr_cfg.anchor_prim_path}"
    )
    assert xr_cfg.fixed_anchor_height is True, f"{embodiment_name}: fixed_anchor_height should be True"
    assert xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED, (
        f"{embodiment_name}: anchor_rotation_mode should be FOLLOW_PRIM_SMOOTHED"
    )

    # Anchor offsets are relative to the pelvis prim, not recomputed from world initial pose.
    robot_pose = Pose(position_xyz=(0.5, 1.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    embodiment.set_initial_pose(robot_pose)
    xr_cfg_after = embodiment.get_xr_cfg()
    np.testing.assert_allclose(
        xr_cfg_after.anchor_pos,
        _EXPECTED_ANCHOR_POS,
        rtol=1e-5,
        err_msg=f"{embodiment_name}: anchor_pos should stay fixed after set_initial_pose",
    )
    np.testing.assert_allclose(
        xr_cfg_after.anchor_rot,
        _EXPECTED_ANCHOR_ROT,
        rtol=1e-5,
        err_msg=f"{embodiment_name}: anchor_rot should stay fixed after set_initial_pose",
    )

    return True


def _test_gr1_pink_xr_anchor(simulation_app) -> bool:
    return _assert_pelvis_relative_xr_cfg("gr1_pink", simulation_app)


def _test_g1_wbc_pink_xr_anchor(simulation_app) -> bool:
    return _assert_pelvis_relative_xr_cfg("g1_wbc_pink", simulation_app)


def test_gr1_pink_xr_anchor_pose():
    """GR1T2 Pink uses a fixed pelvis-relative XR anchor."""
    result = run_simulation_app_function(
        _test_gr1_pink_xr_anchor,
        headless=HEADLESS,
    )
    assert result, "gr1_pink XR anchor test failed"


def test_g1_wbc_pink_xr_anchor_pose():
    """G1 WBC Pink uses the same pelvis-relative XR anchor pattern as GR1T2."""
    result = run_simulation_app_function(
        _test_g1_wbc_pink_xr_anchor,
        headless=HEADLESS,
    )
    assert result, "g1_wbc_pink XR anchor test failed"


if __name__ == "__main__":
    test_gr1_pink_xr_anchor_pose()
    test_g1_wbc_pink_xr_anchor_pose()
