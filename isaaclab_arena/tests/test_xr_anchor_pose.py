# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for XR anchor pose configuration in embodiments."""

import numpy as np

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_gr1t2_xr_anchor_pose(simulation_app) -> bool:
    """Test GR1T2 XR anchor pose at origin and with robot transformation."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.utils.pose import Pose
    
    # Test 1: XR anchor at origin (no initial pose)
    asset_registry = AssetRegistry()
    embodiment = asset_registry.get_asset_by_name("gr1_pink")()
    xr_cfg = embodiment.get_xr_cfg()
    
    expected_pos = embodiment._xr_offset.position_xyz
    expected_rot = embodiment._xr_offset.rotation_wxyz
    
    assert xr_cfg.anchor_pos == expected_pos, \
        f"XR anchor position should match offset at origin: expected {expected_pos}, got {xr_cfg.anchor_pos}"
    assert xr_cfg.anchor_rot == expected_rot, \
        f"XR anchor rotation should match offset at origin: expected {expected_rot}, got {xr_cfg.anchor_rot}"
    
    print("✓ GR1T2 XR anchor at origin: PASSED")
    
    # Test 2: XR anchor with robot position and rotation
    robot_pose = Pose(
        position_xyz=(1.0, 2.0, 0.0),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0)  # No rotation
    )
    embodiment.set_initial_pose(robot_pose)
    xr_cfg = embodiment.get_xr_cfg()
    
    # Expected position: robot_pos + offset
    expected_pos = (
        robot_pose.position_xyz[0] + embodiment._xr_offset.position_xyz[0],  # 1.0 + (-0.5) = 0.5
        robot_pose.position_xyz[1] + embodiment._xr_offset.position_xyz[1],  # 2.0 + 0.0 = 2.0
        robot_pose.position_xyz[2] + embodiment._xr_offset.position_xyz[2],  # 0.0 + (-1.0) = -1.0
    )
    
    np.testing.assert_allclose(
        xr_cfg.anchor_pos,
        expected_pos,
        rtol=1e-5,
        err_msg=f"XR anchor position incorrect with robot pose: expected {expected_pos}, got {xr_cfg.anchor_pos}"
    )
    
    print("✓ GR1T2 XR anchor with robot translation: PASSED")
    
    # Test 3: XR anchor with robot rotation
    robot_pose_rotated = Pose(
        position_xyz=(0.0, 0.0, 0.0),
        rotation_wxyz=(0.70711, 0.0, 0.0, 0.70711)  # 90° rotation around Z
    )
    embodiment.set_initial_pose(robot_pose_rotated)
    xr_cfg_rotated = embodiment.get_xr_cfg()
    
    # Rotation should be composed, not same as offset
    assert xr_cfg_rotated.anchor_rot != embodiment._xr_offset.rotation_wxyz, \
        "XR anchor rotation should be composed with robot rotation"
    
    print("✓ GR1T2 XR anchor with robot rotation: PASSED")
    
    # Test 4: Dynamic recomputation
    pose1 = Pose(position_xyz=(1.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    embodiment.set_initial_pose(pose1)
    xr_cfg1 = embodiment.get_xr_cfg()
    
    pose2 = Pose(position_xyz=(2.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    embodiment.set_initial_pose(pose2)
    xr_cfg2 = embodiment.get_xr_cfg()
    
    assert xr_cfg1.anchor_pos != xr_cfg2.anchor_pos, \
        "XR anchor should change when robot pose changes"
    
    pos_diff = tuple(
        xr_cfg2.anchor_pos[i] - xr_cfg1.anchor_pos[i]
        for i in range(3)
    )
    expected_diff = (1.0, 0.0, 0.0)
    
    np.testing.assert_allclose(
        pos_diff,
        expected_diff,
        rtol=1e-5,
        err_msg="XR anchor position difference should match robot movement"
    )
    
    print("✓ GR1T2 XR anchor dynamic recomputation: PASSED")
        
    
    return True


def _test_g1_xr_anchor_pose(simulation_app) -> bool:
    """Test G1 XR anchor pose at origin and with robot transformation."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.utils.pose import Pose
    
    # Test 1: XR anchor at origin (no initial pose)
    asset_registry = AssetRegistry()
    embodiment = asset_registry.get_asset_by_name("g1_wbc_pink")()
    xr_cfg = embodiment.get_xr_cfg()
    
    expected_pos = embodiment._xr_offset.position_xyz
    expected_rot = embodiment._xr_offset.rotation_wxyz
    
    assert xr_cfg.anchor_pos == expected_pos, \
        f"XR anchor position should match offset at origin: expected {expected_pos}, got {xr_cfg.anchor_pos}"
    assert xr_cfg.anchor_rot == expected_rot, \
        f"XR anchor rotation should match offset at origin: expected {expected_rot}, got {xr_cfg.anchor_rot}"
    
    print("✓ G1 XR anchor at origin: PASSED")
    
    # Test 2: XR anchor with robot position
    robot_pose = Pose(
        position_xyz=(0.5, 1.0, 0.0),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0)  # No rotation
    )
    embodiment.set_initial_pose(robot_pose)
    xr_cfg = embodiment.get_xr_cfg()
    
    # G1 offset is (0.0, 0.0, -1.0)
    expected_pos = (
        robot_pose.position_xyz[0] + embodiment._xr_offset.position_xyz[0],  # 0.5 + 0.0 = 0.5
        robot_pose.position_xyz[1] + embodiment._xr_offset.position_xyz[1],  # 1.0 + 0.0 = 1.0
        robot_pose.position_xyz[2] + embodiment._xr_offset.position_xyz[2],  # 0.0 + (-1.0) = -1.0
    )
    
    np.testing.assert_allclose(
        xr_cfg.anchor_pos,
        expected_pos,
        rtol=1e-5,
        err_msg=f"XR anchor position incorrect with robot pose: expected {expected_pos}, got {xr_cfg.anchor_pos}"
    )
    
    print("✓ G1 XR anchor with robot translation: PASSED")

    return True


def _test_xr_anchor_multiple_positions(simulation_app) -> bool:
    """Test XR anchor with multiple different robot positions."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.utils.pose import Pose
    
    asset_registry = AssetRegistry()
    embodiment = asset_registry.get_asset_by_name("gr1_pink")()
    xr_cfg = embodiment.get_xr_cfg()
    test_positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 0.0),
    ]
    
    for pos in test_positions:
        robot_pose = Pose(
            position_xyz=pos,
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0)
        )
        embodiment.set_initial_pose(robot_pose)
        
        xr_cfg = embodiment.get_xr_cfg()
        
        # Verify XR anchor moved with robot
        expected_pos = tuple(
            robot_pos + offset_pos
            for robot_pos, offset_pos in zip(pos, embodiment._xr_offset.position_xyz)
        )
        
        np.testing.assert_allclose(
            xr_cfg.anchor_pos,
            expected_pos,
            rtol=1e-5,
            err_msg=f"XR anchor incorrect for robot at {pos}: expected {expected_pos}, got {xr_cfg.anchor_pos}"
        )
        
        print(f"✓ XR anchor correct for robot at {pos}: PASSED")

    return True


# Public test functions that pytest will discover
def test_gr1t2_xr_anchor_pose():
    """Test GR1T2 XR anchor pose behavior."""
    result = run_simulation_app_function(
        _test_gr1t2_xr_anchor_pose,
        headless=HEADLESS,
    )
    assert result, "GR1T2 XR anchor pose test failed"


def test_g1_xr_anchor_pose():
    """Test G1 XR anchor pose behavior."""
    result = run_simulation_app_function(
        _test_g1_xr_anchor_pose,
        headless=HEADLESS,
    )
    assert result, "G1 XR anchor pose test failed"


def test_xr_anchor_multiple_positions():
    """Test XR anchor with multiple robot positions."""
    result = run_simulation_app_function(
        _test_xr_anchor_multiple_positions,
        headless=HEADLESS,
    )
    assert result, "Multiple positions XR anchor test failed"


if __name__ == "__main__":
    test_gr1t2_xr_anchor_pose()
    test_g1_xr_anchor_pose()
    test_xr_anchor_multiple_positions()
