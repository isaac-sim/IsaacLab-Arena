# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check physical clearance and command-based grasp detection."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_gripper_released(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    import pytest

    from isaaclab_arena.tasks.predicates.gripper import gripper_released

    for device in ("cpu", "cuda:0"):
        for dtype in (torch.float32, torch.float64):
            # Closed, grasping, opening but still touching, below/exactly at clearance, and physically open.
            joint_positions = torch.tensor(
                [[0.0], [0.015625], [0.015625], [0.0166015625], [0.017578125], [0.01953125], [0.0625]],
                device=device,
                dtype=dtype,
            )
            jaw_gaps = 2.0 * joint_positions[:, 0]
            gripper = SimpleNamespace(get_opening_width_m=lambda _world: jaw_gaps)
            action = SimpleNamespace(processed_actions=torch.zeros_like(joint_positions))
            env = SimpleNamespace(
                arena_world=SimpleNamespace(),
                action_manager=SimpleNamespace(get_term={"gripper": action}.__getitem__),
            )
            params = dict(
                gripper=gripper,
                grasp_width_m=0.03125,
                release_clearance_m=0.00390625,
            )
            expected = [False, False, False, False, False, True, True]
            result = gripper_released(env, **params)
            assert result.tolist() == expected
            assert result.device == joint_positions.device and result.dtype == torch.bool

            # An opening command cannot release an object before the jaws actually move.
            action.processed_actions[:] = 0.0625
            assert gripper_released(env, **params).tolist() == expected
            jaw_gaps[2] = 0.125
            assert gripper_released(env, **params)[2]

            with pytest.raises(AssertionError, match="clearance"):
                gripper_released(env, **{**params, "release_clearance_m": -0.001})
    return True


def test_gripper_released() -> None:
    assert run_function_with_persistent_simulation_app(_test_gripper_released)


def _test_gripper_not_grasping(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    import pytest

    from isaaclab_arena.tasks.predicates.gripper import gripper_not_grasping

    # Binary fractions keep exact boundary checks independent of rounding.
    width, band, margin = 1 / 32, 1 / 256, 1 / 1024
    for device in ("cpu", "cuda:0"):
        for dtype in (torch.float32, torch.float64):
            gaps = torch.tensor([0, width, width, width, width - band, width + band], device=device, dtype=dtype)
            errors = torch.tensor([0, 2 * margin, margin, -margin, 2 * margin, 2 * margin], device=device, dtype=dtype)
            gripper = SimpleNamespace(
                get_opening_width_m=lambda _world: gaps,
                get_closing_error_m=lambda _env: errors,
            )
            env = SimpleNamespace(arena_world=SimpleNamespace())
            params = dict(gripper=gripper, grasp_width_m=width, gap_band_m=band, stall_margin_m=margin)
            result = gripper_not_grasping(env, **params)
            assert result.tolist() == [True, False, True, True, True, True]
            assert result.device == gaps.device and result.dtype == torch.bool
            for key, value in (("grasp_width_m", 0.0), ("gap_band_m", 0.0), ("stall_margin_m", -1.0)):
                with pytest.raises(AssertionError, match=key):
                    gripper_not_grasping(env, **{**params, key: value})
    return True


def test_gripper_not_grasping() -> None:
    assert run_function_with_persistent_simulation_app(_test_gripper_not_grasping)


def _test_shipped_grippers_not_grasping(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.embodiments.gripper import PandaGripper, RobotiqGripper
    from isaaclab_arena.environments.arena_world import ArenaWorld
    from isaaclab_arena.tasks.predicates.gripper import gripper_not_grasping

    for device in ("cpu", "cuda:0"):
        for dtype in (torch.float32, torch.float64):
            # Closed empty, stalled at object width, matched target, opening, and fully open.
            measured_finger = torch.tensor([0.0, 0.015, 0.015, 0.015, 0.04], device=device, dtype=dtype)
            target_finger = torch.tensor([0.0, 0.0, 0.015, 0.04, 0.04], device=device, dtype=dtype)
            for gripper in (PandaGripper(), RobotiqGripper(), RobotiqGripper(driver_joint_name="custom_driver")):
                if isinstance(gripper, PandaGripper):
                    measured = measured_finger
                    target = target_finger
                    joint_names = [gripper.right_finger_joint_name, "unused", gripper.left_finger_joint_name]
                else:
                    # Invert the linkage to create known linear displacements for the revolute driver.
                    measured = 0.715 - torch.asin((2 * measured_finger - 0.01) / 0.1143)
                    target = 0.715 - torch.asin((2 * target_finger - 0.01) / 0.1143)
                    joint_names = ["unused_mimic", "unused", gripper.driver_joint_name or "finger_joint"]
                data = SimpleNamespace(
                    joint_names=joint_names,
                    joint_pos=SimpleNamespace(
                        torch=torch.stack([measured, torch.ones_like(measured), measured], dim=1)
                    ),
                    joint_pos_target=SimpleNamespace(
                        torch=torch.stack([torch.ones_like(target), torch.ones_like(target), target], dim=1)
                    ),
                )
                left_pad = torch.zeros((5, 3), device=device, dtype=dtype)
                left_pad[:, 1] = measured_finger
                sensor_data = SimpleNamespace(
                    target_frame_names=["tool_rightfinger", "tool_leftfinger"],
                    target_pos_w=SimpleNamespace(torch=torch.stack([-left_pad, left_pad], dim=1)),
                )
                world = ArenaWorld(
                    SimpleNamespace(
                        num_envs=5,
                        articulations={"robot": SimpleNamespace(data=data)},
                        sensors={"ee_frame": SimpleNamespace(data=sensor_data)},
                    )
                )
                env = SimpleNamespace(arena_world=world)
                torch.testing.assert_close(
                    gripper.get_closing_error_m(env), measured_finger - target_finger, atol=1e-8, rtol=1e-5
                )
                result = gripper_not_grasping(env, gripper, grasp_width_m=0.03, gap_band_m=0.001, stall_margin_m=0.0002)
                assert result.tolist() == [True, False, True, True, True]
                assert result.device == measured.device and result.dtype == torch.bool

                # Read current targets on each call, independent of action-term names and column order.
                data.joint_pos_target.torch = data.joint_pos.torch.clone()
                assert gripper_not_grasping(
                    env, gripper, grasp_width_m=0.03, gap_band_m=0.001, stall_margin_m=0.0002
                ).all()
    return True


def test_shipped_grippers_not_grasping() -> None:
    assert run_function_with_persistent_simulation_app(_test_shipped_grippers_not_grasping)
