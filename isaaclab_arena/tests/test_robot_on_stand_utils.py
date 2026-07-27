# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``robot_on_stand_utils``: cached USD compose and runtime stand hierarchy."""

from __future__ import annotations

import torch
import traceback
from pathlib import Path

import pytest
import warp as wp

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

_HEIGHT_ATOL = 1e-3
_ROOT_WRITE_ATOL = 5e-3
_ROOT_WRITE_DELTA_X = 1.0
_CUSTOM_DROID_STAND_HEIGHT_M = 2.0
_HEADLESS = True


def _assert_compose_on_stand_usd(
    robot_usd_path: str,
    mount,
    *,
    stand_height_m: float,
    output_basename: str,
    payload_child_name: str,
    check_orient_180z: bool = False,
) -> str:
    from pxr import Usd, UsdGeom

    from isaaclab_arena.embodiments.robot_on_stand_utils import (
        ROBOT_BASE_PRIM_PATH,
        STAND_PRIM_PATH,
        compose_on_stand_usd,
    )

    usd_path = compose_on_stand_usd(
        robot_usd_path,
        mount,
        stand_height_m=stand_height_m,
        output_basename=output_basename,
    )
    assert Path(usd_path).is_file()

    stage = Usd.Stage.Open(usd_path)
    assert stage is not None
    stand = stage.GetPrimAtPath(STAND_PRIM_PATH)
    assert stand.IsValid(), f"missing stand prim at {STAND_PRIM_PATH!r}"
    robot_base = stage.GetPrimAtPath(ROBOT_BASE_PRIM_PATH)
    assert robot_base.IsValid(), f"missing robot base prim at {ROBOT_BASE_PRIM_PATH!r}"
    assert stand.GetParent() == robot_base

    payload = stage.GetPrimAtPath(f"{STAND_PRIM_PATH}/{payload_child_name}")
    assert payload.IsValid(), f"missing payload child {payload_child_name!r}"
    assert abs(_stand_world_height_m(stage) - stand_height_m) < _HEIGHT_ATOL
    if check_orient_180z:
        mesh_xf = UsdGeom.Xformable(payload).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        assert mesh_xf[0][0] < 0.0 and mesh_xf[1][1] < 0.0, mesh_xf
    return usd_path


def _stand_world_height_m(stage) -> float:
    from pxr import Usd, UsdGeom

    from isaaclab_arena.embodiments.robot_on_stand_utils import STAND_PRIM_PATH

    stand = stage.GetPrimAtPath(STAND_PRIM_PATH)
    assert stand.IsValid(), f"missing stand prim at {STAND_PRIM_PATH!r}"
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    return float(cache.ComputeWorldBound(stand).ComputeAlignedRange().GetSize()[2])


def _stand_top_z(stage) -> float:
    from pxr import Usd, UsdGeom

    from isaaclab_arena.embodiments.robot_on_stand_utils import STAND_PRIM_PATH

    stand = stage.GetPrimAtPath(STAND_PRIM_PATH)
    assert stand.IsValid(), f"missing stand prim at {STAND_PRIM_PATH!r}"
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    return float(cache.ComputeWorldBound(stand).ComputeAlignedRange().GetMax()[2])


def _assert_prim_parent(prim_path: str, expected_parent_name: str) -> None:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid(), f"missing prim at {prim_path}"
    parent = prim.GetParent()
    assert (
        parent and parent.GetName() == expected_parent_name
    ), f"{prim_path} parent should be {expected_parent_name}, got {parent.GetPath() if parent else None}"


def _link0_world_xy(env) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    body_ids, _ = robot.find_bodies("panda_link0")
    return wp.to_torch(robot.data.body_pos_w)[0, body_ids[0], :2].detach().cpu().float()


def _write_root_x_delta(env, delta_x: float) -> None:
    robot = env.unwrapped.scene["robot"]
    root_pose = wp.to_torch(robot.data.root_pose_w).clone()
    root_pose[:, 0] += delta_x
    env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(
        torch.zeros(env.unwrapped.num_envs, 6, device=env.unwrapped.device), env_ids=env_ids
    )


def _assert_root_write_moves_link0(env) -> None:
    before = _link0_world_xy(env)
    _write_root_x_delta(env, _ROOT_WRITE_DELTA_X)
    env.unwrapped.sim.step(render=False)
    after = _link0_world_xy(env)
    assert abs(float(after[0] - before[0]) - _ROOT_WRITE_DELTA_X) < _ROOT_WRITE_ATOL, (before, after)


def _test_droid_on_stand_usd_compose(simulation_app) -> bool:
    """Droid compose: height, align, orient, distinct paths per height, embodiment wiring."""
    from pxr import Usd

    from isaaclab_arena.embodiments.droid.droid import (
        _DEFAULT_STAND_HEIGHT_M,
        _DROID_ROBOT_USD_PATH,
        _DROID_STAND_MOUNT,
        DroidAbsoluteJointPositionEmbodiment,
    )

    try:
        default_usd = _assert_compose_on_stand_usd(
            _DROID_ROBOT_USD_PATH,
            _DROID_STAND_MOUNT,
            stand_height_m=_DEFAULT_STAND_HEIGHT_M,
            output_basename="droid_franka_robotiq_on_stand",
            payload_child_name="franka_table",
            check_orient_180z=True,
        )

        custom_usd = _assert_compose_on_stand_usd(
            _DROID_ROBOT_USD_PATH,
            _DROID_STAND_MOUNT,
            stand_height_m=_CUSTOM_DROID_STAND_HEIGHT_M,
            output_basename="droid_franka_robotiq_on_stand",
            payload_child_name="franka_table",
        )
        assert custom_usd != default_usd
        assert abs(_stand_top_z(Usd.Stage.Open(custom_usd)) - _stand_top_z(Usd.Stage.Open(default_usd))) < _HEIGHT_ATOL

        default_emb = DroidAbsoluteJointPositionEmbodiment()
        custom_emb = DroidAbsoluteJointPositionEmbodiment(stand_height_m=_CUSTOM_DROID_STAND_HEIGHT_M)
        assert not hasattr(default_emb.scene_config, "stand")
        assert custom_emb.scene_config.robot.spawn.usd_path != default_emb.scene_config.robot.spawn.usd_path
        assert abs(custom_emb.scene_config.robot.init_state.pos[2]) < 1e-6
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_franka_on_stand_usd_compose(simulation_app) -> bool:
    """Franka compose reaches the legacy default stand height."""
    from isaaclab_arena.embodiments.franka.franka import (
        _FRANKA_DEFAULT_STAND_HEIGHT_M,
        _FRANKA_ROBOT_USD_PATH,
        _FRANKA_STAND_MOUNT,
    )

    try:
        _assert_compose_on_stand_usd(
            _FRANKA_ROBOT_USD_PATH,
            _FRANKA_STAND_MOUNT,
            stand_height_m=_FRANKA_DEFAULT_STAND_HEIGHT_M,
            output_basename="franka_panda_on_stand",
            payload_child_name="Stand",
        )
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    return True


def _test_franka_stand_under_link0_follows_root_write(simulation_app) -> bool:
    """Franka runtime: stand under link0; root write moves PhysX ``panda_link0``."""
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    env = None
    try:
        arena_env = IsaacLabArenaEnvironment(name="franka_stand_follow", embodiment=FrankaIKEmbodiment(), scene=Scene())
        args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
        env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
        env.reset()
        env.unwrapped.sim.step(render=False)

        stand_path = "/World/envs/env_0/Robot/panda_link0/stand_instanceable"
        _assert_prim_parent(stand_path, "panda_link0")
        _assert_root_write_moves_link0(env)
        _assert_prim_parent(stand_path, "panda_link0")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        if env is not None:
            env.close()
    return True


def _test_droid_stand_and_externals_under_link0(simulation_app) -> bool:
    """Droid runtime: stand and external cameras under ``panda_link0``; root write moves link0."""
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    env = None
    try:
        arena_env = IsaacLabArenaEnvironment(
            name="droid_stand_follow",
            embodiment=DroidAbsoluteJointPositionEmbodiment(enable_cameras=True, stand_height_m=0.8),
            scene=Scene(),
        )
        args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
        env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
        env.reset()
        env.unwrapped.sim.step(render=True)

        stand_path = "/World/envs/env_0/Robot/panda_link0/stand_instanceable"
        cam_paths = (
            "/World/envs/env_0/Robot/panda_link0/external_camera",
            "/World/envs/env_0/Robot/panda_link0/external_camera_2",
        )
        _assert_prim_parent(stand_path, "panda_link0")
        for path in cam_paths:
            _assert_prim_parent(path, "panda_link0")
        _assert_root_write_moves_link0(env)
        _assert_prim_parent(stand_path, "panda_link0")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        if env is not None:
            env.close()
    return True


def test_droid_on_stand_usd_compose():
    result = run_simulation_app_function(_test_droid_on_stand_usd_compose, headless=_HEADLESS)
    assert result, f"Test {test_droid_on_stand_usd_compose.__name__} failed"


def test_franka_on_stand_usd_compose():
    result = run_simulation_app_function(_test_franka_on_stand_usd_compose, headless=_HEADLESS)
    assert result, f"Test {test_franka_on_stand_usd_compose.__name__} failed"


def test_franka_stand_under_link0_follows_root_write():
    result = run_simulation_app_function(_test_franka_stand_under_link0_follows_root_write, headless=_HEADLESS)
    assert result, f"Test {test_franka_stand_under_link0_follows_root_write.__name__} failed"


@pytest.mark.with_cameras
def test_droid_stand_and_externals_under_link0():
    result = run_simulation_app_function(
        _test_droid_stand_and_externals_under_link0,
        headless=_HEADLESS,
        enable_cameras=True,
    )
    assert result, f"Test {test_droid_stand_and_externals_under_link0.__name__} failed"
