# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SimApp test that the cuRobo IK oracle separates a reachable grasp from an unreachable one.

Builds a minimal DROID scene with two objects, anchors one inside the arm's workspace (at the
current end-effector position) and pushes the other far out of reach, then asserts the batched IK
feasibility check marks exactly the near object reachable.

The check launches its own SimulationApp, so the pytest entry point runs it in a subprocess to keep
it isolated from the shared persistent-SimApp tests. Runs only in the cuRobo image (build with
``./docker/run_docker.sh -c``); the base image has no cuRobo. Run it with::

    /isaac-sim/python.sh -m pytest -sv -m curobo_deps isaaclab_arena_curobo/tests/
"""

from __future__ import annotations

import sys
import torch

import pytest
import warp as wp

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

# The test entry point above only shells out via run_subprocess, so module import stays light. The
# isaaclab / isaaclab_mimic / env-building imports below are kept function-local because they must
# load only after ``_main`` launches the SimulationApp (the Arena/Isaac Lab app-first convention);
# torch and warp have no such constraint and live at module top.

# red_cube lands as a cuRobo collision obstacle; dex_cube's USD is instanceable, so cuRobo's
# get_obstacles_from_stage doesn't extract collision geometry for it. The sync test needs the object
# that is actually present in the collision world (red_cube) to read its synced pose back.
OBSTACLE_OBJECT = "red_cube"
NON_OBSTACLE_OBJECT = "dex_cube"
GRASP_Z_OFFSET = 0.05
# Lateral shift (m) that puts the out-of-reach object well outside the Franka's ~0.85 m reach.
UNREACHABLE_OFFSET_M = 3.0


@pytest.mark.curobo_deps
@pytest.mark.with_subprocess
def test_curobo_ik_one_reachable_one_not():
    """The IK oracle marks the in-reach object reachable and the out-of-reach object not."""
    run_subprocess([TestConstants.python_path, __file__, "reachability"])


@pytest.mark.curobo_deps
@pytest.mark.with_subprocess
def test_sync_object_poses_in_robot_base_frame():
    """Syncing pushes each object into cuRobo's world in the robot base frame with a wxyz quat."""
    run_subprocess([TestConstants.python_path, __file__, "sync"])


def _build_droid_two_object_env(args_cli, robot_initial_pose=None):
    """Build a single-env DROID scene holding the two graspable test objects, reset and ready.

    Args:
        args_cli: Parsed Arena CLI namespace used to configure the env builder.
        robot_initial_pose: Optional world-frame pose for the robot base. When set (e.g. offset and
            yawed), it makes the world->robot-base transform non-trivial so a frame-conversion test
            can actually exercise it; defaults to the embodiment's own initial pose.

    Returns the unwrapped env and the embodiment driving it.
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    background = registry.get_asset_by_name("table")()
    light = registry.get_asset_by_name("light")()
    obstacle_object = registry.get_asset_by_name(OBSTACLE_OBJECT)()
    non_obstacle_object = registry.get_asset_by_name(NON_OBSTACLE_OBJECT)()
    # Spawn them apart so they don't start interpenetrating; the test overrides both poses after reset.
    obstacle_object.set_initial_pose(Pose(position_xyz=(0.45, -0.1, 0.2)))
    non_obstacle_object.set_initial_pose(Pose(position_xyz=(0.45, 0.1, 0.2)))

    scene = Scene(assets=[background, light, obstacle_object, non_obstacle_object])
    embodiment = DroidAbsoluteJointPositionEmbodiment()
    if robot_initial_pose is not None:
        embodiment.set_initial_pose(robot_initial_pose)

    env = (
        ArenaEnvBuilder(
            IsaacLabArenaEnvironment(
                name="test_curobo_ik_reachability",
                embodiment=embodiment,
                scene=scene,
                task=None,
            ),
            arena_env_builder_cfg_from_argparse(args_cli),
        )
        .make_registered()
        .unwrapped
    )
    env.reset()
    return env, embodiment


def _set_object_world_xyz(env, name, xyz) -> None:
    """Teleport a scene object to a world-frame position (orientation left identity)."""
    from isaaclab_arena.utils.pose import Pose

    env_ids = torch.tensor([0], device=env.device)
    pose = Pose(position_xyz=tuple(float(v) for v in xyz)).to_tensor(device=env.device).unsqueeze(0)
    env.scene[name].write_root_pose_to_sim(pose, env_ids=env_ids)


def _set_object_world_pose(env, name, xyz, quat_xyzw) -> None:
    """Teleport a scene object to a world-frame pose, orientation given as (x, y, z, w)."""
    from isaaclab_arena.utils.pose import Pose

    env_ids = torch.tensor([0], device=env.device)
    pose = (
        Pose(position_xyz=tuple(float(v) for v in xyz), rotation_xyzw=tuple(float(v) for v in quat_xyzw))
        .to_tensor(device=env.device)
        .unsqueeze(0)
    )
    pose[0, :3] += env.scene.env_origins[0, :]
    env.scene[name].write_root_pose_to_sim(pose, env_ids=env_ids)


def _read_world_model_object_pose(planner, object_name):
    """Return the object's pose stored in cuRobo's collision world as [x, y, z, qw, qx, qy, qz].

    Mirrors the planner's own bidirectional path matching. Returns None if the object is absent.
    """
    object_path = planner._get_object_mappings()[object_name]
    world_model = planner.motion_gen.world_coll_checker.world_model
    if isinstance(world_model, list):
        world_model = world_model[planner.env_id]
    for primitive_type in planner.primitive_types:
        for primitive in getattr(world_model, primitive_type):
            name = str(primitive.name) if primitive.name else ""
            if name and (object_path == name or object_path in name or name in object_path):
                return list(primitive.pose)
    return None


def _run_sync_pose_check(args_cli) -> bool:
    """Sync one object, then check its world-model pose is the robot-base-frame pose in wxyz order."""
    import isaaclab.utils.math as math_utils

    # Offset + 90 deg yaw on the robot base so the world->base transform is non-trivial: a bug that
    # left poses in world frame (or used the wrong quaternion order) would then diverge from expected.
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_curobo.curobo_planner_utils import make_curobo_planner, sync_object_poses_in_robot_base_frame

    env, embodiment = _build_droid_two_object_env(
        args_cli,
        robot_initial_pose=Pose(position_xyz=(0.2, -0.15, 0.0), rotation_xyzw=(0.0, 0.0, 0.70710678, 0.70710678)),
    )
    planner = make_curobo_planner(env, embodiment, env_id=0)

    # Only the obstacle object actually lands in cuRobo's collision world (the other is instanceable
    # and isn't extracted), so it's the one whose synced pose we can read back and check.
    sync_object = OBSTACLE_OBJECT
    # Put the object at a known world pose with a non-identity (45 deg about Z) orientation.
    _set_object_world_pose(env, sync_object, (0.55, 0.15, 0.30), (0.0, 0.0, 0.38268343, 0.92387953))
    sync_object_poses_in_robot_base_frame(planner)

    stored = _read_world_model_object_pose(planner, sync_object)
    assert stored is not None, f"{sync_object} not found in cuRobo world model after sync"
    stored = torch.tensor(stored, dtype=torch.float32)

    # Independently recompute the expected robot-base-frame pose in cuRobo's [x, y, z, qw, qx, qy, qz].
    robot = env.scene["robot"]
    t_W_R = wp.to_torch(robot.data.root_pos_w)[0, :3].float().cpu()
    q_W_R_xyzw = wp.to_torch(robot.data.root_quat_w)[0, :4].float().cpu()
    t_W_O = wp.to_torch(env.scene[sync_object].data.root_pos_w)[0, :3].float().cpu()
    q_W_O_xyzw = wp.to_torch(env.scene[sync_object].data.root_quat_w)[0, :4].float().cpu()

    R_R_W = math_utils.matrix_from_quat(q_W_R_xyzw.unsqueeze(0))[0].T
    q_R_W_xyzw = math_utils.quat_inv(q_W_R_xyzw.unsqueeze(0))[0]
    t_R_O = (R_R_W @ (t_W_O - t_W_R).unsqueeze(-1)).squeeze(-1)
    q_R_O_xyzw = math_utils.quat_mul(q_R_W_xyzw.unsqueeze(0), q_W_O_xyzw.unsqueeze(0))[0]
    expected = torch.cat((t_R_O, math_utils.convert_quat(q_R_O_xyzw, to="wxyz"))).cpu()

    pos_close = torch.allclose(stored[:3], expected[:3], atol=2e-3)
    # Quaternion double cover: q and -q are the same rotation.
    quat_close = torch.allclose(stored[3:], expected[3:], atol=2e-3) or torch.allclose(
        stored[3:], -expected[3:], atol=2e-3
    )
    # Guard against a regression that leaves poses in world frame: with the base offset + yawed, the
    # base-frame position must differ clearly from the world-frame position.
    in_base_frame = not torch.allclose(stored[:3], t_W_O, atol=1e-2)

    print(f"stored   = {[round(v, 4) for v in stored.tolist()]}", flush=True)
    print(f"expected = {[round(v, 4) for v in expected.tolist()]}", flush=True)
    print(f"world_pos = {[round(v, 4) for v in t_W_O.tolist()]}", flush=True)
    print(f"pos_close={pos_close} quat_close={quat_close} in_base_frame={in_base_frame}", flush=True)
    return bool(pos_close and quat_close and in_base_frame)


def _run_reachability_check(args_cli) -> bool:
    """Place one object in-reach and one out-of-reach; return True iff IK agrees with that split."""
    from isaaclab_arena_curobo.curobo_planner_utils import (
        make_curobo_planner,
        sync_object_poses_in_robot_base_frame,
        top_down_grasp_pose_in_robot_frame,
    )
    from isaaclab_arena_curobo.ik_utils import check_ik_feasibility_batch_goal_poses

    env, embodiment = _build_droid_two_object_env(args_cli)
    planner = make_curobo_planner(env, embodiment, env_id=0)

    # Anchor reachability to the arm's own workspace: the current end-effector position is reachable by
    # construction, so one object goes there and the other is shifted well out of reach.
    robot = env.scene["robot"]
    ee_idx = robot.data.body_names.index(planner.config.ee_link_name)
    ee_pos_w = wp.to_torch(robot.data.body_pos_w)[0, ee_idx, :3].clone()

    far_pos_w = ee_pos_w.clone()
    far_pos_w[1] += UNREACHABLE_OFFSET_M
    _set_object_world_xyz(env, OBSTACLE_OBJECT, ee_pos_w)
    _set_object_world_xyz(env, NON_OBSTACLE_OBJECT, far_pos_w)

    sync_object_poses_in_robot_base_frame(planner)

    grasp_poses = torch.stack([
        top_down_grasp_pose_in_robot_frame(env, OBSTACLE_OBJECT, GRASP_Z_OFFSET),
        top_down_grasp_pose_in_robot_frame(env, NON_OBSTACLE_OBJECT, GRASP_Z_OFFSET),
    ])
    feasible, pos_err, rot_err = check_ik_feasibility_batch_goal_poses(planner, grasp_poses)
    print(
        f"reach: feasible={bool(feasible[0])} pos_err={float(pos_err[0]):.4f}m rot_err={float(rot_err[0]):.4f}rad",
        flush=True,
    )
    print(
        f"far:   feasible={bool(feasible[1])} pos_err={float(pos_err[1]):.4f}m rot_err={float(rot_err[1]):.4f}rad",
        flush=True,
    )

    return bool(feasible[0].item()) and not bool(feasible[1].item())


def _main() -> int:
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    mode = sys.argv[1] if len(sys.argv) > 1 else "reachability"
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--headless"])
    with SimulationAppContext(args_cli):
        passed = _run_sync_pose_check(args_cli) if mode == "sync" else _run_reachability_check(args_cli)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(_main())
