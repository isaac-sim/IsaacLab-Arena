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

    MPLCONFIGDIR=/tmp/mpl /isaac-sim/python.sh -m pytest -sv -m curobo_deps isaaclab_arena_curobo/tests/
"""

from __future__ import annotations

import sys

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

NEAR_OBJECT = "dex_cube"
FAR_OBJECT = "red_cube"
GRASP_Z_OFFSET = 0.05
# Lateral shift (m) that puts the far object well outside the Franka's ~0.85 m reach.
UNREACHABLE_OFFSET_M = 3.0


@pytest.mark.curobo_deps
@pytest.mark.with_subprocess
def test_curobo_ik_one_reachable_one_not():
    """The IK oracle marks the in-reach object reachable and the out-of-reach object not."""
    run_subprocess([TestConstants.python_path, __file__])


def _build_droid_two_object_env(args_cli):
    """Build a single-env DROID scene holding the two graspable test objects, reset and ready.

    Returns the unwrapped env and the embodiment driving it.
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    background = registry.get_asset_by_name("table")()
    light = registry.get_asset_by_name("light")()
    near_object = registry.get_asset_by_name(NEAR_OBJECT)()
    far_object = registry.get_asset_by_name(FAR_OBJECT)()
    # Spawn them apart so they don't start interpenetrating; the test overrides both poses after reset.
    near_object.set_initial_pose(Pose(position_xyz=(0.45, -0.1, 0.2)))
    far_object.set_initial_pose(Pose(position_xyz=(0.45, 0.1, 0.2)))

    scene = Scene(assets=[background, light, near_object, far_object])
    embodiment = DroidAbsoluteJointPositionEmbodiment()

    env = (
        ArenaEnvBuilder(
            IsaacLabArenaEnvironment(
                name="test_curobo_ik_reachability",
                embodiment=embodiment,
                scene=scene,
                task=None,
            ),
            args_cli,
        )
        .make_registered()
        .unwrapped
    )
    env.reset()
    return env, embodiment


def _set_object_world_xyz(env, name, xyz) -> None:
    """Teleport a scene object to a world-frame position (orientation left identity)."""
    import torch

    from isaaclab_arena.utils.pose import Pose

    env_ids = torch.tensor([0], device=env.device)
    pose = Pose(position_xyz=tuple(float(v) for v in xyz)).to_tensor(device=env.device).unsqueeze(0)
    pose[0, :3] += env.scene.env_origins[0, :]
    env.scene[name].write_root_pose_to_sim(pose, env_ids=env_ids)


def _run_reachability_check(args_cli) -> bool:
    """Place one object in-reach and one out-of-reach; return True iff IK agrees with that split."""
    import torch

    import warp as wp

    from isaaclab_arena_curobo.curobo_planner_utils import make_curobo_planner, top_down_grasp_pose_in_robot_frame
    from isaaclab_arena_curobo.ik_utils import check_ik_feasibility_batch_goal_poses

    env, embodiment = _build_droid_two_object_env(args_cli)
    planner = make_curobo_planner(env, embodiment, env_id=0)

    # Anchor reachability to the arm's own workspace: the current end-effector position is reachable by
    # construction, so the near object goes there and the far object is shifted out of reach.
    robot = env.scene["robot"]
    ee_idx = robot.data.body_names.index(planner.config.ee_link_name)
    ee_pos_w = wp.to_torch(robot.data.body_pos_w)[0, ee_idx, :3].clone()

    far_pos_w = ee_pos_w.clone()
    far_pos_w[1] += UNREACHABLE_OFFSET_M
    _set_object_world_xyz(env, NEAR_OBJECT, ee_pos_w)
    _set_object_world_xyz(env, FAR_OBJECT, far_pos_w)

    # Mirror the validation pipeline: sync obstacles into the (robot-base-frame) collision world.
    planner._sync_object_poses_with_isaaclab()

    grasp_poses = torch.stack([
        top_down_grasp_pose_in_robot_frame(env, NEAR_OBJECT, GRASP_Z_OFFSET),
        top_down_grasp_pose_in_robot_frame(env, FAR_OBJECT, GRASP_Z_OFFSET),
    ])
    feasible, pos_err, rot_err = check_ik_feasibility_batch_goal_poses(planner, grasp_poses)
    print(f"near: feasible={bool(feasible[0])} pos_err={float(pos_err[0]):.4f}m rot_err={float(rot_err[0]):.4f}rad")
    print(f"far:  feasible={bool(feasible[1])} pos_err={float(pos_err[1]):.4f}m rot_err={float(rot_err[1]):.4f}rad")

    return bool(feasible[0].item()) and not bool(feasible[1].item())


def _main() -> int:
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--headless"])
    with SimulationAppContext(args_cli):
        passed = _run_reachability_check(args_cli)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(_main())
