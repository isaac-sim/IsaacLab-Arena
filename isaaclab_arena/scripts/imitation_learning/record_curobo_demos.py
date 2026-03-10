# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""
Record demonstrations using CuRobo plan execution for droid_v2_tabletop_pick_and_place.

Runs the CuRobo pick-and-place planner for each episode and exports (state, action)
trajectories to HDF5 for imitation learning. Uses the same core flow as
run_droid_v2_tabletop_curobo_pick_place.py (SimulationAppContext + shared utils).

State-dependent IK (why replay can diverge):
  The env uses Differential IK: joint targets are computed from *current* EEF pose and
  joint state each step. That happens in IsaacLab in:
    isaaclab/envs/mdp/actions/task_space_actions.py
  - DifferentialInverseKinematicsAction.process_actions (lines ~196-197): reads current
    EEF pose via _compute_frame_pose() and passes it to the IK controller.
  - apply_actions (lines ~201-211): again reads current ee_pos_curr, ee_quat_curr and
    joint_pos, then compute(joint_pos_des) and set_joint_position_target(joint_pos_des).
  So the same (delta_pose, gripper) action can produce different joint targets if the
  current state differs (e.g. after first grasp, object attachment changes dynamics).

Replay: This script uses ActionStateRecorderManagerCfg (processed_actions, states, etc.)
  and a post-step recorder that writes "joint_targets" (7 arm + 1 gripper). To replay
  without re-running IK, use replay_demos.py with --use_joint_targets to apply the
  recorded joint_targets directly.
"""

"""Launch Isaac Sim Simulator first."""

# record curobo demos, use the following command:
# python isaaclab_arena/scripts/imitation_learning/record_curobo_demos.py --device cpu droid_v2_tabletop_pick_and_place --num_demos 1 --dataset_file /datasets/curobo.hdf5 --grasp_z_offset 0.17 --pick_order alphabet_soup_can_hope_robolab ketchup_bottle_hope_robolab tomato_soup_can --grasp_orientation object_yaw --post_place_clearance 0.0 --grasp_z_offset 0.17 --approach_distance 0.08 --retreat_distance 0.08 --debug_planner --debug_goal 

import argparse
import contextlib
import os
from copy import deepcopy
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli
from isaaclab_arena.scripts.curobo.curobo_cli_args import add_script_args_to_subparsers as add_curobo_script_args_to_subparsers

# Parser: example env is a subcommand, so all script options must be on each subparser to work after the subcommand
parser = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(parser)
for action in parser._actions:
    choices = getattr(action, "choices", None)
    if isinstance(choices, dict):
        for subparser in choices.values():
            subparser.add_argument("--dataset_file", type=str, required=True, help="File path to export recorded demos.")
        add_curobo_script_args_to_subparsers(parser)  # adds --num_demos, --gripper_settle_steps, etc.
        break
args_cli = parser.parse_args()


def setup_output_directories(args_cli) -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir, output_file_name


def _create_post_step_joint_targets_recorder():
    """Create a RecorderTerm that records 7 arm + 1 gripper joint targets for replay (--use_joint_targets)."""
    from isaaclab.managers.recorder_manager import RecorderTerm

    class PostStepJointTargetsRecorder(RecorderTerm):
        def record_post_step(self):
            robot = self._env.scene["robot"]
            joint_ids = []
            for name in self._env.action_manager.active_terms:
                t = self._env.action_manager.get_term(name)
                jids = getattr(t, "_joint_ids", None)
                if jids is None:
                    continue
                if isinstance(jids, slice):
                    joint_ids.extend(range(robot.num_joints))
                else:
                    joint_ids.extend(list(jids))
            return "joint_targets", robot.data.joint_pos_target[:, joint_ids].clone()

    return PostStepJointTargetsRecorder


def create_environment_config(
    args_cli,
    output_dir: str,
    output_file_name: str,
):
    """Build env config with recorders; same arena builder as run_droid_v2, plus recorder settings."""
    import omni.log
    from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode, RecorderTermCfg

    try:
        arena_builder = get_arena_builder_from_cli(args_cli)
        env_name, env_cfg = arena_builder.build_registered()
    except Exception as e:
        omni.log.error(f"Failed to parse environment configuration: {e}")
        exit(1)
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn("No success termination term was found in the environment.")
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    # Record 7 arm + 1 gripper joint targets for --use_joint_targets replay (avoids 6D vs 7 joint mismatch)
    env_cfg.recorders.record_post_step_joint_targets = RecorderTermCfg(
        class_type=_create_post_step_joint_targets_recorder()
    )
    return env_cfg, env_name, success_term


def main() -> None:
    if getattr(args_cli, "example_environment", None) != "droid_v2_tabletop_pick_and_place":
        raise ValueError(
            "This script records demos with CuRobo for droid_v2_tabletop_pick_and_place only. "
            "Use --example_environment droid_v2_tabletop_pick_and_place."
        )

    with SimulationAppContext(args_cli) as ctx:
        import gymnasium as gym
        import torch
        import omni.log
        from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
        from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_arena.scripts.curobo.curobo_pick_place_utils import (
            GRIPPER_CLOSE_CMD,
            GRIPPER_OPEN_CMD,
            auto_pick_order,
            compute_grasp_and_place_poses,
            compute_placement_slots,
            compute_retreat_pose,
            execute_gripper_action,
            fix_planner_object_sync_frame,
            get_bin_interior_center,
            get_current_eef_pose,
            get_object_pos,
            get_object_quat,
            make_planner_cfg,
            plan_and_execute,
        )

        output_dir, output_file_name = setup_output_directories(args_cli)
        env_cfg, env_name, _success_term = create_environment_config(args_cli, output_dir, output_file_name)
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        env.reset()

        robot = env.scene["robot"]
        planner_cfg = make_planner_cfg(args_cli)
        planner = CuroboPlanner(env=env, robot=robot, config=planner_cfg, env_id=0)
        fix_planner_object_sync_frame(planner)

        goal_pose_visualizer = None
        ee_visualizer = None
        assert args_cli.debug_goal, "debug_goal must be set to True for goal pose visualization"
        if not getattr(args_cli, "headless", False):
            marker_cfg = deepcopy(FRAME_MARKER_CFG)
            marker_cfg.prim_path = "/World/Visuals/curobo_goal_pose"
            frame_marker = marker_cfg.markers.get("frame")
            if frame_marker is not None:
                setattr(frame_marker, "scale", (0.08, 0.08, 0.08))
            goal_pose_visualizer = VisualizationMarkers(marker_cfg)
            ee_marker_cfg = deepcopy(FRAME_MARKER_CFG)
            ee_marker_cfg.prim_path = "/World/Visuals/curobo_ee_actual"
            ee_frame_marker = ee_marker_cfg.markers.get("frame")
            if ee_frame_marker is not None:
                setattr(ee_frame_marker, "scale", (0.05, 0.05, 0.05))
            ee_visualizer = VisualizationMarkers(ee_marker_cfg)

        num_demos = args_cli.num_demos if args_cli.num_demos > 0 else float("inf")
        current_recorded_demo_count = 0

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while (num_demos == float("inf") or current_recorded_demo_count < num_demos) and ctx.app_launcher.app.is_running():
                total_label = int(num_demos) if num_demos != float("inf") else "∞"
                print(f"\n--- Demo {current_recorded_demo_count + 1}/{total_label} ---")
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()

                pick_order = auto_pick_order(env, getattr(args_cli, "pick_order", None))
                if getattr(args_cli, "max_objects", None) is not None:
                    pick_order = pick_order[: args_cli.max_objects]
                if len(pick_order) == 0:
                    omni.log.warn("No pickable objects in scene; skipping episode.")
                    continue

                bin_pos = get_bin_interior_center(env, "blue_sorting_bin", env_id=0, verbose=False)
                placement_slots = compute_placement_slots(
                    bin_pos, len(pick_order), args_cli.bin_half_x, args_cli.bin_half_y, verbose=False
                )
                results = {}

                for idx, object_name in enumerate(pick_order):
                    object_pos = get_object_pos(env, object_name, env_id=0, verbose=False)
                    object_quat = get_object_quat(env, object_name, env_id=0)
                    grasp_pose, place_pose = compute_grasp_and_place_poses(
                        object_pos,
                        object_quat,
                        placement_slots[idx],
                        args_cli.grasp_orientation,
                        args_cli.grasp_z_offset,
                        args_cli.place_z_offset,
                        env.device,
                    )

                    if getattr(args_cli, "debug_goal", False):
                        grasp_xyz = grasp_pose[:3, 3].cpu().tolist()
                        place_xyz = place_pose[:3, 3].cpu().tolist()
                        eef_pos = get_current_eef_pose(env, planner, env_id=0)[:3, 3]
                        dist_grasp = torch.norm(grasp_pose[:3, 3] - eef_pos).item()
                        print(
                            f"[DEBUG GOAL] {object_name} | grasp_xyz={grasp_xyz} | "
                            f"place_xyz={place_xyz} | EEF->grasp dist={dist_grasp:.4f}m"
                        )

                    grasp_success = plan_and_execute(
                        env,
                        planner,
                        target_pose=grasp_pose,
                        gripper_action=GRIPPER_OPEN_CMD,
                        expected_attached_object=None,
                        stage=f"{object_name}:grasp",
                        goal_pose_visualizer=goal_pose_visualizer,
                        ee_visualizer=ee_visualizer,
                        debug_goal=args_cli.debug_goal,
                    )
                    results[object_name] = {"grasp": grasp_success}
                    if not grasp_success:
                        continue

                    execute_gripper_action(
                        env, planner, GRIPPER_CLOSE_CMD,
                        steps=args_cli.gripper_settle_steps, env_id=0,
                    )

                    place_success = plan_and_execute(
                        env,
                        planner,
                        target_pose=place_pose,
                        gripper_action=GRIPPER_CLOSE_CMD,
                        expected_attached_object=object_name,
                        stage=f"{object_name}:place",
                        goal_pose_visualizer=goal_pose_visualizer,
                        ee_visualizer=ee_visualizer,
                        debug_goal=getattr(args_cli, "debug_goal", False),
                    )
                    results[object_name]["place"] = place_success
                    if not place_success:
                        execute_gripper_action(
                            env, planner, GRIPPER_OPEN_CMD,
                            steps=args_cli.gripper_settle_steps, env_id=0,
                        )
                        continue

                    execute_gripper_action(
                        env, planner, GRIPPER_OPEN_CMD,
                        steps=args_cli.gripper_settle_steps, env_id=0,
                    )

                    if args_cli.post_place_clearance > 0:
                        retreat_pose = compute_retreat_pose(
                            place_pose, args_cli.post_place_clearance, env.device
                        )
                        retreat_success = plan_and_execute(
                            env,
                            planner,
                            target_pose=retreat_pose,
                            gripper_action=GRIPPER_OPEN_CMD,
                            expected_attached_object=None,
                            stage=f"{object_name}:retreat",
                            goal_pose_visualizer=goal_pose_visualizer,
                            ee_visualizer=ee_visualizer,
                            debug_goal=args_cli.debug_goal,
                        )
                        results[object_name]["retreat"] = retreat_success

                all_success = all(
                    stages.get("grasp") and stages.get("place", False) for stages in results.values()
                )
                if all_success and results:
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes(
                        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes([0])
                    current_recorded_demo_count += 1
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                else:
                    print("Episode had failures; not exporting (EXPORT_SUCCEEDED_ONLY).")

                if env.sim.is_stopped():
                    break

        env.close()
        print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
        print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
