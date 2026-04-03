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
# v2
# python isaaclab_arena/scripts/imitation_learning/record_curobo_demos.py droid_v2_tabletop_pick_and_place --num_demos 1 --dataset_file /datasets/curobo.hdf5 --grasp_z_offset 0.17 --pick_order alphabet_soup_can_hope_robolab ketchup_bottle_hope_robolab tomato_soup_can --grasp_orientation object_yaw --post_place_clearance 0.0 --grasp_z_offset 0.17 --approach_distance 0.08 --retreat_distance 0.08 --debug_planner --debug_goal 
# v3
# python isaaclab_arena/scripts/imitation_learning/record_curobo_demos.py droid_v3_tabletop_pick_and_place --grasp_z_offset 0.17 --approach_distance 0.08 --retreat_distance 0.08 --debug_planner --debug_goal --grasp_orientation object_yaw --post_place_clearance 0.0 --num_demos 1 --dataset_file /datasets/curobo_v3_statesonly.hdf5

import argparse
import contextlib
import os
import torch
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
        add_curobo_script_args_to_subparsers(parser)  # adds --num_demos, etc.
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

def _create_post_step_eef_pose_target_recorder():
    """Create a RecorderTerm that records the EEF pose target for replay (--use_eef_pose_target)."""
    from isaaclab.managers.recorder_manager import RecorderTerm

    class PostStepEefPoseTargetRecorder(RecorderTerm):
        def record_post_step(self):
            # record eef pose after the action is applied
            robot = self._env.scene["robot"]
            eef_idx = robot.data.body_names.index("base_link")  #robotiq gripper base link
            eef_pos = robot.data.body_pos_w[:, eef_idx, :].clone()
            eef_quat = robot.data.body_quat_w[:, eef_idx, :].clone()
            return "eef_pose_target", torch.cat((eef_pos, eef_quat), dim=1)
    return PostStepEefPoseTargetRecorder

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
    is_v3 = getattr(args_cli, "example_environment", "") == "droid_v3_tabletop_pick_and_place"
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
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    # Record 7 arm + 1 gripper joint targets for --use_joint_targets replay (avoids 6D vs 7 joint mismatch)
    env_cfg.recorders.record_post_step_joint_targets = RecorderTermCfg(
        class_type=_create_post_step_joint_targets_recorder()
    )
    env_cfg.recorders.record_post_step_eef_pose_target = RecorderTermCfg(
        class_type=_create_post_step_eef_pose_target_recorder()
    )
    return env_cfg, env_name, success_term, arena_builder, is_v3


def main() -> None:

    with SimulationAppContext(args_cli) as ctx:
        import gymnasium as gym
        import torch
        import omni.log
        from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
        from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        import isaaclab.utils.math as math_utils
        from isaaclab_arena.scripts.curobo.curobo_pick_place_utils import (
            DOWN_FACING_QUAT_WXYZ,
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
            pose_from_pos_quat,
        )
        from isaaclab_arena.scripts.curobo.ik_utils import (
            check_ik_feasibility,
            get_current_joint_config,
        )

        output_dir, output_file_name = setup_output_directories(args_cli)
        env_cfg, env_name, _success_term, arena_builder, is_v3 = create_environment_config(args_cli, output_dir, output_file_name)
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        env.reset()

        robot = env.scene["robot"]
        planner_cfg = make_planner_cfg(args_cli)
        planner = CuroboPlanner(env=env, robot=robot, config=planner_cfg, env_id=0)
        fix_planner_object_sync_frame(planner)

        # V3: build IK cost function once (closure over stable references)
        robot_pos_w = robot.data.root_pos_w[0, :3]
        robot_quat_w = robot.data.root_quat_w[0, :4]
        R_w2r = math_utils.matrix_from_quat(robot_quat_w.unsqueeze(0))[0].T
        IK_FAIL_PENALTY = 100.0
        down_quat = DOWN_FACING_QUAT_WXYZ.to(env.device)
        ik_pos_th = args_cli.ik_pos_threshold
        ik_rot_th = args_cli.ik_rot_threshold

        def _world_to_robot(pos_w_tuple):
            xyz = torch.tensor(pos_w_tuple, device=env.device, dtype=torch.float32)
            return (R_w2r @ (xyz - robot_pos_w).unsqueeze(-1)).squeeze(-1)

        def _ik_cost_fn(name, init_pos_w, target_pos_w, prev_jc):
            """Sequential IK penalty: evaluates grasp then place as a chain."""
            if prev_jc is None:
                prev_jc = get_current_joint_config(planner)

            jc = prev_jc

            grasp_xyz = _world_to_robot(init_pos_w)
            grasp_xyz[2] += args_cli.grasp_z_offset
            ok, pe, re, sol = check_ik_feasibility(
                planner, pose_from_pos_quat(grasp_xyz, down_quat), seed_config=jc,
                position_threshold=ik_pos_th, rotation_threshold=ik_rot_th,
            )
            if not ok:
                print(f"[IK] {name} grasp INFEASIBLE (pos={pe:.4f}m, rot={re:.4f}rad)")
                return 2 * IK_FAIL_PENALTY, prev_jc
            jc = sol

            place_xyz = _world_to_robot(target_pos_w)
            place_xyz[2] += args_cli.place_z_offset
            ok, pe, re, sol = check_ik_feasibility(
                planner, pose_from_pos_quat(place_xyz, down_quat), seed_config=jc,
                position_threshold=ik_pos_th, rotation_threshold=ik_rot_th,
            )
            if not ok:
                print(f"[IK] {name} place INFEASIBLE (pos={pe:.4f}m, rot={re:.4f}rad)")
                return IK_FAIL_PENALTY, prev_jc
            jc = sol

            return 0.0, jc

        goal_pose_visualizer = None
        ee_visualizer = None

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

        num_demos = args_cli.num_demos if args_cli.num_demos > 0 else 1
        current_attempt = 0
        current_recorded_demo_count = 0

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while current_attempt < num_demos and ctx.app_launcher.app.is_running():
                current_attempt += 1
                print(f"\n--- Attempt {current_attempt}/{num_demos} | Successful demos: {current_recorded_demo_count} ---")
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()

                # ── Plan pick order ───────────────────────────────────
                if is_v3:
                    # RelationSolver needs gradients; temporarily exit inference_mode
                    with torch.inference_mode(False):
                        target_positions = arena_builder.example_env.generate_target_positions()
                    planner.update_world()

                    pick_order, ik_penalty = arena_builder.example_env.plan_pick_order(
                        verbose=getattr(args_cli, "debug_planner", False),
                        ik_cost_fn=_ik_cost_fn,
                    )
                    if getattr(args_cli, "max_objects", None) is not None:
                        pick_order = pick_order[: args_cli.max_objects]

                    if ik_penalty > 0:
                        print(f"[SKIP] Layout IK-infeasible (penalty={ik_penalty:.1f}), resetting")
                        continue

                    placement_slots = {}
                    for obj_name in pick_order:
                        world_pos = torch.tensor(target_positions[obj_name], device=env.device)
                        placement_slots[obj_name] = (R_w2r @ (world_pos - robot_pos_w).unsqueeze(-1)).squeeze(-1)
                else:
                    pick_order = auto_pick_order(env, getattr(args_cli, "pick_order", None))
                    if getattr(args_cli, "max_objects", None) is not None:
                        pick_order = pick_order[: args_cli.max_objects]

                    bin_pos = get_bin_interior_center(env, "blue_sorting_bin", env_id=0, verbose=False)
                    slot_list = compute_placement_slots(
                        bin_pos, len(pick_order), args_cli.bin_half_x, args_cli.bin_half_y, verbose=False
                    )
                    placement_slots = {name: slot_list[i] for i, name in enumerate(pick_order)}

                if len(pick_order) == 0:
                    omni.log.warn("No pickable objects in scene; skipping episode.")
                    continue

                print(f"Pick order: {pick_order}")
                results = {}

                for idx, object_name in enumerate(pick_order):
                    object_pos = get_object_pos(env, object_name, env_id=0, verbose=False)
                    object_quat = get_object_quat(env, object_name, env_id=0)
                    grasp_pose, place_pose = compute_grasp_and_place_poses(
                        object_pos,
                        object_quat,
                        placement_slots[object_name],
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
                        print(f"[SKIP] '{object_name}' grasp failed, aborting attempt")
                        break

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
                        print(f"[SKIP] '{object_name}' place failed, aborting attempt")
                        break

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
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([[all_success]], dtype=torch.bool, device=env.device)
                )
                env.recorder_manager.export_episodes([0])
                current_recorded_demo_count += 1
                status = "SUCCESS" if all_success else "FAILED"
                print(f"[{status}] Recorded demo {current_recorded_demo_count}/{num_demos}")

                if env.sim.is_stopped():
                    break

        env.close()
        print(f"\n{'=' * 80}")
        print(f"RECORDING COMPLETE: {current_recorded_demo_count}/{current_attempt} attempts succeeded")
        if current_recorded_demo_count > 0:
            print(f"Demonstrations saved to: {args_cli.dataset_file}")
        else:
            print(f"No successful demonstrations recorded.")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
