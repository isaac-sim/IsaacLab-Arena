# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""
Record demonstrations using CuRobo plan execution for droid_v2_tabletop_pick_and_place.

Runs the CuRobo pick-and-place planner for each episode and exports (state, action)
trajectories to HDF5 for imitation learning. Uses the same core flow as
run_droid_v2_tabletop_curobo_pick_place.py (SimulationAppContext + shared utils).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import os
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
            subparser.add_argument(
                "--num_demos", type=int, default=1, help="Number of demonstrations to record. Set to 0 for infinite."
            )
        add_curobo_script_args_to_subparsers(parser)
        break
args_cli = parser.parse_args()


def setup_output_directories(args_cli) -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir, output_file_name


def create_environment_config(
    args_cli,
    output_dir: str,
    output_file_name: str,
):
    """Build env config with recorders; same arena builder as run_droid_v2, plus recorder settings."""
    import omni.log
    from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode

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
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_arena.scripts.curobo.curobo_pick_place_utils import (
            GRIPPER_CLOSE_CMD,
            GRIPPER_OPEN_CMD,
            DOWN_FACING_QUAT_WXYZ,
            auto_pick_order,
            compute_grasp_quat,
            compute_placement_slots,
            execute_gripper_action_recordable,
            fix_planner_object_sync_frame,
            get_bin_interior_center,
            get_object_pos,
            get_object_quat,
            make_planner_cfg,
            plan_and_execute,
            pose_from_pos_quat,
        )

        output_dir, output_file_name = setup_output_directories(args_cli)
        env_cfg, env_name, _success_term = create_environment_config(args_cli, output_dir, output_file_name)
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        env.reset()

        robot = env.scene["robot"]
        planner_cfg = make_planner_cfg(args_cli)
        planner = CuroboPlanner(env=env, robot=robot, config=planner_cfg, env_id=0)
        fix_planner_object_sync_frame(planner)

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
                    grasp_quat = compute_grasp_quat(args_cli.grasp_orientation, object_quat, env.device)
                    grasp_xyz = object_pos.clone()
                    grasp_xyz[2] += args_cli.grasp_z_offset
                    grasp_pose = pose_from_pos_quat(grasp_xyz, grasp_quat)
                    slot_center_xyz = placement_slots[idx]
                    place_xyz = slot_center_xyz.clone()
                    place_xyz[2] += args_cli.place_z_offset
                    place_pose = pose_from_pos_quat(place_xyz, DOWN_FACING_QUAT_WXYZ.to(env.device))

                    grasp_success = plan_and_execute(
                        env,
                        planner,
                        target_pose=grasp_pose,
                        gripper_action=GRIPPER_OPEN_CMD,
                        expected_attached_object=None,
                        stage=f"{object_name}:grasp",
                        use_env_step_batch=True,
                    )
                    results[object_name] = {"grasp": grasp_success}
                    if not grasp_success:
                        continue

                    execute_gripper_action_recordable(
                        env, planner, GRIPPER_CLOSE_CMD, args_cli.gripper_settle_steps, env_id=0
                    )

                    place_success = plan_and_execute(
                        env,
                        planner,
                        target_pose=place_pose,
                        gripper_action=GRIPPER_CLOSE_CMD,
                        expected_attached_object=object_name,
                        stage=f"{object_name}:place",
                        use_env_step_batch=True,
                    )
                    results[object_name]["place"] = place_success
                    if not place_success:
                        execute_gripper_action_recordable(
                            env, planner, GRIPPER_OPEN_CMD, args_cli.gripper_settle_steps, env_id=0
                        )
                        continue

                    execute_gripper_action_recordable(
                        env, planner, GRIPPER_OPEN_CMD, args_cli.gripper_settle_steps, env_id=0
                    )

                    if args_cli.post_place_clearance > 0:
                        retreat_xyz = place_xyz.clone()
                        retreat_xyz[2] += args_cli.post_place_clearance
                        retreat_pose = pose_from_pos_quat(retreat_xyz, DOWN_FACING_QUAT_WXYZ.to(env.device))
                        retreat_success = plan_and_execute(
                            env,
                            planner,
                            target_pose=retreat_pose,
                            gripper_action=GRIPPER_OPEN_CMD,
                            expected_attached_object=None,
                            stage=f"{object_name}:retreat",
                            use_env_step_batch=True,
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
