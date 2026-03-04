#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# From Neel: This script is an example of how to use the curobo planner to pick and place objects.
# It is not a complete pick and place policy, but it is a good starting point

# can run this command: python isaaclab_arena/scripts/curobo/run_droid_v2_tabletop_curobo_pick_place.py droid_v2_tabletop_pick_and_place --grasp_z_offset 0.17 --approach_distance 0.08 --retreat_distance 0.08 --debug_planner --debug_goal --pick_order tomato_soup_can --grasp_orientation object_yaw --post_place_clearance 0.0

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.scripts.curobo.curobo_cli_args import add_script_args, add_script_args_to_subparsers
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser


def _save_rerun_checkpoint(planner, label: str) -> None:
    """Best-effort explicit Rerun save checkpoint."""
    try:
        viz = getattr(planner, "plan_visualizer", None)
        save_path = getattr(viz, "save_path", None)
        if viz is None or save_path is None:
            return
        import rerun as rr

        rr.save(save_path)
        print(f"[RERUN] checkpoint saved after {label}: {save_path}")
    except Exception as exc:
        print(f"[RERUN] checkpoint save failed after {label}: {exc}")


def _run_sanity_check(
    planner,
    env,
    sphere_dump_dir: Path | None,
    sphere_dump_png: bool,
    goal_pose_visualizer=None,
    ee_visualizer=None,
    debug_goal: bool = False,
) -> None:
    curr_pose = get_current_eef_pose(env, planner)
    lifted_pose = curr_pose.clone()
    lifted_pose[2, 3] += 0.05
    ok = plan_and_execute(
        env=env,
        planner=planner,
        target_pose=lifted_pose,
        gripper_action=GRIPPER_OPEN_CMD,
        expected_attached_object=None,
        stage="sanity_lift_5cm",
        sphere_dump_dir=sphere_dump_dir,
        sphere_dump_png=sphere_dump_png,
        goal_pose_visualizer=goal_pose_visualizer,
        ee_visualizer=ee_visualizer,
        debug_goal=debug_goal,
        use_env_step_batch=False,
    )
    print(f"[SANITY] lift-from-home planning success: {ok}")
    if ok:
        _save_rerun_checkpoint(planner, "sanity_lift_5cm")


def main() -> None:
    """Main function to run the script."""
    args_parser = get_isaaclab_arena_cli_parser()
    args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
    add_script_args_to_subparsers(args_parser)
    args_cli = args_parser.parse_args()

    if args_cli.example_environment != "droid_v2_tabletop_pick_and_place":
        raise ValueError("This script is intended for the droid_v2_tabletop_pick_and_place environment.")

    with SimulationAppContext(args_cli):
        from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        import isaaclab.utils.math as math_utils
        from isaaclab_arena.scripts.curobo.curobo_pick_place_utils import (
            DOWN_FACING_QUAT_WXYZ,
            GRIPPER_CLOSE_CMD,
            GRIPPER_OPEN_CMD,
            action_from_pose,
            auto_pick_order,
            compute_grasp_quat,
            compute_placement_slots,
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

        sphere_dump_dir = Path(args_cli.dump_spheres_dir) if args_cli.dump_spheres_dir else None

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

        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()
        env.reset()

        robot = env.scene["robot"]
        robot_world_pos = robot.data.root_pos_w[0, :3]
        print(f"[DEBUG INIT] Robot world position: {robot_world_pos}")
        print("[DEBUG INIT] CuRobo coordinate system: robot base at origin, all goals/objects in robot-base frame")
        joint_names = robot.joint_names
        for i, name in enumerate(joint_names):
            s = robot.data.default_joint_stiffness[0, i].item()
            d = robot.data.default_joint_damping[0, i].item()
            if "finger" in name or "knuckle" in name or "panda" in name:
                print(f"[DEBUG JOINT] {name}: default_stiffness={s:.4f}, default_damping={d:.4f}")

        planner_cfg = make_planner_cfg(args_cli)
        planner = CuroboPlanner(
            env=env,
            robot=robot,
            config=planner_cfg,
            env_id=0,
        )
        fix_planner_object_sync_frame(planner)

        # ToDo (Neel): WIP rerun recording saving (does not work yet)
        if args_cli.rerun_recording_path is not None:
            import datetime as _dt

            rerun_path = Path(args_cli.rerun_recording_path).expanduser().resolve()
            if rerun_path.suffix.lower() != ".rrd":
                if rerun_path.exists() and rerun_path.is_dir():
                    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    rerun_path = rerun_path / f"curobo_plan_{ts}.rrd"
                else:
                    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    rerun_path = (
                        rerun_path / f"curobo_plan_{ts}.rrd"
                        if rerun_path.suffix == ""
                        else rerun_path.with_suffix(".rrd")
                    )

            rerun_path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(planner, "plan_visualizer"):
                planner.plan_visualizer.save_path = str(rerun_path)
                print(f"[RERUN] recording will be saved to: {rerun_path}")
            else:
                print("[RERUN] visualize_plan is disabled; no recording will be produced.")

        if args_cli.run_sanity_check:
            _run_sanity_check(
                planner,
                env,
                sphere_dump_dir,
                args_cli.dump_spheres_png,
                goal_pose_visualizer,
                ee_visualizer=ee_visualizer,
                debug_goal=args_cli.debug_goal,
            )

        pick_order = auto_pick_order(env, explicit_order=args_cli.pick_order)
        if args_cli.max_objects is not None:
            pick_order = pick_order[: args_cli.max_objects]
        if len(pick_order) == 0:
            raise RuntimeError("No pickable objects were found in scene.rigid_objects.")

        print(f"Resolved pick order: {pick_order}")

        bin_pos = get_bin_interior_center(
            env, "blue_sorting_bin", 0, 0
        )
        print(f"Bin interior center (robot frame): {bin_pos}")

        # Pre-compute placement slots so every object has a guaranteed in-bin position
        placement_slots = compute_placement_slots(
            bin_pos,
            len(pick_order),
            args_cli.bin_half_x,
            args_cli.bin_half_y,
        )

        # Track success/failure for each object
        results = {}  # object_name -> {stage: success_bool}

        for idx, object_name in enumerate(pick_order):
            object_pos = get_object_pos(env, object_name)
            object_quat = get_object_quat(env, object_name)
            print(f"\n{'=' * 80}")
            print(f"[{idx + 1}/{len(pick_order)}] Planning for object '{object_name}' at {object_pos}")
            print(f"{'=' * 80}")

            results[object_name] = {}

            grasp_quat = compute_grasp_quat(args_cli.grasp_orientation, object_quat, env.device)
            print(f"[GRASP] orientation={args_cli.grasp_orientation}, quat={grasp_quat.tolist()}")

            grasp_xyz = object_pos.clone()
            grasp_xyz[2] += args_cli.grasp_z_offset
            grasp_pose = pose_from_pos_quat(grasp_xyz, grasp_quat)

            # Compute reachability metrics
            current_eef_pose = get_current_eef_pose(env, planner)
            current_eef_pos = current_eef_pose[:3, 3]
            distance_to_goal = torch.norm(grasp_xyz - current_eef_pos).item()

            print(f"[DEBUG GOAL] Current EEF (robot-frame): {current_eef_pos}")
            print(f"[DEBUG GOAL] Grasp XYZ (robot-frame): {grasp_xyz}")
            print(f"[DEBUG GOAL] Distance to goal: {distance_to_goal:.3f}m")
            if args_cli.grasp_z_offset != 0.0:
                print(f"[DEBUG GOAL] grasp_z_offset applied: {args_cli.grasp_z_offset}m")
            print(f"[DEBUG GOAL] Approach distance: {args_cli.approach_distance}m (cuRobo multi-phase)")

            slot_center_xyz = placement_slots[idx]

            place_xyz = slot_center_xyz.clone()
            place_xyz[2] += args_cli.place_z_offset
            place_pose = pose_from_pos_quat(place_xyz, DOWN_FACING_QUAT_WXYZ.to(env.device))

            # Single grasp motion (approach_distance adds multi-phase planning automatically)
            grasp_success = plan_and_execute(
                env,
                planner,
                target_pose=grasp_pose,
                gripper_action=GRIPPER_OPEN_CMD,
                expected_attached_object=None,
                stage=f"{object_name}:grasp",
                sphere_dump_dir=sphere_dump_dir,
                sphere_dump_png=args_cli.dump_spheres_png,
                goal_pose_visualizer=goal_pose_visualizer,
                ee_visualizer=ee_visualizer,
                debug_goal=args_cli.debug_goal,
            )
            results[object_name]["grasp"] = grasp_success
            if not grasp_success:
                print(f"[SKIP] Skipping remaining stages for '{object_name}' due to grasp planning failure")
                continue

            execute_gripper_action(
                env,
                planner,
                gripper_binary_action=GRIPPER_CLOSE_CMD,
                steps=args_cli.gripper_settle_steps,
                env_id=0,
            )

            # Plan directly to place pose (approach_distance ideally should handles final approach)
            place_success = plan_and_execute(
                env,
                planner,
                target_pose=place_pose,
                gripper_action=GRIPPER_CLOSE_CMD,
                expected_attached_object=object_name,
                stage=f"{object_name}:place",
                sphere_dump_dir=sphere_dump_dir,
                sphere_dump_png=args_cli.dump_spheres_png,
                goal_pose_visualizer=goal_pose_visualizer,
                ee_visualizer=ee_visualizer,
                debug_goal=args_cli.debug_goal,
            )
            results[object_name]["place"] = place_success
            if not place_success:
                print(f"[SKIP] Skipping retreat for '{object_name}' due to place failure")
                execute_gripper_action(
                    env, planner, gripper_binary_action=GRIPPER_OPEN_CMD, steps=args_cli.gripper_settle_steps, env_id=0
                )
                continue

            execute_gripper_action(
                env,
                planner,
                gripper_binary_action=GRIPPER_OPEN_CMD,
                steps=args_cli.gripper_settle_steps,
                env_id=0,
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
                    sphere_dump_dir=sphere_dump_dir,
                    sphere_dump_png=args_cli.dump_spheres_png,
                    goal_pose_visualizer=goal_pose_visualizer,
                    ee_visualizer=ee_visualizer,
                    debug_goal=args_cli.debug_goal,
                )
                results[object_name]["retreat"] = retreat_success

        # Print summary
        print(f"\n{'=' * 80}")
        print("PICK AND PLACE RESULTS SUMMARY")
        print(f"{'=' * 80}")

        successful_objects = []
        failed_objects = []

        for obj_name, stages in results.items():
            all_success = all(stages.values()) if stages else False
            if all_success:
                successful_objects.append(obj_name)
                print(f"{obj_name}: SUCCESS (all stages completed)")
            else:
                failed_objects.append(obj_name)
                failed_stage = next((stage for stage, success in stages.items() if not success), "unknown")
                print(f"{obj_name}: FAILED at stage '{failed_stage}'")
                for stage, success in stages.items():
                    status = "Successful" if success else "Failed"
                    print(f" {stage}: {status}")

        print(f"\nTotal: {len(successful_objects)}/{len(results)} objects successfully picked and placed")

        if failed_objects:
            print(f"\nFailed objects: {', '.join(failed_objects)}")
        if successful_objects:
            print(f"Successful objects: {', '.join(successful_objects)}")

        print(f"{'=' * 80}\n")

        if len(successful_objects) == 0:
            print("[INFO] All objects failed; skipping final EEF query.")
            env.close()
            return

        curr_pose = get_current_eef_pose(env, planner)
        curr_quat = math_utils.quat_from_matrix(curr_pose[:3, :3].unsqueeze(0))[0]
        curr_pos = curr_pose[:3, 3]
        print(f"Final EEF pose: pos={curr_pos}, quat_wxyz={curr_quat}")

        for _ in range(10):
            env.step(action_from_pose(env, planner, curr_pose, GRIPPER_OPEN_CMD))

        env.close()


if __name__ == "__main__":
    main()
