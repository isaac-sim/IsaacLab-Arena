#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# From Neel: This script is an example of how to use the curobo planner to pick and place objects.
# It is not a complete pick and place policy, but it is a good starting point

# can run this command: python isaaclab_arena/scripts/curobo/run_droid_v2_tabletop_curobo_pick_place.py droid_v2_tabletop_pick_and_place --grasp_z_offset 0.17 --approach_distance 0.08 --retreat_distance 0.08 --debug_planner --debug_goal --pick_order tomato_soup_can --grasp_orientation object_yaw --post_place_clearance 0.0

# (Neel: New command for IK based A star with num demos and max attempts)
# python isaaclab_arena/scripts/curobo/run_droid_v2_tabletop_curobo_pick_place.py droid_v3_tabletop_pick_and_place --grasp_z_offset 0.17 --approach_distance 0.08 --retreat_distance 0.08 --debug_planner --debug_goal --grasp_orientation object_yaw --post_place_clearance 0.0 --num_demos 3

from __future__ import annotations

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

    allowed_envs = {"droid_v2_tabletop_pick_and_place", "droid_v3_tabletop_pick_and_place"}
    if args_cli.example_environment not in allowed_envs:
        raise ValueError(f"This script requires one of {allowed_envs}, got '{args_cli.example_environment}'.")

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

        bin_pos = get_bin_interior_center(env, "blue_sorting_bin", 0, 0)
        print(f"Bin interior center (robot frame): {bin_pos}")

        is_v3 = args_cli.example_environment == "droid_v3_tabletop_pick_and_place"

        # V3: build the IK cost function once (closure over stable references)
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
            """Sequential IK penalty: evaluates grasp then place as a chain.

            If grasp IK fails, place IK is skipped (it would be meaningless
            since the arm can't reach the grasp pose). The full penalty is
            returned and the joint config stays unchanged so the next object
            in the A* chain doesn't get a misleading seed.
            """
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

        # ── Demo collection loop ──────────────────────────────────────────
        num_demos = args_cli.num_demos
        max_attempts = args_cli.max_attempts or num_demos * 10
        successes = 0

        for attempt in range(1, max_attempts + 1):
            if successes >= num_demos:
                break

            print(f"\n{'#' * 80}")
            print(f"ATTEMPT {attempt} | Successful demos: {successes}/{num_demos}")
            print(f"{'#' * 80}")

            if attempt > 1:
                env.reset()

            # ── Plan pick order ───────────────────────────────────────────
            if is_v3:
                target_positions = arena_builder.example_env.generate_target_positions()
                planner.update_world()

                pick_order, ik_penalty = arena_builder.example_env.plan_pick_order(
                    verbose=args_cli.debug_planner, ik_cost_fn=_ik_cost_fn,
                )
                if args_cli.max_objects is not None:
                    pick_order = pick_order[: args_cli.max_objects]

                if ik_penalty > 0:
                    print(f"[SKIP] Layout IK-infeasible (penalty={ik_penalty:.1f}), resetting")
                    continue

                placement_slots = {}
                for obj_name in pick_order:
                    world_pos = torch.tensor(target_positions[obj_name], device=env.device)
                    placement_slots[obj_name] = (R_w2r @ (world_pos - robot_pos_w).unsqueeze(-1)).squeeze(-1)
            else:
                pick_order = auto_pick_order(env, explicit_order=args_cli.pick_order)
                if args_cli.max_objects is not None:
                    pick_order = pick_order[: args_cli.max_objects]
                slot_list = compute_placement_slots(
                    bin_pos, len(pick_order), args_cli.bin_half_x, args_cli.bin_half_y,
                )
                placement_slots = {name: slot_list[i] for i, name in enumerate(pick_order)}

            if not pick_order:
                print("[WARN] No pickable objects found, skipping attempt")
                continue

            print(f"Pick order: {pick_order}")

            # ── Execute pick-place for each object ────────────────────────
            results: dict[str, dict[str, bool]] = {}

            for idx, object_name in enumerate(pick_order):
                object_pos = get_object_pos(env, object_name)
                object_quat = get_object_quat(env, object_name)
                print(f"\n{'=' * 80}")
                print(f"[{idx + 1}/{len(pick_order)}] Object '{object_name}' at {object_pos}")
                print(f"{'=' * 80}")

                results[object_name] = {}

                grasp_pose, place_pose = compute_grasp_and_place_poses(
                    object_pos,
                    object_quat,
                    placement_slots[object_name],
                    args_cli.grasp_orientation,
                    args_cli.grasp_z_offset,
                    args_cli.place_z_offset,
                    env.device,
                )

                if args_cli.debug_goal:
                    eef_pos = get_current_eef_pose(env, planner)[:3, 3]
                    grasp_xyz = grasp_pose[:3, 3]
                    print(f"[DEBUG GOAL] EEF={eef_pos}, grasp={grasp_xyz}, "
                          f"dist={torch.norm(grasp_xyz - eef_pos).item():.3f}m")

                grasp_success = plan_and_execute(
                    env, planner, target_pose=grasp_pose,
                    gripper_action=GRIPPER_OPEN_CMD, expected_attached_object=None,
                    stage=f"{object_name}:grasp",
                    sphere_dump_dir=sphere_dump_dir, sphere_dump_png=args_cli.dump_spheres_png,
                    goal_pose_visualizer=goal_pose_visualizer, ee_visualizer=ee_visualizer,
                    debug_goal=args_cli.debug_goal,
                )
                results[object_name]["grasp"] = grasp_success
                if not grasp_success:
                    print(f"[SKIP] '{object_name}' grasp failed")
                    continue

                execute_gripper_action(
                    env, planner, gripper_binary_action=GRIPPER_CLOSE_CMD,
                    steps=args_cli.gripper_settle_steps, env_id=0,
                )

                place_success = plan_and_execute(
                    env, planner, target_pose=place_pose,
                    gripper_action=GRIPPER_CLOSE_CMD, expected_attached_object=object_name,
                    stage=f"{object_name}:place",
                    sphere_dump_dir=sphere_dump_dir, sphere_dump_png=args_cli.dump_spheres_png,
                    goal_pose_visualizer=goal_pose_visualizer, ee_visualizer=ee_visualizer,
                    debug_goal=args_cli.debug_goal,
                )
                results[object_name]["place"] = place_success
                if not place_success:
                    print(f"[SKIP] '{object_name}' place failed")
                    execute_gripper_action(
                        env, planner, gripper_binary_action=GRIPPER_OPEN_CMD,
                        steps=args_cli.gripper_settle_steps, env_id=0,
                    )
                    continue

                execute_gripper_action(
                    env, planner, gripper_binary_action=GRIPPER_OPEN_CMD,
                    steps=args_cli.gripper_settle_steps, env_id=0,
                )

                if args_cli.post_place_clearance > 0:
                    retreat_pose = compute_retreat_pose(
                        place_pose, args_cli.post_place_clearance, env.device
                    )
                    retreat_ok = plan_and_execute(
                        env, planner, target_pose=retreat_pose,
                        gripper_action=GRIPPER_OPEN_CMD, expected_attached_object=None,
                        stage=f"{object_name}:retreat",
                        sphere_dump_dir=sphere_dump_dir, sphere_dump_png=args_cli.dump_spheres_png,
                        goal_pose_visualizer=goal_pose_visualizer, ee_visualizer=ee_visualizer,
                        debug_goal=args_cli.debug_goal,
                    )
                    results[object_name]["retreat"] = retreat_ok

            # ── Per-attempt summary ───────────────────────────────────────
            ok_objs = [n for n, s in results.items() if s and all(s.values())]
            fail_objs = [n for n in results if n not in ok_objs]
            all_success = len(ok_objs) == len(pick_order) and len(pick_order) > 0

            print(f"\n{'=' * 80}")
            print(f"ATTEMPT {attempt} SUMMARY")
            print(f"{'=' * 80}")
            for obj_name in ok_objs:
                print(f"  {obj_name}: SUCCESS")
            for obj_name in fail_objs:
                stages = results.get(obj_name, {})
                failed_at = next((s for s, v in stages.items() if not v), "no stages")
                print(f"  {obj_name}: FAILED at '{failed_at}'")
            print(f"  Objects: {len(ok_objs)}/{len(pick_order)} succeeded")

            if all_success:
                successes += 1
                print(f"  >> Demo {successes}/{num_demos} recorded")

        # ── Final summary ─────────────────────────────────────────────────
        print(f"\n{'#' * 80}")
        print(f"COLLECTION COMPLETE: {successes}/{num_demos} demos in {attempt} attempts")
        if successes < num_demos:
            print(f"WARNING: reached max attempts ({max_attempts}) before target")
        print(f"{'#' * 80}")

        env.close()


if __name__ == "__main__":
    main()
