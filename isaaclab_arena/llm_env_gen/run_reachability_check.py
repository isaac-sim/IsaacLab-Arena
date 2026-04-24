# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Driver: bring up an Arena env, check IK reachability of the pick and
place grasp poses, and optionally visualize them with Isaac Lab frame-axes
markers.

All primitives live in
:mod:`isaaclab_arena.llm_env_gen.reachability_utils`; this module only
owns the SimulationApp lifecycle, the env bring-up, cuRobo planner
construction, and the top-level IK / marker orchestration.

Both task shapes are supported:

* **Flat** — ``avocadoPnPbowltable`` (top-level task is a ``PickAndPlaceTask``).
* **Sequential** — ``franka_put_and_close_door`` (first subtask is the PnP).

Examples (inside the Arena container with cuRobo installed)::

    /isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_reachability_check.py \\
        --viz kit --num_envs 1 avocadoPnPbowltable

    /isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_reachability_check.py \\
        --viz kit --num_envs 1 franka_put_and_close_door --object cracker_box

Exit codes:

* 0 — pick and place both feasible
* 2 — any target unreachable (pose-space failure)
* 3 — any target in_collision (pose reachable but config rejected)
"""

from __future__ import annotations

import json
import sys
import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.random import set_seed
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

from isaaclab_arena.llm_env_gen.reachability_utils import (
    IK_STATUS_FEASIBLE,
    IK_STATUS_IN_COLLISION,
    IK_STATUS_UNREACHABLE,
    add_ik_reachability_cli_args,
    assert_franka_embodiment,
    build_curobo_target_pose,
    check_ik_feasibility,
    classify_ik_status,
    find_pick_and_place_task,
    format_xyz,
    get_object_pos_in_robot_frame,
    get_object_world_pos,
    get_robot_world_pos,
)


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    add_ik_reachability_cli_args(parser)
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli):
        # Env subparsers register after the SimApp boots so they can
        # introspect the registry for env-specific flags (e.g. --object).
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        args_cli = parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)
        assert_franka_embodiment(arena_builder.arena_env)

        pnp_task = find_pick_and_place_task(arena_builder.arena_env)
        pick_name = pnp_task.pick_up_object.name
        dest = getattr(pnp_task, "destination_location", None)
        dest_name = getattr(dest, "name", None) if dest is not None else None

        env_name = getattr(arena_builder.arena_env, "name", type(arena_builder.arena_env).__name__)
        print(
            f"[reachability] Env: '{env_name}' | pick: '{pick_name}' | place: '{dest_name}'",
            flush=True,
        )

        env, _ = arena_builder.make_registered_and_return_cfg()
        if args_cli.seed is not None:
            set_seed(args_cli.seed, env)

        env_id = int(args_cli.env_id)
        num_envs = int(args_cli.num_envs)
        assert 0 <= env_id < num_envs, f"--env_id {env_id} out of range for --num_envs {num_envs}"

        # Optional target markers — only when a viewer is open. We use
        # the stock FRAME_MARKER_CFG (RGB axes USD) but scale it ~3x
        # larger than Arena's built-in FrameTransformer markers, which
        # are at scale 0.1. Sphere markers were tried and collided
        # visually with Arena's contact-sensor visualizer (also
        # spheres), so we're back to axes — just much bigger.
        markers: dict = {}
        if not getattr(args_cli, "headless", False):
            from copy import deepcopy

            from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers

            def _make_big_frame(prim_path: str, scale: float = 0.3) -> VisualizationMarkers:
                cfg = deepcopy(FRAME_MARKER_CFG)
                cfg.prim_path = prim_path
                frame = cfg.markers.get("frame")
                if frame is not None:
                    # Arena's ee_frame uses (0.1, 0.1, 0.1). 0.3 puts us
                    # at 3x their size so ours is unmistakable.
                    setattr(frame, "scale", (scale, scale, scale))
                return VisualizationMarkers(cfg)

            markers["pick_hand"] = _make_big_frame("/World/Visuals/reach_pick_hand")
            if dest_name is not None:
                markers["place_hand"] = _make_big_frame("/World/Visuals/reach_place_hand")

        try:
            env.reset()
            zero = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(zero)

            robot_world_pos = get_robot_world_pos(env, env_id)
            print(
                f"[reachability] Robot base world pos: {format_xyz(robot_world_pos)}",
                flush=True,
            )

            # World → robot-base frame for the IK target. Printing both
            # frames so any mismatch is obvious in the log.
            pick_world = get_object_world_pos(env, pick_name, env_id)
            pick_pos_local = get_object_pos_in_robot_frame(env, pick_name, env_id)
            print(
                f"[reachability] Pick  '{pick_name}': world={format_xyz(pick_world)} "
                f"robot_frame={format_xyz(pick_pos_local)}",
                flush=True,
            )

            dest_world = None
            dest_pos_local = None
            if dest_name is not None:
                # Some envs wire the destination as an ObjectReference to a
                # prim nested inside an articulation (e.g. the microwave's
                # interior disc in franka_put_and_close_door). Those are
                # not in scene.rigid_objects, so the lookup raises. Treat
                # that as "skip the place check" rather than failing the
                # whole run — pick reachability is still useful.
                try:
                    dest_world = get_object_world_pos(env, dest_name, env_id)
                    dest_pos_local = get_object_pos_in_robot_frame(env, dest_name, env_id)
                except AssertionError as exc:
                    print(
                        f"[reachability] Place '{dest_name}': not a rigid_object — "
                        f"skipping place IK check. ({exc})",
                        flush=True,
                    )
                    dest_name = None
                else:
                    print(
                        f"[reachability] Place '{dest_name}': world={format_xyz(dest_world)} "
                        f"robot_frame={format_xyz(dest_pos_local)}",
                        flush=True,
                    )

            # Lazy import so --help works without cuRobo installed.
            try:
                from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
                from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
            except ImportError as exc:
                print(
                    "[reachability] ERROR: cuRobo / isaaclab_mimic is not installed in this container.\n"
                    "  Re-run `./docker/run_docker.sh -c` to build with cuRobo.",
                    file=sys.stderr,
                )
                raise exc

            planner_cfg = CuroboPlannerCfg.franka_config()
            planner_cfg.debug_planner = bool(args_cli.debug_planner)
            if args_cli.position_threshold is not None:
                planner_cfg.position_threshold = float(args_cli.position_threshold)
            if args_cli.rotation_threshold is not None:
                planner_cfg.rotation_threshold = float(args_cli.rotation_threshold)
            # Disable cuRobo's rerun-backed visualization hooks; our
            # Isaac-Lab markers (above) are the viz surface.
            planner_cfg.visualize_plan = False
            planner_cfg.visualize_spheres = False

            print("[reachability] Initializing cuRobo planner from USD stage...", flush=True)
            planner = CuroboPlanner(
                env=env.unwrapped,
                robot=env.unwrapped.scene["robot"],
                config=planner_cfg,
                env_id=env_id,
            )
            # Sync object poses into cuRobo's collision world before IK.
            planner.update_world()

            pose_kwargs = dict(
                top_down_offset=float(args_cli.top_down_offset),
                hand_to_tcp_z=float(args_cli.hand_to_tcp_z),
                grasp_axis=str(args_cli.grasp_axis),
                device=planner.tensor_args.device,
            )
            pick_target, _ = build_curobo_target_pose(object_pos_local=pick_pos_local, **pose_kwargs)
            print(
                f"[reachability] Pick  top-down hand target: "
                f"{format_xyz(pick_target.position.squeeze(0).detach().cpu())}",
                flush=True,
            )
            place_target = None
            if dest_pos_local is not None:
                place_target, _ = build_curobo_target_pose(object_pos_local=dest_pos_local, **pose_kwargs)
                print(
                    f"[reachability] Place top-down hand target: "
                    f"{format_xyz(place_target.position.squeeze(0).detach().cpu())}",
                    flush=True,
                )

            pos_thresh = float(planner_cfg.position_threshold)
            rot_thresh = float(planner_cfg.rotation_threshold)

            pick_feasible, pick_pos_err, pick_rot_err, pick_q = check_ik_feasibility(planner, pick_target)
            pick_status = classify_ik_status(
                pick_feasible, pick_pos_err, pick_rot_err, pos_thresh, rot_thresh
            )
            print(
                f"[reachability] Pick  status: {pick_status} "
                f"(pos_err={pick_pos_err:.4f} m, rot_err={pick_rot_err:.4f} rad)",
                flush=True,
            )

            place_status = None
            place_pos_err = float("nan")
            place_rot_err = float("nan")
            if place_target is not None:
                # Warm-seed the place IK with the pick's solution so both
                # ends converge from a consistent basin when poses are close.
                place_feasible, place_pos_err, place_rot_err, _ = check_ik_feasibility(
                    planner, place_target, seed_config=pick_q
                )
                place_status = classify_ik_status(
                    place_feasible, place_pos_err, place_rot_err, pos_thresh, rot_thresh
                )
                print(
                    f"[reachability] Place status: {place_status} "
                    f"(pos_err={place_pos_err:.4f} m, rot_err={place_rot_err:.4f} rad)",
                    flush=True,
                )

            # Drop a marker at each hand (cuRobo) grasp pose in world
            # frame. Uses robot-base → world conversion: the target was
            # built in robot-base frame, so world = robot_world_pos +
            # target (assumes robot base has no yaw; see
            # get_object_pos_in_robot_frame docstring).
            if markers:
                device = robot_world_pos.device

                def _world_from_robot_frame(vec3: torch.Tensor) -> torch.Tensor:
                    return (robot_world_pos + vec3.to(device)).unsqueeze(0)

                def _marker_quat(target):
                    return target.quaternion.to(device).reshape(1, 4)

                markers["pick_hand"].visualize(
                    translations=_world_from_robot_frame(pick_target.position.squeeze(0)),
                    orientations=_marker_quat(pick_target),
                )

                if "place_hand" in markers and place_target is not None:
                    markers["place_hand"].visualize(
                        translations=_world_from_robot_frame(place_target.position.squeeze(0)),
                        orientations=_marker_quat(place_target),
                    )

                # Tick the sim so the markers render before env.close().
                # Dwell is configurable via --dwell_steps so the user can
                # keep the Kit viewer up long enough to inspect.
                for _ in range(int(args_cli.dwell_steps)):
                    env.step(zero)

            overall_feasible = pick_status == IK_STATUS_FEASIBLE and (
                place_status is None or place_status == IK_STATUS_FEASIBLE
            )
            statuses = [s for s in (pick_status, place_status) if s is not None]
            overall_status = (
                IK_STATUS_FEASIBLE
                if overall_feasible
                else IK_STATUS_IN_COLLISION
                if IK_STATUS_IN_COLLISION in statuses and IK_STATUS_UNREACHABLE not in statuses
                else IK_STATUS_UNREACHABLE
            )
            print(f"[reachability] Overall: {overall_status}", flush=True)

            if args_cli.json:
                payload: dict = {
                    "overall_feasible": overall_feasible,
                    "overall_status": overall_status,
                    "pick": {
                        "object": pick_name,
                        "status": pick_status,
                        "position_error_m": pick_pos_err,
                        "rotation_error_rad": pick_rot_err,
                    },
                    "place": (
                        None
                        if place_status is None
                        else {
                            "object": dest_name,
                            "status": place_status,
                            "position_error_m": place_pos_err,
                            "rotation_error_rad": place_rot_err,
                        }
                    ),
                }
                print(json.dumps(payload))

            if overall_feasible:
                return 0
            if overall_status == IK_STATUS_IN_COLLISION:
                return 3
            return 2

        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
