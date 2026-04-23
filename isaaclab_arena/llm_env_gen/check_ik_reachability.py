# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""IK reachability and motion-planning feasibility check for Arena pick-and-place tasks.

Builds any registered Arena environment (with a Franka embodiment and a
:class:`PickAndPlaceTask`), constructs a cuRobo motion planner the same way
``isaaclab_mimic`` does, then asks two questions about the current initial
condition:

1. **IK reachability** — is there a joint configuration that places the
   end-effector at a top-down grasp pose above the pick object?
2. **Motion-plan feasibility** — is there a collision-free trajectory from
   the robot's initial joint state to that grasp pose given the extracted
   scene (tables, backgrounds, other objects)?

The grasp pose is a top-down grasp: the panda_hand Z-axis points in world
``-Z`` and the TCP is positioned ``--top_down_offset`` meters above the
object center.

Prerequisites
-------------
* ``curobo`` and ``rerun-sdk`` (optional, only needed if you enable
  visualization) installed in the container. See
  ``submodules/IsaacLab/docs/source/overview/imitation-learning/skillgen.rst``.
* The task must be a :class:`PickAndPlaceTask` (or subclass) and the
  embodiment must be a Franka variant, e.g. ``franka_ik``.

Example
-------
Inside the running Arena container::

    /isaac-sim/python.sh isaaclab_arena/llm_env_gen/check_ik_reachability.py \\
        --headless \\
        --num_envs 1 \\
        --top_down_offset 0.05 \\
        kitchen_pick_and_place \\
        --object cracker_box \\
        --embodiment franka_ik
"""

from __future__ import annotations

import argparse
import sys
import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.random import set_seed
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser


def add_ik_reachability_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add IK-reachability-specific CLI arguments to the top-level parser."""
    group = parser.add_argument_group(
        "IK Reachability Check",
        "Arguments for the cuRobo-based IK reachability and motion-plan feasibility check.",
    )
    group.add_argument(
        "--top_down_offset",
        type=float,
        default=0.02,
        help=(
            "Z-offset (meters) of the TCP above the pick object center for the top-down grasp. "
            "Should be roughly half the object's height for a realistic grasp (default: 0.02)."
        ),
    )
    group.add_argument(
        "--hand_to_tcp_z",
        type=float,
        default=0.1034,
        help=(
            "Offset (meters) along panda_hand Z from the hand frame to the TCP between the fingers. "
            "Used to convert the TCP target into cuRobo's EE (panda_hand) target. "
            "Default matches the stock Franka Panda."
        ),
    )
    group.add_argument(
        "--grasp_axis",
        choices=["x", "y"],
        default="x",
        help=(
            "Axis of the 180-degree rotation that makes the hand Z-axis point downward. "
            "With 'x' the fingers close along world-Y; with 'y' they close along world-X."
        ),
    )
    group.add_argument(
        "--env_id",
        type=int,
        default=0,
        help="Environment index to check (0 to --num_envs-1).",
    )
    group.add_argument(
        "--debug_planner",
        action="store_true",
        default=False,
        help="Enable cuRobo planner debug logging.",
    )
    group.add_argument(
        "--position_threshold",
        type=float,
        default=None,
        metavar="METERS",
        help=(
            "Override cuRobo IK position convergence threshold in meters "
            "(default from franka_config: 0.005 m). Increase to 0.01-0.02 to check "
            "borderline reachability without tightening rotation."
        ),
    )
    group.add_argument(
        "--rotation_threshold",
        type=float,
        default=None,
        metavar="RAD",
        help=(
            "Override cuRobo IK rotation convergence threshold in radians "
            "(default from franka_config: 0.05 rad). Increase to 0.3-0.8 to diagnose "
            "whether the orientation is simply wrong vs the pose being unreachable."
        ),
    )
    group.add_argument(
        "--ik_only",
        action="store_true",
        default=False,
        help="Skip the motion-planning step; only run the IK reachability check.",
    )


def _top_down_quaternion_wxyz(axis: str) -> tuple[float, float, float, float]:
    """Return a ``(w, x, y, z)`` quaternion whose rotation points panda_hand Z downward."""
    if axis == "x":
        return (0.0, 1.0, 0.0, 0.0)  # 180 deg about world-X
    if axis == "y":
        return (0.0, 0.0, 1.0, 0.0)  # 180 deg about world-Y
    raise ValueError(f"Unsupported grasp axis: {axis!r}")


def _resolve_pick_object_name(arena_env) -> str:
    """Extract the Isaac Lab scene-entity name of the pick-up object from the task."""
    task = arena_env.task
    assert task is not None, "Environment must provide a task for the IK reachability check."
    assert hasattr(task, "pick_up_object"), (
        f"Task {type(task).__name__} does not expose a 'pick_up_object'; "
        "this script only supports PickAndPlaceTask-derived tasks."
    )
    pick = task.pick_up_object
    assert hasattr(pick, "name") and isinstance(
        pick.name, str
    ), f"Task.pick_up_object must expose a string 'name' attribute; got {type(pick).__name__}."
    return pick.name


def _assert_franka_embodiment(arena_env) -> None:
    embodiment = arena_env.embodiment
    assert embodiment is not None, "Environment must declare an embodiment."
    name = getattr(embodiment, "name", type(embodiment).__name__).lower()
    assert (
        "franka" in name
    ), f"This script only supports Franka embodiments (got '{name}'). Pass --embodiment franka_ik or franka_joint_pos."


def _get_pick_object_local_pos(env, pick_object_name: str, env_id: int) -> torch.Tensor:
    """Return the pick object's position expressed in the robot's env-local frame."""
    import warp as wp

    rigid_objects = env.unwrapped.scene.rigid_objects
    assert (
        pick_object_name in rigid_objects
    ), f"Pick object '{pick_object_name}' not found in scene.rigid_objects. Available: {list(rigid_objects.keys())}"
    obj = rigid_objects[pick_object_name]
    env_origin = env.unwrapped.scene.env_origins[env_id]
    pos_w = wp.to_torch(obj.data.root_pos_w)[env_id]
    return pos_w - env_origin


def _build_curobo_target_pose(
    object_pos_local: torch.Tensor,
    top_down_offset: float,
    hand_to_tcp_z: float,
    grasp_axis: str,
    device: torch.device,
):
    """Construct the ``panda_hand`` target Pose (cuRobo ``Pose``) and its 4x4 equivalent.

    Returns a tuple ``(curobo_pose, hand_target_4x4)`` where the 4x4 is in robot-local
    frame with wxyz rotation order converted to a rotation matrix.
    """
    import isaaclab.utils.math as PoseUtils
    from curobo.types.math import Pose

    tcp_target = object_pos_local.to(device=device, dtype=torch.float32).clone()
    tcp_target[2] = tcp_target[2] + top_down_offset

    # With panda_hand pointing straight down, the TCP sits hand_to_tcp_z below the hand
    # in world coordinates -> the hand origin is that far *above* the TCP.
    hand_target_pos = tcp_target.clone()
    hand_target_pos[2] = hand_target_pos[2] + hand_to_tcp_z

    qw, qx, qy, qz = _top_down_quaternion_wxyz(grasp_axis)
    quat_wxyz = torch.tensor([qw, qx, qy, qz], device=device, dtype=torch.float32)

    curobo_pose = Pose(
        position=hand_target_pos.unsqueeze(0),  # (1, 3)
        quaternion=quat_wxyz.unsqueeze(0),  # (1, 4) wxyz
    )

    rot_matrix = PoseUtils.matrix_from_quat(quat_wxyz)
    hand_target_4x4 = PoseUtils.make_pose(hand_target_pos, rot_matrix)
    return curobo_pose, hand_target_4x4


def _run_ik_reachability(planner, target_pose) -> dict[str, float | bool]:
    """Call cuRobo's IK solver. Returns a summary dict with success and errors."""
    ik_solver = planner.motion_gen.ik_solver
    result = ik_solver.solve_single(target_pose)

    success_tensor = getattr(result, "success", None)
    if success_tensor is None:
        return {"success": False, "position_error": float("nan"), "rotation_error": float("nan")}

    success = bool(success_tensor.any().item())

    def _safe_min(attr: str) -> float:
        tensor = getattr(result, attr, None)
        if tensor is None:
            return float("nan")
        try:
            return float(tensor.min().item())
        except (RuntimeError, ValueError):
            return float("nan")

    return {
        "success": success,
        "position_error": _safe_min("position_error"),
        "rotation_error": _safe_min("rotation_error"),
    }


def _run_motion_plan(planner, hand_target_4x4: torch.Tensor) -> dict[str, object]:
    """Sync the world and plan a single-phase motion to the target. Returns a summary."""
    planner.update_world()
    ok = planner.plan_motion(hand_target_4x4)
    num_waypoints = 0
    if ok and planner.current_plan is not None:
        num_waypoints = int(len(planner.current_plan.position))
    return {"success": bool(ok), "num_waypoints": num_waypoints}


def _format_xyz(t: torch.Tensor) -> str:
    return f"({t[0].item():+.3f}, {t[1].item():+.3f}, {t[2].item():+.3f})"


def _log_initial_scene(planner, target_pose) -> None:
    """Log robot spheres, world meshes, and target pose to Rerun unconditionally.

    Normally ``PlanVisualizer.visualize_plan`` is only called on planning success.
    This helper forces a snapshot so the scene is always visible in the viewer,
    even when the plan fails or is skipped.
    """
    try:
        from curobo.types.robot import JointState
        from curobo.types.state import WorldConfig

        viz = planner.plan_visualizer
        cu_js: JointState = planner._get_current_joint_state_for_curobo()
        sphere_list = planner.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)[0]

        try:
            world_scene = WorldConfig.get_scene_graph(planner.motion_gen.world_coll_checker.world_model)
        except Exception:
            world_scene = None

        # Build a trivial single-waypoint "plan" so visualize_plan can render the robot pose.
        dummy_plan = cu_js.unsqueeze(0) if hasattr(cu_js, "unsqueeze") else cu_js

        viz.visualize_plan(
            plan=dummy_plan,
            target_pose=target_pose.position.squeeze(0),
            robot_spheres=sphere_list,
            attached_spheres=None,
            ee_positions=None,
            world_scene=world_scene,
        )
        print("[ik-check] Initial scene logged to Rerun.")
    except Exception as exc:
        print(f"[ik-check] Warning: could not log initial scene to Rerun: {exc}")


def main() -> int:
    args_parser = get_isaaclab_arena_cli_parser()
    add_ik_reachability_cli_args(args_parser)
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        # Add the env subparsers last (they require known args to dispatch).
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_cli = args_parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)

        _assert_franka_embodiment(arena_builder.arena_env)
        pick_object_name = _resolve_pick_object_name(arena_builder.arena_env)

        env, _ = arena_builder.make_registered_and_return_cfg()

        if args_cli.seed is not None:
            set_seed(args_cli.seed, env)

        env_id = int(args_cli.env_id)
        num_envs = int(args_cli.num_envs)
        assert 0 <= env_id < num_envs, f"--env_id {env_id} out of range for --num_envs {num_envs}"

        try:
            # Reset + one zero action so the physics / observation buffers are populated
            # before we query object poses and initialize cuRobo.
            env.reset()
            zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(zero_action)

            object_pos_local = _get_pick_object_local_pos(env, pick_object_name, env_id)
            print(
                f"[ik-check] Pick object '{pick_object_name}' local pos (env {env_id}): {_format_xyz(object_pos_local)}"
            )

            # Lazy import so '--help' works without cuRobo installed, and we can give a
            # friendly error if the user is missing the install.
            try:
                from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
                from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
            except ImportError as e:
                print(
                    "[ik-check] ERROR: cuRobo / isaaclab_mimic is not installed in this container.\n"
                    "  See submodules/IsaacLab/docs/source/overview/imitation-learning/skillgen.rst"
                    " for install instructions.",
                    file=sys.stderr,
                )
                raise e

            planner_cfg = CuroboPlannerCfg.franka_config()
            planner_cfg.debug_planner = bool(args_cli.debug_planner)
            if args_cli.position_threshold is not None:
                planner_cfg.position_threshold = float(args_cli.position_threshold)
                print(f"[ik-check] position_threshold overridden to {planner_cfg.position_threshold} m")
            if args_cli.rotation_threshold is not None:
                planner_cfg.rotation_threshold = float(args_cli.rotation_threshold)
                print(f"[ik-check] rotation_threshold overridden to {planner_cfg.rotation_threshold} rad")
            planner_cfg.visualize_plan = True
            planner_cfg.visualize_spheres = True

            print("[ik-check] Initializing cuRobo planner (this extracts obstacles from the USD stage)...")
            planner = CuroboPlanner(
                env=env.unwrapped,
                robot=env.unwrapped.scene["robot"],
                config=planner_cfg,
                env_id=env_id,
            )

            target_pose, hand_target_4x4 = _build_curobo_target_pose(
                object_pos_local=object_pos_local,
                top_down_offset=float(args_cli.top_down_offset),
                hand_to_tcp_z=float(args_cli.hand_to_tcp_z),
                grasp_axis=str(args_cli.grasp_axis),
                device=planner.tensor_args.device,
            )
            print(
                "[ik-check] Top-down target hand pos (robot-local): "
                f"{_format_xyz(target_pose.position.squeeze(0).detach().cpu())}"
                f"  quat_wxyz=({args_cli.grasp_axis}-axis 180 deg)"
            )

            ik_summary = _run_ik_reachability(planner, target_pose)
            print(
                f"[ik-check] IK reachable: {ik_summary['success']} "
                f"(pos_err={ik_summary['position_error']:.4f} m, rot_err={ik_summary['rotation_error']:.4f} rad)"
            )

            # Log initial robot spheres + world meshes unconditionally so Rerun
            # shows something even if the motion plan fails.
            _log_initial_scene(planner, target_pose)

            plan_summary: dict[str, object]
            if args_cli.ik_only or not ik_summary["success"]:
                plan_summary = {"success": False, "num_waypoints": 0, "skipped": True}
                reason = "ik_only flag" if args_cli.ik_only else "IK was infeasible"
                print(f"[ik-check] Skipping motion-plan check ({reason}).")
            else:
                plan_summary = _run_motion_plan(planner, hand_target_4x4)
                print(
                    f"[ik-check] Collision-free plan found: {plan_summary['success']} "
                    f"(waypoints={plan_summary['num_waypoints']})"
                )

            overall = bool(ik_summary["success"]) and (args_cli.ik_only or bool(plan_summary.get("success", False)))
            print(f"[ik-check] Overall feasibility: {overall}")

            return 0 if overall else 2

        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
