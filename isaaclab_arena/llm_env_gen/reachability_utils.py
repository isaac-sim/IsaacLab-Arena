# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reachability primitives shared by the IK feasibility driver.

This module hosts **only** pure functions and constants — no CLI entry
point, no :class:`SimulationAppContext` lifecycle, no top-level I/O. The
companion driver
:mod:`isaaclab_arena.llm_env_gen.run_reachability_check` composes these
primitives into the full env-bring-up → IK-check flow.

Scope: Franka pick-and-place tasks. The grasp pose is a top-down grasp
(panda_hand Z-axis points world ``-Z``; TCP positioned ``top_down_offset``
meters above the object center).
"""

from __future__ import annotations

import argparse
import math
import torch

# ---------------------------------------------------------------------------
# Status constants (stable strings — safe to emit in JSON payloads)
# ---------------------------------------------------------------------------

IK_STATUS_FEASIBLE = "feasible"
IK_STATUS_UNREACHABLE = "unreachable"
IK_STATUS_IN_COLLISION = "in_collision"


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


def add_ik_reachability_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register reachability-check CLI arguments on the given parser.

    Call from the driver before :func:`SimulationAppContext` is entered,
    so ``--help`` works without cuRobo installed.
    """
    group = parser.add_argument_group(
        "IK Reachability Check",
        "Arguments for the cuRobo-based IK reachability check.",
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
        "--json",
        action="store_true",
        default=False,
        help=(
            "Print a final single-line JSON payload to stdout with per-target IK results. "
            "Useful for piping into rejection-sampling wrappers: `... | tail -n 1 | jq`."
        ),
    )
    group.add_argument(
        "--dwell_steps",
        type=int,
        default=300,
        help=(
            "Number of zero-action sim steps to take after rendering the target markers "
            "so the Kit viewer stays up for inspection. Only runs when a viewer is open "
            "(i.e. not --headless). Default 300 (~10s); bump for longer inspection."
        ),
    )
    group.add_argument(
        "--door_approach_offset",
        type=float,
        default=0.10,
        help=(
            "Distance (meters) from the openable object's center along the door-facing axis "
            "where the panda_hand target is placed for open-door IK checks. "
            "Increase if the hand clips the object; decrease to check a tighter approach. "
            "Default: 0.10 m."
        ),
    )
    group.add_argument(
        "--door_facing_axis",
        choices=["-x", "+x", "-y", "+y"],
        default="-x",
        help=(
            "World-frame axis the openable object's door faces (the direction pointing "
            "away from the object toward the robot). '-x' is the default for envs generated "
            "with kitchen background + RotateAroundSolution(yaw=-pi/2). "
            "Default: '-x'."
        ),
    )


# ---------------------------------------------------------------------------
# Task / env introspection
# ---------------------------------------------------------------------------


def assert_franka_embodiment(arena_env) -> None:
    """Raise unless the env's embodiment name contains 'franka'."""
    embodiment = arena_env.embodiment
    assert embodiment is not None, "Environment must declare an embodiment."
    name = getattr(embodiment, "name", type(embodiment).__name__).lower()
    assert (
        "franka" in name
    ), f"This script only supports Franka embodiments (got '{name}'). Pass --embodiment franka_ik or franka_joint_pos."


def resolve_pick_object_name(arena_env) -> str:
    """Flat PickAndPlaceTask only: return ``task.pick_up_object.name``."""
    task = arena_env.task
    assert task is not None, "Environment must provide a task for the IK reachability check."
    assert hasattr(task, "pick_up_object"), (
        f"Task {type(task).__name__} does not expose a 'pick_up_object'; "
        "this helper only supports PickAndPlaceTask-derived tasks. "
        "Use find_pick_and_place_task(...) to handle sequential wrappers."
    )
    pick = task.pick_up_object
    assert hasattr(pick, "name") and isinstance(
        pick.name, str
    ), f"Task.pick_up_object must expose a string 'name' attribute; got {type(pick).__name__}."
    return pick.name


def resolve_destination_name(arena_env) -> str:
    """Flat PickAndPlaceTask only: return ``task.destination_location.name``."""
    task = arena_env.task
    assert task is not None, "Environment must provide a task for the IK reachability check."
    assert hasattr(task, "destination_location"), (
        f"Task {type(task).__name__} does not expose a 'destination_location'; "
        "this helper only supports PickAndPlaceTask-derived tasks."
    )
    dest = task.destination_location
    assert hasattr(dest, "name") and isinstance(
        dest.name, str
    ), f"Task.destination_location must expose a string 'name' attribute; got {type(dest).__name__}."
    return dest.name


def find_pick_and_place_task(arena_env):
    """Return the ``PickAndPlaceTask``-like object for the env.

    Walks one level of subtasks so sequential wrappers
    (e.g. ``FrankaPutAndCloseDoorTask``) that keep the PnP as a subtask are
    also supported. The returned object must expose ``pick_up_object.name``.
    """
    task = arena_env.task
    assert task is not None, "Environment must provide a task for the IK check."

    # Case 1 — flat: the top-level task IS a PickAndPlaceTask.
    if hasattr(task, "pick_up_object"):
        return task

    # Case 2 — sequential: scan subtasks for the first PnP-shaped one.
    subtasks = getattr(task, "subtasks", None)
    if subtasks:
        for sub in subtasks:
            if hasattr(sub, "pick_up_object"):
                return sub

    raise AssertionError(
        f"Task {type(task).__name__} has no 'pick_up_object' (direct or in subtasks); "
        "this helper needs a PickAndPlaceTask somewhere in the task chain."
    )


def find_open_close_door_task(arena_env):
    """Return the OpenDoorTask / CloseDoorTask object for the env.

    Uses duck-typing: any task (or subtask) that exposes ``openable_object``
    is accepted, so no Isaac Sim import is required here.

    Raises ``AssertionError`` if no such task is found.
    """
    task = arena_env.task
    assert task is not None, "Environment must provide a task for the IK check."

    # Case 1 — flat: the top-level task exposes openable_object.
    if hasattr(task, "openable_object"):
        return task

    # Case 2 — sequential wrapper: scan subtasks.
    subtasks = getattr(task, "subtasks", None)
    if subtasks:
        for sub in subtasks:
            if hasattr(sub, "openable_object"):
                return sub

    raise AssertionError(
        f"Task {type(task).__name__} has no 'openable_object' (direct or in subtasks); "
        "this helper needs an OpenDoorTask or CloseDoorTask somewhere in the task chain."
    )


def get_object_world_pos(env, object_name: str, env_id: int) -> torch.Tensor:
    """Return the named rigid-body scene object's position in **world** coordinates.

    Shape: ``(3,)``.
    """
    import warp as wp

    rigid_objects = env.unwrapped.scene.rigid_objects
    assert (
        object_name in rigid_objects
    ), f"Object '{object_name}' not found in scene.rigid_objects. Available: {list(rigid_objects.keys())}"
    obj = rigid_objects[object_name]
    return wp.to_torch(obj.data.root_pos_w)[env_id]


def get_articulation_world_pos(env, object_name: str, env_id: int) -> torch.Tensor:
    """Return the named articulation's root position in **world** coordinates.

    Shape: ``(3,)``.
    """
    import warp as wp

    articulations = env.unwrapped.scene.articulations
    assert (
        object_name in articulations
    ), f"Object '{object_name}' not found in scene.articulations. Available: {list(articulations.keys())}"
    art = articulations[object_name]
    return wp.to_torch(art.data.root_pos_w)[env_id, :3]


def get_scene_object_world_pos(env, object_name: str, env_id: int) -> torch.Tensor:
    """Return the world position of any scene object, trying rigid_objects then articulations.

    Shape: ``(3,)``.
    """
    try:
        return get_object_world_pos(env, object_name, env_id)
    except AssertionError:
        return get_articulation_world_pos(env, object_name, env_id)


def get_robot_world_pos(env, env_id: int) -> torch.Tensor:
    """Return the robot base's world position (``scene['robot'].root_pos_w``)."""
    import warp as wp

    robot = env.unwrapped.scene["robot"]
    return wp.to_torch(robot.data.root_pos_w)[env_id, :3]


def get_object_pos_in_robot_frame(env, object_name: str, env_id: int) -> torch.Tensor:
    """Return the object's position expressed in the robot's base frame.

    Accepts both rigid bodies and articulations (tries rigid_objects first,
    then articulations). cuRobo's IK solver interprets target poses in the
    robot-base frame, so this is what callers should feed into
    :func:`build_curobo_target_pose` and :func:`build_curobo_door_approach_pose`.

    Assumes the robot base has no yaw / roll / pitch rotation (true for
    the Franka-on-stand assets we've validated). Generalizing requires
    rotating ``(world_pos - robot_world_pos)`` by ``root_quat_w.inverse()``.
    """
    return get_scene_object_world_pos(env, object_name, env_id) - get_robot_world_pos(env, env_id)


# ---------------------------------------------------------------------------
# Grasp-pose construction
# ---------------------------------------------------------------------------


def top_down_quaternion_wxyz(axis: str) -> tuple[float, float, float, float]:
    """Return a ``(w, x, y, z)`` quaternion whose rotation points panda_hand Z downward."""
    if axis == "x":
        return (0.0, 1.0, 0.0, 0.0)  # 180 deg about world-X
    if axis == "y":
        return (0.0, 0.0, 1.0, 0.0)  # 180 deg about world-Y
    raise ValueError(f"Unsupported grasp axis: {axis!r}")


def build_curobo_target_pose(
    object_pos_local: torch.Tensor,
    top_down_offset: float,
    hand_to_tcp_z: float,
    grasp_axis: str,
    device: torch.device,
):
    """Construct the ``panda_hand`` target Pose (cuRobo) and its 4x4 equivalent.

    Returns a tuple ``(curobo_pose, hand_target_4x4)`` in the robot-local
    frame. ``curobo_pose.position`` and ``.quaternion`` are shape ``(1, 3)``
    and ``(1, 4)`` (wxyz) respectively.
    """
    import isaaclab.utils.math as PoseUtils
    from curobo.types.math import Pose

    tcp_target = object_pos_local.to(device=device, dtype=torch.float32).clone()
    tcp_target[2] = tcp_target[2] + top_down_offset

    # With panda_hand pointing straight down, the TCP sits hand_to_tcp_z below the hand
    # in world coordinates -> the hand origin is that far *above* the TCP.
    hand_target_pos = tcp_target.clone()
    hand_target_pos[2] = hand_target_pos[2] + hand_to_tcp_z

    qw, qx, qy, qz = top_down_quaternion_wxyz(grasp_axis)
    quat_wxyz = torch.tensor([qw, qx, qy, qz], device=device, dtype=torch.float32)

    curobo_pose = Pose(
        position=hand_target_pos.unsqueeze(0),
        quaternion=quat_wxyz.unsqueeze(0),
    )

    rot_matrix = PoseUtils.matrix_from_quat(quat_wxyz)
    hand_target_4x4 = PoseUtils.make_pose(hand_target_pos, rot_matrix)
    return curobo_pose, hand_target_4x4


def door_approach_quaternion_wxyz(door_facing_axis: str) -> tuple[float, float, float, float]:
    """Return a ``(w, x, y, z)`` quaternion whose panda_hand Z-axis points INTO the door.

    The hand approaches horizontally: panda_hand Z points opposite to the
    door-facing direction (i.e. toward the door, from the robot's side).

    ``door_facing_axis`` is the world-frame axis the door face points along
    (toward the robot):

    * ``"-x"`` — door faces world -X → hand Z points +X  (rotate +90° about Y)
    * ``"+x"`` — door faces world +X → hand Z points -X  (rotate -90° about Y)
    * ``"-y"`` — door faces world -Y → hand Z points +Y  (rotate -90° about X)
    * ``"+y"`` — door faces world +Y → hand Z points -Y  (rotate +90° about X)
    """
    _SQRT2_INV = 0.7071067811865476
    mapping = {
        "-x": (_SQRT2_INV, 0.0, _SQRT2_INV, 0.0),  # R_y(+90°): Z → +X
        "+x": (_SQRT2_INV, 0.0, -_SQRT2_INV, 0.0),  # R_y(-90°): Z → -X
        "-y": (_SQRT2_INV, -_SQRT2_INV, 0.0, 0.0),  # R_x(-90°): Z → +Y
        "+y": (_SQRT2_INV, _SQRT2_INV, 0.0, 0.0),  # R_x(+90°): Z → -Y
    }
    if door_facing_axis not in mapping:
        raise ValueError(f"Unsupported door_facing_axis: {door_facing_axis!r}")
    return mapping[door_facing_axis]


def _door_facing_unit_vec(door_facing_axis: str) -> list[float]:
    """Unit vector (x, y, z) for the door-facing axis string."""
    return {
        "-x": [-1.0, 0.0, 0.0],
        "+x": [1.0, 0.0, 0.0],
        "-y": [0.0, -1.0, 0.0],
        "+y": [0.0, 1.0, 0.0],
    }[door_facing_axis]


def build_curobo_door_approach_pose(
    object_pos_local: torch.Tensor,
    door_approach_offset: float,
    door_facing_axis: str,
    device: torch.device,
):
    """Construct a horizontal front-approach ``panda_hand`` target Pose (cuRobo).

    The hand is placed ``door_approach_offset`` meters in front of the
    openable object's center, along the door-facing direction (toward the
    robot). The orientation is a horizontal approach — panda_hand Z-axis
    points INTO the door (opposite to ``door_facing_axis``).

    Returns a tuple ``(curobo_pose, hand_target_4x4)`` in the robot-local
    frame. ``curobo_pose.position`` and ``.quaternion`` are shape ``(1, 3)``
    and ``(1, 4)`` (wxyz) respectively.
    """
    import isaaclab.utils.math as PoseUtils
    from curobo.types.math import Pose

    facing_vec = torch.tensor(_door_facing_unit_vec(door_facing_axis), device=device, dtype=torch.float32)
    pos = object_pos_local.to(device=device, dtype=torch.float32).clone()
    hand_target_pos = pos + door_approach_offset * facing_vec

    qw, qx, qy, qz = door_approach_quaternion_wxyz(door_facing_axis)
    quat_wxyz = torch.tensor([qw, qx, qy, qz], device=device, dtype=torch.float32)

    curobo_pose = Pose(
        position=hand_target_pos.unsqueeze(0),
        quaternion=quat_wxyz.unsqueeze(0),
    )

    rot_matrix = PoseUtils.matrix_from_quat(quat_wxyz)
    hand_target_4x4 = PoseUtils.make_pose(hand_target_pos, rot_matrix)
    return curobo_pose, hand_target_4x4


# ---------------------------------------------------------------------------
# IK feasibility
# ---------------------------------------------------------------------------


def check_ik_feasibility(
    planner,
    target_pose,
    seed_config: torch.Tensor | None = None,
) -> tuple[bool, float, float, torch.Tensor | None]:
    """Check whether ``target_pose`` is IK-feasible for the planner's robot.

    Args:
        planner: ``CuroboPlanner`` instance (provides the IK solver).
        target_pose: cuRobo ``Pose`` in the robot's base frame.
        seed_config: Optional joint config to seed the IK solver. Pass the
            previous call's ``joint_solution`` to evaluate sequential
            reachability (warm-start helps when successive targets are close).

    Returns:
        ``(feasible, pos_err, rot_err, joint_solution)`` where
        ``joint_solution`` is the best joint config (shape ``(dof,)``) when
        feasible, or ``None`` otherwise. ``pos_err`` / ``rot_err`` are the
        minimum errors across all IK seeds.
    """
    ik_solver = planner.motion_gen.ik_solver
    ik_seed = None
    if seed_config is not None:
        ik_seed = seed_config
        while ik_seed.dim() < 3:
            ik_seed = ik_seed.unsqueeze(0)
    result = ik_solver.solve_single(target_pose, seed_config=ik_seed)

    success_tensor = getattr(result, "success", None)
    if success_tensor is None:
        return False, float("nan"), float("nan"), None

    feasible = bool(success_tensor.any().item())

    def _safe_min(attr: str) -> float:
        tensor = getattr(result, attr, None)
        if tensor is None:
            return float("nan")
        try:
            return float(tensor.min().item())
        except (RuntimeError, ValueError):
            return float("nan")

    pos_err = _safe_min("position_error")
    rot_err = _safe_min("rotation_error")

    joint_solution = None
    if feasible and getattr(result, "solution", None) is not None:
        dof = result.solution.shape[-1]
        pos_err_flat = result.position_error.view(-1)
        best_idx = int(pos_err_flat.argmin().item())
        joint_solution = result.solution.view(-1, dof)[best_idx].detach().clone()

    return feasible, pos_err, rot_err, joint_solution


def classify_ik_status(
    feasible: bool,
    pos_err: float,
    rot_err: float,
    pos_threshold: float,
    rot_threshold: float,
) -> str:
    """Classify an IK result into a human-readable failure mode.

    * :data:`IK_STATUS_FEASIBLE` — cuRobo reported ``success.any() == True``.
    * :data:`IK_STATUS_IN_COLLISION` — pose errors are within cuRobo's
      convergence thresholds but ``success`` is still False. cuRobo found a
      joint config that matches the target pose but rejected it on
      collision or joint-limit grounds.
    * :data:`IK_STATUS_UNREACHABLE` — pose errors exceed threshold (or are
      NaN). No joint config can match the target pose regardless of
      collisions.

    The distinction is a best-effort heuristic; cuRobo folds
    pose / collision / limits into a single ``success`` flag and does not
    expose per-check reasons in the public ``IKResult``.
    """
    if feasible:
        return IK_STATUS_FEASIBLE
    if math.isnan(pos_err) or math.isnan(rot_err):
        return IK_STATUS_UNREACHABLE
    if pos_err < pos_threshold and rot_err < rot_threshold:
        return IK_STATUS_IN_COLLISION
    return IK_STATUS_UNREACHABLE


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def format_xyz(t: torch.Tensor) -> str:
    """Compact ``(x, y, z)`` pretty-printer used in log lines."""
    return f"({t[0].item():+.3f}, {t[1].item():+.3f}, {t[2].item():+.3f})"
