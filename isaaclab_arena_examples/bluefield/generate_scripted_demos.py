# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record pick-and-place demonstrations from a scripted Cartesian waypoint policy, without teleop.

The G1 Pink-IK embodiments take end-effector poses rather than joint targets, so a demonstration is
fully determined by a sequence of wrist poses plus a binary hand state -- no motion planner and no
human operator. This script samples a grasp per episode, builds a smooth minimum-jerk trajectory
through it, replays it, and exports the episodes the task's own success term accepts.

Every episode's sampled parameters are logged to a JSON sidecar next to the dataset (successes *and*
failures), so a later pass can mine which grasps are reliable and narrow the sampling ranges.

Example::

    python isaaclab_arena_examples/bluefield/generate_scripted_demos.py --viz kit --enable_cameras \\
        --dataset_file ./datasets/Bluefield_recon/bluefield_g1_scripted.hdf5 --num_demos 50 \\
        bluefield_g1_fixed_base_pick_and_place
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--dataset_file", type=str, required=True, help="File path to export recorded demos.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of *successful* demos to record.")
parser.add_argument(
    "--max_attempts",
    type=int,
    default=0,
    help="Cap on total episodes attempted. 0 derives a cap of 4x --num_demos.",
)
parser.add_argument("--seed_offset", type=int, default=0, help="Added to the episode index to seed its sampling.")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Consecutive steps the success term must hold before an episode is exported.",
)
parser.add_argument(
    "--max_apple_push_m",
    type=float,
    default=0.024,
    help=(
        "Reject an episode whose object is shoved further than this across the table before being"
        " lifted. The task's success term ignores such episodes; a real hand would not recover from"
        " them. Set to 0 to disable."
    ),
)

# NOTE: registered last, because the example-environment subparser must parse after the global flags.
add_example_environments_cli_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation, Slerp
from typing import Any

import isaaclab.utils.math as PoseUtils
import omni.log
import warp as wp
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

from isaaclab_arena.terms.recorders import ArenaEnvRecorderManagerCfg

# --- Action layout of the G1 Pink-IK embodiments (see G1DecoupledWBCPinkAction.process_actions) ---
LEFT_HAND_IDX = 0
RIGHT_HAND_IDX = 1
LEFT_WRIST_POS_SLICE = slice(2, 5)
LEFT_WRIST_QUAT_SLICE = slice(5, 9)
RIGHT_WRIST_POS_SLICE = slice(9, 12)
RIGHT_WRIST_QUAT_SLICE = slice(12, 16)
NAVIGATE_CMD_SLICE = slice(16, 19)
BASE_HEIGHT_IDX = 19
TORSO_RPY_SLICE = slice(20, 23)
ACTION_DIM = 23

HAND_OPEN = 0.0
HAND_CLOSED = 1.0

# Wrist poses are expressed in the pelvis frame, quaternions as (x, y, z, w).
PELVIS_LINK_NAME = "pelvis"
LEFT_WRIST_LINK_NAME = "left_wrist_yaw_link"

# --- Grasp geometry, measured from datasets/Bluefield_recon/bluefield_g1_spacemouse_3demos.hdf5 ---
# Those three teleop episodes ran on the current table and all succeeded. Everything here is either
# object-relative or a fixed rest pose, so moving the table does not invalidate them; re-derive them
# if the hand or the wrist link changes.

# Apple centre in the wrist's local frame when the hand closes (per-axis spread under 1.5 cm). This
# is what makes the grasp model work: for any wrist orientation, put the wrist at
# ``object - R * PALM_OFFSET`` and the palm lands on the object.
PALM_OFFSET_LOCAL_M = np.array([0.139, -0.038, -0.015])

# Rest poses, identical at t=0 across all three demos. The two arms are exact mirrors.
LEFT_HOME_POS_M = np.array([0.1744, 0.2889, 0.1269])
LEFT_HOME_QUAT_XYZW = np.array([0.1208, -0.0308, 0.2454, 0.9614])
RIGHT_REST_POS_M = np.array([0.1744, -0.2889, 0.1269])
RIGHT_REST_QUAT_XYZW = np.array([-0.1208, -0.0308, -0.2454, 0.9614])

# Lower-body channels are constant for the welded-pelvis embodiment: no navigation, no torso lean.
BASE_HEIGHT_CMD_M = 0.71

# Grasp orientation (roll, pitch, yaw in degrees, extrinsic xyz, pelvis frame). Centred on the teleop
# demos but widened to roughly the spread of the 208 galileo ones, so the export carries a comparable
# range of grasps. Only galileo's spread is transplantable, not its absolute poses: its scene state
# and actions sit in frames that differ by the 8 cm the replay's --robot_base_offset compensates.
GRASP_ROLL_RANGE_DEG = (5.0, 45.0)
GRASP_PITCH_RANGE_DEG = (-12.0, 20.0)
GRASP_YAW_RANGE_DEG = (5.0, 35.0)

# Yaw folded into the wrist between the lift and the release, and the floor it has to keep the wrist
# above. Both are what stop the arm folding over the torso: the palm sits ~0.14 m along the wrist's
# local +x, so ``wrist_y ~= palm_y - 0.14*sin(yaw)``, and holding the grasp yaw all the way to a
# plate on the midline drags the wrist across it. Both human datasets yaw over and bottom out at
# wrist_y +0.027.
RELEASE_YAW_DELTA_RANGE_DEG = (-35.0, -15.0)
MIN_WRIST_Y_M = 0.03

# Per-axis jitter added to the palm target, on top of the orientation spread.
PALM_JITTER_M = 0.01
# Straight-line back-off from the grasp for the pre-grasp pose, and how steeply the insertion then
# descends. The pitch spans a level approach that closes around the apple from the side through a
# steep press from above; the human datasets sit at -31 deg (teleop) and -55 deg (galileo).
APPROACH_DISTANCE_RANGE_M = (0.04, 0.07)
APPROACH_PITCH_RANGE_DEG = (-70.0, 0.0)
# Vertical lift after the hand closes, and palm height above the dish centre at release. Both floors
# buy margin: failures cluster at the bottom of whatever lift range is in force, and at the top of
# the release range, where the apple bounces off the dish and rolls clear.
LIFT_HEIGHT_RANGE_M = (0.06, 0.09)
RELEASE_HEIGHT_RANGE_M = (0.09, 0.11)
# Per-episode pace multiplier on every interpolated segment. Bounded below at 1.0 because the segment
# length it scales already honours MAX_STEP_DISPLACEMENT_M.
SEGMENT_DURATION_SCALE_RANGE = (1.0, 1.25)
# Steps held still while the fingers close on / release the apple. The close range brackets the ~20
# steps the teleop operator waited before lifting; the open range outlasts the apple's ~8-step fall.
CLOSE_SETTLE_STEPS_RANGE = (12, 20)
OPEN_SETTLE_STEPS_RANGE = (8, 14)

# Peak per-step wrist displacement, which sets the trajectory's pace and length. The teleop demos ran
# at a p95 of 7-9 mm/step, so this stays inside the regime Pink IK was shown to track. Grasp accuracy
# does not suffer: minimum-jerk decelerates to zero at every waypoint, so the peak is mid-segment and
# the wrist has settled by the time the hand closes.
MAX_STEP_DISPLACEMENT_M = 0.009
# Sanity floor on any interpolated segment; minimum-jerk stays smooth well below it.
MIN_SEGMENT_STEPS = 8
# Steps held at home after the trajectory ends, so the apple can settle for the success term's
# contact-force and velocity predicates.
POST_TRAJECTORY_SETTLE_STEPS = 60
# Rise above its spawn height that counts the apple as lifted, for the ``--max_apple_push_m`` check.
LIFT_DETECT_HEIGHT_M = 0.01

# Pink IK's steady-state tracking error was 3-5 mm in the demos; a frame mismatch would be far larger.
FRAME_CHECK_TOLERANCE_M = 0.02


def _min_jerk(u: np.ndarray) -> np.ndarray:
    """Minimum-jerk time scaling on ``u`` in [0, 1]: zero velocity and acceleration at both ends."""
    return 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3


def _segment_steps(distance_m: float, duration_scale: float) -> int:
    """Steps for a segment of ``distance_m``, sized to keep the peak displacement under the cap.

    A minimum-jerk profile peaks at 1.875x its mean speed, which is what sets the step count;
    ``duration_scale`` then only ever lengthens it.
    """
    capped_steps = int(np.ceil(1.875 * distance_m / MAX_STEP_DISPLACEMENT_M))
    return max(MIN_SEGMENT_STEPS, capped_steps, int(round(capped_steps * duration_scale)))


def _interpolate(
    start_pos: np.ndarray,
    start_quat_xyzw: np.ndarray,
    end_pos: np.ndarray,
    end_quat_xyzw: np.ndarray,
    num_steps: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Minimum-jerk position blend and slerp orientation blend, excluding the start pose.

    Both channels share the same time scaling, so position and orientation stay synchronised and
    consecutive segments join with zero velocity -- no discontinuity for Pink IK to chase.
    """
    slerp = Slerp([0.0, 1.0], Rotation.from_quat(np.stack([start_quat_xyzw, end_quat_xyzw])))
    scaled = _min_jerk(np.linspace(0.0, 1.0, num_steps + 1)[1:])
    return [(start_pos + (end_pos - start_pos) * s, slerp(s).as_quat()) for s in scaled]


def _wrist_pose_for_palm(palm_target_p: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Wrist position (pelvis frame) that puts the palm on ``palm_target_p`` at orientation ``quat_xyzw``."""
    return palm_target_p - Rotation.from_quat(quat_xyzw).apply(PALM_OFFSET_LOCAL_M)


def _release_rpy_clear_of_midline(release_rpy_deg: np.ndarray, palm_target_p: np.ndarray) -> np.ndarray:
    """Yaw the release orientation further negative until the wrist clears ``MIN_WRIST_Y_M``.

    ``wrist_y`` falls monotonically as yaw rises over the range in play, so walking yaw down finds
    the first orientation that keeps the wrist left of the midline while the palm stays on target.
    """
    rpy = release_rpy_deg.copy()
    for yaw in np.arange(release_rpy_deg[2], release_rpy_deg[2] - 60.0, -2.0):
        rpy[2] = yaw
        wrist = _wrist_pose_for_palm(palm_target_p, Rotation.from_euler("xyz", rpy, degrees=True).as_quat())
        if wrist[1] >= MIN_WRIST_Y_M:
            break
    return rpy


def sample_episode_params(rng: np.random.Generator) -> dict[str, Any]:
    """Draw one episode's grasp and trajectory-shape parameters."""
    return {
        "grasp_rpy_deg": [
            float(rng.uniform(*GRASP_ROLL_RANGE_DEG)),
            float(rng.uniform(*GRASP_PITCH_RANGE_DEG)),
            float(rng.uniform(*GRASP_YAW_RANGE_DEG)),
        ],
        "palm_jitter_m": rng.uniform(-PALM_JITTER_M, PALM_JITTER_M, size=3).tolist(),
        "release_yaw_delta_deg": float(rng.uniform(*RELEASE_YAW_DELTA_RANGE_DEG)),
        "approach_distance_m": float(rng.uniform(*APPROACH_DISTANCE_RANGE_M)),
        "approach_pitch_deg": float(rng.uniform(*APPROACH_PITCH_RANGE_DEG)),
        "lift_height_m": float(rng.uniform(*LIFT_HEIGHT_RANGE_M)),
        "release_height_m": float(rng.uniform(*RELEASE_HEIGHT_RANGE_M)),
        "duration_scale": float(rng.uniform(*SEGMENT_DURATION_SCALE_RANGE)),
        "close_settle_steps": int(rng.integers(*CLOSE_SETTLE_STEPS_RANGE)),
        "open_settle_steps": int(rng.integers(*OPEN_SETTLE_STEPS_RANGE)),
    }


def build_trajectory(
    apple_p: np.ndarray,
    plate_center_p: np.ndarray,
    params: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Build the full wrist trajectory as ``(position, quaternion_xyzw, hand_state)`` per step.

    Args:
        apple_p: Apple centre in the pelvis frame.
        plate_center_p: Plate dish centre in the pelvis frame.
        params: One draw from ``sample_episode_params``.
    """
    grasp_rpy = np.asarray(params["grasp_rpy_deg"], dtype=np.float64)
    grasp_quat = Rotation.from_euler("xyz", grasp_rpy, degrees=True).as_quat()
    duration_scale = params["duration_scale"]

    palm_target = apple_p + np.asarray(params["palm_jitter_m"])
    grasp_wrist = _wrist_pose_for_palm(palm_target, grasp_quat)
    # Insert along the hand's own approach axis (local +x), tilted down by the sampled pitch so the
    # hand descends onto the apple instead of sliding into it. The axis is flattened first, so the
    # pitch is measured against the table rather than compounding with the grasp orientation.
    approach_axis = Rotation.from_quat(grasp_quat).apply(np.array([1.0, 0.0, 0.0]))
    approach_axis[2] = 0.0
    approach_axis /= np.linalg.norm(approach_axis)
    pitch = np.radians(params["approach_pitch_deg"])
    insertion_axis = approach_axis * np.cos(pitch) + np.array([0.0, 0.0, np.sin(pitch)])
    pregrasp_wrist = grasp_wrist - insertion_axis * params["approach_distance_m"]
    lift_wrist = grasp_wrist + np.array([0.0, 0.0, params["lift_height_m"]])
    release_palm = plate_center_p + np.array([0.0, 0.0, params["release_height_m"]])
    release_rpy = _release_rpy_clear_of_midline(
        grasp_rpy + np.array([0.0, 0.0, params["release_yaw_delta_deg"]]), release_palm
    )
    release_quat = Rotation.from_euler("xyz", release_rpy, degrees=True).as_quat()
    # Solved against the yawed-over orientation, so the palm still ends up centred on the dish.
    release_wrist = _wrist_pose_for_palm(release_palm, release_quat)
    # Approach the release from above, so the apple crosses the rim rather than swinging into it.
    pre_release_wrist = release_wrist + np.array([0.0, 0.0, params["lift_height_m"]])

    trajectory: list[tuple[np.ndarray, np.ndarray, float]] = []

    def move_to(from_pos, from_quat, to_pos, to_quat, hand):
        steps = _segment_steps(float(np.linalg.norm(to_pos - from_pos)), duration_scale)
        trajectory.extend((p, q, hand) for p, q in _interpolate(from_pos, from_quat, to_pos, to_quat, steps))

    def hold(pos, quat, hand, steps):
        trajectory.extend((pos.copy(), quat.copy(), hand) for _ in range(steps))

    move_to(LEFT_HOME_POS_M, LEFT_HOME_QUAT_XYZW, pregrasp_wrist, grasp_quat, HAND_OPEN)
    move_to(pregrasp_wrist, grasp_quat, grasp_wrist, grasp_quat, HAND_OPEN)
    hold(grasp_wrist, grasp_quat, HAND_CLOSED, params["close_settle_steps"])
    move_to(grasp_wrist, grasp_quat, lift_wrist, grasp_quat, HAND_CLOSED)
    # Yaw over across the transfer rather than at the plate, matching the human demos and keeping the
    # rotation slow enough not to shake the apple loose.
    move_to(lift_wrist, grasp_quat, pre_release_wrist, release_quat, HAND_CLOSED)
    move_to(pre_release_wrist, release_quat, release_wrist, release_quat, HAND_CLOSED)
    hold(release_wrist, release_quat, HAND_OPEN, params["open_settle_steps"])
    move_to(release_wrist, release_quat, LEFT_HOME_POS_M, LEFT_HOME_QUAT_XYZW, HAND_OPEN)
    hold(LEFT_HOME_POS_M, LEFT_HOME_QUAT_XYZW, HAND_OPEN, POST_TRAJECTORY_SETTLE_STEPS)

    return trajectory


def assemble_action(wrist_pos: np.ndarray, wrist_quat_xyzw: np.ndarray, hand_state: float, device: str) -> torch.Tensor:
    """Pack one scripted waypoint into the embodiment's 23-D action tensor."""
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[LEFT_HAND_IDX] = hand_state
    action[RIGHT_HAND_IDX] = HAND_OPEN
    action[LEFT_WRIST_POS_SLICE] = wrist_pos
    action[LEFT_WRIST_QUAT_SLICE] = wrist_quat_xyzw
    action[RIGHT_WRIST_POS_SLICE] = RIGHT_REST_POS_M
    action[RIGHT_WRIST_QUAT_SLICE] = RIGHT_REST_QUAT_XYZW
    action[NAVIGATE_CMD_SLICE] = 0.0
    action[BASE_HEIGHT_IDX] = BASE_HEIGHT_CMD_M
    action[TORSO_RPY_SLICE] = 0.0
    return torch.from_numpy(action).to(device).unsqueeze(0)


def _np(tensor) -> np.ndarray:
    """Torch tensor to a float64 numpy array."""
    return tensor.cpu().numpy().astype(np.float64)


def _link_state(robot, link_name: str) -> torch.Tensor:
    """Position + quaternion of a named body link, for env 0.

    Read through ``body_link_state_w`` so the frame matches the one the ``*_eef_pos`` observations
    are built in -- which is the frame the Pink-IK action targets are interpreted in.
    """
    return wp.to_torch(robot.data.body_link_state_w)[0, robot.data.body_names.index(link_name), :7]


def pelvis_frame(env) -> tuple[np.ndarray, np.ndarray]:
    """Return the pelvis link's world position and rotation matrix."""
    state = _link_state(env.scene["robot"], PELVIS_LINK_NAME)
    return _np(state[:3]), _np(PoseUtils.matrix_from_quat(state[3:7].unsqueeze(0))[0])


def to_pelvis_frame(position_w: np.ndarray, pelvis_pos: np.ndarray, pelvis_rot: np.ndarray) -> np.ndarray:
    """Express a world-frame position in the pelvis frame."""
    return pelvis_rot.T @ (position_w - pelvis_pos)


def object_position_w(env, scene_name: str) -> np.ndarray:
    """World-frame root position of a scene rigid object."""
    return _np(wp.to_torch(env.scene[scene_name].data.root_pos_w)[0, :3])


def verify_frame_convention(env, obs: dict, pelvis_pos: np.ndarray, pelvis_rot: np.ndarray) -> None:
    """Fail loudly if our pelvis-frame maths disagrees with the env's own eef observation.

    Every waypoint is expressed in the pelvis frame, so a quaternion-convention or link-name mistake
    would silently produce trajectories that miss the apple rather than erroring.
    """
    ours = to_pelvis_frame(_np(_link_state(env.scene["robot"], LEFT_WRIST_LINK_NAME)[:3]), pelvis_pos, pelvis_rot)
    theirs = _np(obs["policy"]["left_eef_pos"][0])
    error = float(np.linalg.norm(ours - theirs))
    assert error < FRAME_CHECK_TOLERANCE_M, (
        f"pelvis-frame mismatch: left wrist at {ours} vs {theirs} from left_eef_pos"
        f" ({error * 1000:.1f} mm apart); the scripted waypoints are not in the frame actions use."
    )


def plate_center_offset_xy() -> np.ndarray:
    """Env-local XY offset from the destination's USD origin to its dish centre.

    Only the bluefield task's clay plate has a measured offset; anything else is treated as centred
    on its own origin.
    """
    from isaaclab_arena_environments.bluefield_g1_fixed_base_pick_and_place_environment import (
        PLATE_ORIGIN_TO_CENTER_XY,
        TUNED_DESTINATION_NAME,
    )

    if args_cli.destination != TUNED_DESTINATION_NAME:
        return np.zeros(2)
    return np.asarray(PLATE_ORIGIN_TO_CENTER_XY, dtype=np.float64)


def setup_output_paths() -> tuple[str, str, str]:
    """Create the dataset directory and return ``(directory, dataset stem, sidecar path)``."""
    output_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file))
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, output_file_name, os.path.join(output_dir, f"{output_file_name}_scripted_params.json")


def create_environment_config(output_dir: str, output_file_name: str):
    """Build the env config for scripted recording, with its name, kwargs and success term."""
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg, env_kwargs = arena_builder.build_registered()

    assert hasattr(env_cfg.terminations, "success"), (
        "The environment has no 'success' termination term, so scripted episodes could never be "
        "validated. Use a task that defines one."
    )
    # Pull success out of the termination manager and poll it by hand: an episode has to keep running
    # past the first success so the recorder can close it out deliberately.
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    if args_cli.enable_cameras:
        env_cfg.recorders = ArenaEnvRecorderManagerCfg()
    else:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, env_name, env_kwargs, success_term


def reset_environment(env) -> dict:
    """Reset sim, recorder and env for a fresh episode, returning the first observation."""
    env.sim.reset()
    env.recorder_manager.reset()
    obs, _ = env.reset()
    return obs


def run_episode(env, success_term, params: dict[str, Any]) -> dict[str, Any]:
    """Play one scripted episode, exporting it if the task succeeds without shoving the object.

    Returns the sidecar record for the attempt: the sampled parameters, the object layout it was
    built for, and the outcome.
    """
    obs = reset_environment(env)
    pelvis_pos, pelvis_rot = pelvis_frame(env)
    verify_frame_convention(env, obs, pelvis_pos, pelvis_rot)

    apple_p = to_pelvis_frame(object_position_w(env, args_cli.object), pelvis_pos, pelvis_rot)
    plate_pos_w = object_position_w(env, args_cli.destination)
    plate_pos_w[:2] += plate_center_offset_xy()
    plate_center_p = to_pelvis_frame(plate_pos_w, pelvis_pos, pelvis_rot)

    trajectory = build_trajectory(apple_p, plate_center_p, params)
    record = {
        **params,
        "apple_pelvis_frame_m": apple_p.tolist(),
        "plate_center_pelvis_frame_m": plate_center_p.tolist(),
        "num_steps": len(trajectory),
    }

    apple_start_w = object_position_w(env, args_cli.object)
    max_push_m = 0.0
    lifted = False

    success_step_count = 0
    for step_index, (wrist_pos, wrist_quat, hand_state) in enumerate(trajectory):
        _, _, terminated, truncated, _ = env.step(assemble_action(wrist_pos, wrist_quat, hand_state, env.device))

        # Shove only counts until the first lift. The latch matters: an apple sitting in the dish is
        # barely higher than one on the table, so a height test alone would resume counting there and
        # charge the whole carry as shove.
        apple_now_w = object_position_w(env, args_cli.object)
        if not lifted:
            lifted = apple_now_w[2] > apple_start_w[2] + LIFT_DETECT_HEIGHT_M
            max_push_m = max(max_push_m, float(np.linalg.norm(apple_now_w[:2] - apple_start_w[:2])))
            # Shove is final once the apple is off the table, so a doomed episode can stop here
            # rather than play out the carry it is going to be rejected for anyway.
            if lifted and args_cli.max_apple_push_m and max_push_m > args_cli.max_apple_push_m:
                return {
                    **record,
                    "outcome": "apple_pushed",
                    "steps_taken": step_index + 1,
                    "max_apple_push_m": max_push_m,
                }

        if bool(success_term.func(env, **success_term.params)[0]):
            success_step_count += 1
        else:
            success_step_count = 0

        if success_step_count >= args_cli.num_success_steps:
            record = {**record, "max_apple_push_m": max_push_m, "steps_taken": step_index + 1}
            if args_cli.max_apple_push_m and max_push_m > args_cli.max_apple_push_m:
                return {**record, "outcome": "apple_pushed"}
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            return {**record, "outcome": "success"}

        if bool(terminated) or bool(truncated):
            # The only termination left enabled is the object-dropped guard, so this is a failed grasp.
            return {**record, "outcome": "terminated", "steps_taken": step_index + 1, "max_apple_push_m": max_push_m}

    return {**record, "outcome": "no_success", "steps_taken": len(trajectory), "max_apple_push_m": max_push_m}


def main() -> None:
    """Sample, replay and export scripted demonstrations until the requested count succeeds."""
    output_dir, output_file_name, sidecar_path = setup_output_paths()
    env_cfg, env_name, env_kwargs, success_term = create_environment_config(output_dir, output_file_name)

    env = gym.make(env_name, cfg=env_cfg, **env_kwargs).unwrapped
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    reapply_viewer_cfg(env)

    max_attempts = args_cli.max_attempts if args_cli.max_attempts > 0 else 4 * args_cli.num_demos
    records: list[dict[str, Any]] = []
    num_success = 0

    for attempt in range(max_attempts):
        if num_success >= args_cli.num_demos:
            break
        # Seed per attempt so a run is reproducible and a single episode can be replayed in isolation.
        params = sample_episode_params(np.random.default_rng(args_cli.seed_offset + attempt))
        record = run_episode(env, success_term, params)
        record["attempt"] = attempt
        records.append(record)
        num_success += record["outcome"] == "success"

        print(
            f"[{attempt + 1}/{max_attempts}] {record['outcome']:13s}"
            f" grasp_rpy={np.round(params['grasp_rpy_deg'], 1).tolist()}"
            f" approach={params['approach_pitch_deg']:+5.1f}deg"
            f" push={1000 * record.get('max_apple_push_m', 0.0):5.1f}mm"
            f" -- {num_success}/{args_cli.num_demos} recorded"
        )
        # Rewritten every attempt so an interrupted run still leaves a usable log.
        with open(sidecar_path, "w") as sidecar_file:
            json.dump(records, sidecar_file, indent=2)

    if num_success < args_cli.num_demos:
        omni.log.warn(
            f"Exhausted {max_attempts} attempts with only {num_success}/{args_cli.num_demos} successes. "
            "Raise --max_attempts, relax --max_apple_push_m, or narrow the sampling ranges using the sidecar."
        )
    print(
        f"\nRecorded {num_success} demos in {len(records)} attempts "
        f"({num_success / max(1, len(records)):.0%} success rate)\n"
        f"  dataset: {args_cli.dataset_file}\n"
        f"  sidecar: {sidecar_path}"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
