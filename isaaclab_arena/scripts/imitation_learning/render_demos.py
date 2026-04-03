# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Script to replay demonstrations and record camera (and state/actions) to a new HDF5.

Replays from an input dataset (same as replay_demos) but with cameras enabled and
recorders writing to --output_dataset. Output HDF5 contains initial_state, states,
actions, and external_camera_obs (external_camera_rgb) per step.

Example:
  python isaaclab_arena/scripts/imitation_learning/render_demos.py \
    --dataset_file /datasets/curobo_v3_statesonly.hdf5 --output_dataset /datasets/curobo_v3_with_camera.hdf5 \
    --use_joint_targets --enable_cameras droid_v3_tabletop_pick_and_place --embodiment droid_abs_joint_pos
"""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--dataset_file", type=str, required=True, help="Input dataset to replay.")
parser.add_argument(
    "--output_dataset",
    type=str,
    required=True,
    help="Output HDF5 path for replayed episodes with camera (and state/actions) recorded.",
)
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="Episode indices to replay; empty = all.",
)
parser.add_argument(
    "--use_joint_targets",
    action="store_true",
    default=False,
    help="Replay from joint_targets in the input HDF5 (e.g. from record_curobo_demos).",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
add_example_environments_cli_args(parser)

args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def _create_pre_step_external_camera_observations_recorder():
    """RecorderTerm that records external_camera_rgb each step (same as record_curobo_demos)."""
    from isaaclab.managers.recorder_manager import RecorderTerm

    class PreStepExternalCameraRecorder(RecorderTerm):
        def record_pre_step(self):
            cam_obs = self._env.obs_buf["camera_obs"]
            return "external_camera_obs", cam_obs["external_camera_rgb"].clone()

    return PreStepExternalCameraRecorder


def _create_post_step_joint_targets_recorder():
    """RecorderTerm that records 7 arm + 1 gripper joint targets for replay (--use_joint_targets)."""
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
    """RecorderTerm that records the EEF pose target for replay."""
    from isaaclab.managers.recorder_manager import RecorderTerm

    class PostStepEefPoseTargetRecorder(RecorderTerm):
        def record_post_step(self):
            robot = self._env.scene["robot"]
            eef_idx = robot.data.body_names.index("base_link")
            eef_pos = robot.data.body_pos_w[:, eef_idx, :].clone()
            eef_quat = robot.data.body_quat_w[:, eef_idx, :].clone()
            return "eef_pose_target", torch.cat((eef_pos, eef_quat), dim=1)

    return PostStepEefPoseTargetRecorder


def _create_render_env_config(args_cli, output_dir: str, output_file_name: str):
    """Build env config with recorders for output_dataset (state, actions, external_camera_obs)."""
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode, RecorderTermCfg
    from isaaclab.utils import configclass

    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    env_cfg.terminations = {}
    env_cfg.observations.policy.concatenate_terms = False

    @configclass
    class RenderRecorderManagerCfg(ActionStateRecorderManagerCfg):
        record_pre_step_external_camera_obs = RecorderTermCfg(
            class_type=_create_pre_step_external_camera_observations_recorder()
        )
        record_post_step_joint_targets = RecorderTermCfg(
            class_type=_create_post_step_joint_targets_recorder()
        )
        record_post_step_eef_pose_target = RecorderTermCfg(
            class_type=_create_post_step_eef_pose_target_recorder()
        )

    env_cfg.recorders = RenderRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    return env_cfg, env_name


def main():
    global is_paused

    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"Input dataset not found: {args_cli.dataset_file}")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name_from_file = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()
    if episode_count == 0:
        print("No episodes in the dataset.")
        return

    episode_indices = args_cli.select_episodes if args_cli.select_episodes else list(range(episode_count))
    num_envs = args_cli.num_envs

    # Output path for recorded HDF5
    output_dir = os.path.dirname(args_cli.output_dataset)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_dataset))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg, env_name = _create_render_env_config(args_cli, output_dir, output_file_name)
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume. Replaying and recording camera to', args_cli.output_dataset)

    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)

    env.reset()
    teleop_interface.reset()

    episode_names = list(dataset_file_handler.get_episode_names())
    use_joint_targets = args_cli.use_joint_targets

    def get_next_command(episode_data):
        return episode_data.get_next_joint_target() if use_joint_targets else episode_data.get_next_action()

    replayed_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {i: EpisodeData() for i in range(num_envs)}
            first_loop = True
            has_next_action = True
            while has_next_action:
                actions = idle_action.clone()
                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = get_next_command(env_episode_data_map[env_id])
                    if env_next_action is None:
                        # Export current episode for this env if we had one
                        if env_episode_data_map[env_id].get_initial_state() is not None:
                            env.recorder_manager.record_pre_reset([env_id], force_export_or_skip=False)
                            env.recorder_manager.set_success_to_episodes(
                                [env_id], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                            )
                            env.recorder_manager.export_episodes([env_id])
                            replayed_count += 1
                            print(f"  Exported episode {replayed_count} to {args_cli.output_dataset}")

                        next_idx = None
                        while episode_indices:
                            next_idx = episode_indices.pop(0)
                            if next_idx < episode_count:
                                break
                            next_idx = None
                        if next_idx is None:
                            continue

                        print(f"  Loading episode #{next_idx} into env_{env_id}")
                        episode_data = dataset_file_handler.load_episode(episode_names[next_idx], env.device)
                        env_episode_data_map[env_id] = episode_data
                        env.recorder_manager.reset([env_id])
                        initial_state = episode_data.get_initial_state()
                        env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                        env_next_action = get_next_command(env_episode_data_map[env_id])
                        has_next_action = True
                    else:
                        has_next_action = True
                    actions[env_id] = env_next_action

                if first_loop:
                    first_loop = False
                else:
                    if not has_next_action:
                        # No more episodes/actions; exit inner loop instead of entering pause loop
                        break
                    while is_paused:
                        env.sim.render()
                        continue
                env.step(actions)
            print("[render_demos] Exiting replay loop.", flush=True)
            break

    print(f"Finished. Replayed and recorded {replayed_count} episode(s) to {args_cli.output_dataset}", flush=True)
    dataset_file_handler.close()
    print("[render_demos] Input dataset closed.", flush=True)
    print("[render_demos] Closing env...", flush=True)
    env.close()
    print("[render_demos] Env closed.", flush=True)


if __name__ == "__main__":
    main()
    print("[render_demos] Main returned. Closing simulation app...", flush=True)
    simulation_app.close()
    print("[render_demos] Done.", flush=True)
