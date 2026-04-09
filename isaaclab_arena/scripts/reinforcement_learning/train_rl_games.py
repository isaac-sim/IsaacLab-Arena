# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Train an RL agent with RL-Games using the Arena environment builder.

Launch Isaac Sim Simulator first.
"""

import math
from pathlib import Path

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--agent_cfg_path",
    type=Path,
    required=True,
    help="Path to the RL-Games agent YAML configuration file.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume from.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max training epochs.")
parser.add_argument("--experiment_name", type=str, default=None, help="Override experiment name in config.")
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

if getattr(args_cli, "enable_pinocchio", False):
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import yaml
from datetime import datetime

import omni.log
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab_arena_environments.cli import get_arena_builder_from_cli


def main():
    """Train with RL-Games agent."""
    try:
        arena_builder = get_arena_builder_from_cli(args_cli)
        env_name, env_cfg = arena_builder.build_registered()
    except Exception as e:
        omni.log.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    with open(args_cli.agent_cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    if args_cli.distributed:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        agent_cfg["params"]["seed"] += int(os.getenv("RANK", "0"))
        agent_cfg["params"]["config"]["device"] = f"cuda:{local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        env_cfg.sim.device = f"cuda:{local_rank}"

    if args_cli.max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations

    if args_cli.experiment_name is not None:
        agent_cfg["params"]["config"]["name"] = args_cli.experiment_name

    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    env_cfg.seed = agent_cfg["params"]["seed"]

    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join("logs", "rl_games", config_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)

    env_cfg.log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(env_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
