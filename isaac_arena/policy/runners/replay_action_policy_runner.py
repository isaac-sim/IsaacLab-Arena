# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gymnasium as gym
import torch
import tqdm

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.isaaclab_utils.simulation_app import SimulationAppContext
from isaac_arena.policy.replay_action_policy import ReplayActionPolicy


def main():
    args_parser = get_isaac_arena_cli_parser()
    args_parser.add_argument_group("Replay Runner", "Arguments for the zero action runner")
    args_parser.add_argument(
        "--episode_name",
        type=str,
        default=None,
        help="Name of the episode to replay. If not provided, the first episode will be replayed.",
    )
    args_parser.add_argument(
        "--replay_file_path", type=str, required=True, help="Path to the HDF5 file containing the episode."
    )

    args_cli = args_parser.parse_args()

    # Start the simulation app
    with SimulationAppContext(args_cli):

        # Imports have to follow simulation startup.
        from isaac_arena.environments.compile_env import get_arena_env_cfg

        # Scene variation
        env_cfg, env_name = get_arena_env_cfg(args_cli)
        env = gym.make(env_name, cfg=env_cfg)
        env.reset()

        # Create a replay action episode policy.
        policy = ReplayActionPolicy(args_cli.replay_file_path, args_cli.episode_name)

        # Run some zero actions.
        for action_index in tqdm.tqdm(range(len(policy))):
            with torch.inference_mode():
                actions = policy.get_action_from_index(action_index)
                env.step(actions)

        # Close the environment.
        env.close()


if __name__ == "__main__":
    main()
