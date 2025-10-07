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

import torch
import tqdm

from isaac_arena.examples.example_environments.cli import get_arena_builder_from_cli
from isaac_arena.examples.policy_runner_cli import create_policy, setup_policy_argument_parser
from isaac_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def main():
    """Script to run an Isaac Arena environment with a zero-action agent."""
    # Set up argument parser
    args_parser = setup_policy_argument_parser()
    args_cli = args_parser.parse_args()

    # Start the simulation app
    with SimulationAppContext(args_cli):
        # Build scene
        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()
        obs, _ = env.reset()

        # NOTE(xinjieyao, 2025-09-29): General rule of thumb is to have as many non-standard python
        # library imports after app launcher as possible, otherwise they will likely stall the sim
        # app. Given current SimulationAppContext setup, use lazy import to handle policy-related
        # deps inside create_policy() function to bringup sim app.
        policy, num_steps = create_policy(args_cli)

        for _ in tqdm.tqdm(range(num_steps)):
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)
                if terminated.any() or truncated.any():
                    obs, _ = env.reset()
        # Close the environment.
        env.close()


if __name__ == "__main__":
    main()
