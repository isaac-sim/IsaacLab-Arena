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
from collections.abc import Callable

from isaaclab.envs.manager_based_env import ManagerBasedEnv


def step_zeros_and_call(
    env: ManagerBasedEnv, num_steps: int, function: Callable[[ManagerBasedEnv, torch.Tensor], None] | None = None
) -> None:
    """Step through the environment with zero actions for a specified number of steps."""
    for _ in tqdm.tqdm(range(num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.device)
            _, _, terminated, _, _ = env.step(actions)
            if function is not None:
                function(env, terminated)
