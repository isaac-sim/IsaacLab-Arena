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

# %%

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

# %%


def reload_arena_modules():
    import importlib
    import os
    import sys

    # Clear Python's bytecode cache
    if hasattr(importlib, "invalidate_caches"):
        importlib.invalidate_caches()

    # Get all isaac_arena modules currently loaded
    isaac_arena_modules = [
        (name, module) for name, module in sys.modules.items() if name.startswith("isaac_arena") and module is not None
    ]

    if len(isaac_arena_modules) == 0:
        print("No isaac_arena modules found")
        return

    # We skip this step due to import issues in our asset registry
    # # Reload modules from bottom to top (deepest/leaf modules first)
    # isaac_arena_modules.sort(key=lambda x: x[0].count('.'), reverse=True)

    for module_name, module in isaac_arena_modules:
        try:
            # Delete the .pyc cache file to force recompilation from source
            module_cached = getattr(module, "__cached__", None)
            if module_cached and os.path.exists(module_cached):
                os.remove(module_cached)

            print(f"Reloading {module_name}")
            importlib.reload(module)

        except Exception as e:
            print(f"[WARNING] Failed to reload {module_name}: {e}")


# %%

reload_arena_modules()
from isaac_arena.examples.example_environments.cli import (
    get_arena_builder_from_cli,
    get_isaac_arena_example_environment_cli_parser,
)

args_parser = get_isaac_arena_example_environment_cli_parser()

# GR1 Open Microwave
args_cli = args_parser.parse_args([
    "gr1_open_microwave",
    "--object",
    "cracker_box",
])

# Pick and Place
# args_cli = args_parser.parse_args([
#     "kitchen_pick_and_place",
#     "--object",
#     "cracker_box",
#     "--background",
#     "kitchen",
#     "--embodiment",
#     "franka",
# ])

arena_builder = get_arena_builder_from_cli(args_cli)
env = arena_builder.make_registered()
env.reset()

# %%

# Run some zero actions.
NUM_STEPS = 100
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%

from isaaclab.sim import SimulationContext

simulation_context = SimulationContext.instance()
simulation_context._disable_app_control_on_stop_handle = True
simulation_context.stop()
simulation_context.clear_instance()
env.close()
import omni.timeline

omni.timeline.get_timeline_interface().stop()
