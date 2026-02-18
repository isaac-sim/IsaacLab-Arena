# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher(headless=False, enable_cameras=True)

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.embodiments.droid.droid import DroidEmbodiment
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.policy.gr00t_remote_client_policy import Gr00tFrankaClient
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("rubiks_cube_bowl_srl")()
rubiks_cube = ObjectReference(
    name="rubiks_cube",
    prim_path="{ENV_REGEX_NS}/rubiks_cube_bowl_srl/rubiks_cube",
    parent_asset=background,
    object_type=ObjectType.RIGID,
)
bowl = ObjectReference(
    name="bowl",
    prim_path="{ENV_REGEX_NS}/rubiks_cube_bowl_srl/bowl",
    parent_asset=background,
    object_type=ObjectType.RIGID,
)
light = asset_registry.get_asset_by_name("light")()

task = PickAndPlaceTask(
    pick_up_object=rubiks_cube,
    destination_location=bowl,
    background_scene=background,
)

embodiment = DroidEmbodiment(enable_cameras=True, use_joint_position_actions=True)

scene = Scene(assets=[background, light, rubiks_cube, bowl])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="gr00t_droid_pick_and_place",
    scene=scene,
    task=task,
    embodiment=embodiment,
)

args_cli = get_isaaclab_arena_cli_parser().parse_args(["--enable_cameras"])
args_cli.solve_relations = True
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
obs, _ = env.reset()

# Match robolab's viewport camera angle
env.sim.set_camera_view(eye=(1.5, 0.0, 1.0), target=(0.2, 0.0, 0.0))

# %%

client = Gr00tFrankaClient(
    remote_host="localhost",
    remote_port=5555,
    open_loop_horizon=10,
)

instruction = task.get_task_description() or "pick up the rubiks cube and place it in the bowl"
print(f"Task instruction: {instruction}")

# %%

NUM_STEPS = 1000
for step in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        ret = client.infer(obs, instruction)
        action = torch.tensor(ret["action"], dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated.any() or truncated.any():
            env_ids = (terminated | truncated).nonzero().flatten()
            print(
                f"Resetting at step {step} — "
                f"terminated: {terminated.nonzero().flatten().tolist()}, "
                f"truncated: {truncated.nonzero().flatten().tolist()}"
            )
            break
            # client.reset()

# Compute and print metrics
from isaaclab_arena.metrics.metrics import compute_metrics

metrics = compute_metrics(env)
if metrics is not None:
    print(f"\nMetrics: {metrics}")
else:
    print("\nNo metrics registered for this environment.")

# %%
