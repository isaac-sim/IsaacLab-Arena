# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

print("Launching simulation app once in notebook")
# headless=False enables the Kit viewer window so we can visually verify per-env
# randomizations (e.g. object tints). Set headless=True for CI / non-GUI runs.
args_cli = get_isaaclab_arena_cli_parser().parse_args([])
args_cli.headless = False
args_cli.visualizer = "kit"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# %%

import isaaclab.envs.mdp as mdp  # noqa: F401  (kept for the commented-out replacement term below)
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.examples.tint_events import randomize_visual_diffuse_tint
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka_ik")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()

cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
cracker_box.add_relation(IsAnchor())
tomato_soup_can.add_relation(On(cracker_box))

scene = Scene(assets=[background, cracker_box, tomato_soup_can])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
)

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
args_cli.solve_relations = True
# Bump num_envs so we can visually verify per-env color variation.
args_cli.num_envs = 4
args_cli.visualizer = "kit"
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)

# Build the env_cfg so we can inject extra event terms before registration.
env_cfg = env_builder.compose_manager_cfg()

# %%

# Per-env visual tint via a custom event (see ``tint_events.py``).
#
# This *keeps* the asset's original diffuse texture and multiplies a random
# color onto it, rather than replacing the material with a flat OmniPBR
# instance like ``mdp.randomize_visual_color`` does.
#
# Requirements:
# * ``scene.replicate_physics`` must be False (Arena default) — with replication
#   on, every env shares a single source material and per-env tinting is
#   impossible. The event itself also asserts this.
# * ``mode="prestartup"`` → each env gets a stable tint for the entire run.
#   Use ``mode="reset"`` instead to resample on every episode reset.
# * The colors dict specifies uniform ranges per channel. Narrow ranges near
#   1.0 (e.g. (0.4, 1.0)) give subtle, photo-realistic tints; wide ranges
#   like (0.0, 1.0) look more aggressive.
assert (
    env_cfg.scene.replicate_physics is False
), "randomize_visual_diffuse_tint requires replicate_physics=False; got True."
# env_cfg.events.cracker_box_tint = EventTermCfg(
#     func=randomize_visual_diffuse_tint,
#     mode="reset",
#     params={
#         "asset_cfg": SceneEntityCfg(cracker_box.name),
#         "colors": {"r": (0.4, 1.0), "g": (0.4, 1.0), "b": (0.4, 1.0)},
#     },
# )
# env_cfg.events.tomato_soup_can_tint = EventTermCfg(
#     func=randomize_visual_diffuse_tint,
#     mode="prestartup",
#     params={
#         "asset_cfg": SceneEntityCfg(tomato_soup_can.name),
#         "colors": {"r": (0.4, 1.0), "g": (0.4, 1.0), "b": (0.4, 1.0)},
#     },
# )

# --- Previous behavior: replace the material entirely (texture is lost) ---
env_cfg.events.cracker_box_color = EventTermCfg(
    func=mdp.randomize_visual_color,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg(cracker_box.name),
        "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        "mesh_name": "",
        "event_name": "cracker_box_color",
    },
)
# env_cfg.events.tomato_soup_can_color = EventTermCfg(
#     func=mdp.randomize_visual_color,
#     mode="prestartup",
#     params={
#         "asset_cfg": SceneEntityCfg(tomato_soup_can.name),
#         "colors": [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)],
#         "mesh_name": "",
#         "event_name": "tomato_soup_can_color",
#     },
# )

# %%

env = env_builder.make_registered(env_cfg)
env.reset()

# %%

# Run some zero actions.
NUM_STEPS = 500
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%


from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app
from isaaclab_arena.utils.reload_modules import reload_arena_modules

# Run this to tear down the simulation app, for rebuilding the environment, without requiring a restart.
reload_arena_modules()
teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)

# %%
