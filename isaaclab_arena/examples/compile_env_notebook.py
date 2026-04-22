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

import isaaclab.envs.mdp as mdp  # noqa: F401  (kept for the commented-out in-place tint notes below)
from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: F401  (kept for the commented-out in-place tint notes below)

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.examples.tint_events import randomize_visual_diffuse_tint  # noqa: F401  (kept for the commented-out in-place tint notes below)
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations import UniformSampler

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka_ik")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()

cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
cracker_box.add_relation(IsAnchor())
tomato_soup_can.add_relation(On(cracker_box))

# --- New-style variation configuration ---------------------------------------
#
# Every ``Object`` ships with a registry of built-in variations (currently just
# ``"color"``), pre-configured with a sensible default sampler. Calling
# :meth:`~isaaclab_arena.variations.variation_base.VariationBase.enable` alone
# is enough to get reasonable behaviour; :meth:`set_sampler` is only needed to
# narrow or replace the default distribution.
cracker_box.get_variation("color").enable()  # uses the default full-RGB sampler

# Uncomment to also randomize the soup can with a tighter (pastel) range:
# tomato_soup_can.get_variation("color").enable()
# tomato_soup_can.get_variation("color").set_sampler(UniformSampler(low=(0.4,) * 3, high=(1.0,) * 3))

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

# ``compose_manager_cfg`` collects every enabled variation from the scene and
# merges their event terms into ``env_cfg.events`` automatically (see
# ``ArenaEnvBuilder._compose_variations_event_cfg``). No manual plumbing.
env_cfg = env_builder.compose_manager_cfg()
assert (
    env_cfg.scene.replicate_physics is False
), "Per-env color variation requires replicate_physics=False; got True."

# %%

# --- In-place diffuse tint (still a TODO, see 2026_04_21_color_variation_status.md) --
#
# The in-place tint path (``randomize_visual_diffuse_tint``) preserves the
# asset's diffuse texture but currently doesn't produce a visible change.
# Left here so it's easy to A/B once the shader-path fix lands.
# env_cfg.events.cracker_box_tint = EventTermCfg(
#     func=randomize_visual_diffuse_tint,
#     mode="reset",
#     params={
#         "asset_cfg": SceneEntityCfg(cracker_box.name),
#         "colors": {"r": (0.4, 1.0), "g": (0.4, 1.0), "b": (0.4, 1.0)},
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
