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
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# %%

import isaaclab.envs.mdp as mdp  # noqa: F401  (kept for the commented-out in-place tint notes below)
from isaaclab.managers import (  # noqa: F401  (kept for the commented-out in-place tint notes below)
    EventTermCfg,
    SceneEntityCfg,
)

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

# UniformSampler / UniformSamplerCfg are referenced only by the commented-out
# imperative-path block below; kept around so uncommenting that block is a
# single edit rather than a "and re-add the import" combo.
from isaaclab_arena.variations import (  # noqa: F401
    CameraDecalibrationVariation,
    UniformSampler,
    UniformSamplerCfg,
)

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
# embodiment = asset_registry.get_asset_by_name("franka_ik")()
# enable_cameras=True spawns the droid camera rig (incl. ``wrist_camera``) so
# the camera-decalibration variation attached below has something to act on.
embodiment = asset_registry.get_asset_by_name("droid_differential_ik")(enable_cameras=True)
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
dome_light = asset_registry.get_asset_by_name("light")()

cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
cracker_box.add_relation(IsAnchor())
tomato_soup_can.add_relation(On(cracker_box))

# --- Embodiment-level variation: wrist-camera decalibration -----------------
#
# Embodiments are :class:`~isaaclab_arena.assets.asset.Asset` instances, so they
# host variations the same way scene assets do — via ``add_variation``.
# ``ArenaEnvBuilder.get_all_variations`` then merges these into the same
# ``{asset_name: [variation, ...]}`` dict the Hydra schema and the variation
# recorder consume. The override path is keyed on the embodiment's ``name``
# (``"droid_differential_ik"`` here), shown two cells below.
embodiment.add_variation(CameraDecalibrationVariation("wrist_camera"))

# --- Variation configuration --------------------------------------------------
#
# Every ``Object`` ships with a registry of built-in variations (currently just
# ``"color"``), pre-configured with a sensible default sampler. Two surfaces
# can drive a variation:
#
# * **Imperative** (Python): call ``variation.set_sampler(...) / .enable()``
#   directly on the variation object. Kept here as commented-out reference
#   code so it's easy to compare against the structured-config path.
# * **Structured / Hydra** (cfg-driven): assemble a list of dotted-path
#   override strings that mirror the schema returned by
#   ``env_builder.get_variations_schema()`` and hand it to
#   ``env_builder.apply_hydra_variation_overrides(...)``. This is the form
#   that survives serialisation / CLI overrides and is exercised below.
#
# Both objects randomize along a single RGB axis so the per-env tint is obvious
# at a glance: the cracker box varies red, the tomato soup can varies blue.

# Imperative path (commented out, replaced by the structured-config overrides
# applied after ``ArenaEnvBuilder`` construction below):
#
# cracker_box_color = cracker_box.get_variation("color")
# cracker_box_color.set_sampler(UniformSampler(low=[0.2, 0.2, 0.0], high=[1.0, 1.0, 0.0]))
# cracker_box_color.enable()
#
# tomato_soup_can_color = tomato_soup_can.get_variation("color")
# tomato_soup_can_color.set_sampler(UniformSamplerCfg(low=[0.0, 0.2, 0.2], high=[0.0, 1.0, 1.0]))
# tomato_soup_can_color.enable()

scene = Scene(assets=[background, cracker_box, tomato_soup_can, dome_light])
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
args_cli.enable_cameras = True
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)

# --- Inspecting the dynamic variations schema --------------------------------
#
# Before applying any overrides, dump the structured-config schema that
# ``ArenaEnvBuilder.get_variations_schema`` builds from the scene. The schema
# uses each variation's existing ``*Cfg`` directly as a per-variation node —
# ``enabled`` lives on ``VariationBaseCfg`` so every variation cfg already
# carries it. The schema therefore lists every variation knob attached to the
# scene (enabled or not); each entry is pre-populated from the variation's
# current cfg, which at this point is the constructor default (e.g. ``color``
# disabled, full-RGB-uniform sampler). The override paths printed here line
# up one-to-one with the dotted keys we hand to
# ``apply_hydra_variation_overrides`` in the next cell.
from omegaconf import OmegaConf  # noqa: E402

variations_schema = env_builder.get_variations_schema()
if variations_schema is None:
    print("Scene has no variations attached.")
else:
    print(OmegaConf.to_yaml(OmegaConf.structured(variations_schema)))

# --- Structured / Hydra-driven variation overrides ---------------------------
#
# Replaces the imperative ``set_sampler / enable`` calls commented out above.
# Each override string is a dotted path into the schema printed in the
# previous cell:
#
#   <asset_name>.<variation_name>.<cfg_field>=<value>
#
# Hydra validates the paths against the structured-config schema at compose
# time, so typos / unknown fields (e.g. ``cracker_box.colour.enabled=true``)
# are rejected up front rather than silently ignored. The list below mirrors
# the two color variations the imperative path used to set up: the cracker
# box varies red, the tomato soup can varies blue.
# hydra_variation_overrides = [
#     "cracker_box.color.enabled=true",
#     "cracker_box.color.sampler.low=[0.2,0.2,0.0]",
#     "cracker_box.color.sampler.high=[1.0,1.0,0.0]",
#     "tomato_soup_can.color.enabled=true",
#     "tomato_soup_can.color.sampler.low=[0.0,0.2,0.2]",
#     "tomato_soup_can.color.sampler.high=[0.0,1.0,1.0]",
# ]
hydra_variation_overrides = [
    "cracker_box.color.enabled=true",
    "cracker_box.color.sampler.low=[0.2,0.2,0.2]",
    "cracker_box.color.sampler.high=[1.0,1.0,1.0]",
    "tomato_soup_can.color.enabled=true",
    "tomato_soup_can.color.sampler.low=[0.2,0.2,0.2]",
    "tomato_soup_can.color.sampler.high=[1.0,1.0,1.0]",
    "light.hdr_image.enabled=true",
    # Wrist-camera decalibration: ±5 cm per axis (deliberately exaggerated vs.
    # the default ±5 mm so the per-env offsets are obvious in the viewport).
    "droid_differential_ik.camera_decalibration.enabled=true",
    # "droid_differential_ik.camera_decalibration.sampler.low=[-0.05,-0.05,-0.05]",
    # "droid_differential_ik.camera_decalibration.sampler.high=[0.05,0.05,0.05]",
    "droid_differential_ik.camera_decalibration.sampler.low=[0.0,0.0,0.0]",
    "droid_differential_ik.camera_decalibration.sampler.high=[0.0,0.0,0.1]",
]
env_builder.apply_hydra_variation_overrides(hydra_variation_overrides)

# Re-dump the schema so we can confirm the overrides landed on the live
# variation cfgs (``enabled: true`` plus the narrowed sampler bounds).
print(OmegaConf.to_yaml(OmegaConf.structured(env_builder.get_variations_schema())))

# ``compose_manager_cfg`` collects every enabled variation from the scene and
# merges their event terms into ``env_cfg.events`` automatically (see
# ``ArenaEnvBuilder._compose_variations_event_cfg``). No manual plumbing.
env_cfg = env_builder.compose_manager_cfg()
assert env_cfg.scene.replicate_physics is False, "Per-env color variation requires replicate_physics=False; got True."


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


env = env_builder.make_registered(env_cfg)
env.reset()

# %%

RESET_EVERY_N_STEPS = 10

# Run some zero actions.
NUM_STEPS = 100
for step in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)
    if step % RESET_EVERY_N_STEPS == 0:
        env.reset()

#%%

# camera = env.unwrapped.scene['wrist_camera']
# t_parent_C, q_parent_C_wxyz = camera._view.get_local_poses()

# #%%

# import torch

# from isaaclab.utils.math import quat_apply, quat_inv

# camera_decalibration_ROS = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device)
# camera_decalibration_OpenGL = torch.tensor([
#     -camera_decalibration_ROS[0],
#     camera_decalibration_ROS[1],
#     -camera_decalibration_ROS[2],
# ], device=env.unwrapped.device)
# # camera_decalibration_OpenGL = camera_decalibration_ROS
# t_Cnew_C_in_C = camera_decalibration_OpenGL.repeat(4, 1)
# # t_Cnew_C_in_C = torch.tensor(
# #     [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=env.unwrapped.device)

# # Local pose of the camera relative to its parent prim.
# # Convention: get_local_poses returns (translation, quaternion) where the quaternion
# # is in (w, x, y, z) and represents the camera's OpenGL frame (-Z forward, +Y up)
# # expressed in the parent prim's frame, i.e. R_parent_cameraOpenGL.
# # camera = env.unwrapped.scene['wrist_camera']
# # t_parent_C, q_parent_C_wxyz = camera._view.get_local_poses()

# q_parent_C_xyzw = torch.roll(q_parent_C_wxyz, shifts=-1, dims=-1)

# t_Cnew_C_in_parent = quat_apply(q_parent_C_xyzw, t_Cnew_C_in_C)
# t_parent_Cnew = t_parent_C + t_Cnew_C_in_parent

# # dummy_local_translation = torch.tensor([0.01, 0.0, -0.070], device=env.unwrapped.device).repeat(4, 1)
# camera._view.set_local_poses(translations=t_parent_Cnew, orientations=None, indices=None)
# # camera._view.set_local_poses(translations=dummy_local_translation, orientations=q_parent_C_wxyz, indices=None)

# # Run some zero actions.
# # env.reset()
# NUM_STEPS = 10
# for step in tqdm.tqdm(range(NUM_STEPS)):
#     with torch.inference_mode():
#         actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
#         env.step(actions)

#%%


# %%

# --- Variation recorder inspection -------------------------------------------
#
# ``ArenaEnvBuilder`` attaches a fresh ``VariationRecorder`` to
# ``env.unwrapped`` after ``gym.make`` (see
# ``arena_env_builder.make_registered_and_return_cfg``), so every enabled
# variation has a record of the values its sampler actually produced. Each
# record bundles the variation's source id (``<asset>.<variation>``), the cfg
# that drove it (here, the Hydra-overridden bounds set above), and the ordered
# list of sample tensors — one entry per ``SamplerBase.sample()`` call, shape
# ``(num_envs, *event_shape)``. Useful as a quick sanity check that the
# distribution we asked for is what the policy actually saw.
print(env.unwrapped.variations_recorder)

# %%


from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app
from isaaclab_arena.utils.reload_modules import reload_arena_modules

# Run this to tear down the simulation app, for rebuilding the environment, without requiring a restart.
reload_arena_modules()
teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)

# %%
