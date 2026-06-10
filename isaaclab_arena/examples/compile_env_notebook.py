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

args_cli = get_isaaclab_arena_cli_parser().parse_args(["--viz", "kit", "--enable_cameras"])
print("Launching simulation app once in notebook")
simulation_app = AppLauncher(args_cli)


# %%

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
franka = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=True)
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
dome_light = asset_registry.get_asset_by_name("light")()

cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
cracker_box.add_relation(IsAnchor())
tomato_soup_can.add_relation(On(cracker_box))

scene = Scene(assets=[background, cracker_box, tomato_soup_can, dome_light])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=franka,
    scene=scene,
    task=PickAndPlaceTask(cracker_box, tomato_soup_can, background, episode_length_s=1.0),
)

dome_light.get_variation("hdr_image").enable()
franka.get_variation("camera_extrinsics_wrist_cam").enable()

env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()


# %%

# Run some zero actions.
RESET_ON_EVERY_N_STEPS = 1000
NUM_EPISODES = 2
NUM_STEPS = 100
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        _, _, terminated, truncated, _ = env.step(actions)
    if terminated.any() or truncated.any():
        print(f"Terminated or truncated at step {_}")
        NUM_EPISODES -= 1
        if NUM_EPISODES == 0:
            break

# %%

from isaaclab_arena.metrics.metric_data import MetricsDataCollection

metrics_data_collection = env.unwrapped.compute_metrics()
print(f"metrics_data_collection: {metrics_data_collection}")

# %%

from copy import deepcopy

from isaaclab_arena.metrics.metric_data import MetricData

metrics_1 = deepcopy(metrics_data_collection)
metrics_2 = deepcopy(metrics_data_collection)

metrics_per_run: list[MetricsDataCollection] = [metrics_1, metrics_2]

# %%

import numpy as np
from typing import Any

# Check that all metrics have the same names
metric_names = [metric_data.term_name for metric_data in metrics_per_run[0].metric_data_entries]
for metrics_data_collection in metrics_per_run:
    for metric_data in metrics_data_collection.metric_data_entries:
        assert metric_data.term_name in metric_names, f"Metric {metric_data.term_name} not found in all runs"

# Total number of episodes
total_num_episodes = sum(metrics_data_collection.num_episodes for metrics_data_collection in metrics_per_run)
print(f"total_num_episodes: {total_num_episodes}")

# Aggregate the recorded data, for all runs, for each metric name
metric_name_to_aggregated_data: dict[str, list[np.ndarray]] = {}
print(f"test: {metric_name_to_aggregated_data}")
for metrics_data_collection in metrics_per_run:
    for metric_data in metrics_data_collection.metric_data_entries:
        if metric_data.term_name not in metric_name_to_aggregated_data:
            metric_name_to_aggregated_data[metric_data.term_name] = list(metric_data.recorded_data)
        else:
            metric_name_to_aggregated_data[metric_data.term_name].extend(metric_data.recorded_data)
print(f"test2: {metric_name_to_aggregated_data['success_rate']}")

# Re-compute the metric values
metric_cfgs = {metric_data.term_name: metric_data.term_cfg for metric_data in metrics_per_run[0].metric_data_entries}
metric_name_to_aggregated_metric_values: dict[str, Any] = {}
for metric_name, recorded_data in metric_name_to_aggregated_data.items():
    metric_cfg = metric_cfgs[metric_name]
    metric = metric_cfg.compute_metric_func(recorded_data, **metric_cfg.params)
    metric_name_to_aggregated_metric_values[metric_name] = metric

# Assemble a new MetricsDataCollection with the aggregated metric values
metric_data_entries: list[MetricData] = []
for metric_name in metric_names:
    metric_data_entry = MetricData(
        term_name=metric_name,
        term_cfg=metric_cfgs[metric_name],
        recorded_data=metric_name_to_aggregated_data[metric_name],
        metric_value=metric_name_to_aggregated_metric_values[metric_name],
    )
    metric_data_entries.append(metric_data_entry)

metrics_data_collection_aggregated = MetricsDataCollection(
    num_episodes=total_num_episodes, metric_data_entries=metric_data_entries
)

print(f"metrics_data_collection_aggregated: {metrics_data_collection_aggregated}")

# print(f"metric_name_to_aggregated_metric_values: {metric_name_to_aggregated_metric_values}")


# %%

# %%

from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app
from isaaclab_arena.utils.reload_modules import reload_arena_modules

# Run this to tear down the simulation app, for rebuilding the environment, without requiring a restart.
reload_arena_modules()
teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)

# %%
