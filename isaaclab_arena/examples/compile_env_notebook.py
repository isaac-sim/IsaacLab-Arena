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
simulation_app = AppLauncher()

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.assets.object_set import RigidObjectSet
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()

# cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
# cracker_box.add_relation(IsAnchor())
# tomato_soup_can.add_relation(On(cracker_box))

object_set = RigidObjectSet(
    name="object_set",
    objects=[cracker_box, tomato_soup_can],
)
object_set.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

# scene = Scene(assets=[background, cracker_box, tomato_soup_can])
scene = Scene(assets=[background, object_set])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
)

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
# args_cli.solve_relations = True
args_cli.num_envs = 2
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
name, cfg = env_builder.build_registered()


import os

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from isaaclab_arena.utils.usd_helpers import get_asset_usd_path_from_prim_path


def _resolve_object_name_for_usd_path(usd_path: str, objects: list) -> str:
    """Resolve a referenced USD path to an object name. Raises if no object matches."""
    ref_basename = os.path.basename(usd_path)
    for obj in objects:
        obj_name = getattr(obj, "name", None)
        obj_usd = getattr(obj, "usd_path", None)
        if obj_name is None:
            continue
        if obj_usd and ref_basename == os.path.basename(obj_usd):
            return obj_name
        if ref_basename == obj_name + ".usd" or ref_basename.startswith(obj_name + "_"):
            return obj_name
    raise AssertionError(f"No object name for USD path {usd_path!r} (basename {ref_basename!r})")


def set_object_set_pose_by_usd(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_by_object: dict[str, Pose],
    objects: list,
) -> None:
    """Reset object_set instances to poses by object name. Prim must have a USD reference; object name must be in pose_by_object."""
    if env_ids is None or env_ids.numel() == 0:
        return
    asset = env.scene[asset_cfg.name]
    stage = env.scene.stage
    matching_prims = sim_utils.find_matching_prims(asset.cfg.prim_path, stage)
    # device = env.device if isinstance(env.device, torch.device) else torch.device(env.device)

    poses_list = []
    for i in range(env_ids.numel()):
        env_id = int(env_ids[i].item())
        prim = matching_prims[env_id]
        prim_path = prim.GetPath().pathString
        # Get the USD path from the prim path in this env_idx
        usd_path = get_asset_usd_path_from_prim_path(prim_path, stage)
        assert usd_path is not None, f"Prim at {prim_path} has no USD reference."
        # Resolve the object name from the USD path
        object_name = _resolve_object_name_for_usd_path(usd_path, objects)
        assert (
            object_name in pose_by_object
        ), f"Object name {object_name!r} not in pose_by_object (keys: {list(pose_by_object.keys())})."
        # Get the pose for the object
        pose = pose_by_object[object_name]
        # Add the pose to the list of poses to set.
        pose_t = pose.to_tensor(device=env.device).unsqueeze(0)
        pose_t[:, :3] += env.scene.env_origins[env_id]
        poses_list.append(pose_t)
    pose_t_xyz_q_wxyz = torch.cat(poses_list, dim=0)
    asset.write_root_pose_to_sim(pose_t_xyz_q_wxyz, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(pose_t_xyz_q_wxyz.shape[0], 6, device=env.device), env_ids=env_ids)


# Key by object name (no USD paths visible). Use the same objects as in the object set.
pose_by_object = {
    cracker_box.name: Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    tomato_soup_can.name: Pose(position_xyz=(0.35, 0.50, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
}

new_reset_pose_event = EventTermCfg(
    func=set_object_set_pose_by_usd,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("object_set"),
        "pose_by_object": pose_by_object,
        "objects": object_set.objects,
    },
)

print("Original event:")
print(cfg.events.object_set)
print("New event:")
cfg.events.object_set = new_reset_pose_event
print(cfg.events.object_set)

env = env_builder.make_registered(cfg)
env.reset()

# %%
# Run some zero actions.
RESET_EVERY_N_STEPS = 100
NUM_STEPS = 500
for idx in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)
        if idx % RESET_EVERY_N_STEPS == 0:
            env.reset()


# %%


# %%
