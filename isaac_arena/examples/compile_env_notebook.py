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

from isaac_arena.assets.asset_registry import AssetRegistry
from isaac_arena.assets.object_reference import ObjectReference
from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.environments.compile_env import ArenaEnvBuilder
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.scene.scene import Scene

# from isaac_arena.tasks.dummy_task import DummyTask
from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaac_arena.utils.pose import Pose

asset_registry = AssetRegistry()


background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
microwave = asset_registry.get_asset_by_name("microwave")()

cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
microwave.set_initial_pose(Pose(position_xyz=(0.4, 0.4, 0.3), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

destination_location = ObjectReference(
    name="destination_location",
    prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
    parent_asset=background,
)

scene = Scene(assets=[background, cracker_box, microwave])
isaac_arena_environment = IsaacArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    task=PickAndPlaceTask(cracker_box, destination_location, background),
    teleop_device=None,
)

# args_cli = get_isaac_arena_cli_parser().parse_args([])
# env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
# env = env_builder.make_registered()
# env.reset()


args_cli = get_isaac_arena_cli_parser().parse_args([])
args_cli.num_envs = 2
env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
env_cfg = env_builder.compose_manager_cfg()


# %

from dataclasses import MISSING

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaac_arena.assets.asset import Asset
from isaac_arena.terms.events import set_object_pose_per_env
from isaac_arena.utils.pose import Pose

# UP TO HERE: MAKE A TEST.

# Poses for envs 1 and 2.
pose_list = [
    Pose(position_xyz=(0.4, 0.0, 0.2), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    # Pose(position_xyz=(-0.4, 0.0, 0.2), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    Pose(position_xyz=(0.4, 0.4, 0.2), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
]


initial_pose = cracker_box.get_initial_pose()
env_cfg.events.reset_pick_up_object_pose = EventTermCfg(
    func=set_object_pose_per_env,
    mode="reset",
    params={
        "pose_list": pose_list,
        "asset_cfg": SceneEntityCfg(cracker_box.name),
    },
)

# %

env = env_builder.make_registered(env_cfg)
env.reset()

# %%

test = env.scene["cracker_box"]
print(test)

print(env.scene.get_state(is_relative=True)["rigid_object"]["cracker_box"]["root_pose"])

# %%

from isaac_arena.assets.object_base import ObjectBase, ObjectType


def get_object_pose(asset: ObjectBase, env: ManagerBasedEnv, is_relative: bool = True):
    # We require that the asset has been added to the scene under its name.
    assert asset.name in env.scene.keys(), f"Asset {asset.name} not found in scene"
    if asset.object_type == ObjectType.RIGID:
        object_pose = env.scene[asset.name].data.root_pose_w.clone()
    elif asset.object_type == ObjectType.ARTICULATION:
        object_pose = env.scene[asset.name].data.root_pose_w.clone()
    elif asset.object_type == ObjectType.BASE:
        object_pose = torch.cat(test.get_world_poses(), dim=-1)
    else:
        raise ValueError(f"Function not implemented for object type: {asset.object_type}")
    if is_relative:
        object_pose[:, :3] -= env.scene.env_origins
    return object_pose


print(get_object_pose(cracker_box, env, is_relative=True))
print(get_object_pose(background, env, is_relative=True))

# %%

test = env.scene["kitchen"]
# test.get_world_poses()[1].shape

# %%

# Run some zero actions.
NUM_STEPS = 100
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%
