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
# args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "2"])
# simulation_app = AppLauncher(args)
simulation_app = AppLauncher()

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.assets.object_set import RigidObjectSet

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
sweet_potato = asset_registry.get_asset_by_name("sweet_potato")()
jug = asset_registry.get_asset_by_name("jug")()

# cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
# cracker_box.add_relation(IsAnchor())
# tomato_soup_can.add_relation(On(cracker_box))

object_set = RigidObjectSet(
    name="object_set",
    # objects=[cracker_box, tomato_soup_can]
    objects=[sweet_potato, jug]
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
args_cli.solve_relations = True
args_cli.num_envs = 2
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

# %%

# Run some zero actions.
NUM_STEPS = 500
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)


#%%

import pathlib

from pxr import Usd, Gf, UsdGeom


from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body_from_stage
from isaaclab_arena.utils.usd_helpers import open_stage



def get_cache_dir() -> pathlib.Path:
    home_path = pathlib.Path.home()
    print(f"Home path: {home_path}")
    cache_dir = home_path / ".cache" / "isaaclab_arena"
    print(f"Cache dir: {cache_dir}")
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created cache dir: {cache_dir}")
    return cache_dir


def get_arena_asset_cache_path(asset: Asset, scale: tuple[float, float, float] | None = None) -> pathlib.Path:
    cache_dir = get_cache_dir()
    if scale is not None:
        scale_str = "_".join([str(s) for s in scale])
        return cache_dir / f"{asset.name}_{scale_str}.usd"
    else:
        return cache_dir / f"{asset.name}.usd"


def rescale_root(stage: Usd.Stage, asset: Asset) -> None:
    root_prim = stage.GetDefaultPrim()
    xformable = UsdGeom.Xformable(root_prim)
    scale_attr = root_prim.GetAttribute("xformOp:scale")
    if scale_attr.IsValid():
        UsdGeom.XformOp(scale_attr).Set(Gf.Vec3f(*asset.scale))
    else:
        xformable.AddScaleOp().Set(Gf.Vec3f(*asset.scale))


def rename_rigid_body(stage: Usd.Stage, new_name: str = "rigid_body") -> None:
    shallowest_rigid_body = find_shallowest_rigid_body_from_stage(stage)
    print(f"Shallowest rigid body: {shallowest_rigid_body}")
    prim = stage.GetPrimAtPath(shallowest_rigid_body)
    print(f"Prim: {prim}")
    assert prim.IsValid()
    prim_spec = stage.GetRootLayer().GetPrimAtPath(shallowest_rigid_body)
    print(f"Prim spec: {prim_spec}")
    prim_spec.name = new_name


def rescale_and_save_to_cache(asset: Asset) -> pathlib.Path:
    cache_path = get_arena_asset_cache_path(asset, asset.scale)
    # print(f"Rescaling and saving {asset.name} to cache at {cache_path}")
    with open_stage(asset.usd_path) as stage:
        rescale_root(stage, asset)
        rename_rigid_body(stage)
        stage.Export(str(cache_path))
    return cache_path


#%%

from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body


def get_all_rigid_bodies(assets: list[Asset]) -> list[str]:
    rigid_body_paths = []
    depths = []
    for asset in assets:
        shallowest_rigid_body = find_shallowest_rigid_body(asset.usd_path)
        rigid_body_paths.append(shallowest_rigid_body)
        depth = shallowest_rigid_body.count("/") - 1
        depths.append(depth)
    return rigid_body_paths, depths


def is_asset_modification_needed(assets: list[Asset]) -> bool:
    # If any asset is scaled, we need to modify the assets
    for asset in assets:
        if asset.scale != (1.0, 1.0, 1.0):
            return True
    # If all assets have rigid bodies at the root, we don't need to modify the assets
    rigid_body_paths, depths = get_all_rigid_bodies(assets)
    if all(depth == 0 for depth in depths):
        return False
    # Otherwise, we need to modify the assets
    return True


assets = [sweet_potato, jug]

new_usd_paths = []
if is_asset_modification_needed(assets):
    _, rigid_body_depths = get_all_rigid_bodies(assets)
    assert all(depth == 0 for depth in rigid_body_depths), "All assets should have rigid bodies at the root"
    for asset in assets:
        new_usd_path = rescale_and_save_to_cache(asset)
        new_usd_paths.append(new_usd_path)


#%%

cracker_box = asset_registry.get_asset_by_name("cracker_box")()
rescaled_cracker_box_path = rescale_and_save_to_cache(cracker_box)
print(f"Rescaled cracker box path: {rescaled_cracker_box_path}")

sweet_potato = asset_registry.get_asset_by_name("sweet_potato")()
rescaled_sweet_potato_path = rescale_and_save_to_cache(sweet_potato)
print(f"Rescaled sweet potato path: {rescaled_sweet_potato_path}")

# %%
