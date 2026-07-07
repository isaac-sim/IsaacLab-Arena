# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%
from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Example notebook: the mesh-only ``In`` relation, mirroring RoboLab "put the cube in the bowl".

Spawns an item already inside a real container at reset. ``In`` requires ``CollisionMode.MESH`` and a
container cavity; here the cavity is derived automatically from the bowl's own mesh (its opening is
capped into a watertight interior), so no per-asset authoring is needed.
"""

# NOTE: When running as a notebook, first run this cell to launch the simulation app:
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

# %%

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaacsim import SimulationApp


def run_in_relation_demo(
    num_steps: int = 10000,
    reset_every_n_steps: int = 100,
    container: str = "serving_bowl_vomp_robolab",
    item: str = "rubiks_cube_hot3d_robolab",
):
    """Spawn ``item`` inside ``container`` via the ``In`` relation and step the sim.

    Args:
        num_steps: Number of simulation steps to run.
        reset_every_n_steps: Reset the environment every N steps (fresh In placement each reset).
        container: Registered container asset used as the (anchored) In parent.
        item: Registered object placed inside the container.
    """
    import torch
    import tqdm

    import warp as wp

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams
    from isaaclab_arena.relations.relations import In, IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    ground_plane = asset_registry.get_asset_by_name("ground_plane")()
    light = asset_registry.get_asset_by_name("light")()

    # The container is the In parent, so it must be a fixed anchor with a pure-Z (here identity) pose.
    container_object = asset_registry.get_asset_by_name(container)()
    container_object.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    container_object.add_relation(IsAnchor())

    # The item spawns inside the container's cavity (derived from the container mesh).
    item_object = asset_registry.get_asset_by_name(item)()
    item_object.add_relation(In(container_object))

    scene = Scene(assets=[ground_plane, light, container_object, item_object])

    # In is mesh-only: select MESH collision mode. Solve once (resolve_on_reset=False): MESH placement
    # caches non-picklable Warp meshes that a per-reset pooled placement event cannot serialize.
    placer_params = ObjectPlacerParams(
        solver_params=RelationSolverParams(
            collision_mode=CollisionMode.MESH, verbose=False, save_position_history=False
        ),
        resolve_on_reset=False,
    )
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="isaac_sim_in_relation_demo",
        scene=scene,
        placer_params=placer_params,
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, ArenaEnvBuilderCfg())
    env = env_builder.make_registered()
    env.reset()

    # Report placement so it's clear the item spawned inside the container.
    origin = env.unwrapped.scene.env_origins[0]
    container_xyz = (wp.to_torch(env.unwrapped.scene[container_object.name].data.root_pos_w)[0] - origin).tolist()
    item_xyz = (wp.to_torch(env.unwrapped.scene[item_object.name].data.root_pos_w)[0] - origin).tolist()
    print(f"[In demo] container '{container_object.name}' at {container_xyz}")
    print(f"[In demo] item '{item_object.name}' at {item_xyz} (initialized inside the container)")

    for step in tqdm.tqdm(range(num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
        if reset_every_n_steps > 0 and (step + 1) % reset_every_n_steps == 0:
            env.reset()


def smoke_test_in_relation(simulation_app: SimulationApp) -> bool:
    """Smoke test: spawn an item in a bowl via In and step a couple frames."""
    run_in_relation_demo(num_steps=2)
    return True


# %%
# When running as a notebook (after launching simulation_app), uncomment and run:
run_in_relation_demo()

# %%
