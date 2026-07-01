# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%
from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Example notebook: placing objects on a kitchen counter while avoiding background fixtures.

The Lightwheel Robocasa kitchen is loaded as the background. Objects are placed On the main
counter, and the collision obstacles are discovered automatically: a placement region is
built from the counter anchor, then background fixtures (the sink protruding through the
counter, ...) whose bounding box intersects that region are passed to the placer. A correct
solve spreads the objects onto the flat counter areas flanking the sink.
"""

# NOTE: When running as a notebook, first run this cell to launch the simulation app:
import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

# %%

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaacsim import SimulationApp

# The counter top is the placement surface; other fixtures (sink, ...) are discovered as obstacles.
_KITCHEN = "lightwheel_robocasa_kitchen"
_COUNTER_PRIM = "{ENV_REGEX_NS}/" + _KITCHEN + "/counter_main_main_group"


def run_kitchen_background_collision_demo(
    num_steps: int = 1000,
    reset_every_n_steps: int = 100,
    hold_overlapping_steps: int = 150,
):
    """Place objects on the kitchen counter while avoiding the sink and stove.

    Objects start clustered (overlapping) on a clear stretch of the counter; after each reset
    the relation solver runs with the discovered background fixtures as fixed collision
    obstacles and spreads the objects onto the flat counter areas around the sink.

    Args:
        num_steps: Number of simulation steps to run.
        reset_every_n_steps: Reset the environment every N steps (to re-solve and see new layouts).
        hold_overlapping_steps: After each reset, render this many frames (no physics) showing the
            overlapping start pose before the solver runs.
    """
    import torch
    import tqdm

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.background_colliders import build_placement_region, find_background_colliders
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name(_KITCHEN)()
    light = asset_registry.get_asset_by_name("light")()

    counter_reference = ObjectReference(name="counter", prim_path=_COUNTER_PRIM, parent_asset=background)
    counter_reference.add_relation(IsAnchor())

    # Cluster the objects on a clear stretch of counter (left of the sink) so they start overlapping.
    same_pose = Pose(position_xyz=(0.45, -0.3, 0.97), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    object_names = ["cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle", "mug"]
    objects = []
    for name in object_names:
        obj = asset_registry.get_asset_by_name(name)()
        obj.add_relation(On(counter_reference, clearance_m=0.02))
        obj.set_initial_pose(same_pose)
        objects.append(obj)

    # Step 1: region above the counter the objects can occupy. Step 2: background fixtures that
    # reach into it. The solver treats these as fixed obstacles, never optimizing them.
    region = build_placement_region([counter_reference], objects)
    collision_objects = find_background_colliders(background, region, anchors=[counter_reference])
    print(f"Discovered background collision obstacles: {[c.name for c in collision_objects]}")

    scene = Scene(assets=[background, counter_reference, *objects, light])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="kitchen_background_collision_demo",
        scene=scene,
    )

    # Build without solving relations so objects start overlapping; we run the solver after reset.
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--no-solve-relations"])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()

    objects_with_relations = [counter_reference, *objects]
    num_envs = env.unwrapped.scene.num_envs
    env_ids = torch.arange(num_envs, device=env.unwrapped.device)
    qw, qx, qy, qz = (1.0, 0.0, 0.0, 0.0)

    def apply_overlapping_pose_then_solve_and_display():
        x, y, z = same_pose.position_xyz
        root_pose = (
            torch.tensor([x, y, z, qw, qx, qy, qz], device=env.unwrapped.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(num_envs, 7)
            .clone()
        )
        root_pose[:, :3] += env.unwrapped.scene.env_origins[env_ids]
        for obj in objects:
            if obj.name in env.unwrapped.scene.rigid_objects:
                env.unwrapped.scene.rigid_objects[obj.name].write_root_pose_to_sim(root_pose.clone(), env_ids=env_ids)
            else:
                print(f"\nObject {obj.name!r} is not in scene.rigid_objects, skipping overlapping pose.")
        for _ in range(hold_overlapping_steps):
            env.unwrapped.sim.render()

        # Run the solver with the sink and stove as fixed background obstacles, then apply the poses.
        placer = ObjectPlacer()
        (result,) = placer.place(objects=objects_with_relations, collision_objects=collision_objects)
        for obj in objects:
            if obj.name not in env.unwrapped.scene.rigid_objects:
                print(f"\nObject {obj.name!r} is not in scene.rigid_objects, skipping solved pose.")
                continue
            x, y, z = result.positions[obj]
            root_pose = (
                torch.tensor([x, y, z, qw, qx, qy, qz], device=env.unwrapped.device, dtype=torch.float32)
                .unsqueeze(0)
                .expand(num_envs, 7)
                .clone()
            )
            root_pose[:, :3] += env.unwrapped.scene.env_origins[env_ids]
            env.unwrapped.scene.rigid_objects[obj.name].write_root_pose_to_sim(root_pose, env_ids=env_ids)
        env.unwrapped.scene.write_data_to_sim()
        env.unwrapped.sim.step(render=True)
        env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)

    env.reset()
    apply_overlapping_pose_then_solve_and_display()

    for step in tqdm.tqdm(range(num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

        if reset_every_n_steps > 0 and (step + 1) % reset_every_n_steps == 0:
            env.reset()
            apply_overlapping_pose_then_solve_and_display()


def smoke_test_kitchen_background_collision(simulation_app: SimulationApp) -> bool:
    """Smoke test: run the kitchen background-collision demo (minimal steps)."""
    run_kitchen_background_collision_demo(num_steps=2)
    return True


# %%
# When running as a notebook (after launching simulation_app), uncomment and run:
# run_kitchen_background_collision_demo()

# %%
