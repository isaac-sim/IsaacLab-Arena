# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%
from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Example notebook demonstrating NoCollision with real Isaac Sim objects."""

# NOTE: When running as a notebook, first run this cell to launch the simulation app:
import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

# %%

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaacsim import SimulationApp


def run_isaac_sim_no_collision_demo(
    num_steps: int = 10000,
    reset_every_n_steps: int = 100,
    hold_overlapping_steps: int = 150,
):
    """Run the NoCollision demo with Isaac Sim objects.

    Args:
        num_steps: Number of simulation steps to run.
        reset_every_n_steps: Reset the environment every N steps (to see placer re-solve and new layouts).
        hold_overlapping_steps: After each reset, render this many frames without physics (overlapping pose
            held) so you can capture the frame; then the solver runs and results are displayed.
    """
    import torch
    import tqdm

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.relations import IsAnchor, NoCollision, On
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    ground_plane = asset_registry.get_asset_by_name("ground_plane")()
    table_background = asset_registry.get_asset_by_name("office_table")()
    light = asset_registry.get_asset_by_name("light")()

    tabletop_reference = ObjectReference(
        name="table",
        prim_path="{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
        parent_asset=table_background,
    )
    # Mark the ObjectReference as the anchor for relation solving (not subject to optimization).
    tabletop_reference.add_relation(IsAnchor())

    # Same initial pose for all three so they appear overlapping in the sim; we run the solver after reset.
    same_pose = Pose(position_xyz=(0.0, 0.0, 0.77), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))

    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    cracker_box.add_relation(On(tabletop_reference, clearance_m=0.02))
    cracker_box.set_initial_pose(same_pose)

    mug = asset_registry.get_asset_by_name("mug")()
    mug.add_relation(On(tabletop_reference, clearance_m=0.02))
    mug.set_initial_pose(same_pose)

    tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
    tomato_soup_can.add_relation(On(tabletop_reference, clearance_m=0.02))
    tomato_soup_can.set_initial_pose(same_pose)

    cracker_box.add_relation(NoCollision(mug))
    cracker_box.add_relation(NoCollision(tomato_soup_can))
    mug.add_relation(NoCollision(tomato_soup_can))

    scene = Scene(assets=[ground_plane, table_background, tabletop_reference, cracker_box, mug, tomato_soup_can, light])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="isaac_sim_no_collision_demo",
        scene=scene,
    )

    # Build without solving relations so objects start in the same (overlapping) place in the sim.
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--no-solve-relations"])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()

    objects_with_relations = [tabletop_reference, cracker_box, mug, tomato_soup_can]
    unwrapped = env.unwrapped
    identity_quat = (1.0, 0.0, 0.0, 0.0)

    def _write_overlapping_pose_to_sim():
        """Hackily set all three placeable objects to the same (overlapping) pose in the sim."""
        num_envs = unwrapped.scene.num_envs
        env_ids = torch.arange(num_envs, device=unwrapped.device)
        zero_vel = torch.zeros(num_envs, 6, device=unwrapped.device)
        x, y, z = same_pose.position_xyz
        root_pose = torch.tensor(
            [x, y, z, identity_quat[0], identity_quat[1], identity_quat[2], identity_quat[3]],
            device=unwrapped.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(num_envs, 7)
        root_pose[:, :3] += unwrapped.scene.env_origins[env_ids]
        for obj in (cracker_box, mug, tomato_soup_can):
            name = obj.name
            if name in unwrapped.scene.rigid_objects:
                unwrapped.scene.rigid_objects[name].write_root_pose_to_sim(root_pose.clone(), env_ids=env_ids)
                unwrapped.scene.rigid_objects[name].write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    def _pause_for_capture():
        """Render only (no physics) for hold_overlapping_steps so you can capture the overlapping frame."""
        for _ in range(hold_overlapping_steps):
            unwrapped.sim.render()

    def _run_solver_apply_poses_and_display():
        """Run the solver, apply solved poses to the sim, then one physics step + update to display results."""
        placer = ObjectPlacer()
        result = placer.place(objects=objects_with_relations)
        num_envs = unwrapped.scene.num_envs
        env_ids = torch.arange(num_envs, device=unwrapped.device)
        zero_vel = torch.zeros(num_envs, 6, device=unwrapped.device)
        for obj in (cracker_box, mug, tomato_soup_can):
            name = obj.name
            if name not in unwrapped.scene.rigid_objects:
                continue
            x, y, z = result.positions[obj]
            root_pose = torch.tensor(
                [x, y, z, identity_quat[0], identity_quat[1], identity_quat[2], identity_quat[3]],
                device=unwrapped.device,
                dtype=torch.float32,
            ).unsqueeze(0).expand(num_envs, 7)
            root_pose[:, :3] += unwrapped.scene.env_origins[env_ids]
            unwrapped.scene.rigid_objects[name].write_root_pose_to_sim(root_pose, env_ids=env_ids)
            unwrapped.scene.rigid_objects[name].write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=True)
        unwrapped.scene.update(dt=unwrapped.physics_dt)

    def _hold_then_solve_and_display():
        """Set overlapping pose, pause for capture, run solver, enable physics and display results."""
        _write_overlapping_pose_to_sim()
        _pause_for_capture()
        _run_solver_apply_poses_and_display()

    env.reset()
    _hold_then_solve_and_display()

    # Run simulation steps with periodic resets
    for step in tqdm.tqdm(range(num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
            env.step(actions)

        if reset_every_n_steps > 0 and (step + 1) % reset_every_n_steps == 0:
            env.reset()
            _hold_then_solve_and_display()


# %%
# When running as a notebook (after launching simulation_app), uncomment and run:
run_isaac_sim_no_collision_demo()

# %%
