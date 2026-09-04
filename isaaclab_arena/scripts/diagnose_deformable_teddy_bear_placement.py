# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reproduce and classify teddy-bear penetration in the DROID deformable environment.

Run inside the Arena development container:

    /isaac-sim/python.sh isaaclab_arena/scripts/diagnose_deformable_teddy_bear_placement.py
"""

from __future__ import annotations

import argparse

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def _parse_args() -> argparse.Namespace:
    """Parse simulation and diagnostic arguments."""
    parser = get_isaaclab_arena_cli_parser()
    parser.set_defaults(num_envs=1, presets="physx", solve_relations=True)
    parser.add_argument("--diagnostic_steps", type=int, default=120)
    parser.add_argument("--report_every", type=int, default=10)
    parser.add_argument("--penetration_tolerance", type=float, default=1.0e-3)
    args = parser.parse_args()
    assert args.num_envs == 1, "This diagnostic requires --num_envs 1"
    assert args.presets in (None, "default", "physx"), "The teddy-bear asset is PhysX-only"
    assert args.diagnostic_steps >= 0, "--diagnostic_steps must be non-negative"
    assert args.report_every > 0, "--report_every must be positive"
    return args


def _env_zero_pose(asset):
    """Return environment zero's solved pose."""
    from isaaclab_arena.utils.pose import Pose, PosePerEnv

    pose = asset.get_initial_pose()
    if isinstance(pose, PosePerEnv):
        return pose.poses[0]
    assert isinstance(pose, Pose), f"Expected a solved Pose or PosePerEnv, got {type(pose).__name__}"
    return pose


def _run_diagnostic(args: argparse.Namespace) -> None:
    """Build the exact environment and report planned versus simulated clearances."""
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.droid_deformable_pick_and_place_environment import (
        DroidDeformablePickAndPlaceEnvironment,
        DroidDeformablePickAndPlaceEnvironmentCfg,
    )

    arena_env = DroidDeformablePickAndPlaceEnvironment().build(
        DroidDeformablePickAndPlaceEnvironmentCfg(
            pick_object="teddy_bear",
            embodiment="droid_differential_ik",
        )
    )
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(
            num_envs=1,
            presets="physx",
            solve_relations=True,
        ),
    )
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env = builder.make_registered(env_cfg, env_kwargs)

    try:
        env.reset()
        unwrapped = env.unwrapped
        bear_description = arena_env.scene.assets["pick_object"]
        table_description = arena_env.scene.assets["table"]
        bear = unwrapped.scene["pick_object"]

        bear_bbox = bear_description.get_bounding_box().to(unwrapped.device)
        table_bbox = table_description.get_bounding_box().to(unwrapped.device)
        bear_layout_pose = _env_zero_pose(bear_description)
        table_pose_w = unwrapped.arena_world.get_pose_w("table")[0, :3]
        table_top_w = table_pose_w[2] + table_bbox.max_point[0, 2]
        predicted_bear_bottom_w = bear_layout_pose.position_xyz[2] + bear_bbox.min_point[0, 2].item()
        predicted_clearance = predicted_bear_bottom_w - table_top_w.item()

        print(f"inferred_bear_bbox_min={bear_bbox.min_point[0].tolist()}")
        print(f"inferred_bear_bbox_max={bear_bbox.max_point[0].tolist()}")
        print(f"solved_bear_pose={bear_layout_pose}")
        print(f"table_top_w={table_top_w.item():.6f}")
        print(f"solver_predicted_bottom_w={predicted_bear_bottom_w:.6f}")
        print(f"solver_predicted_clearance={predicted_clearance:.6f}")

        minimum_clearance = float("inf")
        initial_clearance = None
        initial_offset_error = None
        sim_dt = unwrapped.sim.get_physics_dt()

        with torch.inference_mode():
            for step in range(args.diagnostic_steps + 1):
                nodal_pos_w = bear.data.nodal_pos_w.torch[0]
                centroid_z = nodal_pos_w[:, 2].mean()
                nodal_min_z = nodal_pos_w[:, 2].min()
                actual_clearance = (nodal_min_z - table_top_w).item()
                nodal_min_from_centroid = (nodal_min_z - centroid_z).item()
                bbox_min_from_layout_origin = bear_bbox.min_point[0, 2].item()
                offset_error = nodal_min_from_centroid - bbox_min_from_layout_origin
                minimum_clearance = min(minimum_clearance, actual_clearance)

                if step == 0:
                    initial_clearance = actual_clearance
                    initial_offset_error = offset_error
                if step == 0 or step == args.diagnostic_steps or step % args.report_every == 0:
                    print(
                        f"step={step:04d} centroid_z={centroid_z.item():.6f} "
                        f"nodal_min_z={nodal_min_z.item():.6f} clearance={actual_clearance:.6f} "
                        f"nodal_min_from_centroid={nodal_min_from_centroid:.6f} "
                        f"bbox_min_from_layout_origin={bbox_min_from_layout_origin:.6f}"
                    )

                if step < args.diagnostic_steps:
                    unwrapped.scene.write_data_to_sim()
                    unwrapped.sim.step(render=False)
                    unwrapped.scene.update(sim_dt)

        assert initial_clearance is not None and initial_offset_error is not None
        tolerance = args.penetration_tolerance
        print(f"minimum_clearance={minimum_clearance:.6f}")
        if predicted_clearance >= -tolerance and initial_clearance < -tolerance:
            print(
                "ROOT_CAUSE=placement uses USD-root-relative bounds, but deformable pose writes interpret the "
                "solved position as a nodal-centroid target"
            )
        elif initial_clearance >= -tolerance and minimum_clearance < -tolerance:
            print("ROOT_CAUSE=the initially valid placement penetrates only after PhysX simulation")
        elif minimum_clearance < -tolerance:
            print("ROOT_CAUSE=the relation solver itself planned a penetrating placement")
        else:
            print("NOT_REPRODUCED=no table penetration exceeded the configured tolerance")
    finally:
        env.close()


def main() -> None:
    """Launch Isaac Sim and run the placement diagnostic."""
    args = _parse_args()
    with SimulationAppContext(args):
        _run_diagnostic(args)


if __name__ == "__main__":
    main()
