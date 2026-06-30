# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline IK-reachability validator for pooled placement layouts.

Builds any registered (or graph-spec) Arena environment, then asks cuRobo whether the robot can
reach a top-down grasp at every movable object of each candidate layout, logging the verdict.
Requires the cuRobo-enabled image (build with ``./docker/run_docker.sh -c``).

Run inside the container (example with a registered example environment):

    /isaac-sim/python.sh isaaclab_arena_curobo/scripts/run_placement_pool_ik_validation.py --headless \\
        --num_envs 4 <example_environment_name>

Or against a graph spec:

    /isaac-sim/python.sh isaaclab_arena_curobo/scripts/run_placement_pool_ik_validation.py --headless \\
        --num_envs 4 --env_graph_spec_yaml path/to/spec.yaml
"""

from __future__ import annotations

import argparse

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def add_ik_validation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add IK-reachability tuning flags. Registered before the env subparser so they stay top-level."""
    group = parser.add_argument_group("IK Validation Arguments")
    group.add_argument(
        "--grasp_z_offset",
        type=float,
        default=0.02,
        help="Height (m) above each object center for the top-down grasp pose.",
    )
    group.add_argument(
        "--ik_pos_threshold",
        type=float,
        default=0.01,
        help="Max IK position error (m) for a grasp to count as reachable.",
    )
    group.add_argument(
        "--ik_rot_threshold",
        type=float,
        default=0.1,
        help="Max IK rotation error (rad) for a grasp to count as reachable.",
    )


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    # Add our flags before the env subparser, which must be registered last (its subcommand flags parse
    # after the top-level ones).
    add_ik_validation_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        from isaaclab_arena_curobo.curobo_planner_utils import make_curobo_planner
        from isaaclab_arena_curobo.placement_pool_ik_validation import print_ik_validation_results, validate_pool_ik
        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_cli = args_parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()
        # Reset starts the sim timeline (so physics can be stepped) and applies one layout per env; the
        # validator overwrites poses per candidate, so that initial layout does not bias the results.
        env.reset()

        embodiment = arena_builder.arena_env.embodiment
        assert embodiment is not None, "IK validation requires an environment with a robot embodiment."
        # CuroboPlanner is single-env: one collision world bound to one env, so we serialize every candidate through env 0.
        planner = make_curobo_planner(env.unwrapped, embodiment, env_id=0)

        validation_results = validate_pool_ik(
            env,
            planner,
            grasp_z_offset=args_cli.grasp_z_offset,
            ik_pos_threshold=args_cli.ik_pos_threshold,
            ik_rot_threshold=args_cli.ik_rot_threshold,
        )
        assert validation_results is not None, (
            "The selected environment has no pooled placement, so there are no candidates to validate. "
            "Pooled placement is created when objects declare placement relations (e.g. On)."
        )
        print_ik_validation_results(validation_results)

        env.close()


if __name__ == "__main__":
    main()
