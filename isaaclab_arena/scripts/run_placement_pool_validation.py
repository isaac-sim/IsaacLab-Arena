# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline placement-pool validator.

Builds any registered (or graph-spec) Arena environment, then physics-validates every candidate layout
its placement pool holds and logs the per-candidate result. This is a standalone diagnostic, distinct
from the rollout-time settle check in the policy runner: that one verifies and re-selects only the layout
applied to each env on reset, whereas this sweeps the whole pool so you can see which candidates are
physically stable before running an evaluation.

Run inside the container (example with a registered example environment):

    /isaac-sim/python.sh isaaclab_arena/scripts/run_placement_pool_validation.py --headless \\
        --num_envs 4 <example_environment_name>

Or against a graph spec:

    /isaac-sim/python.sh isaaclab_arena/scripts/run_placement_pool_validation.py --headless \\
        --num_envs 4 --env_graph_spec_yaml path/to/spec.yaml
"""

from __future__ import annotations

import argparse

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def add_pool_validation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add settle-check tuning flags. Registered before the env subparser so they stay top-level."""
    group = parser.add_argument_group("Pool Validation Arguments")
    group.add_argument(
        "--settle_steps",
        type=int,
        default=5,
        help=(
            "Environment steps to advance before reading back object velocities for each layout. "
            "Converted to physics substeps internally (x the env's decimation)."
        ),
    )
    group.add_argument(
        "--lin_vel_thresh",
        type=float,
        default=0.1,
        help="Max per-object linear speed (m/s) for a layout to count as settled.",
    )
    group.add_argument(
        "--ang_vel_thresh",
        type=float,
        default=0.1,
        help="Max per-object angular speed (rad/s) for a layout to count as settled.",
    )
    group.add_argument(
        "--render",
        action="store_true",
        help=(
            "Render each settle step so the sweep is visible in the GUI (pair with --viz kit). "
            "Off by default; has no visible effect under --headless."
        ),
    )


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    # Add our flags before the env subparser, which must be registered last (its subcommand flags parse
    # after the top-level ones).
    add_pool_validation_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
        from isaaclab_arena.relations.placement_pool_validation import log_validation_results, validate_pool_layouts
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

        settle_params = PhysicsSettleParams(
            num_steps=args_cli.settle_steps,
            lin_vel_thresh=args_cli.lin_vel_thresh,
            ang_vel_thresh=args_cli.ang_vel_thresh,
        )
        results = validate_pool_layouts(env, settle_params=settle_params, render=args_cli.render)
        assert results is not None, (
            "The selected environment has no pooled placement, so there are no candidates to validate. "
            "Pooled placement is created when objects declare placement relations (e.g. On)."
        )
        log_validation_results(results)

        env.close()


if __name__ == "__main__":
    main()
