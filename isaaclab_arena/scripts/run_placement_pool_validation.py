# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline placement-pool validator.

Builds any registered (or graph-spec) Arena environment, then physics-validates every candidate layout
its placement pool holds and logs the validation results. Exits non-zero when any layout fails its
required checks or does not settle.

Run inside the container (example with a registered example environment):

    python isaaclab_arena/scripts/run_placement_pool_validation.py \\
        --num_envs 4 <example_environment_name>

Or against a graph spec:

    python isaaclab_arena/scripts/run_placement_pool_validation.py \\
        --num_envs 4 --env_spec path/to/spec.yaml
"""

from __future__ import annotations

import argparse
import math

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def add_pool_validation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add settle-check tuning flags. Registered before the env subparser so they stay top-level."""
    group = parser.add_argument_group("Pool Validation Arguments")
    group.add_argument(
        "--settle_time_s",
        type=float,
        default=2.0,
        help="Simulated time (s) to advance before reading back object velocities for each layout.",
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
            "Off by default; has no visible effect when no visualizer is selected."
        ),
    )


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    # Add our flags before the env subparser, which must be registered last (its subcommand flags parse
    # after the top-level ones).
    add_pool_validation_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        from isaaclab_arena.offline_placement.pool_validation import (
            do_all_layouts_pass_and_settle,
            print_validation_results,
            validate_pool_layouts,
        )
        from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_cli = args_parser.parse_args()
        assert args_cli.settle_time_s > 0, f"--settle_time_s must be positive, got {args_cli.settle_time_s}."

        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()
        # Reset starts the sim timeline (so physics can be stepped) and applies one layout per env; the
        # validator overwrites poses per candidate, so that initial layout does not bias the results.
        env.reset()

        step_dt = env.unwrapped.step_dt
        settle_steps = math.ceil(args_cli.settle_time_s / step_dt)
        print(f"Settling each layout for {settle_steps} environment steps ({settle_steps * step_dt:.2f} s).")
        settle_params = PhysicsSettleParams(
            num_steps=settle_steps,
            lin_vel_thresh=args_cli.lin_vel_thresh,
            ang_vel_thresh=args_cli.ang_vel_thresh,
        )
        validation_results = validate_pool_layouts(env, settle_params=settle_params, render=args_cli.render)
        assert validation_results is not None, (
            "The selected environment has no pooled placement, so there are no candidates to validate. "
            "Recorded replay bypasses the pool; omit recorded placement replay to validate solver candidates. "
            "Pooled placement also requires placement relations and resolve_on_reset=True."
        )
        print_validation_results(validation_results)
        all_layouts_valid = do_all_layouts_pass_and_settle(validation_results)

        env.close()
        # Raising inside the app context exits non-zero; a normal exit can report success regardless.
        if not all_layouts_valid:
            raise RuntimeError("Some placement pool layouts failed validation; see the summary above.")


if __name__ == "__main__":
    main()
