# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time IK-reachability filtering for pooled placement, via a throwaway env.

Composes an Arena env cfg once, filters its placement pool down to IK-reachable layouts on a THROWAWAY
env (cuRobo IK + reject-&-refill), closes that env, then builds the real eval env from the SAME filtered
cfg. Because validation runs on the throwaway env, the eval env's recorder, pool cursor, and run-time
variation RNG start pristine -- the eval env only ever places layouts the robot can reach.

Requires the cuRobo-enabled image (build with ``./docker/run_docker.sh -c``).

Run inside the container (example with a registered example environment):

    /isaac-sim/python.sh isaaclab_arena_curobo/scripts/run_ik_filtered_rollout.py --headless \\
        --num_envs 4 --rollout_steps 30 <example_environment_name>

Or against a graph spec:

    /isaac-sim/python.sh isaaclab_arena_curobo/scripts/run_ik_filtered_rollout.py --headless \\
        --num_envs 4 --env_graph_spec_yaml path/to/spec.yaml
"""

from __future__ import annotations

import argparse

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def add_ik_filter_arguments(parser: argparse.ArgumentParser) -> None:
    """Add IK-filter tuning flags. Registered before the env subparser so they stay top-level."""
    group = parser.add_argument_group("IK Placement Filter Arguments")
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
    group.add_argument(
        "--target_reachable_per_env",
        type=int,
        default=None,
        help="Reachable layouts to retain per env. Defaults to the pool's per-env stored count.",
    )
    group.add_argument(
        "--max_refill_rounds",
        type=int,
        default=5,
        help="Max solve-and-recheck rounds before returning best-effort.",
    )
    group.add_argument(
        "--rollout_steps",
        type=int,
        default=0,
        help="Zero-action steps to run on the eval env as a smoke test (0 to skip).",
    )


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    # Add our flags before the env subparser, which must be registered last (its subcommand flags parse
    # after the top-level ones).
    add_ik_filter_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        import torch

        from isaaclab_arena.relations.placement_events import get_placement_pool, get_placement_pool_from_cfg
        from isaaclab_arena.relations.placement_validation import PlacementCheck
        from isaaclab_arena_curobo.curobo_planner_utils import make_curobo_planner
        from isaaclab_arena_curobo.placement_pool_ik_validation import filter_pool_by_ik_reachability
        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_cli = args_parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)
        embodiment = arena_builder.arena_env.embodiment
        assert embodiment is not None, "IK filtering requires an environment with a robot embodiment."

        # Compose the env cfg once; the placement pool is embedded in its placement-reset event. Filtering
        # this cfg's pool (below) is what makes every env later built from the same cfg draw only reachable
        # layouts -- regardless of whether env construction deep-copies the cfg.
        name, env_cfg, env_kwargs = arena_builder.build_registered()
        pool = get_placement_pool_from_cfg(env_cfg)
        assert pool is not None, (
            "The selected environment has no pooled placement, so there is nothing to filter. "
            "Pooled placement is created when objects declare placement relations (e.g. On)."
        )

        # Phase 1: filter the pool on a THROWAWAY env, then discard it. Reset starts the sim timeline so the
        # robot articulation and collision world initialize; the filter overwrites object poses per candidate.
        throwaway_env = arena_builder.make_registered(env_cfg, env_kwargs)
        throwaway_env.reset()
        # CuroboPlanner is single-env: one collision world bound to env 0, so every candidate is serialized there.
        planner = make_curobo_planner(throwaway_env.unwrapped, embodiment, env_id=0)
        filter_pool_by_ik_reachability(
            throwaway_env,
            planner,
            placement_pool=pool,
            target_reachable_per_env=args_cli.target_reachable_per_env,
            grasp_z_offset=args_cli.grasp_z_offset,
            ik_pos_threshold=args_cli.ik_pos_threshold,
            ik_rot_threshold=args_cli.ik_rot_threshold,
            max_refill_rounds=args_cli.max_refill_rounds,
        )
        throwaway_env.close()

        # Phase 2: build the real eval env from the SAME (now filtered) cfg and confirm the filtered pool
        # propagated -- every stored layout should be IK-reachable.
        eval_env = arena_builder.make_registered(env_cfg, env_kwargs)
        eval_pool = get_placement_pool(eval_env)
        assert eval_pool is not None, "Eval env unexpectedly has no placement pool."
        layouts_per_env = eval_pool.layouts_per_env()
        total = sum(len(layouts) for layouts in layouts_per_env)
        reachable = sum(
            1
            for layouts in layouts_per_env
            for layout in layouts
            if layout.validation_results.validation_results.get(PlacementCheck.IK_REACHABLE)
        )
        print(
            f"Eval env pool: {reachable}/{total} layouts IK-reachable; per-env counts "
            f"{[len(layouts) for layouts in layouts_per_env]}."
        )
        assert reachable == total, "Eval env pool still contains IK-unreachable layouts after filtering."

        # Phase 3: optional zero-action smoke rollout to confirm the filtered eval env resets and steps.
        if args_cli.rollout_steps > 0:
            eval_env.reset()
            num_envs = eval_env.unwrapped.num_envs
            action_dim = eval_env.action_space.shape[-1]
            zero_actions = torch.zeros((num_envs, action_dim), device=eval_env.unwrapped.device)
            for _ in range(args_cli.rollout_steps):
                eval_env.step(zero_actions)
            print(f"Smoke rollout: stepped the eval env {args_cli.rollout_steps} times with zero actions.")

        eval_env.close()


if __name__ == "__main__":
    main()
