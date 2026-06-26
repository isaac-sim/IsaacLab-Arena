# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Boot a registered (or graph-spec) Arena environment with a live viewer and idle-step it.

Unlike the placement-pool validator, this does no checking: it builds the environment, resolves the
scene on reset, and keeps stepping zero actions so the layout can be inspected and iterated on in the
GUI.

Run (GUI):

    python isaaclab_arena/scripts/view_environment.py --viz kit --num_envs 1 <example_environment_name>

Re-roll the randomized layout each reset with --resolve_on_reset, or pin it with --placement_seed.
"""

from __future__ import annotations

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli) as simulation_app:
        import torch
        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_cli = args_parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()
        env.reset()

        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        while simulation_app.is_running():
            with torch.inference_mode():
                env.step(action)

        env.close()


if __name__ == "__main__":
    main()
