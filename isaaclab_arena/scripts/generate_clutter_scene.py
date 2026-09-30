# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate offline clutter placements with configurable post-physics checks."""

import argparse
from pathlib import Path

from isaaclab_arena.offline_placement.clutter_generation import (
    ClutterGenerationCfg,
    generate_clutter_layouts,
    load_generation_config,
)


def parse_generation_args() -> tuple[argparse.Namespace, ClutterGenerationCfg]:
    """Parse launcher flags and Hydra generation settings."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    launcher_args, overrides = parser.parse_known_args()
    assert_hydra_overrides(overrides, parser)
    cfg = load_generation_config(overrides)
    assert not Path(cfg.output).exists(), f"Output already exists: {cfg.output}"
    assert cfg.env_spec is not None, "Set env_spec to the source environment YAML"
    return launcher_args, cfg


def main() -> None:
    """Generate clutter records from an environment YAML."""
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    launcher_args, cfg = parse_generation_args()
    with SimulationAppContext(launcher_args):
        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

        arena_env = ArenaEnvGraphSpec.from_yaml(cfg.env_spec).to_arena_env()
        generate_clutter_layouts(arena_env, cfg, device=launcher_args.device)


if __name__ == "__main__":
    main()
