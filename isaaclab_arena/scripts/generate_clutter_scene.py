# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate offline clutter placements with configurable post-physics checks."""

import argparse
import importlib
from pathlib import Path

from isaaclab_arena.offline_placement.clutter_generation import ClutterGenerationCfg, generate_clutter_layouts


def main() -> None:
    """Launch offline generation with Hydra overrides and Isaac Lab launcher flags."""
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore
    from isaaclab.app import AppLauncher
    from omegaconf import OmegaConf

    from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    launcher_args, overrides = parser.parse_known_args()
    assert_hydra_overrides(overrides, parser)
    ConfigStore.instance().store(name="clutter_generation", node=ClutterGenerationCfg)
    with initialize(version_base=None, config_path=None):
        cfg = OmegaConf.to_object(compose(config_name="clutter_generation", overrides=overrides))
    assert not Path(cfg.output).exists(), f"Output already exists: {cfg.output}"
    assert cfg.env_spec is not None, "Set env_spec to the source environment YAML"
    with SimulationAppContext(launcher_args):
        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

        for entry_point in cfg.register:
            module_name, separator, function_name = entry_point.partition(":")
            assert separator and module_name and function_name, "register entries must be module:function"
            getattr(importlib.import_module(module_name), function_name)()
        arena_env = ArenaEnvGraphSpec.from_yaml(cfg.env_spec).to_arena_env()
        generate_clutter_layouts(arena_env, cfg, device=launcher_args.device)


if __name__ == "__main__":
    main()
