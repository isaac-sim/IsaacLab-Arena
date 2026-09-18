# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Settle ClutterOn objects offline and save placement records in the episode variations envelope."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING

from isaaclab_arena.relations.clutter.validation import ClutterSettleParams


@dataclass
class ClutterGenerationCfg:
    """Offline scene source, output and sampling settings."""

    env_spec: str = MISSING
    """Environment graph YAML path."""
    output: str = MISSING
    """Output placement JSONL path; must not already exist."""
    num_envs: int = 1
    """Number of parallel physics environments."""
    num_layouts: int | None = None
    """Total layouts to generate; None generates one per environment."""
    seed: int = 42
    """Initial release-sampling seed."""
    attempts: int = 5
    """Maximum settling trials per environment."""
    presets: str | None = None
    """Physics backend override: physx or newton."""
    register: list[str] = field(default_factory=list)
    """Optional component registration entry points, each module:function."""
    settle: ClutterSettleParams = field(default_factory=ClutterSettleParams)
    """Physics time budget, rest thresholds and containment tolerances."""


def generate_scene(cfg: ClutterGenerationCfg, device: str = "cuda:0") -> Path:
    """Generate settled placement records and return the output path.

    Args:
        cfg: Scene source, output and settling settings.
        device: Simulation device, such as cuda:0 or cpu.
    """
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import (
        build_arena_env_with_assets_from_graph_spec,
    )
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.clutter.geometry import dynamic_rigid_object_keys
    from isaaclab_arena.relations.clutter.settle import groups_from_assets, settle_clutter
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts

    output = Path(cfg.output)
    assert not output.exists(), f"Output already exists: {output}"
    for entry_point in cfg.register:
        module_name, separator, function_name = entry_point.partition(":")
        assert separator and module_name and function_name, "register entries must be module:function"
        getattr(importlib.import_module(module_name), function_name)()
    spec = ArenaEnvGraphSpec.from_yaml(cfg.env_spec)
    assert not spec.object_sets, "Resolve object sets to concrete assets before offline settling"
    arena_env, assets = build_arena_env_with_assets_from_graph_spec(spec)
    groups_from_assets(list(assets.values()))
    num_layouts = cfg.num_layouts if cfg.num_layouts is not None else cfg.num_envs
    assert (
        cfg.num_envs > 0 and num_layouts > 0 and cfg.attempts > 0
    ), "num_envs, num_layouts and attempts must be positive"
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(
            num_envs=cfg.num_envs,
            seed=cfg.seed,
            device=device,
            presets=cfg.presets,
            solve_relations=False,
        ),
    )
    env = builder.make_registered()
    try:
        nodes = [spec.background, spec.embodiment, *spec.objects]
        node_by_key = {assets[node.id].get_scene_key(): node.id for node in nodes}
        assert len(node_by_key) == len(nodes), "Graph nodes must map to distinct scene keys"
        keys = dynamic_rigid_object_keys(env.unwrapped.scene)
        unsupported = set(keys) - set(node_by_key)
        assert not unsupported, f"Cannot cache rigid objects without concrete graph nodes: {unsupported}"
        env.reset()
        layouts = []
        for start in range(0, num_layouts, cfg.num_envs):
            batch = settle_clutter(
                env,
                list(assets.values()),
                seed=cfg.seed + start * cfg.attempts * arena_env.placer_params.max_placement_attempts,
                attempts=cfg.attempts,
                params=cfg.settle,
                placer_params=arena_env.placer_params,
            )
            layouts.extend(batch[: num_layouts - start])
        cache = PlacementLayouts({node_by_key[key]: [layout[key] for layout in layouts] for key in keys})
        cache.write_episode_jsonl(output)
        print(f"Saved {cache.num_layouts} settled layouts: {output}")
        return output
    finally:
        env.close()


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
    with SimulationAppContext(launcher_args):
        generate_scene(cfg, device=launcher_args.device)


if __name__ == "__main__":
    main()
