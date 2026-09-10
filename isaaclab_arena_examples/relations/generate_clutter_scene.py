# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate a physics-settled scene YAML from an Arena graph with cluttered_on relations."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def generate_scene(args: argparse.Namespace) -> list[Path]:
    """Build and settle the input graph, then save one prepared layout per environment."""
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import (
        build_arena_env_with_assets_from_graph_spec,
    )
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_events import get_placement_pool
    from isaaclab_arena.relations.settled_scene import settled_scene_spec

    for entry_point in args.register:
        module_name, separator, function_name = entry_point.partition(":")
        assert separator and module_name and function_name, "--register expects module:function"
        getattr(importlib.import_module(module_name), function_name)()
    spec = ArenaEnvGraphSpec.from_yaml(args.env_spec)
    assert any(r.kind == "cluttered_on" for r in spec.relations), "Input must declare cluttered_on relations"
    assert not spec.object_sets, "Fixed scene export currently requires concrete objects, not object sets"
    output = Path(args.output)
    paths = (
        [output]
        if args.num_envs == 1
        else [output.with_name(f"{output.stem}_env_{i}{output.suffix}") for i in range(args.num_envs)]
    )
    assert all(not path.exists() for path in paths), "Output already exists; choose a new output path"
    arena_env, assets_by_node_id = build_arena_env_with_assets_from_graph_spec(spec)
    arena_env.placer_params.min_unique_layouts_per_env = args.layouts_per_env
    arena_env.placer_params.allow_best_loss_fallbacks = False
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(
            num_envs=args.num_envs,
            placement_seed=args.seed,
            seed=args.seed,
            device=args.device,
            presets=args.presets,
        ),
    )
    cfg, kwargs = builder.compose_manager_cfg()
    env = builder.make_registered(env_cfg=cfg, env_kwargs=kwargs)
    try:
        pool = get_placement_pool(env)
        assert pool is not None, "Clutter placement has no layout pool"
        layouts = pool.sample_for_envs(list(range(args.num_envs)))
        # Validate all outputs before writing any file.
        scenes = [settled_scene_spec(spec, layouts[i], assets_by_node_id) for i in range(args.num_envs)]
        for path, scene in zip(paths, scenes):
            path.parent.mkdir(parents=True, exist_ok=True)
            scene.write_yaml(path)
            print(f"Saved settled scene: {path}")
        return paths
    finally:
        env.close()


def main() -> None:
    """Parse arguments, launch Isaac Sim, and export the prepared scenes."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_spec", type=Path, required=True, help="Arena environment graph YAML")
    parser.add_argument("--output", type=Path, required=True, help="Output scene.yaml path")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--presets", choices=("physx", "newton"), default=None)
    parser.add_argument(
        "--register", action="append", default=[], help="Optional component registration module:function"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layouts_per_env", type=int, default=5, help="Candidate piles per environment before rejection"
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    assert args.num_envs > 0 and args.layouts_per_env > 0, "Environment and layout counts must be positive"
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(args):
        generate_scene(args)


if __name__ == "__main__":
    main()
