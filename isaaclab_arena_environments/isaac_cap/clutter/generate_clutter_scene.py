# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Settle ClutterOn objects offline and save a companion placement-layout YAML."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def generate_scene(args: argparse.Namespace) -> Path:
    """Generate a companion pose cache from an environment's ClutterOn relations."""
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import (
        build_arena_env_with_assets_from_graph_spec,
    )
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena_environments.isaac_cap.clutter.geometry import dynamic_rigid_object_keys
    from isaaclab_arena_environments.isaac_cap.clutter.settle import groups_from_assets, settle_clutter
    from isaaclab_arena_environments.isaac_cap.clutter.validation import ClutterSettleParams

    output = Path(args.output)
    assert not output.exists(), f"Output already exists: {output}"
    for entry_point in args.register:
        module_name, separator, function_name = entry_point.partition(":")
        assert separator and module_name and function_name, "--register expects module:function"
        getattr(importlib.import_module(module_name), function_name)()
    spec = ArenaEnvGraphSpec.from_yaml(args.env_spec)
    assert spec.placement_layouts is None, "Remove placement_layouts from the source environment before regenerating"
    assert not spec.object_sets, "Resolve object sets to concrete assets before offline settling"
    arena_env, assets = build_arena_env_with_assets_from_graph_spec(spec)
    groups_from_assets(list(assets.values()))
    params = ClutterSettleParams(
        timeout_s=args.timeout_s,
        poll_interval_s=args.poll_interval_s,
        move_thresh_m=args.move_thresh_m,
        turn_thresh_deg=args.turn_thresh_deg,
        required_quiet_windows=args.required_quiet_windows,
        fall_through_tolerance_m=args.fall_through_tolerance_m,
        containment_margin_m=args.containment_margin_m,
        passive_move_thresh_m=args.passive_move_thresh_m,
        passive_turn_thresh_deg=args.passive_turn_thresh_deg,
    )
    num_layouts = args.num_layouts if args.num_layouts is not None else args.num_envs
    assert (
        args.num_envs > 0 and num_layouts > 0 and args.attempts > 0
    ), "num_envs, num_layouts and attempts must be positive"
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(
            num_envs=args.num_envs,
            seed=args.seed,
            device=args.device,
            presets=args.presets,
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
        for start in range(0, num_layouts, args.num_envs):
            batch = settle_clutter(
                env,
                list(assets.values()),
                seed=args.seed + start * args.attempts * arena_env.placer_params.max_placement_attempts,
                attempts=args.attempts,
                params=params,
                placer_params=arena_env.placer_params,
            )
            layouts.extend(batch[: num_layouts - start])
        cache = PlacementLayouts({node_by_key[key]: [layout[key] for layout in layouts] for key in keys})
        cache.write_yaml(output)
        print(f"Saved {cache.num_layouts} settled layouts: {output}")
        return output
    finally:
        env.close()


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _registration_entry_point(value: str) -> str:
    module, separator, function = value.partition(":")
    if not module or not separator or not function:
        raise argparse.ArgumentTypeError("expected module:function")
    return value


def main() -> None:
    """Launch the offline generation process."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_spec", type=Path, required=True, help="Concrete Arena scene YAML")
    parser.add_argument("--output", type=Path, required=True, help="Output companion pose YAML")
    parser.add_argument(
        "--num_layouts", type=_positive_int, default=None, help="Total cached layouts; defaults to num_envs"
    )
    parser.add_argument("--num_envs", type=_positive_int, default=1, help="Parallel offline environments")
    parser.add_argument("--presets", choices=("physx", "newton"), default=None, help="Arena physics preset")
    parser.add_argument(
        "--register",
        type=_registration_entry_point,
        action="append",
        default=[],
        help="Optional component registration module:function",
    )
    parser.add_argument("--seed", type=int, default=42, help="Release sampling seed")
    parser.add_argument("--attempts", type=_positive_int, default=5, help="Maximum offline trials per environment")
    parser.add_argument("--timeout_s", type=float, default=10.0, help="Simulated seconds per trial")
    parser.add_argument("--poll_interval_s", type=float, default=0.4, help="Simulated seconds between pose checks")
    parser.add_argument("--move_thresh_m", type=float, default=0.002, help="Maximum motion per quiet window, in metres")
    parser.add_argument(
        "--turn_thresh_deg", type=float, default=2.0, help="Maximum rotation per quiet window, in degrees"
    )
    parser.add_argument(
        "--required_quiet_windows", type=_positive_int, default=2, help="Consecutive quiet windows required"
    )
    parser.add_argument(
        "--fall_through_tolerance_m", type=float, default=0.01, help="Allowed depth below the support, in metres"
    )
    parser.add_argument(
        "--containment_margin_m", type=float, default=0.0, help="Allowed overhang beyond the support, in metres"
    )
    parser.add_argument(
        "--passive_move_thresh_m",
        type=float,
        default=0.002,
        help="Maximum total neighbor/support/link drift, in metres",
    )
    parser.add_argument(
        "--passive_turn_thresh_deg", type=float, default=2.0, help="Maximum total passive rotation, in degrees"
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    assert not args.output.exists(), f"Output already exists: {args.output}"
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(args):
        generate_scene(args)


if __name__ == "__main__":
    main()
