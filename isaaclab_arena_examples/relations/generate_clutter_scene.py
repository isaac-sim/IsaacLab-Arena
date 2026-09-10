# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Settle clutter offline and save fixed scene YAML files for ordinary runtime loading."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def generate_scene(args: argparse.Namespace) -> list[Path]:
    """Generate one cached layout per environment from a concrete scene graph."""
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import (
        build_arena_env_with_assets_from_graph_spec,
    )
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_examples.relations.clutter.cache import (
        scene_with_cached_poses,
        validate_cache_directory,
        write_scene_cache,
    )
    from isaaclab_arena_examples.relations.clutter.drop_poses import DropOrder
    from isaaclab_arena_examples.relations.clutter.geometry import dynamic_rigid_object_keys
    from isaaclab_arena_examples.relations.clutter.settle import ClutterGroup, settle_clutter
    from isaaclab_arena_examples.relations.clutter.validation import ClutterSettleParams

    for entry_point in args.register:
        module_name, separator, function_name = entry_point.partition(":")
        assert separator and module_name and function_name, "--register expects module:function"
        getattr(importlib.import_module(module_name), function_name)()
    spec = ArenaEnvGraphSpec.from_yaml(args.env_spec)
    assert not spec.relations, "Input must use concrete poses; resolve placement relations before offline settling"
    assert not spec.object_sets, "Resolve object sets to concrete assets before offline settling"
    paths = _output_paths(Path(args.output), args.num_envs)
    validate_cache_directory(paths[0].parent)
    arena_env, assets = build_arena_env_with_assets_from_graph_spec(spec)
    assert args.support in assets, f"Unknown support node {args.support!r}"
    assert all(key in assets for key in args.objects), f"Unknown clutter nodes: {set(args.objects) - set(assets)}"
    group = ClutterGroup(
        support=assets[args.support].get_scene_key(),
        objects=tuple(assets[key].get_scene_key() for key in args.objects),
        spread=args.spread,
        gap_m=args.gap_m,
        clearance_m=args.clearance_m,
        random_yaw=not args.keep_rotation,
        drop_order=DropOrder(args.drop_order),
    )
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
        exportable = {assets[node.id].get_scene_key() for node in [spec.background, spec.embodiment, *spec.objects]}
        unsupported = set(dynamic_rigid_object_keys(env.unwrapped.scene)) - exportable
        assert not unsupported, f"Cannot cache rigid objects without concrete graph nodes: {unsupported}"
        env.reset()
        layouts = settle_clutter(
            env,
            [group],
            seed=args.seed,
            attempts=args.attempts,
            params=ClutterSettleParams(
                timeout_s=args.timeout_s,
                poll_interval_s=args.poll_interval_s,
                move_thresh_m=args.move_thresh_m,
                turn_thresh_deg=args.turn_thresh_deg,
                required_quiet_windows=args.required_quiet_windows,
                fall_through_tolerance_m=args.fall_through_tolerance_m,
                containment_margin_m=args.containment_margin_m,
                passive_move_thresh_m=args.passive_move_thresh_m,
                passive_turn_thresh_deg=args.passive_turn_thresh_deg,
            ),
        )
        # Serialize every result before writing any file.
        scenes = [scene_with_cached_poses(spec, poses, assets) for poses in layouts]
        for path, scene in zip(paths, scenes):
            write_scene_cache(scene, path)
            print(f"Saved settled scene: {path}")
        return paths
    finally:
        env.close()


def _output_paths(output: Path, num_envs: int) -> list[Path]:
    """Resolve cache filenames and reject existing outputs."""
    paths = (
        [output]
        if num_envs == 1
        else [output.with_name(f"{output.stem}_env_{i}{output.suffix}") for i in range(num_envs)]
    )
    existing = [str(path) for path in paths if path.exists()]
    assert not existing, f"Output already exists: {existing}"
    return paths


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
    parser.add_argument("--output", type=Path, required=True, help="Output scene.yaml path")
    parser.add_argument("--support", required=True, help="Graph node identifying the support")
    parser.add_argument("--objects", nargs="+", required=True, help="Graph nodes identifying clutter objects")
    parser.add_argument("--num_envs", type=_positive_int, default=1, help="Independent offline layouts to generate")
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
    parser.add_argument("--spread", type=float, default=1.0, help="Fraction of support width/depth used for releases")
    parser.add_argument("--gap_m", type=float, default=0.03, help="Vertical gap above overlapping release footprints")
    parser.add_argument("--clearance_m", type=float, default=0.01, help="Initial clearance above the support")
    parser.add_argument("--keep_rotation", action="store_true", help="Retain authored rotations without sampling yaw")
    parser.add_argument(
        "--drop_order",
        choices=("as_listed", "flattest_first", "shuffle"),
        default="as_listed",
        help="Release planning order",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    _output_paths(args.output, args.num_envs)
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(args):
        generate_scene(args)


if __name__ == "__main__":
    main()
