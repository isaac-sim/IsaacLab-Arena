# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

r"""Launch an Arena environment and execute a physical force-control demo.

Run a complete checked-in JSONL recipe inside the Arena container::

    isaaclab.sh -p isaaclab_arena_examples/agentic_environment_generation/force_demo_runner.py \
      --env_spec isaaclab_arena_environments/kitchen_bench/\
kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.yaml \
      --commands isaaclab_arena_environments/kitchen_bench/\
kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2_force_control.jsonl \
      --output artifacts/force_control/open_cabinet \
      --visualizer kit

For live mode, create the JSONL file first, including any ``setup_joint`` records, and add
``--follow_commands``. Append newline-terminated runtime commands from another shell; the runner
waits for each complete line and stops on ``{"action":"finish"}``::

    isaaclab.sh -p isaaclab_arena_examples/agentic_environment_generation/force_demo_runner.py \
      --env_spec isaaclab_arena_environments/kitchen_bench/\
kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.yaml \
      --commands /tmp/force_demo.jsonl --follow_commands --visualizer kit

Then append commands from another shell::

    printf '%s\n' '{"action":"inspect"}' '{"action":"finish"}' >> /tmp/force_demo.jsonl

Success prints ``TASK_SUCCESS`` and writes ``validation.json`` with ``"success": [true]``.
Failure exits nonzero after printing ``FORCE_CONTROL_FAILURE`` and writes a
``failure_<reason>_...json`` diagnostic, plus a PNG when viewport capture is available. Recorded
runs also produce ``demo.mp4`` and progress/state artifacts under ``--output``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser
from isaaclab_arena_examples.agentic_environment_generation.force_demo.force_control_recipe import (
    ForceControlRecipe,
    load_force_control_recipe,
    read_recipe_lines,
    validate_force_control_command,
)
from isaaclab_arena_examples.agentic_environment_generation.force_demo.physical_command_runner import (
    PhysicalCommandRunner,
    configure_physical_command_runner,
)


def main() -> None:
    """Launch an Arena environment and execute a physical force-control demo."""
    parser = get_isaaclab_arena_cli_parser()
    _add_runner_args(parser)
    parser.set_defaults(device="cpu", visualizer=["kit"])
    parser.allow_abbrev = False
    original_cli_args = sys.argv[1:]
    app_args, _ = parser.parse_known_args(original_cli_args)
    _assert_runner_args(app_args)
    # Load RTX/Replicator for Lab's Kit recorder without adding scene cameras.
    if not app_args.no_record:
        app_args.enable_cameras = True

    with SimulationAppContext(app_args):
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        args, hydra_overrides = parser.parse_known_args(original_cli_args)
        assert_hydra_overrides(hydra_overrides, parser)
        _assert_runner_args(args)
        recipe = load_force_control_recipe(args.commands, require_runtime_commands=not args.follow_commands)
        builder = get_arena_builder_from_cli(args, hydra_overrides=hydra_overrides)
        assert builder.resolved_physics_backend is PhysicsBackend.PHYSX, "force_demo_runner requires PhysX"
        _run_recipe(builder, args, recipe)


def _assert_runner_args(args: argparse.Namespace) -> None:
    assert args.num_envs == 1, "force_demo_runner supports exactly one environment"
    assert not args.distributed, "force_demo_runner does not support distributed execution"
    assert not args.list_variations, "force_demo_runner does not support --list_variations"
    if not args.no_record:
        assert args.visualizer is not None and "kit" in args.visualizer, "recording requires the Kit visualizer"


def _add_runner_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Physical Force-Control Demo Runner")
    group.add_argument("--commands", type=Path, required=True, help="JSONL force-control recipe.")
    group.add_argument("--output", type=Path, default=Path("artifacts/force_control"))
    group.add_argument("--follow_commands", action="store_true", help="Wait for commands appended to the JSONL file.")
    group.add_argument("--no_record", action="store_true", help="Do not capture viewport video or marker images.")
    group.add_argument("--allow_incomplete", action="store_true", help="Write a report without requiring task success.")


def _follow_commands(path: Path, render: Callable[[], None]) -> Iterator[dict]:
    """Yield complete validated commands as they are appended to a JSONL recipe."""
    cursor = 0
    runtime_command_seen = False
    print("FORCE_CONTROL_READY", str(path), flush=True)
    while True:
        lines = read_recipe_lines(path, complete_only=True) if path.exists() else []
        if cursor == len(lines):
            render()
            time.sleep(0.05)
            continue
        line_number = cursor + 1
        line = lines[cursor]
        cursor += 1
        if not line.strip():
            continue
        command = validate_force_control_command(json.loads(line), line_number)
        if command["action"] == "setup_joint":
            assert not runtime_command_seen, "setup_joint commands must precede runtime commands"
            continue
        runtime_command_seen = True
        yield command
        if command["action"] == "finish":
            return
        print("FORCE_CONTROL_DONE", cursor, flush=True)


def _run_recipe(builder, args: argparse.Namespace, recipe: ForceControlRecipe) -> None:
    """Build one environment, execute its recipe, and always close it."""
    env_cfg, env_kwargs = builder.build_env_cfg()
    env_cfg.sim.enable_scene_query_support = True
    env_cfg.recorders = {}
    env_cfg.episode_recorders = {}
    configure_physical_command_runner(env_cfg, recipe.joint_setups)
    env_cfg.video_recorders = []
    if not args.no_record:
        from isaaclab_arena_examples.agentic_environment_generation.force_demo.force_control_video import (
            configure_video,
        )

        configure_video(env_cfg, args.output)
    env = builder.make_registered(env_cfg, env_kwargs)
    try:
        env.unwrapped.configure_episode_limit(1)
        env.reset()
        commands = (
            _follow_commands(args.commands, env.unwrapped.sim.render) if args.follow_commands else recipe.commands
        )
        report_mode = "followed_force_commands" if args.follow_commands else "force_control_recipe"
        PhysicalCommandRunner(env, args).run(commands, recipe.joint_setups, report_mode)
    finally:
        env.close()


if __name__ == "__main__":
    main()
