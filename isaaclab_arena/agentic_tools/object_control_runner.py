# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

r"""Launch an Arena environment and execute an object-control demo.

Run a complete checked-in JSONL recipe inside the Arena container::

    isaaclab.sh -p isaaclab_arena/agentic_tools/object_control_runner.py \
      --env_spec isaaclab_arena_environments/kitchen_bench/\
kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.yaml \
      --commands isaaclab_arena_environments/kitchen_bench/\
kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2_object_control.jsonl \
      --output artifacts/object_control/open_cabinet \
      --visualizer kit

For live mode, create the JSONL file first, including any ``setup_joint`` records, and add
``--follow_commands``. Append newline-terminated runtime commands from another shell; the runner
waits for each complete line and stops on ``{"action":"finish"}``::

    isaaclab.sh -p isaaclab_arena/agentic_tools/object_control_runner.py \
      --env_spec isaaclab_arena_environments/kitchen_bench/\
kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.yaml \
      --commands /tmp/object_control.jsonl --follow_commands --visualizer kit

Then append commands from another shell::

    printf '%s\n' '{"action":"inspect"}' '{"action":"finish"}' >> /tmp/object_control.jsonl

Viewport recording is enabled by default; pass ``--no_record`` to disable it.
Success prints ``TASK_SUCCESS`` and writes ``validation.json`` with ``"success": [true]``.
Failure exits nonzero after printing ``OBJECT_CONTROL_FAILURE`` and writes a
``failure_<reason>_...json`` diagnostic, plus a PNG when viewport capture is available. Recorded
runs also produce ``demo.mp4`` and progress/state artifacts under ``--output``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from isaaclab_arena.agentic_tools.object_control.command_executor import (
    ObjectCommandExecutor,
    configure_object_command_executor,
)
from isaaclab_arena.agentic_tools.object_control.object_control_recipe import (
    FinishCommand,
    ObjectControlRecipe,
    RuntimeCommand,
    SetupJointCommand,
    load_object_control_recipe,
    parse_object_control_line,
    read_recipe_lines,
)
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from isaaclab_arena.utils.timer import Timer, get_timer_stats_json
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser


def main() -> None:
    """Launch an Arena environment and execute an object-control demo."""
    started = time.perf_counter()
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

    startup_started = time.perf_counter()
    with SimulationAppContext(app_args):
        startup_seconds = time.perf_counter() - startup_started
        try:
            with Timer("scene_spec"):
                parser = get_isaaclab_arena_environments_cli_parser(parser)
                args, hydra_overrides = parser.parse_known_args(original_cli_args)
                assert_hydra_overrides(hydra_overrides, parser)
                _assert_runner_args(args)
                recipe = load_object_control_recipe(args.commands, require_runtime_commands=not args.follow_commands)
                builder = get_arena_builder_from_cli(args, hydra_overrides=hydra_overrides)
                assert builder.resolved_physics_backend is PhysicsBackend.PHYSX, "object_control_runner requires PhysX"
                from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

                # This runner starts exactly one episode; unused pool layouts only add solver work.
                builder.arena_env.placer_params = replace(
                    builder.arena_env.placer_params or ObjectPlacerParams(), min_unique_layouts_per_env=1
                )
            _run_recipe(builder, args, recipe)
        finally:
            app_args.output.mkdir(parents=True, exist_ok=True)
            timings = {
                "simulation_app_startup_s": startup_seconds,
                "main_wall_time_s": time.perf_counter() - started,
                "timers": get_timer_stats_json("object_control_runner"),
            }
            (app_args.output / "timings.json").write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
            print("OBJECT_CONTROL_TIMINGS", json.dumps(timings), flush=True)


def _assert_runner_args(args: argparse.Namespace) -> None:
    assert args.num_envs == 1, "object_control_runner supports exactly one environment"
    assert not args.distributed, "object_control_runner does not support distributed execution"
    assert not args.list_variations, "object_control_runner does not support --list_variations"
    if not args.no_record:
        assert args.visualizer is not None and "kit" in args.visualizer, "recording requires the Kit visualizer"


def _add_runner_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Object Control Runner")
    group.add_argument("--commands", type=Path, required=True, help="JSONL object-control recipe.")
    group.add_argument("--output", type=Path, default=Path("artifacts/object_control"))
    group.add_argument("--follow_commands", action="store_true", help="Wait for commands appended to the JSONL file.")
    group.add_argument("--no_record", action="store_true", help="Do not capture viewport video or marker images.")
    group.add_argument("--allow_incomplete", action="store_true", help="Write a report without requiring task success.")


def _follow_commands(
    path: Path,
    render: Callable[[], None],
    startup_joint_setups: tuple[SetupJointCommand, ...],
) -> Iterator[RuntimeCommand]:
    """Yield complete validated commands as they are appended to a JSONL recipe."""
    cursor = 0
    runtime_command_seen = False
    print("OBJECT_CONTROL_READY", str(path), flush=True)
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
        command = parse_object_control_line(line, line_number)
        if isinstance(command, SetupJointCommand):
            assert not runtime_command_seen, "setup_joint commands must precede runtime commands"
            assert (
                command in startup_joint_setups
            ), f"Recipe line {line_number}: add setup_joint before starting the runner"
            continue
        runtime_command_seen = True
        yield command
        if isinstance(command, FinishCommand):
            return
        print("OBJECT_CONTROL_DONE", cursor, flush=True)


def _run_recipe(builder, args: argparse.Namespace, recipe: ObjectControlRecipe) -> None:
    """Build one environment, execute its recipe, and always close it."""
    with Timer("scene_config"):
        env_cfg, env_kwargs = builder.build_env_cfg()
    _configure_demo_timing(env_cfg)
    env_cfg.sim.enable_scene_query_support = True
    env_cfg.recorders = {}
    env_cfg.episode_recorders = {}
    configure_object_command_executor(env_cfg, recipe.joint_setups)
    env_cfg.video_recorders = []
    if not args.no_record:
        from isaaclab_arena.agentic_tools.object_control.object_control_video import configure_video

        configure_video(env_cfg, args.output)
    with Timer("scene_loading"):
        env = builder.make_registered(env_cfg, env_kwargs)
    try:
        env.unwrapped.configure_episode_limit(1)
        with Timer("scene_reset"):
            env.reset()
        commands = (
            _follow_commands(args.commands, env.unwrapped.sim.render, recipe.joint_setups)
            if args.follow_commands
            else recipe.commands
        )
        report_mode = "followed_object_commands" if args.follow_commands else "object_control_recipe"
        with Timer("force_demo"):
            ObjectCommandExecutor(
                env,
                args.output,
                record_video=not args.no_record,
                allow_incomplete=args.allow_incomplete,
            ).run(commands, recipe.joint_setups, report_mode)
    finally:
        with Timer("environment_close"):
            env.close()


def _configure_demo_timing(env_cfg) -> None:
    """Use 120 Hz physics and control, preserving configured temporal hold durations."""
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    previous_step_dt = env_cfg.sim.dt * env_cfg.decimation
    # Update pose targets every physics tick and honor recipe step budgets exactly. At the default
    # 15 Hz control rate, targets change only every eight ticks and 60-step waits round up to 64;
    # both changes altered banana contacts enough to fail the placement support predicate.
    env_cfg.sim.dt = 1 / 120
    env_cfg.decimation = 1
    steps_per_previous_step = previous_step_dt / env_cfg.sim.dt
    if env_cfg.task_progress is None:
        return
    # The task may reuse these configs elsewhere; adjust only this demo's copy.
    env_cfg.task_progress = deepcopy(env_cfg.task_progress)
    temporal_predicates = {}
    for criteria in env_cfg.task_progress.success_criteria:
        for sequence in criteria.canonical_predicate_sequences.values():
            for predicate, _ in sequence:
                if isinstance(predicate, TrueForConsecutiveStepsCfg):
                    temporal_predicates[id(predicate)] = predicate
    for event in env_cfg.task_progress.tracked_events:
        for predicate in event.predicate_sequence:
            if isinstance(predicate, TrueForConsecutiveStepsCfg):
                temporal_predicates[id(predicate)] = predicate
    for predicate in temporal_predicates.values():
        predicate.required_steps = max(1, math.ceil(predicate.required_steps * steps_per_previous_step))


if __name__ == "__main__":
    main()
