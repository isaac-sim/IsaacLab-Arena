# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end agentic environment generation and execution.

Usage::

    # Resolve an environment intent spec into an initial environment graph spec and a linked environment graph spec:
    python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py --mode resolve --prompt ...

    # Build a gym env from a linked environment graph spec YAML and run the zero-action policy:
    python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py --mode build --headless \\
        --num_envs 1 --linked_env_graph_spec_yaml <env>_linked.yaml

    # Resolve and build in one process:
    python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py --mode full --headless \\
        --num_envs 1 --prompt ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.agentic_environment_generation.spec_io import DEFAULT_AGENTIC_OUTPUT_DIR, write_env_graph_specs
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec, ArenaEnvInitialGraphSpec

DEFAULT_PROMPT = "Franka picks up a cube from the maple table and places it into a bowl on the table."


def add_agentic_env_gen_runner_cli_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Agentic Environment Generation Runner")
    group.add_argument(
        "--mode",
        type=str,
        choices=("full", "resolve", "build"),
        default="full",
        help=(
            "Which phases to run: 'resolve' (no Isaac Sim), 'build' (needs --linked_env_graph_spec_yaml), "
            "or 'full' (resolve and build in one process; default)."
        ),
    )
    group.add_argument(
        "--linked_env_graph_spec_yaml",
        type=Path,
        default=None,
        help="Linked environment graph spec YAML to build from (required for --mode build).",
    )
    group.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Natural-language env description passed to the generation agent.",
    )
    group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the LLM model id (default: agent's built-in default).",
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature (default: 0.2).",
    )
    group.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="Number of simulation steps to run with the zero-action policy (default: 20).",
    )
    group.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_AGENTIC_OUTPUT_DIR,
        help="Directory for the generated YAML files (default: isaaclab_arena_environments/agent_generated).",
    )


def generate_env_intent_spec(args_cli: argparse.Namespace) -> EnvironmentIntentSpec:
    """Generate an environment intent spec from a prompt."""
    from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
        EnvironmentGenerationAgent,
        build_asset_catalogue,
        build_relation_catalogue,
        build_task_catalogue,
    )

    print(f"\n[runner] prompt: {args_cli.prompt!r}", flush=True)

    asset_catalog = build_asset_catalogue()
    relation_catalog = build_relation_catalogue()
    task_catalog = build_task_catalogue()

    agent_kwargs = {"model": args_cli.model} if args_cli.model else {}
    agent = EnvironmentGenerationAgent(**agent_kwargs)
    intent_spec, _raw_response = agent.generate_spec(
        args_cli.prompt,
        asset_catalog=asset_catalog,
        relation_catalog=relation_catalog,
        task_catalog=task_catalog,
        temperature=args_cli.temperature,
    )
    print(f"[runner] agent reasoning: {intent_spec.reasoning}", flush=True)
    return intent_spec


def compile_env_intent_spec(env_intent_spec: EnvironmentIntentSpec) -> ArenaEnvInitialGraphSpec:
    """Compile an EnvironmentIntentSpec into an initial environment graph spec."""
    from isaaclab_arena.agentic_environment_generation.intent_compiler import IntentCompiler

    compiler = IntentCompiler()
    initial_env_graph_spec = compiler.compile(env_intent_spec)

    print(
        f"[runner] compiled → {len(initial_env_graph_spec.nodes)} nodes, "
        f"{len(initial_env_graph_spec.tasks)} tasks, "
        f"env_name={initial_env_graph_spec.env_name!r}",
        flush=True,
    )

    if compiler.has_resolution_errors:
        print("[runner] WARNING: resolution errors detected:", flush=True)
        for event in compiler.resolution_errors:
            chosen = event.chosen or "<none>"
            print(f"  {event.stage:34s} {event.query!s:24s} -> {chosen}", flush=True)
    else:
        print("[runner] all assets resolved without errors.", flush=True)

    return initial_env_graph_spec


def link_env_graph_spec(initial_env_graph_spec: ArenaEnvInitialGraphSpec) -> ArenaEnvGraphSpec:
    """Link an initial environment graph spec into a fully wired environment graph spec."""
    linked_env_graph_spec = initial_env_graph_spec.link()
    print(
        f"[runner] linked → {len(linked_env_graph_spec.state_specs)} state specs,"
        f" {len(linked_env_graph_spec.tasks)} wired tasks",
        flush=True,
    )
    return linked_env_graph_spec


def resolve_env_spec(args_cli: argparse.Namespace) -> Path:
    """Resolve an environment intent spec into an initial environment graph spec and a linked environment graph spec."""
    # step 1: generate the environment intent spec
    env_intent_spec = generate_env_intent_spec(args_cli)
    # step 2: compile the environment intent spec into an initial environment graph spec
    initial_env_graph_spec = compile_env_intent_spec(env_intent_spec)
    # step 3: link the initial environment graph spec into a fully wired environment graph spec
    linked_env_graph_spec = link_env_graph_spec(initial_env_graph_spec)
    # step 4: write the initial and linked environment graph specs to YAML files
    initial_path, linked_path = write_env_graph_specs(initial_env_graph_spec, linked_env_graph_spec, args_cli.out_dir)
    print(f"[runner] wrote initial environment graph spec → {initial_path}", flush=True)
    print(f"[runner] wrote linked environment graph spec  → {linked_path}", flush=True)
    return linked_path


def build_env_from_linked_env_graph_spec(
    linked_env_graph_spec_path: Path, args_cli: argparse.Namespace
) -> ManagerBasedEnv:
    """Build a gymnasium env from a linked environment graph spec YAML."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec

    loaded_env_graph_spec = ArenaEnvGraphSpec.from_yaml(linked_env_graph_spec_path)
    arena_env = loaded_env_graph_spec.to_arena_env()
    builder = ArenaEnvBuilder(arena_env, args_cli)
    env = builder.make_registered()
    print(
        f"[runner] built env {arena_env.name!r} from linked environment graph spec {linked_env_graph_spec_path}",
        flush=True,
    )
    return env


def run_zero_action_policy(env: ManagerBasedEnv, num_steps: int) -> None:
    """Run the zero-action policy for a given number of steps."""
    import torch

    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyArgs

    policy = ZeroActionPolicy(ZeroActionPolicyArgs())
    obs, _ = env.reset()
    policy.reset()
    for step in range(num_steps):
        with torch.inference_mode():
            action = policy.get_action(env, obs)
            obs, _, terminated, truncated, _ = env.step(action)
        if (terminated | truncated).any():
            env_ids = (terminated | truncated).nonzero().flatten()
            print(f"[runner] step {step}: episode done for env_ids {env_ids.tolist()}", flush=True)
            policy.reset(env_ids=env_ids)
    env.close()
    print("[runner] done.", flush=True)


def build_env_and_run_policy(linked_env_graph_spec_path: Path, args_cli: argparse.Namespace) -> None:
    """Run steps 5-6: reload the linked spec, build the gym env, run the zero-action policy.

    Must be called inside an active :class:`SimulationAppContext`: ``to_arena_env`` opens USD
    assets and ``make_registered`` creates the simulation context.
    """
    # step 5: build the gym env from the linked environment graph spec
    env = build_env_from_linked_env_graph_spec(linked_env_graph_spec_path, args_cli)

    # step 6: run the zero-action policy for the given number of steps
    run_zero_action_policy(env, args_cli.num_steps)


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    add_agentic_env_gen_runner_cli_args(parser)
    args_cli = parser.parse_args()

    if args_cli.mode == "resolve":
        resolve_env_spec(args_cli)
        return 0

    elif args_cli.mode == "build":
        assert args_cli.linked_env_graph_spec_yaml is not None, "--mode build requires --linked_env_graph_spec_yaml"
        assert (
            args_cli.linked_env_graph_spec_yaml.is_file()
        ), f"--linked_env_graph_spec_yaml not found: {args_cli.linked_env_graph_spec_yaml}"
        with SimulationAppContext(args_cli):
            build_env_and_run_policy(args_cli.linked_env_graph_spec_yaml, args_cli)
        return 0

    # resolve and build in one process.
    else:
        with SimulationAppContext(args_cli):
            linked_env_graph_spec_path = resolve_env_spec(args_cli)
            build_env_and_run_policy(linked_env_graph_spec_path, args_cli)
    return 0


if __name__ == "__main__":
    sys.exit(main())
