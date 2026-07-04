# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end agentic environment generation and execution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.agentic_environment_generation.spec_io import DEFAULT_AGENTIC_OUTPUT_DIR, write_env_graph_spec
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec

DEFAULT_PROMPT = "Franka picks up a cube from the maple table and places it into a bowl on the table."


def add_agentic_env_gen_runner_cli_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Agentic Environment Generation Runner")
    group.add_argument(
        "--mode",
        type=str,
        choices=("full", "resolve", "build"),
        default="full",
        help=(
            "Which phases to run: 'resolve' (no Isaac Sim), 'build' (needs --env_graph_spec_yaml), "
            "or 'full' (resolve and build in one process; default)."
        ),
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


def generate_env_graph_spec(args_cli: argparse.Namespace) -> ArenaEnvGraphSpec:
    """Generate an environment graph spec from a prompt."""
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
    env_graph_spec, _raw_response = agent.generate_spec(
        args_cli.prompt,
        asset_catalog=asset_catalog,
        relation_catalog=relation_catalog,
        task_catalog=task_catalog,
        temperature=args_cli.temperature,
    )
    print(
        f"[runner] generated → {env_graph_spec.summary()}, env_name={env_graph_spec.env_name!r}",
        flush=True,
    )
    return env_graph_spec


def resolve_env_spec(args_cli: argparse.Namespace) -> Path:
    """Resolve a prompt into an environment graph spec YAML."""
    env_graph_spec = generate_env_graph_spec(args_cli)
    path = write_env_graph_spec(env_graph_spec, args_cli.out_dir)
    print(f"[runner] wrote environment graph spec → {path}", flush=True)
    return path


def build_env_from_env_graph_spec(env_graph_spec_path: Path, args_cli: argparse.Namespace) -> ManagerBasedEnv:
    """Build a gymnasium env from an environment graph spec YAML."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec

    loaded_env_graph_spec = ArenaEnvGraphSpec.from_yaml(env_graph_spec_path)
    arena_env = loaded_env_graph_spec.to_arena_env()
    builder = ArenaEnvBuilder(arena_env, args_cli)
    env = builder.make_registered()
    print(
        f"[runner] built env {arena_env.name!r} from environment graph spec {env_graph_spec_path}",
        flush=True,
    )
    return env


def run_zero_action_policy(env: ManagerBasedEnv, num_steps: int) -> None:
    """Run the zero-action policy for a given number of steps."""
    import torch

    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg

    policy = ZeroActionPolicy(ZeroActionPolicyCfg())
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


def build_env_and_run_policy(env_graph_spec_path: Path, args_cli: argparse.Namespace) -> None:
    """Build the gym env from a graph spec YAML and run the zero-action policy."""
    env = build_env_from_env_graph_spec(env_graph_spec_path, args_cli)
    run_zero_action_policy(env, args_cli.num_steps)


def _resolved_graph_spec_yaml(args_cli: argparse.Namespace) -> Path:
    path_arg = args_cli.env_graph_spec_yaml
    assert path_arg is not None, "--mode build requires --env_graph_spec_yaml"
    path = Path(path_arg)
    assert path.is_file(), f"env graph spec YAML not found: {path}"
    return path


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    add_agentic_env_gen_runner_cli_args(parser)
    args_cli = parser.parse_args()

    if args_cli.mode == "resolve":
        resolve_env_spec(args_cli)
        return 0

    if args_cli.mode == "build":
        with SimulationAppContext(args_cli):
            build_env_and_run_policy(_resolved_graph_spec_yaml(args_cli), args_cli)
        return 0

    with SimulationAppContext(args_cli):
        env_graph_spec_path = resolve_env_spec(args_cli)
        build_env_and_run_policy(env_graph_spec_path, args_cli)
    return 0


if __name__ == "__main__":
    sys.exit(main())
