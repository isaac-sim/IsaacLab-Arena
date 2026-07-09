# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end agentic environment generation and execution.

Usage::

    # Print the Pydantic ArenaEnvGraphSpec JSON schema (no agent call, no Isaac Sim):
    python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py --mode schema

    # Print the catalog sent to the agent (no agent call, no Isaac Sim):
    python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py --mode catalog

    # Resolve a prompt into an environment graph spec YAML (no Isaac Sim):
    python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py --mode resolve --prompt ...

    # Build a gym env from a graph spec YAML and run the zero-action policy:
    python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py --mode build --headless \\
        --num_envs 1 --env_graph_spec_yaml <env>_env_graph.yaml

    # Resolve and build in one process:
    python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py --mode full --headless \\
        --num_envs 1 --prompt ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.agentic_environment_generation.spec_io import (
    DEFAULT_AGENTIC_OUTPUT_DIR,
    write_env_graph_dict,
    write_env_graph_spec,
)
from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

DEFAULT_PROMPT = "Franka picks up a cube from the maple table and places it into a bowl on the table."


def add_agentic_env_gen_runner_cli_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Agentic Environment Generation Runner")
    group.add_argument(
        "--mode",
        type=str,
        choices=("full", "resolve", "build", "schema", "catalog"),
        default="full",
        help=(
            "Which phases to run: 'schema' (print the spec JSON schema and exit), "
            "'catalog' (print the agent catalog and exit), 'resolve' (prompt -> spec YAML, no Isaac Sim), "
            "'build' (needs --env_graph_spec_yaml), or 'full' (resolve and build in one process; default). "
            "'schema' and 'catalog' make no agent call."
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


def resolve_env_spec(args_cli: argparse.Namespace) -> Path:
    """Resolve a prompt into an environment graph spec YAML."""
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
    env_graph_spec, data = agent.generate_spec(
        args_cli.prompt,
        asset_catalog=asset_catalog,
        relation_catalog=relation_catalog,
        task_catalog=task_catalog,
        temperature=args_cli.temperature,
    )
    # last_validation_traces holds one line per failure, e.g.
    #   "embodiment.registry_name: Unknown asset registry_name 'not_a_real_asset'"
    #   "Task 'PickAndPlaceTask' is missing required param 'pick_up_object'"
    if env_graph_spec is None:
        print("\n[runner] validation traces:", flush=True)
        for line in agent.last_validation_traces:
            print(f"  {line}", flush=True)
        invalid_path = write_env_graph_dict(data, args_cli.out_dir)
        print(f"[runner] wrote invalid spec YAML to {invalid_path}", flush=True)
        assert False, f"Agent returned an invalid spec. Validation traces: {agent.last_validation_traces}"
    print_env_graph(env_graph_spec)
    print(
        f"[runner] generated → {env_graph_spec.summary()}, env_name={env_graph_spec.env_name!r}",
        flush=True,
    )
    path = write_env_graph_spec(env_graph_spec, args_cli.out_dir)
    print(f"[runner] wrote environment graph spec → {path}", flush=True)
    return path


def print_schema() -> None:
    """Print the Pydantic ArenaEnvGraphSpec JSON schema."""
    import json

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    print(json.dumps(ArenaEnvGraphSpec.model_json_schema(), indent=2))


def print_catalog() -> None:
    """Print the asset, relation, and task catalogs sent to the agent."""
    from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
        build_asset_catalogue,
        build_relation_catalogue,
        build_task_catalogue,
    )

    print(build_asset_catalogue().to_catalog_string())
    print()
    print(build_relation_catalogue().to_catalog_string())
    print()
    print(build_task_catalogue().to_catalog_string())


def _iter_printable_assets(spec: ArenaEnvGraphSpec):
    yield "embodiment", spec.embodiment.id, spec.embodiment.registry_name, spec.embodiment.params
    yield "background", spec.background.id, spec.background.registry_name, spec.background.params
    for obj in spec.objects:
        yield "object", obj.id, obj.registry_name, obj.params
    for ref in spec.object_references or []:
        yield "object_reference", ref.id, ref.id, ref.params


def print_env_graph(spec: ArenaEnvGraphSpec) -> None:
    """Print the generated graph in a human-readable tabular layout."""
    print(f"\n=== ArenaEnvGraphSpec (env_name={spec.env_name!r}) ===")

    print("\nassets:")
    for role, asset_id, registry_name, params in _iter_printable_assets(spec):
        params_str = f"  params={params}" if params else ""
        print(f"  {asset_id:24s} role={role:18s} registry_name={registry_name}{params_str}")

    print("\nrelations:")
    for relation in spec.relations:
        ref_str = f"  reference={relation.reference}" if relation.reference is not None else ""
        params_str = f"  params={relation.params}" if relation.params else ""
        print(f"  {relation.kind:16s} subject={relation.subject}{ref_str}{params_str}")

    print(f"\ntask: composition={spec.task.composition}")
    print(f"  description: {spec.task.description}")
    for i, task in enumerate(spec.task.subtasks):
        print(f"  [{i}] kind={task.kind}")
        print(f"    params: {task.params}")


def build_env_from_env_graph_spec(env_graph_spec_path: Path, args_cli: argparse.Namespace) -> ManagerBasedEnv:
    """Build a gymnasium env from an environment graph spec YAML."""
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    loaded_env_graph_spec = ArenaEnvGraphSpec.from_yaml(env_graph_spec_path)
    arena_env = loaded_env_graph_spec.to_arena_env()
    # TODO(cvolk, 2026-07-06): [typed-config-migration] Pass ArenaEnvBuilderCfg into this function after this
    # runner stops carrying all configuration in one argparse Namespace.
    builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli))
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

    if args_cli.mode == "schema":
        print_schema()
        return 0

    if args_cli.mode == "catalog":
        print_catalog()
        return 0

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
