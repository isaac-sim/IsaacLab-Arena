# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the agent on a prompt and dump the compiled UnresolvedArenaEnvGraphSpec.

Examples:
    # Print the Pydantic EnvironmentIntentSpec JSON schema (no agent call):
    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.try_environment_intent_schema --print-schema

    # Print the catalog sent to the agent (no agent call):
    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.try_environment_intent_schema --print-catalog

    # Call the agent, compile, print, and dump YAML:
    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.try_environment_intent_schema \
        --prompt "franka pick up avocado from the table and place it into a bowl on the table. there are other veggies on the table as distractor"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from isaaclab_arena.agentic_environment_generation.asset_matcher import IntentResolutionTraceEvent
from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
    EnvironmentGenerationAgent,
    build_asset_catalogue,
    build_relation_catalogue,
    build_task_catalogue,
)
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec
from isaaclab_arena.agentic_environment_generation.intent_compiler import IntentCompiler
from isaaclab_arena.environments.arena_env_graph_spec import UnresolvedArenaEnvGraphSpec

_LLM_GENERATED_DIR = Path("isaaclab_arena_environments/llm_generated")

DEFAULT_PROMPT = (
    "franka pick up avocado from the maple table and place it into a bowl on the table. "
    "there are other veggies on the table as distractor"
)
SEQUENTIAL_PROMPT = (
    "franka opens a microwave, picks up avocado on the table, place it into the microwave and close the microwave door."
    " There are other utensils on the table as distractor"
)


def _safe_filename_stem(name: str) -> str:
    """Return a filesystem-safe stem derived from ``env_name``."""
    stem = re.sub(r"[^\w.-]+", "_", name).strip("._")
    return stem or "unnamed_env"


def _format_trace_event(event: IntentResolutionTraceEvent) -> str:
    chosen = event.chosen if event.chosen is not None else "<none>"
    extra = f"  [{event.note}]" if event.note else ""
    return f"  {event.stage:34s} {event.query!s:24s} -> {chosen}{extra}"


def print_unresolved_graph(spec: UnresolvedArenaEnvGraphSpec) -> None:
    """Print the compiled graph in a human-readable tabular layout."""
    print(f"\n=== UnresolvedArenaEnvGraphSpec (env_name={spec.env_name!r}) ===")

    print("\nnodes:")
    for node in spec.nodes:
        params_str = f"  params={node.params}" if node.params else ""
        print(f"  {node.id:24s} type={node.type.value:18s} name={node.name}{params_str}")

    print("\ninitial_state_spec:")
    initial = spec.initial_state_spec
    s_count = len(initial.spatial_constraints)
    t_count = len(initial.task_constraints)
    print(f"  {initial.id:24s} spatial={s_count} task={t_count}")
    for constraint in initial.spatial_constraints:
        ref_str = f"  reference={constraint.reference}" if constraint.reference is not None else ""
        params_str = f"  params={constraint.params}" if constraint.params else ""
        print(f"    {constraint.kind:16s} subject={constraint.subject}{ref_str}{params_str}")
    for constraint in initial.task_constraints:
        print(f"    {constraint.type.value:16s} parent={constraint.parent}  child={constraint.child}")

    print("\ntasks:")
    for i, task in enumerate(spec.tasks):
        print(f"  [{i}] kind={task.kind}")
        print(f"    params: {task.params}")
        if task.description:
            print(f"    description: {task.description}")


def print_resolution_trace(compiler: IntentCompiler) -> None:
    """Print compiler trace events and any resolution errors."""
    print("\n=== trace ===")
    for event in compiler.trace:
        print(_format_trace_event(event))

    if compiler.has_resolution_errors:
        print("\n=== resolution errors ===")
        for event in compiler.resolution_errors:
            print(_format_trace_event(event))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--print-schema", action="store_true")
    parser.add_argument("--print-catalog", action="store_true")
    args = parser.parse_args()

    if args.print_schema:
        print(json.dumps(EnvironmentIntentSpec.model_json_schema(), indent=2))
        return

    asset_catalog = build_asset_catalogue()
    relation_catalog = build_relation_catalogue()
    task_catalog = build_task_catalogue()
    if args.print_catalog:
        print(asset_catalog.to_catalog_string())
        print()
        print(relation_catalog.to_catalog_string())
        print()
        print(task_catalog.to_catalog_string())
        return

    agent = EnvironmentGenerationAgent(model=args.model)
    spec, raw = agent.generate_spec(
        args.prompt,
        asset_catalog=asset_catalog,
        relation_catalog=relation_catalog,
        task_catalog=task_catalog,
        temperature=args.temperature,
    )

    print("=== raw agent response ===")
    print(raw)

    # Surface the forced chain-of-thought field.
    print("\n=== agent reasoning ===")
    print(spec.reasoning)

    print("\n=== parsed EnvironmentIntentSpec ===")
    print(spec.model_dump_json(indent=2))

    compiler = IntentCompiler()
    env_graph_spec = compiler.compile(spec)
    print_unresolved_graph(env_graph_spec)
    print_resolution_trace(compiler)

    out_path = _LLM_GENERATED_DIR / f"{_safe_filename_stem(env_graph_spec.env_name)}_proposal.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env_graph_spec.write_yaml(out_path)
    print(f"\n=== wrote UnresolvedArenaEnvGraphSpec YAML to {out_path} ===")


if __name__ == "__main__":
    main()
