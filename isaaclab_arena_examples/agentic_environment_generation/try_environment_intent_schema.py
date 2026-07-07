# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the agent on a prompt and dump the ArenaEnvGraphSpec.

Examples::

    # Print the Pydantic ArenaEnvGraphSpec JSON schema (no agent call):
    python isaaclab_arena_examples/agentic_environment_generation/try_environment_intent_schema.py --print-schema

    # Print the catalog sent to the agent (no agent call):
    python isaaclab_arena_examples/agentic_environment_generation/try_environment_intent_schema.py --print-catalog

    # Call the agent, print, and dump YAML:
    python isaaclab_arena_examples/agentic_environment_generation/try_environment_intent_schema.py \\
        --prompt "franka pick up avocado from the table and place it into a bowl on the table. there are other veggies on the table as distractor"
"""

from __future__ import annotations

import argparse
import json

from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
    EnvironmentGenerationAgent,
    build_asset_catalogue,
    build_relation_catalogue,
    build_task_catalogue,
)
from isaaclab_arena.agentic_environment_generation.spec_io import (
    DEFAULT_AGENTIC_OUTPUT_DIR,
    write_env_graph_dict,
    write_env_graph_spec,
)
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec

DEFAULT_PROMPT = (
    "franka pick up avocado from the maple table and place it into a bowl on the table. "
    "there are other veggies on the table as distractor"
)
SEQUENTIAL_PROMPT = (
    "franka opens a microwave, picks up avocado on the table, place it into the microwave and close the microwave door."
    " There are other utensils on the table as distractor"
)


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
    for i, task in enumerate(spec.task.tasks):
        print(f"  [{i}] kind={task.kind}")
        print(f"    params: {task.params}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--print-schema", action="store_true")
    parser.add_argument("--print-catalog", action="store_true")
    args = parser.parse_args()

    if args.print_schema:
        print(json.dumps(ArenaEnvGraphSpec.model_json_schema(), indent=2))
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
    spec, data = agent.generate_spec(
        args.prompt,
        asset_catalog=asset_catalog,
        relation_catalog=relation_catalog,
        task_catalog=task_catalog,
        temperature=args.temperature,
    )
    print("=== agent response ===")
    print(json.dumps(data, indent=2))

    if spec is None:
        print("=== validation traces ===")
        for line in agent.last_validation_traces:
            print(line)
        out_path = write_env_graph_dict(data, DEFAULT_AGENTIC_OUTPUT_DIR)
        print(f"\n=== wrote invalid ArenaEnvGraphSpec YAML to {out_path} ===")
        return

    print("\n=== parsed ArenaEnvGraphSpec ===")
    print(spec.model_dump_json(indent=2))

    print_env_graph(spec)

    out_path = write_env_graph_spec(spec, DEFAULT_AGENTIC_OUTPUT_DIR)
    print(f"\n=== wrote ArenaEnvGraphSpec YAML to {out_path} ===")


if __name__ == "__main__":
    main()
