# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the agent on a prompt and dump the resolved ArenaEnvGraphSpec.

Examples:
    # Print the Pydantic EnvironmentIntentSpec JSON schema (no agent call):
    /isaac-sim/python.sh -m isaaclab_arena.agentic_environment_generation.try_environment_intent_schema --print-schema

    # Print the catalog sent to the agent (no agent call):
    /isaac-sim/python.sh -m isaaclab_arena.agentic_environment_generation.try_environment_intent_schema --print-catalog

    # Call the agent, resolve, print, and dump YAML:
    /isaac-sim/python.sh -m isaaclab_arena.agentic_environment_generation.try_environment_intent_schema \
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
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec

DEFAULT_PROMPT = (
    "franka pick up avocado from the maple table and place it into a bowl on the table. "
    "there are other veggies on the table as distractor"
)
SEQUENTIAL_PROMPT = (
    "franka opens a microwave, picks up avocado on the table, place it into the microwave and close the microwave door."
    " There are other utensils on the table as distractor"
)


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


if __name__ == "__main__":
    main()
