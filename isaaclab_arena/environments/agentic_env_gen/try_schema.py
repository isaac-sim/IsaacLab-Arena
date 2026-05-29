# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the agent on a prompt and dump the resolved ArenaEnvGraphSpec.

Requires NV_API_KEY environment variable.

Examples:
    # Print the Pydantic EnvIntentSpec JSON schema (no agent call):
    /isaac-sim/python.sh -m isaaclab_arena.environments.agentic_env_gen.try_schema --print-schema

    # Print the catalog sent to the agent (no agent call):
    /isaac-sim/python.sh -m isaaclab_arena.environments.agentic_env_gen.try_schema --print-catalog

    # Call the agent, resolve, print, and dump YAML:
    /isaac-sim/python.sh -m isaaclab_arena.environments.agentic_env_gen.try_schema \
        --prompt "franka pick up avocado from the table and place it into a bowl on the table. there are other veggies on the table as distractor"
"""

from __future__ import annotations

import argparse
import json

DEFAULT_PROMPT = (
    "franka pick up avocado from the table and place it into a bowl on the table. "
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
    parser.add_argument(
        "--background",
        type=str,
        default="maple_table_robolab",
        help=(
            "Override the background chosen by the agent (e.g. 'office_table' "
            "or 'kitchen'). Default is 'maple_table_robolab' because its "
            "tabletop ObjectReference yields a clean bbox and stable "
            "placement, unlike the rotated plain 'table' background. Pass "
            "an empty string ('') to keep the agent's choice."
        ),
    )
    args = parser.parse_args()

    from isaaclab_arena.environments.agentic_env_gen.env_intent_spec import EnvIntentSpec

    if args.print_schema:
        print(json.dumps(EnvIntentSpec.model_json_schema(), indent=2))
        return

    from isaaclab_arena.environments.agentic_env_gen.env_gen_agent import EnvGenAgent, build_catalog_text

    catalog = build_catalog_text()
    if args.print_catalog:
        print(catalog)
        return

    kwargs = {"model": args.model} if args.model else {}
    agent = EnvGenAgent(**kwargs)
    spec, raw = agent.generate_spec(args.prompt, catalog_text=catalog, temperature=args.temperature)

    print("=== raw agent response ===")
    print(raw)

    # Surface the forced chain-of-thought field on its own so it's easy to
    # spot when debugging a bad spec — without this, ``reasoning`` is
    # buried inside the multi-hundred-line model_dump_json below.
    print("\n=== agent reasoning ===")
    print(spec.reasoning)

    if args.background and args.background != spec.background:
        # Swap the background name wherever it appears so downstream code
        # (resolver, proposer) sees a consistent scene. Rewrite both
        # ``rel.target`` (binary relations like ``on(bowl, table)``) AND
        # ``rel.subject`` (unary relations like ``is_anchor(table)``);
        # missing the subject case would leave the unary constraint
        # pointing at the old background name, after which the resolver
        # would emit a ``relation.initial.unknown_subject`` trace and
        # silently drop the constraint.
        old_bg = spec.background
        new_bg = args.background
        for rel in spec.initial_state_graph:
            if rel.subject == old_bg:
                rel.subject = new_bg
            if rel.target == old_bg:
                rel.target = new_bg
        # Note: tasks don't directly reference background in target (typically None or items),
        # so no background substitution needed in task.target
        spec.background = new_bg
        print(f"\n=== background override applied: {old_bg!r} -> {new_bg!r} ===")

    print("\n=== parsed EnvIntentSpec ===")
    print(spec.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
