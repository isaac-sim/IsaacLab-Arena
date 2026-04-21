# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the LLM parser on a prompt and print the resulting SceneSpec.

Must run inside the Docker container (needs AssetRegistry). Requires
NV_API_KEY and the `openai` pip package.

Examples:
    # Print the Pydantic SceneSpec JSON schema (no LLM call):
    /isaac-sim/python.sh -m isaaclab_arena_examples.llm_env_gen.try_schema --print-schema

    # Print the catalog sent to the LLM (no LLM call):
    /isaac-sim/python.sh -m isaaclab_arena_examples.llm_env_gen.try_schema --print-catalog

    # Call the LLM and print the parsed SceneSpec:
    /isaac-sim/python.sh -m isaaclab_arena_examples.llm_env_gen.try_schema \
        --prompt "franka pick up avocado from the table and place it into a bowl on the table. there are other veggies on the table as distractor"
"""

from __future__ import annotations

import argparse
import json

DEFAULT_PROMPT = (
    "franka pick up avocado from the table and place it into a bowl on the table. "
    "there are other veggies on the table as distractor"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--print-schema", action="store_true")
    parser.add_argument("--print-catalog", action="store_true")
    args = parser.parse_args()

    from isaaclab_arena_examples.llm_env_gen.schema import SceneSpec

    if args.print_schema:
        print(json.dumps(SceneSpec.model_json_schema(), indent=2))
        return

    from isaaclab_arena_examples.llm_env_gen.llm_agent import LLMAgent, build_catalog_text

    catalog = build_catalog_text()
    if args.print_catalog:
        print(catalog)
        return

    kwargs = {"model": args.model} if args.model else {}
    agent = LLMAgent(**kwargs)
    spec, raw = agent.generate_spec(args.prompt, catalog_text=catalog, temperature=args.temperature)

    print("=== raw LLM response ===")
    print(raw)
    print("\n=== parsed SceneSpec ===")
    print(spec.model_dump_json(indent=2))

    from isaaclab_arena_examples.llm_env_gen.resolver import Resolver

    resolved = Resolver().resolve(spec)

    bg = resolved.background.name if resolved.background else "<miss>"
    print("\n=== resolved bindings ===")
    print(f"background : {bg}")
    print(f"embodiment : {resolved.embodiment_name}")
    print("items:")
    for key, cls in resolved.items.items():
        print(f"  {key:20s} -> {cls.name}")

    print("\n=== initial_scene_graph (resolved) ===")
    for rel in resolved.initial_scene_graph:
        print(f"  {rel['kind']}({rel['subject']}, {rel['target']})")

    print("\n=== final_scene_graph (resolved) ===")
    for rel in resolved.final_scene_graph:
        print(f"  {rel['kind']}({rel['subject']}, {rel['target']})")

    print("\n=== trace ===")
    for t in resolved.trace:
        chosen = t.chosen if t.chosen is not None else "<none>"
        extra = f"  [{t.note}]" if t.note else ""
        print(f"  {t.stage:34s} {t.query!s:24s} -> {chosen}{extra}")


if __name__ == "__main__":
    main()
