# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the LLM parser on a prompt and dump the resolved ArenaEnvGraphSpec.

Must run inside the Docker container (needs AssetRegistry). Requires
NV_API_KEY and the `openai` pip package.

Output: the resolved spec is always written to
``isaaclab_arena_environments/llm_generated/<env_name>_proposal.yaml`` (in
addition to being printed to stdout).

Examples:
    # Print the Pydantic SceneSpec JSON schema (no LLM call):
    /isaac-sim/python.sh -m isaaclab_arena.llm_env_gen.try_schema --print-schema

    # Print the catalog sent to the LLM (no LLM call):
    /isaac-sim/python.sh -m isaaclab_arena.llm_env_gen.try_schema --print-catalog

    # Call the LLM, resolve, print, and dump YAML:
    /isaac-sim/python.sh -m isaaclab_arena.llm_env_gen.try_schema \
        --prompt "franka pick up avocado from the table and place it into a bowl on the table. there are other veggies on the table as distractor"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PROMPT = (
    "franka pick up avocado from the table and place it into a bowl on the table. "
    "there are other veggies on the table as distractor"
)
SEQUENTIAL_PROMPT = (
    "franka opens a microwave, picks up avocado on the table, place it into the microwave and close the microwave door."
    " There are other utensils on the table as distractor"
)

# Resolved-spec dumps land here so they're easy to find next to the existing
# auto-generated env modules. Path is computed from this file so it works
# inside the container (/workspaces/isaaclab_arena) and outside.
_LLM_GENERATED_DIR = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments" / "llm_generated"


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
            "Override the background chosen by the LLM (e.g. 'office_table' "
            "or 'kitchen'). Default is 'maple_table_robolab' because its "
            "tabletop ObjectReference yields a clean bbox and stable "
            "placement, unlike the rotated plain 'table' background. Pass "
            "an empty string ('') to keep the LLM's choice."
        ),
    )
    args = parser.parse_args()

    from isaaclab_arena.llm_env_gen.schema import SceneSpec

    if args.print_schema:
        print(json.dumps(SceneSpec.model_json_schema(), indent=2))
        return

    from isaaclab_arena.llm_env_gen.llm_agent import LLMAgent, build_catalog_text

    catalog = build_catalog_text()
    if args.print_catalog:
        print(catalog)
        return

    kwargs = {"model": args.model} if args.model else {}
    agent = LLMAgent(**kwargs)
    spec, raw = agent.generate_spec(args.prompt, catalog_text=catalog, temperature=args.temperature)

    print("=== raw LLM response ===")
    print(raw)

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
        for rel in spec.initial_scene_graph:
            if rel.subject == old_bg:
                rel.subject = new_bg
            if rel.target == old_bg:
                rel.target = new_bg
        # Note: tasks don't directly reference background in target (typically None or items),
        # so no background substitution needed in task.target
        spec.background = new_bg
        print(f"\n=== background override applied: {old_bg!r} -> {new_bg!r} ===")

    print("\n=== parsed SceneSpec ===")
    print(spec.model_dump_json(indent=2))

    from isaaclab_arena.llm_env_gen.resolver import Resolver

    resolver = Resolver()
    env_graph_spec = resolver.resolve(spec)

    print(f"\n=== resolved ArenaEnvGraphSpec (env_name={env_graph_spec.env_name!r}) ===")

    print("\nnodes:")
    for node in env_graph_spec.nodes:
        params_str = f"  params={node.params}" if node.params else ""
        print(f"  {node.id:24s} type={node.type.value:18s} name={node.name}{params_str}")

    print("\nstate_specs:")
    for state_spec in env_graph_spec.state_specs:
        s_count = len(state_spec.spatial_constraints)
        t_count = len(state_spec.task_constraints)
        print(f"  {state_spec.id:24s} spatial={s_count} task={t_count}")
        for c in state_spec.spatial_constraints:
            child_str = f", child={c.child}" if c.child else ""
            params_str = f"  params={c.params}" if c.params else ""
            print(f"    {c.type.value:16s} parent={c.parent}{child_str}{params_str}")
        for c in state_spec.task_constraints:
            print(f"    {c.type.value:16s} parent={c.parent}  child={c.child}")

    print("\ntasks:")
    for task in env_graph_spec.tasks:
        print(
            f"  {task.id:28s} type={task.type:18s} "
            f"initial={task.initial_state_spec_id!r} success={task.success_state_spec_id!r}"
        )
        print(f"    task_args: {task.task_args}")

    print("\n=== trace ===")
    for t in resolver.trace:
        chosen = t.chosen if t.chosen is not None else "<none>"
        extra = f"  [{t.note}]" if t.note else ""
        print(f"  {t.stage:34s} {t.query!s:24s} -> {chosen}{extra}")

    out_path = env_graph_spec.to_yaml(_LLM_GENERATED_DIR / f"{env_graph_spec.env_name}_proposal.yaml")
    print(f"\n=== wrote ArenaEnvGraphSpec YAML to {out_path} ===")


if __name__ == "__main__":
    main()
