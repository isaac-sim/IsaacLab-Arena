# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end agentic environment generation and execution.

Pipeline:

  1. Call :class:`EnvironmentGenerationAgent` on the prompt →
     :class:`~isaaclab_arena.agentic_environment_generation.environment_intent_spec.EnvironmentIntentSpec`.
  2. :class:`~isaaclab_arena.agentic_environment_generation.intent_compiler.IntentCompiler` →
     :class:`~isaaclab_arena.environments.arena_env_graph_spec.ArenaEnvInitialGraphSpec`.
  3. :meth:`ArenaEnvInitialGraphSpec.link` →
     :class:`~isaaclab_arena.environments.arena_env_graph_spec.ArenaEnvGraphSpec`.
  4. Both specs are written to ``--out-dir`` as YAML files. The linked spec is
     reloaded from disk (round-trip sanity check) before building the env.
  5. :meth:`ArenaEnvGraphSpec.to_arena_env` + :class:`ArenaEnvBuilder` → gymnasium env.
  6. Run :class:`~isaaclab_arena.policy.zero_action_policy.ZeroActionPolicy`
     for ``--num-steps`` steps.

All USD / pxr imports happen inside the :class:`SimulationAppContext` block so Kit's
schema extensions are initialized before any asset-registry access.

Usage::

    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.environment_generation_runner \\
        --headless --num_envs 1 \\
        --prompt "franka pick up avocado from the table and place it into a bowl"

    # More steps, custom output dir:
    /isaac-sim/python.sh -m isaaclab_arena_examples.agentic_environment_generation.environment_generation_runner \\
        --headless --num_envs 1 --num-steps 50 --out-dir /tmp/my_envs \\
        --prompt "franka open the microwave door"
"""

from __future__ import annotations

import argparse
import re
import sys
import torch
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

_DEFAULT_PROMPT = (
    "franka pick up avocado from the maple table and place it into a bowl on the table. "
    "there are other veggies on the table as distractor"
)
_DEFAULT_OUT_DIR = Path("isaaclab_arena_environments/llm_generated")


def _safe_stem(name: str) -> str:
    """Return a filesystem-safe stem derived from an env name."""
    stem = re.sub(r"[^\w.-]+", "_", name).strip("._")
    return stem or "unnamed_env"


def _add_runner_cli_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Environment Generation Runner")
    group.add_argument(
        "--prompt",
        type=str,
        default=_DEFAULT_PROMPT,
        help="Natural-language scene description passed to the generation agent.",
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
        "--num-steps",
        type=int,
        default=20,
        help="Number of simulation steps to run with the zero-action policy (default: 20).",
    )
    group.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for the generated YAML files (default: isaaclab_arena_environments/llm_generated).",
    )


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    _add_runner_cli_args(parser)
    args_cli = parser.parse_args()

    with SimulationAppContext(args_cli):
        # ------------------------------------------------------------------ #
        # Step 1 — prompt → EnvironmentIntentSpec                            #
        # ------------------------------------------------------------------ #
        from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
            EnvironmentGenerationAgent,
            build_asset_catalogue,
            build_relation_catalogue,
            build_task_catalogue,
        )
        from isaaclab_arena.agentic_environment_generation.intent_compiler import IntentCompiler
        from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec, ArenaEnvInitialGraphSpec

        print(f"\n[runner] prompt: {args_cli.prompt!r}", flush=True)

        asset_catalog = build_asset_catalogue()
        relation_catalog = build_relation_catalogue()
        task_catalog = build_task_catalogue()

        agent_kwargs = {"model": args_cli.model} if args_cli.model else {}
        agent = EnvironmentGenerationAgent(**agent_kwargs)
        intent_spec, raw_response = agent.generate_spec(
            args_cli.prompt,
            asset_catalog=asset_catalog,
            relation_catalog=relation_catalog,
            task_catalog=task_catalog,
            temperature=args_cli.temperature,
        )
        print(f"[runner] agent reasoning: {intent_spec.reasoning}", flush=True)

        # ------------------------------------------------------------------ #
        # Step 2 — EnvironmentIntentSpec → ArenaEnvInitialGraphSpec          #
        # ------------------------------------------------------------------ #
        compiler = IntentCompiler()
        initial_graph_spec: ArenaEnvInitialGraphSpec = compiler.compile(intent_spec)

        print(
            f"[runner] compiled → {len(initial_graph_spec.nodes)} nodes, "
            f"{len(initial_graph_spec.tasks)} tasks, "
            f"env_name={initial_graph_spec.env_name!r}",
            flush=True,
        )

        if compiler.has_resolution_errors:
            print("[runner] WARNING: resolution errors detected:", flush=True)
            for event in compiler.resolution_errors:
                chosen = event.chosen or "<none>"
                print(f"  {event.stage:34s} {event.query!s:24s} -> {chosen}", flush=True)
        else:
            print("[runner] all assets resolved without errors.", flush=True)

        # ------------------------------------------------------------------ #
        # Step 3 — ArenaEnvInitialGraphSpec.link() → ArenaEnvGraphSpec       #
        # ------------------------------------------------------------------ #
        linked_spec: ArenaEnvGraphSpec = initial_graph_spec.link()
        print(
            f"[runner] linked → {len(linked_spec.state_specs)} state specs, {len(linked_spec.tasks)} wired tasks",
            flush=True,
        )

        # ------------------------------------------------------------------ #
        # Step 4 — dump both specs to YAML, then reload the linked one        #
        # ------------------------------------------------------------------ #
        out_dir: Path = args_cli.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_stem(initial_graph_spec.env_name)

        initial_path = out_dir / f"{stem}_initial.yaml"
        linked_path = out_dir / f"{stem}_linked.yaml"

        initial_graph_spec.write_yaml(initial_path)
        print(f"[runner] wrote initial graph spec → {initial_path}", flush=True)

        linked_spec.write_yaml(linked_path)
        print(f"[runner] wrote linked graph spec  → {linked_path}", flush=True)

        # Reload from disk to verify the YAML round-trip before building the env.
        loaded_spec = ArenaEnvGraphSpec.from_yaml(linked_path)
        print(f"[runner] reloaded linked spec from {linked_path}", flush=True)

        # ------------------------------------------------------------------ #
        # Step 5 — ArenaEnvGraphSpec → gymnasium env                         #
        # ------------------------------------------------------------------ #
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

        arena_env = loaded_spec.to_arena_env()
        builder = ArenaEnvBuilder(arena_env, args_cli)
        env = builder.make_registered()
        print(f"[runner] built env {arena_env.name!r}", flush=True)

        # ------------------------------------------------------------------ #
        # Step 6 — run zero-action policy for --num-steps steps              #
        # ------------------------------------------------------------------ #
        from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyArgs

        policy = ZeroActionPolicy(ZeroActionPolicyArgs())
        obs, _ = env.reset()
        policy.reset()

        print(f"[runner] running zero-action policy for {args_cli.num_steps} steps …", flush=True)
        for step in range(args_cli.num_steps):
            with torch.inference_mode():
                action = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(action)

            if (terminated | truncated).any():
                env_ids = (terminated | truncated).nonzero().flatten()
                print(f"[runner] step {step}: episode done for env_ids {env_ids.tolist()}", flush=True)
                policy.reset(env_ids=env_ids)

        env.close()
        print("[runner] done.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
