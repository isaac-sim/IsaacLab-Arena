# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Auto-generate an Arena env from a natural-language prompt with IK gating.

Pipeline (one SimulationApp boot, N retries):

  1. Run the LLM once on the prompt → ``SceneSpec``.
  2. Resolve once against ``AssetRegistry`` → ``ResolvedScene``.
  3. Inside a single ``SimulationAppContext``, loop up to ``--max-attempts``:
       a. ``write_env(..., attempt=N)`` re-renders the env file with a
          fresh robot-placement sample.
       b. Hot-load the file (pop stale registry + gym entries, then
          ``exec_module`` so ``@register_environment`` re-registers).
       c. Build the env via :class:`ArenaEnvBuilder` and run
          :func:`run_reachability_check.check_reachability_for_arena_builder`.
       d. Exit code 0 stops the loop; 2 (unreachable) and 3 (in_collision)
          bump the attempt index and retry. Anything else propagates.

The seed for the robot-placement sampler is ``(env_name, attempt)`` —
deterministic per (prompt, attempt), but distinct across retries.

Example::

    /isaac-sim/python.sh -m isaaclab_arena.llm_env_gen.auto_generate_env \\
        --headless --num_envs 1 --max-attempts 6 \\
        --prompt "franka pick up avocado from the table and place it into a bowl"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.llm_env_gen.reachability_utils import add_ik_reachability_cli_args
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


_DEFAULT_PROMPT = (
    "franka pick up avocado from the table and place it into a bowl on the table. "
    "there are other veggies on the table as distractor"
)


def _add_auto_cli_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "Auto Env Generation",
        "Prompt-to-env loop with IK-feasibility retries.",
    )
    group.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT, help="Natural-language scene description.")
    group.add_argument(
        "--out-dir",
        type=str,
        default="isaaclab_arena_environments/llm_generated",
        help="Directory the generated env module is written to. Filename is derived from the env name.",
    )
    group.add_argument(
        "--max-attempts",
        type=int,
        default=8,
        help="Maximum robot-placement re-samples before giving up.",
    )
    group.add_argument(
        "--background-override",
        type=str,
        default="maple_table_robolab",
        help=(
            "Force this background instead of the LLM's pick. Empty string keeps the LLM's choice. "
            "Default mirrors try_schema.py — maple_table_robolab gives a clean tabletop bbox."
        ),
    )
    group.add_argument("--llm-model", type=str, default=None, help="Override LLMAgent model id.")
    group.add_argument("--temperature", type=float, default=0.2, help="LLM sampling temperature.")


def _hot_load_env_module(path: Path, env_name: str, attempt: int) -> None:
    """Re-import a generated env file so its ``@register_environment`` decorator runs.

    Two complications make this trickier than a vanilla reload:

    * The file imports ``isaaclab_arena_environments.example_environment_base``,
      which triggers the package's ``__init__.py`` and that init does
      ``pkgutil.iter_modules`` to auto-import every top-level module.
      If a same-named env file is sitting at that top level (e.g. left
      behind by a prior ``try_schema.py`` run), it pre-registers a
      stale class for our env name. By the time our decorator fires it
      sees ``is_registered`` and no-ops, leaving the stale class live.
    * gym's registry caches an ``EnvSpec`` keyed by env name with the
      cfg captured at register time, so we drop it too.

    Strategy: pop both registries up front, exec the new module under a
    unique name, then *forcefully* overwrite the env registry entry
    with our freshly-loaded class — pulled from the module by matching
    ``__module__`` so we never re-bind to an unrelated stale class.

    Imports of ``EnvironmentRegistry`` are deliberately done locally so
    that a prior ``reload_arena_modules()`` call (which swaps the
    registry class object) is honored — using the module-top import
    would write into a stale singleton.
    """
    from isaaclab_arena.assets.registries import EnvironmentRegistry as _ER

    _ER()._components.pop(env_name, None)
    try:
        import gymnasium as gym

        gym.envs.registry.pop(env_name, None)
    except (ImportError, AttributeError, KeyError):
        pass

    mod_name = f"_llm_env_gen_{env_name}_a{attempt}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    assert spec is not None and spec.loader is not None, f"Failed to load module spec for {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    fresh_cls = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and getattr(attr, "name", None) == env_name
            and getattr(attr, "__module__", None) == mod_name
        ):
            fresh_cls = attr
            break
    assert fresh_cls is not None, (
        f"Hot-load could not find an env class with name={env_name!r} in module {mod_name!r}"
    )
    _ER()._components[env_name] = fresh_cls
    # Drop any gym entry the previous attempt's arena_builder.make_registered
    # may have planted; the next make_registered call re-registers with the
    # current arena_env's cfg.
    try:
        import gymnasium as gym

        gym.envs.registry.pop(env_name, None)
    except (ImportError, AttributeError, KeyError):
        pass


def _build_args_for_env(args_cli: argparse.Namespace, env_name: str, embodiment_default: str) -> argparse.Namespace:
    """Synthesize the fields ``ArenaEnvBuilder`` and ``get_env`` read off ``args_cli``.

    We bypass the per-env subparser (we know exactly which env we just
    generated and what its single ``--embodiment`` arg defaults to). Any
    other env-specific kwargs the writer adds in the future will need to
    surface here too.
    """
    args_cli.example_environment = env_name
    if not hasattr(args_cli, "embodiment") or args_cli.embodiment is None:
        args_cli.embodiment = embodiment_default
    return args_cli


def _build_arena_builder(args_cli: argparse.Namespace):
    """Construct an ``ArenaEnvBuilder`` for a freshly registered env.

    Mirrors :func:`isaaclab_arena_environments.cli.get_arena_builder_from_cli`
    but skips its ``ensure_environments_registered`` step — our env is
    already live in the registry and we do not want to re-import the
    bundled package on every retry.

    Imports are local for the same reason as ``_hot_load_env_module``:
    after a ``reload_arena_modules()`` the module-top references are
    stale.
    """
    from isaaclab_arena.assets.registries import EnvironmentRegistry as _ER
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    registry = _ER()
    assert registry.is_registered(args_cli.example_environment), (
        f"env {args_cli.example_environment!r} not registered — hot-load step probably failed"
    )
    env_cls = registry.get_component_by_name(args_cli.example_environment)
    arena_env = env_cls().get_env(args_cli)
    return ArenaEnvBuilder(arena_env, args_cli)


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    add_ik_reachability_cli_args(parser)
    _add_auto_cli_args(parser)
    args_cli = parser.parse_args()

    out_dir = Path(args_cli.out_dir)
    max_attempts = max(1, int(args_cli.max_attempts))

    # Everything below must run *inside* the SimulationAppContext. The
    # asset-library imports triggered by ``build_catalog_text`` pull in
    # ``pxr`` modules; doing that before Kit boots prevents Kit's USD
    # schema extensions from initializing (UsdAPISchemaBase wrapper
    # error, see run_reachability_check.py for the same constraint).
    with SimulationAppContext(args_cli):
        # Stage 1: prompt → SceneSpec → ResolvedScene
        from isaaclab_arena.llm_env_gen.llm_agent import LLMAgent, build_catalog_text
        from isaaclab_arena.llm_env_gen.resolver import Resolver

        catalog = build_catalog_text()
        llm_kwargs = {"model": args_cli.llm_model} if args_cli.llm_model else {}
        agent = LLMAgent(**llm_kwargs)
        spec, _raw = agent.generate_spec(args_cli.prompt, catalog_text=catalog, temperature=args_cli.temperature)

        if args_cli.background_override and args_cli.background_override != spec.background:
            old_bg = spec.background
            new_bg = args_cli.background_override
            for rel in spec.initial_scene_graph:
                if rel.target == old_bg:
                    rel.target = new_bg
            for rel in spec.final_scene_graph:
                if rel.target == old_bg:
                    rel.target = new_bg
            spec.background = new_bg
            print(f"[auto_env] Background override: {old_bg!r} -> {new_bg!r}", flush=True)

        resolved = Resolver().resolve(spec)
        print(
            f"[auto_env] Parsed prompt → spec ({len(resolved.items)} items, embodiment={resolved.embodiment_name})",
            flush=True,
        )

        # Stage 2: rewrite + IK-check loop
        from isaaclab_arena.llm_env_gen.env_writer import write_env
        from isaaclab_arena.llm_env_gen.placement_proposer import block_initial_goal_satisfaction, propose_placement
        from isaaclab_arena.llm_env_gen.run_reachability_check import check_reachability_for_arena_builder

        # Each attempt registers a unique env name (``..._t<attempt>``).
        # Isaac Sim's gym + scene caches are keyed on env name, so
        # re-registering the same name across attempts caused the third
        # attempt onward to silently inherit the second attempt's
        # init_state for both robot and items. Distinct names sidestep
        # that — we pay only one extra registry entry per retry.
        trial_paths: list[Path] = []
        last_status: int | None = None
        winning_attempt: int | None = None

        for attempt in range(max_attempts):
            env_suffix = f"_t{attempt}"
            trial_path = write_env(resolved, spec, out_dir, attempt=attempt, env_suffix=env_suffix)
            trial_paths.append(trial_path)

            # Re-propose just for the placement note (env_name is the
            # suffixed file name). ``write_env`` already ran the same
            # propose, so this is cheap.
            placement = propose_placement(resolved, spec, attempt=attempt)
            if placement.robot_placement is not None:
                rp = placement.robot_placement
                placement_note = f"robot edge={rp.edge}, frac={rp.fraction:.3f}"
            else:
                placement_note = "no robot edge sampling for this background"
            trial_env_name = trial_path.stem
            print(
                f"[auto_env] attempt {attempt + 1}/{max_attempts}: wrote {trial_path.name} "
                f"as env {trial_env_name!r} ({placement_note})",
                flush=True,
            )

            _hot_load_env_module(trial_path, trial_env_name, attempt)

            from isaaclab_arena.assets.registries import EnvironmentRegistry as _ER

            registered_cls = _ER().get_component_by_name(trial_env_name)
            print(
                f"[auto_env] registered class: {registered_cls.__module__}.{registered_cls.__name__}",
                flush=True,
            )

            args_cli = _build_args_for_env(args_cli, trial_env_name, placement.embodiment_default)

            try:
                arena_builder = _build_arena_builder(args_cli)
                last_status = check_reachability_for_arena_builder(arena_builder, args_cli)
            except Exception as exc:  # noqa: BLE001 — surface any rebuild failure and abort the loop
                print(f"[auto_env] attempt {attempt + 1} raised during build/IK: {exc!r}", flush=True)
                raise

            if last_status == 0:
                winning_attempt = attempt
                print(f"[auto_env] feasible on attempt {attempt + 1}: {trial_path}", flush=True)
                break

            print(
                f"[auto_env] attempt {attempt + 1} failed (exit={last_status}); resampling robot placement.",
                flush=True,
            )

        if winning_attempt is None:
            print(
                f"[auto_env] exhausted {max_attempts} attempts without IK-feasible placement "
                f"(last status={last_status}). Trial envs left at {[str(p) for p in trial_paths]}.",
                flush=True,
            )
            return last_status if last_status is not None else 2

        # Promote the winning trial to the canonical env name (drop the
        # ``_t<N>`` suffix). The user gets a single ready-to-use file.
        canonical_path = write_env(resolved, spec, out_dir, attempt=winning_attempt, env_suffix="")
        # Sweep every ``<canonical_stem>_t*.py`` in ``out_dir`` so prior
        # runs' stragglers don't hang around either. The canonical file
        # itself is excluded by the ``_t`` infix.
        canonical_stem = canonical_path.stem
        for stale in canonical_path.parent.glob(f"{canonical_stem}_t*.py"):
            if stale.exists():
                stale.unlink()
        print(f"[auto_env] canonical env written: {canonical_path}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
