# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static listing of a task's progress-objective predicates, independent of any recorded rollout.

Provides a CLI that walks a recordings folder and, for every ``episode_results*.jsonl`` file,
writes a sibling file with one predicates line per input line (see ``annotate_episode_results_dir``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from isaaclab_arena.progress_tracking.progress_tracking_utils import _predicate_repr
from isaaclab_arena.visualization.episode_results_files import find_episode_results_files

STATIC_PREDICATES_SUFFIX = ".static_predicates.jsonl"
"""Appended to an episode_results*.jsonl stem to name its annotated sibling file."""


def _environments_package_root() -> Path:
    """Resolve the installed location of the ``isaaclab_arena_environments`` package."""
    spec = importlib.util.find_spec("isaaclab_arena_environments")
    assert spec is not None and spec.origin is not None, "isaaclab_arena_environments package not found on path"
    return Path(spec.origin).parent


def _task_yaml_path(job_name: str, env_package: str) -> Path:
    """Return the task-graph YAML path for ``job_name`` under ``<env_package>/tasks/``.

    Args:
        job_name: A recorded episode's ``job_name`` field, matching a task yaml filename stem.
        env_package: Subpackage of ``isaaclab_arena_environments`` whose ``tasks/`` directory holds it
            (e.g. ``"robolab"``).
    """
    yaml_path = _environments_package_root() / env_package / "tasks" / f"{job_name}.yaml"
    assert yaml_path.is_file(), f"No task yaml for job_name={job_name!r} under env_package={env_package!r}: {yaml_path}"
    return yaml_path


def build_task_from_job_name(job_name: str, env_package: str) -> Any:
    """Build the (possibly composite) Task instance for ``job_name``.

    Requires a headless ``SimulationApp`` to already be running (see the module docstring): building
    the task resolves its assets' USD files, and remote-hosted ones need Kit's asset resolver.
    """
    # Deferred: the env-graph spec modules pull in isaaclab, and building a concrete task reads USD via
    # pxr, so keep both out of this module's import time.
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import (
        build_task_from_spec,
        instantiate_assets_from_spec,
    )
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    spec = ArenaEnvGraphSpec.from_yaml(_task_yaml_path(job_name, env_package))
    assets_by_node_id = instantiate_assets_from_spec(spec, AssetRegistry())
    return build_task_from_spec(spec.task, assets_by_node_id)


def static_predicates_for_job(job_name: str, env_package: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return every predicate defined by ``job_name``'s progress objectives, whether or not any episode reached it.

    Args:
        job_name: A recorded episode's ``job_name`` field, matching a task yaml filename stem.
        env_package: Subpackage of ``isaaclab_arena_environments`` whose ``tasks/`` directory holds it
            (e.g. ``"robolab"``).

    Returns:
        ``{objective_name: {group_name: [{"index": int, "predicate": str, "score": float}, ...]}}``,
        one entry per group's ordered predicate chain. A composite task's objective names carry the
        same ``subtask_{i}/{name}`` prefixes as the recorded ``progress.objectives`` keys.
    """
    task = build_task_from_job_name(job_name, env_package)
    predicates: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for objective in task.get_progress_objectives():
        predicates[objective.name] = {
            group_name: [
                {"index": index, "predicate": _predicate_repr(predicate), "score": score}
                for index, (predicate, score) in enumerate(chain)
            ]
            for group_name, chain in objective.canonical_predicate_groups.items()
        }
    return predicates


class _JobPredicateCache:
    """Memoizes a per-job-name predicate lookup across every file processed in one run.

    Building a task instance re-reads USD via pxr, so a recordings folder with many files/episodes for
    the same handful of jobs should only pay that cost once per job name.
    """

    def __init__(self, env_package: str, lookup_fn: Callable[[str, str], dict[str, Any]] = static_predicates_for_job):
        self._env_package = env_package
        self._lookup_fn = lookup_fn
        self._cache: dict[str, dict[str, Any] | Exception] = {}

    def get(self, job_name: str) -> dict[str, Any] | Exception:
        """Return the cached predicates dict for ``job_name``, or the exception raised building it."""
        if job_name not in self._cache:
            try:
                self._cache[job_name] = self._lookup_fn(job_name, self._env_package)
            except Exception as exc:  # noqa: BLE001 - reported per-line; must not abort the whole run
                self._cache[job_name] = exc
        return self._cache[job_name]


def _annotated_output_path(episode_results_path: Path) -> Path:
    """Sibling output path, e.g. ``episode_results_rebuild0.jsonl`` -> ``..._rebuild0.static_predicates.jsonl``.

    The result intentionally does not match ``EPISODE_RESULTS_FILENAME_PATTERN``, so re-running this
    tool later does not treat its own output as a source file.
    """
    return episode_results_path.with_name(episode_results_path.stem + STATIC_PREDICATES_SUFFIX)


def _annotate_line(line: str, line_number: int, source_path: Path, cache: _JobPredicateCache) -> dict[str, Any]:
    """Build one output record for one non-blank input line, never raising."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        return {"error": f"{source_path.name} line {line_number}: invalid JSON ({exc.msg})"}
    if not isinstance(record, dict) or "job_name" not in record:
        return {"error": f"{source_path.name} line {line_number}: expected a JSON object with a 'job_name' field"}

    job_name = record["job_name"]
    predicates = cache.get(job_name)
    if isinstance(predicates, Exception):
        return {"error": f"{source_path.name} line {line_number}: {predicates}"}

    return {
        "job_name": job_name,
        "env_id": record.get("env_id"),
        "episode_in_env": record.get("episode_in_env"),
        "predicates": predicates,
    }


def annotate_episode_results_file(
    episode_results_path: str | Path, env_package: str, cache: _JobPredicateCache | None = None
) -> Path:
    """Write a sibling file with one predicates record per line of ``episode_results_path``.

    Preserves exact line-for-line correspondence with the source file: a blank input line yields a
    blank output line, and a line that fails to parse or resolve yields an ``{"error": ...}`` record
    rather than aborting the rest of the file.

    Args:
        episode_results_path: One recorded ``episode_results*.jsonl`` file.
        env_package: Subpackage of ``isaaclab_arena_environments`` whose ``tasks/`` directory defines
            the recorded ``job_name``s (e.g. ``"robolab"``).
        cache: Reused across multiple files by ``annotate_episode_results_dir``; built fresh if omitted.

    Returns:
        The path of the annotated sibling file that was written.
    """
    episode_results_path = Path(episode_results_path)
    cache = cache if cache is not None else _JobPredicateCache(env_package)
    output_path = _annotated_output_path(episode_results_path)

    lines = episode_results_path.read_text(encoding="utf-8").splitlines()
    output_lines = [
        "" if not line.strip() else json.dumps(_annotate_line(line.strip(), line_number, episode_results_path, cache))
        for line_number, line in enumerate(lines, start=1)
    ]

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return output_path


def annotate_episode_results_dir(recordings_root: str | Path, env_package: str) -> list[Path]:
    """Annotate every ``episode_results*.jsonl`` found under ``recordings_root``.

    Args:
        recordings_root: Folder to search recursively for recorded results files.
        env_package: Subpackage of ``isaaclab_arena_environments`` whose ``tasks/`` directory defines
            the recorded ``job_name``s (e.g. ``"robolab"``).

    Returns:
        The paths of the annotated sibling files that were written, one per source file found.
    """
    cache = _JobPredicateCache(env_package)
    return [
        annotate_episode_results_file(path, env_package, cache=cache)
        for path in find_episode_results_files(recordings_root)
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate every episode_results*.jsonl found under a recordings folder with the full "
            "static predicate definitions for each episode's task, regardless of which predicates "
            "actually fired at runtime."
        )
    )
    parser.add_argument("recordings_dir", type=Path, help="Folder to search recursively for episode_results*.jsonl.")
    parser.add_argument(
        "--env-package",
        required=True,
        help=(
            "Subpackage of isaaclab_arena_environments whose tasks/ directory defines the recorded"
            " job_names, e.g. 'robolab'."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Deferred: launching a SimulationApp is real work (tens of seconds), so pay it once for the
    # whole directory scan rather than per job — and keep it out of this module's import time.
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    sim_app_args = get_isaaclab_arena_cli_parser().parse_args([])
    sim_app_args.headless = True
    sim_app_args.enable_cameras = False

    # Report before the `with` block exits: SimulationAppContext.__exit__ calls Kit's app.close(),
    # which can terminate the process immediately, before any code after the block gets to run.
    with SimulationAppContext(sim_app_args):
        output_paths = annotate_episode_results_dir(args.recordings_dir, args.env_package)
        if not output_paths:
            print(f"No episode_results*.jsonl files found under {args.recordings_dir}")
        else:
            for path in output_paths:
                print(path)


if __name__ == "__main__":
    main()
