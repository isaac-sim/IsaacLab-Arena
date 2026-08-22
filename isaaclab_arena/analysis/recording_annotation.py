# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc annotation of a recordings folder with task information the rollout did not capture.

Every ``episode_results*.jsonl`` is the reference file: it names the ``job_name`` whose task yaml
defines the recorded scene, and its ``rebuild`` index names the sibling trajectory dataset. Two
annotations are available, each opt-in on the CLI:

- Static predicates (``--predicates``): a sibling ``*.static_predicates.jsonl`` listing every
  predicate the job's progress objectives define, whether or not it fired (see
  ``annotate_episode_results_dir``).
- Bounding boxes (``--bounding-boxes``): the local (object-frame) AABB of every recorded rigid
  object, written into the trajectory dataset itself (see ``annotate_bounding_boxes_dir``). Read
  them back with ``isaaclab_arena.analysis.aabb_overlap``.

Both need to reach into the ``pxr`` module, which requires a headless ``SimulationApp`` to already be
running. Since launching a ``SimulationApp`` is real work (tens of seconds), both the app launch and
the imports that reach ``pxr`` are deferred into the ``main`` function.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import numpy as np
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.progress_tracking.progress_tracking_utils import _predicate_repr
from isaaclab_arena.visualization.episode_results_files import (
    find_episode_results_files,
    parse_episode_results_filename,
)

STATIC_PREDICATES_SUFFIX = ".static_predicates.jsonl"
"""Appended to an episode_results*.jsonl stem to name its annotated sibling file."""

BOUNDING_BOXES_GROUP = "bounding_boxes"
"""Top-level HDF5 group the local AABBs are written to, alongside the recorder's own ``data``."""

ORIGINAL_DATASET_SUFFIX = ".copy.hdf5"
"""Replaces ``.hdf5`` to name the untouched original kept beside an annotated dataset."""

DEFAULT_MAX_IN_MEMORY_MB = 512
"""Datasets larger than this are refused rather than loaded into memory for annotation."""


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


def build_task_and_assets_from_job_name(job_name: str, env_package: str) -> tuple[Any, dict[str, Any]]:
    """Build ``job_name``'s (possibly composite) Task and the assets its scene instantiates.

    Requires a headless ``SimulationApp`` to already be running (see the module docstring): building
    the task resolves its assets' USD files, and remote-hosted ones need Kit's asset resolver.

    Returns:
        ``(task, assets_by_node_id)``, where the keys of the second element are the scene ids used
        both by the task yaml and by the recorded ``states/rigid_object`` group.
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
    return build_task_from_spec(spec.task, assets_by_node_id), assets_by_node_id


def build_task_from_job_name(job_name: str, env_package: str) -> Any:
    """Build the (possibly composite) Task instance for ``job_name``."""
    task, _ = build_task_and_assets_from_job_name(job_name, env_package)
    return task


def _task_for_objective(task: Any, objective: ProgressObjective) -> Any:
    """Return the subtask instance that produced ``objective``, or ``task`` itself if it is not composite."""
    if objective.parent_subtask_idx is not None:
        return task.subtasks[objective.parent_subtask_idx]
    return task


def _pick_and_place_targets(task: Any) -> dict[str, str] | None:
    """Return ``task``'s pick-up object and destination names, or ``None`` if it is not a ``PickAndPlaceTask``."""
    # Deferred: pulls in isaaclab, which this module otherwise keeps out of import time.
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

    if not isinstance(task, PickAndPlaceTask):
        return None
    return {
        "pick_up_object": task.pick_up_object.name,
        "destination_location": task.destination_location.name,
    }


def static_predicates_for_job(job_name: str, env_package: str) -> dict[str, dict[str, Any]]:
    """Return every predicate defined by ``job_name``'s progress objectives, whether or not any episode reached it.

    Args:
        job_name: A recorded episode's ``job_name`` field, matching a task yaml filename stem.
        env_package: Subpackage of ``isaaclab_arena_environments`` whose ``tasks/`` directory holds it
            (e.g. ``"robolab"``).

    Returns:
        ``{objective_name: {"pick_up_object": str, "destination_location": str, "groups": {...}}}``.
        The ``pick_up_object``/``destination_location`` keys are only present when the objective's
        subtask is a ``PickAndPlaceTask``. ``groups`` maps each group name to its ordered predicate
        chain, as ``[{"index": int, "predicate": str, "score": float}, ...]``. A composite task's
        objective names carry the same ``subtask_{i}/{name}`` prefixes as the recorded
        ``progress.objectives`` keys.
    """
    task = build_task_from_job_name(job_name, env_package)
    predicates: dict[str, dict[str, Any]] = {}
    for objective in task.get_progress_objectives():
        entry: dict[str, Any] = {}
        targets = _pick_and_place_targets(_task_for_objective(task, objective))
        if targets is not None:
            entry.update(targets)
        entry["groups"] = {
            group_name: [
                {"index": index, "predicate": _predicate_repr(predicate), "score": score}
                for index, (predicate, score) in enumerate(chain)
            ]
            for group_name, chain in objective.canonical_predicate_groups.items()
        }
        predicates[objective.name] = entry
    return predicates


def local_bounding_boxes_for_job(job_name: str, env_package: str) -> dict[str, dict[str, Any]]:
    """Return the local (object-frame) AABB of every rigid object in ``job_name``'s scene.

    The box is expressed in the object's own frame at the spawned scale, so it is independent of pose
    and therefore shared by every demo and frame of a recording. A caller places it in the world by
    rotating and translating it with a recorded ``root_pose``.

    Args:
        job_name: A recorded episode's ``job_name`` field, matching a task yaml filename stem.
        env_package: Subpackage of ``isaaclab_arena_environments`` whose ``tasks/`` directory holds it.

    Returns:
        ``{object_name: entry}``, where ``entry`` carries ``min_point``/``max_point`` of shape
        ``(1, 3)`` plus ``usd_path`` and ``scale``, or a single ``error`` string when the object's
        geometry cannot be resolved (e.g. a ``RigidObjectSet``, whose per-env variant assignment is
        not recoverable from the recording).
    """
    # Deferred: reaches isaaclab-backed asset classes and reads USD via pxr.
    from isaaclab_arena.assets.object_set import RigidObjectSet

    _, assets_by_node_id = build_task_and_assets_from_job_name(job_name, env_package)

    boxes: dict[str, dict[str, Any]] = {}
    for asset in assets_by_node_id.values():
        name = getattr(asset, "name", None)
        if name is None or not hasattr(asset, "get_bounding_box"):
            continue
        if isinstance(asset, RigidObjectSet):
            # get_bounding_box() would silently return the tallest member's box, and the per-env
            # variant assignment that says which member each env actually spawned is not recorded.
            boxes[name] = {
                "error": (
                    f"'{name}' is a RigidObjectSet; its per-env variant assignment is not recoverable "
                    "from the recording, so no bounding box was stored."
                )
            }
            continue
        try:
            bbox = asset.get_bounding_box()
        except Exception as exc:  # noqa: BLE001 - reported per object; must not abort the whole job
            boxes[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        boxes[name] = {
            "min_point": bbox.min_point.detach().cpu().numpy().astype(np.float64),
            "max_point": bbox.max_point.detach().cpu().numpy().astype(np.float64),
            "usd_path": str(getattr(asset, "usd_path", "") or ""),
            "scale": np.asarray(getattr(asset, "scale", (1.0, 1.0, 1.0)), dtype=np.float64),
        }
    return boxes


class _JobLookupCache:
    """Memoizes a per-job-name lookup across every file processed in one run.

    Building a task instance re-reads USD via pxr, so a recordings folder with many files/episodes for
    the same handful of jobs should only pay that cost once per job name.
    """

    def __init__(self, env_package: str, lookup_fn: Callable[[str, str], dict[str, Any]]):
        self._env_package = env_package
        self._lookup_fn = lookup_fn
        self._cache: dict[str, dict[str, Any] | Exception] = {}

    def get(self, job_name: str) -> dict[str, Any] | Exception:
        """Return the cached lookup result for ``job_name``, or the exception raised building it."""
        if job_name not in self._cache:
            try:
                self._cache[job_name] = self._lookup_fn(job_name, self._env_package)
            except Exception as exc:  # noqa: BLE001 - reported per-line/per-file; must not abort the run
                self._cache[job_name] = exc
        return self._cache[job_name]


def _annotated_output_path(episode_results_path: Path) -> Path:
    """Sibling output path, e.g. ``episode_results_rebuild0.jsonl`` -> ``..._rebuild0.static_predicates.jsonl``.

    The result intentionally does not match ``EPISODE_RESULTS_FILENAME_PATTERN``, so re-running this
    tool later does not treat its own output as a source file.
    """
    return episode_results_path.with_name(episode_results_path.stem + STATIC_PREDICATES_SUFFIX)


def _annotate_line(line: str, line_number: int, source_path: Path, cache: _JobLookupCache) -> dict[str, Any]:
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
    episode_results_path: str | Path, env_package: str, cache: _JobLookupCache | None = None
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
    cache = cache if cache is not None else _JobLookupCache(env_package, static_predicates_for_job)
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
    cache = _JobLookupCache(env_package, static_predicates_for_job)
    return [
        annotate_episode_results_file(path, env_package, cache=cache)
        for path in find_episode_results_files(recordings_root)
    ]


def job_names_in_episode_results(episode_results_path: str | Path) -> set[str]:
    """Return every ``job_name`` named by the records in one ``episode_results*.jsonl``."""
    job_names: set[str] = set()
    for line in Path(episode_results_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("job_name"), str):
            job_names.add(record["job_name"])
    return job_names


def dataset_path_for_episode_results(episode_results_path: str | Path, job_name: str) -> Path:
    """Return the trajectory dataset written alongside one ``episode_results*.jsonl``.

    Mirrors the ``f"dataset_{name}_rebuild{index}"`` filename the evaluation runner gives the
    recorder, pairing a results file with the dataset from the same rebuild.
    """
    episode_results_path = Path(episode_results_path)
    parsed = parse_episode_results_filename(episode_results_path.name)
    assert parsed is not None, f"Not an episode results filename: {episode_results_path.name}"
    return episode_results_path.with_name(f"dataset_{job_name}_rebuild{parsed.rebuild_index}.hdf5")


def _original_dataset_path(dataset_path: Path) -> Path:
    """Path the untouched original is kept at once its dataset has been annotated."""
    return dataset_path.with_suffix(ORIGINAL_DATASET_SUFFIX)


def _recorded_rigid_object_names(hdf5_file: Any) -> set[str]:
    """Return every rigid object whose per-step ``root_pose`` the dataset recorded."""
    names: set[str] = set()
    for demo in hdf5_file.get("data", {}).values():
        rigid_objects = demo.get("states", {}).get("rigid_object")
        if rigid_objects is not None:
            names.update(rigid_objects.keys())
    return names


def _write_bounding_boxes_group(hdf5_file: Any, boxes: dict[str, dict[str, Any]], job_name: str) -> list[str]:
    """Replace the dataset's bounding-box group with ``boxes``; return the names that carry an error."""
    import datetime

    if BOUNDING_BOXES_GROUP in hdf5_file:
        del hdf5_file[BOUNDING_BOXES_GROUP]
    group = hdf5_file.create_group(BOUNDING_BOXES_GROUP)
    group.attrs["job_name"] = job_name
    group.attrs["computed_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    failed: list[str] = []
    for name, entry in sorted(boxes.items()):
        object_group = group.create_group(name)
        if "error" in entry:
            object_group.attrs["error"] = entry["error"]
            failed.append(name)
            continue
        object_group.create_dataset("min_point", data=entry["min_point"])
        object_group.create_dataset("max_point", data=entry["max_point"])
        object_group.attrs["usd_path"] = entry["usd_path"]
        object_group.attrs["scale"] = entry["scale"]
    return failed


def annotate_bounding_boxes_for_dataset(
    dataset_path: str | Path,
    job_name: str,
    boxes: dict[str, dict[str, Any]],
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    max_in_memory_mb: int = DEFAULT_MAX_IN_MEMORY_MB,
) -> str:
    """Write the local AABBs of ``dataset_path``'s recorded rigid objects into the dataset itself.

    The original file is never opened for writing. It is copied into memory, annotated there, written
    out beside the original, and only then swapped in: the untouched original stays on disk under
    ``ORIGINAL_DATASET_SUFFIX``, and a reader sees either the old or the new file, never a partial
    one. Re-annotating with ``overwrite`` recomputes from that preserved original, so the backup
    stays pristine however many times the tool runs.

    Args:
        dataset_path: The recorded ``dataset_*.hdf5`` to annotate.
        job_name: Job whose task yaml the boxes were resolved from; recorded on the group.
        boxes: Per-object entries as returned by ``local_bounding_boxes_for_job``.
        overwrite: Re-annotate a dataset that already has a preserved original.
        dry_run: Report what would happen without writing anything.
        max_in_memory_mb: Refuse datasets larger than this rather than loading them into memory.

    Returns:
        A one-line human-readable report of what was done to this dataset.
    """
    import h5py

    dataset_path = Path(dataset_path)
    original_path = _original_dataset_path(dataset_path)

    if original_path.exists() and not overwrite:
        return f"skip (already annotated, original at {original_path.name}): {dataset_path}"

    # An earlier annotated run must never become the backup, so always read the pristine original.
    source_path = original_path if original_path.exists() else dataset_path
    size_mb = source_path.stat().st_size / (1024 * 1024)
    if size_mb > max_in_memory_mb:
        return f"skip ({size_mb:.0f} MB exceeds --max-in-memory-mb={max_in_memory_mb}): {dataset_path}"

    source_bytes = source_path.read_bytes()
    buffer = io.BytesIO(source_bytes)
    with h5py.File(buffer, "r+") as hdf5_file:
        format_version = int(hdf5_file.attrs.get("format_version", -1))
        assert format_version == 1, (
            f"{dataset_path.name} has format_version={format_version}; only 1 (XYZW quaternions) is supported. "
            "Convert legacy WXYZ datasets with HDF5DatasetFileHandler.convert_dataset_to_xyzw() first."
        )
        recorded = _recorded_rigid_object_names(hdf5_file)
        stored = {name: entry for name, entry in boxes.items() if name in recorded}
        missing = sorted(recorded - set(boxes))
        if dry_run:
            return (
                f"dry-run: would store {len(stored)} boxes"
                f"{f' (no asset for {missing})' if missing else ''}: {dataset_path}"
            )
        failed = _write_bounding_boxes_group(hdf5_file, stored, job_name)
    annotated_bytes = buffer.getvalue()

    temporary_path = dataset_path.with_name(dataset_path.name + ".tmp")
    with open(temporary_path, "wb") as handle:
        handle.write(annotated_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    if not original_path.exists():
        os.rename(dataset_path, original_path)
    os.replace(temporary_path, dataset_path)

    details = [f"{len(stored) - len(failed)} boxes"]
    if failed:
        details.append(f"{len(failed)} unresolved ({', '.join(failed)})")
    if missing:
        details.append(f"no asset for {', '.join(missing)}")
    return f"wrote {'; '.join(details)} (original at {original_path.name}): {dataset_path}"


def annotate_bounding_boxes_dir(
    recordings_root: str | Path,
    env_package: str,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    max_in_memory_mb: int = DEFAULT_MAX_IN_MEMORY_MB,
    boxes_lookup_fn: Callable[[str, str], dict[str, Any]] = local_bounding_boxes_for_job,
) -> list[str]:
    """Annotate the dataset paired with every ``episode_results*.jsonl`` under ``recordings_root``.

    Results files are the reference: each names its ``job_name`` and, through its ``rebuild`` index,
    its dataset. Several results files (one per rank) can name the same dataset, so datasets are
    annotated once each.

    Args:
        recordings_root: Folder to search recursively for recorded results files.
        env_package: Subpackage of ``isaaclab_arena_environments`` whose ``tasks/`` directory defines
            the recorded ``job_name``s (e.g. ``"robolab"``).
        overwrite: Re-annotate datasets that already have a preserved original.
        dry_run: Report what would happen without writing anything.
        max_in_memory_mb: Refuse datasets larger than this rather than loading them into memory.
        boxes_lookup_fn: Resolves a job's per-object boxes; injectable for tests.

    Returns:
        One human-readable report line per dataset considered.
    """
    cache = _JobLookupCache(env_package, boxes_lookup_fn)
    reports: list[str] = []
    seen: set[Path] = set()

    for results_path in find_episode_results_files(recordings_root):
        job_names = job_names_in_episode_results(results_path)
        if not job_names:
            reports.append(f"skip (no job_name records): {results_path}")
            continue
        if len(job_names) > 1:
            reports.append(f"skip (records name several jobs {sorted(job_names)}): {results_path}")
            continue
        job_name = job_names.pop()

        dataset_path = dataset_path_for_episode_results(results_path, job_name)
        if dataset_path in seen:
            continue
        seen.add(dataset_path)
        if not dataset_path.exists():
            # Runs without trajectory recording produce results and videos but no dataset.
            reports.append(f"skip (no dataset recorded): {dataset_path}")
            continue

        boxes = cache.get(job_name)
        if isinstance(boxes, Exception):
            reports.append(f"skip ({boxes}): {dataset_path}")
            continue

        try:
            reports.append(
                annotate_bounding_boxes_for_dataset(
                    dataset_path,
                    job_name,
                    boxes,
                    overwrite=overwrite,
                    dry_run=dry_run,
                    max_in_memory_mb=max_in_memory_mb,
                )
            )
        except Exception as exc:  # noqa: BLE001 - reported per dataset; must not abort the whole run
            reports.append(f"failed ({type(exc).__name__}: {exc}): {dataset_path}")
    return reports


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate a recordings folder with task information the rollout did not capture, using "
            "each episode_results*.jsonl as the reference file. Pick at least one annotation."
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
    parser.add_argument(
        "--predicates",
        action="store_true",
        help="Write a sibling *.static_predicates.jsonl per results file.",
    )
    parser.add_argument(
        "--bounding-boxes",
        action="store_true",
        help=(
            "Write each recorded object's local AABB into the paired dataset_*.hdf5, keeping the"
            f" untouched original beside it as *{ORIGINAL_DATASET_SUFFIX}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-annotate datasets that already have a preserved original, recomputing from that original.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what --bounding-boxes would do without writing anything.",
    )
    parser.add_argument(
        "--max-in-memory-mb",
        type=int,
        default=DEFAULT_MAX_IN_MEMORY_MB,
        help="Refuse datasets larger than this rather than loading them into memory (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    if not (args.predicates or args.bounding_boxes):
        parser.error("nothing to do: pass --predicates and/or --bounding-boxes")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # See module docstring for why these are deferred; launched once here for the whole directory
    # scan rather than per job.
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    sim_app_args = get_isaaclab_arena_cli_parser().parse_args([])
    sim_app_args.headless = True
    sim_app_args.enable_cameras = False
    with SimulationAppContext(sim_app_args):
        if args.predicates:
            output_paths = annotate_episode_results_dir(args.recordings_dir, args.env_package)
            if not output_paths:
                print(f"No episode_results*.jsonl files found under {args.recordings_dir}")
            else:
                for path in output_paths:
                    print(path)
        if args.bounding_boxes:
            reports = annotate_bounding_boxes_dir(
                args.recordings_dir,
                args.env_package,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                max_in_memory_mb=args.max_in_memory_mb,
            )
            if not reports:
                print(f"No datasets paired with episode_results*.jsonl found under {args.recordings_dir}")
            else:
                for report in reports:
                    print(report)


if __name__ == "__main__":
    main()
