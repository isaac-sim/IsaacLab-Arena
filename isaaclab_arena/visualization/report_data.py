# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scan recorded evaluation results and aggregate them into the evaluation report's data model.

Separated from rendering so the aggregation can be tested without producing HTML. Scanning reads only
the per-episode results and the recorder's mp4 filenames; no video file is opened.
"""

from __future__ import annotations

import functools
import pathlib
import re
from dataclasses import dataclass, field

from isaaclab_arena.evaluation.experiment_manifest import read_experiment_manifest
from isaaclab_arena.evaluation.reconstruct_experiment_manifest import infer_task_and_policy_labels
from isaaclab_arena.evaluation.run_status import RunStatus
from isaaclab_arena.recording.episode_results_files import (
    find_episode_results_files,
    parse_episode_results_rebuild_index,
    read_episode_results,
)
from isaaclab_arena.video.episode_video_files import parse_episode_video_filename

# Record fields rendered explicitly elsewhere (status badge / row label), so excluded from the
# per-episode metadata list to avoid duplication.
_METADATA_EXCLUDED_FIELDS = frozenset({"env_id", "episode_in_env", "success", "job_name", "progress"})

# Strips a predicate's arguments so ``object_on_destination(force_threshold=0.1)`` and its
# per-object variants collapse to the one funnel stage they represent.
_PREDICATE_ARGUMENTS_PATTERN = re.compile(r"\(.*\)$")

# Name used to group Runs when no task label can be established.
UNGROUPED_TASK = "(ungrouped)"


@dataclass
class EpisodeSummary:
    """One recorded episode: its outcome, its progress through the task, and its videos."""

    env_index: int
    """Index of the environment the episode ran in."""

    episode_index: int
    """Episode index within the (job, env), renumbered to be contiguous across rebuilds."""

    video_by_camera: dict[str, str]
    """Camera name -> mp4 path, relative to the scanned root."""

    record: dict = field(default_factory=dict)
    """The matching per-episode results record; empty when no record was found."""

    @property
    def success(self) -> bool | None:
        """Return whether the episode succeeded, or ``None`` when the task records no success term."""
        return self.record.get("success")

    @property
    def score(self) -> float | None:
        """Return the episode's achieved progress score."""
        return (self.record.get("progress") or {}).get("overall_score")

    @property
    def max_score(self) -> float | None:
        """Return the maximum score the episode's objectives could have reached.

        Objectives score one point per completed group, so their group total is the achievable
        maximum. Multi-object tasks declare one objective per object, so this is well above 1.
        """
        objectives = (self.record.get("progress") or {}).get("objectives") or {}
        total = sum(objective.get("total_groups", 0) for objective in objectives.values())
        return float(total) if total > 0 else None

    @property
    def progress_fraction(self) -> float | None:
        """Return the achieved fraction of the episode's achievable score, or ``None`` when unknown."""
        score, max_score = self.score, self.max_score
        if score is None or max_score is None:
            return None
        return max(0.0, min(1.0, score / max_score))

    @property
    def all_objectives_complete(self) -> bool | None:
        """Return whether every objective completed, or ``None`` when progress was not recorded."""
        progress = self.record.get("progress") or {}
        return progress.get("all_complete") if "all_complete" in progress else None

    @property
    def outcome_disagrees_with_progress(self) -> bool:
        """Return whether the success term and the progress objectives reached opposite conclusions.

        These are separate mechanisms — the task's success term is not required to be the conjunction
        of its progress objectives — so they can and do disagree. Worth surfacing rather than hiding,
        because a disagreement usually means one of the two is mis-specified.
        """
        success, complete = self.success, self.all_objectives_complete
        return success is not None and complete is not None and success != complete

    @property
    def metadata(self) -> dict:
        """Return the record fields not already shown as the episode's status or progress."""
        return {
            key: value
            for key, value in self.record.items()
            if key not in _METADATA_EXCLUDED_FIELDS and value is not None
        }


@dataclass
class FunnelStage:
    """How many objective instances reached one stage of a task's success predicate sequence."""

    index: int
    """Position of the predicate within its objective's sequence."""

    name: str
    """Predicate name with its arguments stripped."""

    num_reached: int
    """Objective instances that fired this predicate at least once."""


@dataclass
class PredicateSignal:
    """Whether one predicate of an objective's success sequence fired during a single episode."""

    index: int
    """Position of the predicate within the sequence."""

    name: str
    """Predicate name with its arguments stripped."""

    triggered: bool
    """Whether the predicate fired."""

    step: int | None = None
    """Environment step the predicate fired at, when it did."""

    detail: str = ""
    """Full predicate text including arguments, which name the object for multi-object tasks."""

    blocked: bool = False
    """Whether the objective was still waiting on this predicate when the episode ended."""


@dataclass
class ObjectiveProgress:
    """How far one objective of an episode got through its success predicate sequence."""

    name: str
    """Objective name, e.g. ``pick_and_place`` or ``subtask_3/pick_and_place``."""

    score: float
    """Score the objective achieved."""

    max_score: float
    """Score the objective could have achieved."""

    is_complete: bool
    """Whether the objective completed."""

    signals: list[PredicateSignal]
    """Every predicate of the sequence, in order, marked triggered or not."""

    @property
    def num_triggered(self) -> int:
        """Return how many of the objective's predicates fired."""
        return sum(1 for signal in self.signals if signal.triggered)


def _base_predicate_name(predicate_name: object) -> str:
    """Return a predicate name with its argument list stripped."""
    return _PREDICATE_ARGUMENTS_PATTERN.sub("", str(predicate_name))


def _episode_objectives(episode: EpisodeSummary, sequence: dict[int, str]) -> list[ObjectiveProgress]:
    """Break one episode down into its objectives and their per-predicate signals.

    Args:
        episode: Episode to break down.
        sequence: Predicate position -> name, pooled over the Run so predicates this episode never
            reached are still listed.
    """
    progress = episode.record.get("progress") or {}
    objectives = progress.get("objectives") or {}

    # Events carry the predicates that fired, keyed by the objective they belong to.
    fired: dict[str, dict[int, dict]] = {}
    for event in progress.get("events") or []:
        index = event.get("predicate_index")
        if index is not None:
            fired.setdefault(str(event.get("objective")), {})[index] = event

    ordered_indices = sorted(sequence)
    results = []
    for name in objectives if objectives else sorted(fired):
        detail = objectives.get(name, {}) if objectives else {}
        events_for_objective = fired.get(name, {})
        # An incomplete objective names the predicate it is still waiting on, which is where it stalled.
        blocked_names = {
            _base_predicate_name(predicate)
            for predicate in (detail.get("active_predicates") or {}).values()
            if predicate
        }
        signals = []
        for index in ordered_indices:
            event = events_for_objective.get(index)
            signals.append(
                PredicateSignal(
                    index=index,
                    name=sequence[index],
                    triggered=event is not None,
                    step=event.get("step") if event is not None else None,
                    detail=str(event.get("predicate_name", "")) if event is not None else "",
                    blocked=event is None and sequence[index] in blocked_names,
                )
            )
        total_groups = detail.get("total_groups", 0)
        results.append(
            ObjectiveProgress(
                name=name,
                score=float(detail.get("score", 0.0)),
                max_score=float(total_groups) if total_groups else 1.0,
                is_complete=bool(detail.get("is_complete", False)),
                signals=signals,
            )
        )
    return results


@dataclass
class JobSummary:
    """All recorded episodes for a single Run, with its aggregate outcome."""

    name: str
    """The Run (job) name, which is also its output sub-directory."""

    task: str
    """Task label grouping this Run with Runs of the same task."""

    policy: str
    """Policy label grouping this Run with Runs of the same policy."""

    cameras: list[str]
    """Ordered camera names recorded for this Run."""

    episodes: list[EpisodeSummary]
    """Every recorded episode, ordered by environment then episode index."""

    @property
    def num_episodes(self) -> int:
        """Return the number of recorded episodes."""
        return len(self.episodes)

    @property
    def num_successes(self) -> int:
        """Return the number of episodes whose success term evaluated true."""
        return sum(1 for episode in self.episodes if episode.success is True)

    @property
    def num_scored_episodes(self) -> int:
        """Return the number of episodes carrying a success term."""
        return sum(1 for episode in self.episodes if episode.success is not None)

    @property
    def success_rate(self) -> float | None:
        """Return the fraction of scored episodes that succeeded, or ``None`` when none are scored."""
        scored = self.num_scored_episodes
        return None if scored == 0 else self.num_successes / scored

    @property
    def mean_progress(self) -> float | None:
        """Return the mean achieved progress fraction over episodes that report progress."""
        fractions = [episode.progress_fraction for episode in self.episodes if episode.progress_fraction is not None]
        return None if not fractions else sum(fractions) / len(fractions)

    @property
    def num_videos(self) -> int:
        """Return the number of recorded video files across all episodes."""
        return sum(len(episode.video_by_camera) for episode in self.episodes)

    @functools.cached_property
    def predicate_sequence(self) -> dict[int, str]:
        """Return the task's success predicates by position, pooled over every episode of this Run.

        An episode's own events only name the predicates it actually reached, so the sequence is
        pooled across the Run to recover the ones a given episode never got to.

        Cached, because rendering asks for it once per episode and building it walks every episode.
        """
        names: dict[int, str] = {}
        for episode in self.episodes:
            for event in (episode.record.get("progress") or {}).get("events") or []:
                index = event.get("predicate_index")
                if index is not None:
                    names.setdefault(index, _base_predicate_name(event.get("predicate_name", "")))
        return names

    @property
    def funnel(self) -> list[FunnelStage]:
        """Return how far objective instances got through the task's predicate sequence.

        The unit is one (episode, objective) pair rather than one episode, because a multi-object
        task declares an objective per object and fires the same predicate once per object. Counting
        events instead would report far more than the number of episodes.
        """
        reached: dict[int, set[tuple[int, int, str]]] = {}
        for episode in self.episodes:
            for event in (episode.record.get("progress") or {}).get("events") or []:
                index = event.get("predicate_index")
                if index is None:
                    continue
                instance = (episode.env_index, episode.episode_index, str(event.get("objective")))
                reached.setdefault(index, set()).add(instance)
        names = self.predicate_sequence
        return [
            FunnelStage(index=index, name=names[index], num_reached=len(reached.get(index, ())))
            for index in sorted(names)
        ]

    def objectives_for(self, episode: EpisodeSummary) -> list[ObjectiveProgress]:
        """Return each objective of ``episode`` with which of its predicates fired and which did not.

        Args:
            episode: Episode of this Run to break down.
        """
        return _episode_objectives(episode, self.predicate_sequence)

    @property
    def num_objective_instances(self) -> int:
        """Return the number of (episode, objective) pairs the funnel is measured over."""
        instances = 0
        for episode in self.episodes:
            objectives = (episode.record.get("progress") or {}).get("objectives") or {}
            instances += len(objectives)
        # Fall back to the widest funnel stage when objectives were not recorded, so a stage can
        # never report more instances than the total it is a fraction of.
        widest_stage = max((stage.num_reached for stage in self.funnel), default=0)
        return max(instances, widest_stage)


@dataclass
class TaskSummary:
    """Every Run evaluating one task, one per policy."""

    name: str
    """Task label."""

    jobs: list[JobSummary]
    """Runs of this task, ordered by policy."""

    def job_for_policy(self, policy: str) -> JobSummary | None:
        """Return this task's Run for ``policy``, or ``None`` when it was not evaluated.

        Args:
            policy: Policy label to look up.
        """
        for job in self.jobs:
            if job.policy == policy:
                return job
        return None

    @property
    def num_episodes(self) -> int:
        """Return the total recorded episodes across this task's Runs."""
        return sum(job.num_episodes for job in self.jobs)

    @property
    def best_success_rate(self) -> float | None:
        """Return the highest success rate any policy reached on this task."""
        rates = [job.success_rate for job in self.jobs if job.success_rate is not None]
        return max(rates) if rates else None


@dataclass(frozen=True)
class RunExecutionReport:
    """Record whether one Run process completed and its process exit code."""

    run_name: str
    """Name of the Run that produced this execution result."""

    status: RunStatus
    """Whether the Run process completed or failed."""

    process_exit_code: int
    """Exit code returned by the Run process."""


@dataclass
class ExperimentSummary:
    """A whole evaluation: every Run, grouped by task and policy."""

    title: str
    """Title displayed at the top of the report."""

    tasks: list[TaskSummary]
    """Tasks evaluated, ordered by name."""

    policies: list[str]
    """Distinct policy labels, ordered by name; a single empty label when Runs are ungrouped."""

    run_executions: list[RunExecutionReport] = field(default_factory=list)
    """Run process results, when provided by a parallel Experiment collector."""

    grouping_source: str = "none"
    """How task and policy labels were established: ``manifest``, ``run_names``, or ``none``."""

    @property
    def jobs(self) -> list[JobSummary]:
        """Return every Run across every task."""
        return [job for task in self.tasks for job in task.jobs]

    @property
    def num_episodes(self) -> int:
        """Return the total recorded episodes."""
        return sum(job.num_episodes for job in self.jobs)

    @property
    def num_videos(self) -> int:
        """Return the total recorded video files."""
        return sum(job.num_videos for job in self.jobs)

    @property
    def is_grouped(self) -> bool:
        """Return whether Runs carry real task and policy labels rather than a flat fallback."""
        return self.grouping_source != "none"

    def success_rate_for_policy(self, policy: str) -> float | None:
        """Return the overall success rate for one policy across every task.

        Args:
            policy: Policy label to aggregate.
        """
        jobs = [job for job in self.jobs if job.policy == policy]
        scored = sum(job.num_scored_episodes for job in jobs)
        return None if scored == 0 else sum(job.num_successes for job in jobs) / scored

    def num_episodes_for_policy(self, policy: str) -> int:
        """Return the recorded episodes for one policy across every task.

        Args:
            policy: Policy label to aggregate.
        """
        return sum(job.num_episodes for job in self.jobs if job.policy == policy)

    @property
    def overall_success_rate(self) -> float | None:
        """Return the success rate across every scored episode of the Experiment."""
        scored = sum(job.num_scored_episodes for job in self.jobs)
        return None if scored == 0 else sum(job.num_successes for job in self.jobs) / scored


def _scan_results(root: pathlib.Path) -> dict[str, dict[tuple[int, int, int], dict]]:
    """Scan ``root`` for the recorder's JSONL files, indexed per job by ``(env, rebuild, episode)``.

    Args:
        root: Directory of evaluation results to scan.
    """
    results: dict[str, dict[tuple[int, int, int], dict]] = {}
    for path in find_episode_results_files(root):
        relative = path.relative_to(root)
        job = "" if relative.parent == pathlib.Path(".") else str(relative.parent)
        rebuild = parse_episode_results_rebuild_index(path.name)
        assert rebuild is not None, f"'{path.name}' was matched as a results file but carries no rebuild index"
        job_results = results.setdefault(job, {})
        for record in read_episode_results(path):
            job_results[(int(record["env_id"]), rebuild, int(record["episode_in_env"]))] = record
    return results


def scan_jobs(root: pathlib.Path) -> list[tuple[str, list[str], list[EpisodeSummary]]]:
    """Recursively scan ``root`` for episode results and recorder mp4s, grouped by job.

    Intended for two different output folder structures: the experiment runner writes one per-job
    sub-directory under ``root``, while the policy runner writes directly under ``root``. Files that
    do not match the recorder's naming pattern are ignored.

    Args:
        root: Directory of evaluation results to scan.

    Returns:
        One ``(job name, camera names, episodes)`` tuple per job, ordered by job name.
    """
    # job -> env -> {(rebuild, recorder_episode): {camera: relative_path}}
    raw: dict[str, dict[int, dict[tuple[int, int], dict[str, str]]]] = {}
    cameras_by_job: dict[str, list[str]] = {}
    results = _scan_results(root)

    for path in sorted(root.rglob("*.mp4")):
        parsed = parse_episode_video_filename(path.name)
        if parsed is None:
            continue
        relative = path.relative_to(root)
        job = "" if relative.parent == pathlib.Path(".") else str(relative.parent)
        rebuild = parsed.rebuild_index if parsed.rebuild_index is not None else 0

        envs = raw.setdefault(job, {})
        recordings = envs.setdefault(parsed.env_index, {})
        recordings.setdefault((rebuild, parsed.episode_index), {})[parsed.camera_name] = str(relative)

        cameras = cameras_by_job.setdefault(job, [])
        if parsed.camera_name not in cameras:
            cameras.append(parsed.camera_name)

    jobs = []
    for job in sorted(set(raw) | set(results)):
        episodes = []
        result_keys_by_env: dict[int, set[tuple[int, int]]] = {}
        for env_index, rebuild, recorder_episode in results.get(job, {}):
            result_keys_by_env.setdefault(env_index, set()).add((rebuild, recorder_episode))

        video_envs = raw.get(job, {})
        for env_index in sorted(set(video_envs) | set(result_keys_by_env)):
            # Renumber (rebuild, recorder_episode) pairs into a contiguous, rebuild-agnostic index.
            video_recordings = video_envs.get(env_index, {})
            recording_keys = set(video_recordings) | result_keys_by_env.get(env_index, set())
            for episode_index, recording_key in enumerate(sorted(recording_keys)):
                episodes.append(
                    EpisodeSummary(
                        env_index=env_index,
                        episode_index=episode_index,
                        video_by_camera=video_recordings.get(recording_key, {}),
                        record=results.get(job, {}).get((env_index, *recording_key), {}),
                    )
                )
        jobs.append((job, sorted(cameras_by_job.get(job, [])), episodes))
    return jobs


def resolve_job_labels(root: pathlib.Path, job_names: list[str]) -> tuple[dict[str, tuple[str, str]], str]:
    """Resolve each job's task and policy labels, preferring the Experiment manifest.

    Falls back to factorizing the job names, and finally to leaving every job ungrouped, so a report
    can always be built for output directories that predate manifests.

    Args:
        root: Experiment output directory, which may hold an ``experiment_manifest.json``.
        job_names: Job names discovered by scanning.

    Returns:
        A ``(job name -> (task, policy), grouping source)`` pair.
    """
    manifest = read_experiment_manifest(root)
    if manifest is not None and any(manifest.run_by_name(job_name) is not None for job_name in job_names):
        labels = {}
        for job_name in job_names:
            entry = manifest.run_by_name(job_name)
            labels[job_name] = (entry.task, entry.policy) if entry is not None else (job_name or UNGROUPED_TASK, "")
        return labels, "manifest"

    inferred = infer_task_and_policy_labels(job_names) if len(job_names) > 1 else None
    if inferred is not None:
        return inferred, "run_names"

    return {job_name: (job_name or UNGROUPED_TASK, "") for job_name in job_names}, "none"


def build_experiment_summary(
    root: str | pathlib.Path,
    title: str,
    run_executions: list[RunExecutionReport] | None = None,
) -> ExperimentSummary:
    """Scan ``root`` and aggregate its recorded results into the report's data model.

    Args:
        root: Directory of evaluation results to scan.
        title: Title displayed at the top of the report.
        run_executions: Optional Run process results supplied by a distributed collector.

    Returns:
        The aggregated Experiment, with Runs grouped by task and policy.
    """
    root = pathlib.Path(root)
    scanned = scan_jobs(root)
    failed_run_names = {
        run_execution.run_name for run_execution in (run_executions or []) if run_execution.status is RunStatus.FAILED
    }
    scanned = [entry for entry in scanned if entry[0] not in failed_run_names]

    labels, grouping_source = resolve_job_labels(root, [job_name for job_name, _, _ in scanned])

    jobs_by_task: dict[str, list[JobSummary]] = {}
    for job_name, cameras, episodes in scanned:
        task, policy = labels[job_name]
        jobs_by_task.setdefault(task, []).append(
            JobSummary(name=job_name, task=task, policy=policy, cameras=cameras, episodes=episodes)
        )

    tasks = [
        TaskSummary(name=task, jobs=sorted(jobs_by_task[task], key=lambda job: job.policy))
        for task in sorted(jobs_by_task)
    ]
    policies = sorted({job.policy for task in tasks for job in task.jobs})
    return ExperimentSummary(
        title=title,
        tasks=tasks,
        policies=policies,
        run_executions=list(run_executions or []),
        grouping_source=grouping_source,
    )
