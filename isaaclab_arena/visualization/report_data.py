# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Aggregate recorded evaluation artifacts into the report's data model.

This module is intentionally leaf-only: it reads JSONL records and filenames without importing the
evaluation, video, policy, environment, Isaac Sim, or Isaac Lab stacks.
"""

from __future__ import annotations

import functools
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from isaaclab_arena.visualization.episode_results_files import (
    DataIssue,
    find_episode_results_files,
    find_episode_video_files,
    parse_episode_results_filename,
    parse_episode_video_filename,
    read_episode_results,
)

COMPLETED_STATUS = "completed"
FAILED_STATUS = "failed"
DEFAULT_POLICY_SUFFIXES = ()

# Record fields rendered explicitly elsewhere, so excluded from per-episode metadata.
_METADATA_EXCLUDED_FIELDS = frozenset({"env_id", "episode_in_env", "success", "job_name", "progress"})
_PREDICATE_ARGUMENTS_PATTERN = re.compile(r"\(.*\)$")
_SUBTASK_OBJECTIVE_PATTERN = re.compile(r"^subtask_\d+/(?P<family>.+)$")
UNGROUPED_TASK = "(ungrouped)"


@dataclass(frozen=True)
class EpisodeIdentity:
    result_source: str
    rebuild_index: int
    env_index: int
    recorder_episode_index: int


@dataclass
class EpisodeSummary:
    identity: EpisodeIdentity
    episode_index: int
    video_by_camera: dict[str, str]
    record: dict[str, Any] = field(default_factory=dict)

    @property
    def env_index(self) -> int:
        return self.identity.env_index

    @property
    def rebuild_index(self) -> int:
        return self.identity.rebuild_index

    @property
    def success(self) -> bool | None:
        success = self.record.get("success")
        return success if isinstance(success, bool) else None

    @property
    def max_score(self) -> float | None:
        objectives = _progress_objectives(self.record)
        total = sum(_as_float(objective.get("total_groups")) or 0.0 for objective in objectives.values())
        return total if total > 0 else None

    @property
    def progress_fraction(self) -> float | None:
        score, max_score = _as_float(_progress(self.record).get("overall_score")), self.max_score
        if score is None or max_score is None:
            return None
        return max(0.0, min(1.0, score / max_score))

    @property
    def all_objectives_complete(self) -> bool | None:
        progress = self.record.get("progress")
        if not isinstance(progress, dict) or "all_complete" not in progress:
            return None
        all_complete = progress.get("all_complete")
        return all_complete if isinstance(all_complete, bool) else None

    @property
    def outcome_disagrees_with_progress(self) -> bool:
        success, complete = self.success, self.all_objectives_complete
        return success is not None and complete is not None and success != complete

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.record.items()
            if key not in _METADATA_EXCLUDED_FIELDS and value is not None
        }


@dataclass
class FunnelStage:
    index: int
    name: str
    num_reached: int


@dataclass
class ObjectiveFunnel:
    name: str
    num_instances: int
    stages: list[FunnelStage]


@dataclass
class PredicateSignal:
    index: int
    name: str
    triggered: bool
    step: int | None = None
    detail: str = ""
    blocked: bool = False


@dataclass
class ObjectiveProgress:
    name: str
    family: str
    score: float
    max_score: float
    is_complete: bool
    signals: list[PredicateSignal]
    blocked_predicates: list[str] = field(default_factory=list)

    @property
    def num_triggered(self) -> int:
        return sum(1 for signal in self.signals if signal.triggered)


@dataclass(frozen=True)
class RunExecutionReport:
    """Record whether one Run process completed and its process exit code."""

    run_name: str
    status: object
    process_exit_code: int


@dataclass
class JobSummary:
    name: str
    task: str
    policy: str
    cameras: list[str]
    episodes: list[EpisodeSummary]
    issues: list[DataIssue] = field(default_factory=list)

    _objective_family_by_name: dict[str, str] = field(init=False, repr=False)
    _family_sequences: dict[str, dict[int, str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._objective_family_by_name, family_issues = _build_objective_family_map(self.name, self.episodes)
        self._family_sequences, sequence_issues = _build_family_sequences(
            self.name, self.episodes, self._objective_family_by_name
        )
        self.issues.extend(family_issues)
        self.issues.extend(sequence_issues)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_successes(self) -> int:
        return sum(1 for episode in self.episodes if episode.success is True)

    @property
    def num_scored_episodes(self) -> int:
        return sum(1 for episode in self.episodes if episode.success is not None)

    @property
    def success_rate(self) -> float | None:
        scored = self.num_scored_episodes
        return None if scored == 0 else self.num_successes / scored

    @property
    def mean_progress(self) -> float | None:
        fractions = [episode.progress_fraction for episode in self.episodes if episode.progress_fraction is not None]
        return None if not fractions else sum(fractions) / len(fractions)

    @property
    def num_videos(self) -> int:
        return sum(len(episode.video_by_camera) for episode in self.episodes)

    @functools.cached_property
    def funnels(self) -> list[ObjectiveFunnel]:
        instances_by_family: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
        reached_by_family_index: dict[tuple[str, int], set[tuple[int, int, str]]] = defaultdict(set)
        for episode in self.episodes:
            for objective_name in _episode_objective_names(episode):
                family = self._objective_family_by_name.get(objective_name, objective_name)
                instances_by_family[family].add((episode.env_index, episode.episode_index, objective_name))
            for event in _progress_events(episode.record):
                objective_name = _event_objective_name(event)
                index = _as_int(event.get("predicate_index"))
                if objective_name is None or index is None:
                    continue
                family = self._objective_family_by_name.get(objective_name, objective_name)
                instance = (episode.env_index, episode.episode_index, objective_name)
                instances_by_family[family].add(instance)
                reached_by_family_index[(family, index)].add(instance)

        funnels = []
        for family in sorted(self._family_sequences):
            sequence = self._family_sequences[family]
            stages = [
                FunnelStage(
                    index=index, name=sequence[index], num_reached=len(reached_by_family_index[(family, index)])
                )
                for index in sorted(sequence)
            ]
            funnels.append(
                ObjectiveFunnel(name=family, num_instances=len(instances_by_family.get(family, ())), stages=stages)
            )
        return funnels

    def objectives_for(self, episode: EpisodeSummary) -> list[ObjectiveProgress]:
        objectives = _progress_objectives(episode.record)
        names = list(objectives) if objectives else sorted(_event_objectives(episode.record))
        results = []
        fired = _events_by_objective_and_index(episode.record)
        for name in names:
            detail = objectives.get(name, {}) if objectives else {}
            family = self._objective_family_by_name.get(name, name)
            sequence = self._family_sequences.get(family, {})
            active_names = [
                _base_predicate_name(predicate)
                for predicate in (detail.get("active_predicates") or {}).values()
                if predicate
            ]
            signals = []
            matched_blocked: set[str] = set()
            for index in sorted(sequence):
                event = fired.get(name, {}).get(index)
                blocked = event is None and sequence[index] in active_names
                if blocked:
                    matched_blocked.add(sequence[index])
                signals.append(
                    PredicateSignal(
                        index=index,
                        name=sequence[index],
                        triggered=event is not None,
                        step=_as_int(event.get("step")) if event is not None else None,
                        detail=str(event.get("predicate_name", "")) if event is not None else "",
                        blocked=blocked,
                    )
                )
            total_groups = _as_float(detail.get("total_groups")) if isinstance(detail, dict) else None
            results.append(
                ObjectiveProgress(
                    name=name,
                    family=family,
                    score=_as_float(detail.get("score")) or 0.0 if isinstance(detail, dict) else 0.0,
                    max_score=total_groups if total_groups and total_groups > 0 else 1.0,
                    is_complete=bool(detail.get("is_complete", False)) if isinstance(detail, dict) else False,
                    signals=signals,
                    blocked_predicates=[name for name in active_names if name not in matched_blocked],
                )
            )
        return results


@dataclass
class TaskSummary:
    name: str
    jobs: list[JobSummary]

    def job_for_policy(self, policy: str) -> JobSummary | None:
        for job in self.jobs:
            if job.policy == policy:
                return job
        return None

    @property
    def num_episodes(self) -> int:
        return sum(job.num_episodes for job in self.jobs)


@dataclass
class ExperimentSummary:
    title: str
    tasks: list[TaskSummary]
    policies: list[str]
    run_executions: list[RunExecutionReport] = field(default_factory=list)
    grouping_source: str = "none"
    issues: list[DataIssue] = field(default_factory=list)

    @property
    def jobs(self) -> list[JobSummary]:
        return [job for task in self.tasks for job in task.jobs]

    @property
    def num_episodes(self) -> int:
        return sum(job.num_episodes for job in self.jobs)

    @property
    def num_videos(self) -> int:
        return sum(job.num_videos for job in self.jobs)

    @property
    def is_grouped(self) -> bool:
        return self.grouping_source != "none" and bool(self.policies)

    def success_rate_for_policy(self, policy: str) -> float | None:
        jobs = [job for job in self.jobs if job.policy == policy]
        scored = sum(job.num_scored_episodes for job in jobs)
        return None if scored == 0 else sum(job.num_successes for job in jobs) / scored

    def num_episodes_for_policy(self, policy: str) -> int:
        return sum(job.num_episodes for job in self.jobs if job.policy == policy)

    @property
    def overall_success_rate(self) -> float | None:
        scored = sum(job.num_scored_episodes for job in self.jobs)
        return None if scored == 0 else sum(job.num_successes for job in self.jobs) / scored


@dataclass
class _ScannedJob:
    name: str
    cameras: list[str]
    episodes: list[EpisodeSummary]
    issues: list[DataIssue]


def normalize_run_status(status: object) -> str:
    """Normalize a string or enum-like run status to a lowercase string."""
    value = getattr(status, "value", status)
    return str(value).lower()


def is_failed_execution(execution: RunExecutionReport) -> bool:
    """Return whether a run execution record describes a failed run."""
    return normalize_run_status(execution.status) == FAILED_STATUS


def is_completed_execution(execution: RunExecutionReport) -> bool:
    """Return whether a run execution record describes a completed run."""
    return normalize_run_status(execution.status) == COMPLETED_STATUS


def _base_predicate_name(predicate_name: object) -> str:
    return _PREDICATE_ARGUMENTS_PATTERN.sub("", str(predicate_name))


def _candidate_family_name(objective_name: str) -> str:
    match = _SUBTASK_OBJECTIVE_PATTERN.match(objective_name)
    return objective_name if match is None else match.group("family")


def _build_objective_family_map(
    job_name: str,
    episodes: list[EpisodeSummary],
) -> tuple[dict[str, str], list[DataIssue]]:
    exact_names = sorted(_objective_names(episodes))
    candidates: dict[str, list[str]] = defaultdict(list)
    for name in exact_names:
        candidates[_candidate_family_name(name)].append(name)

    issues = []
    family_by_name: dict[str, str] = {}
    for candidate, names in sorted(candidates.items()):
        if len(names) == 1:
            family_by_name[names[0]] = names[0]
            continue
        if _objective_names_are_compatible(episodes, names):
            for name in names:
                family_by_name[name] = candidate
        else:
            issues.append(
                DataIssue(
                    job_name or ".",
                    f"objective family '{candidate}' has conflicting predicate sequences; showing exact objectives",
                )
            )
            for name in names:
                family_by_name[name] = name
    return family_by_name, issues


def _build_family_sequences(
    job_name: str,
    episodes: list[EpisodeSummary],
    family_by_name: dict[str, str],
) -> tuple[dict[str, dict[int, str]], list[DataIssue]]:
    names_by_family_index: dict[tuple[str, int], set[str]] = defaultdict(set)
    for episode in episodes:
        for event in _progress_events(episode.record):
            objective_name = _event_objective_name(event)
            index = _as_int(event.get("predicate_index"))
            if objective_name is None or index is None:
                continue
            family = family_by_name.get(objective_name, objective_name)
            names_by_family_index[(family, index)].add(_base_predicate_name(event.get("predicate_name", "")))

    issues = []
    sequences: dict[str, dict[int, str]] = defaultdict(dict)
    for (family, index), names in names_by_family_index.items():
        if len(names) > 1:
            issues.append(
                DataIssue(
                    job_name or ".",
                    f"objective family '{family}' has multiple predicate names at index {index}: {sorted(names)}",
                )
            )
        sequences[family][index] = sorted(names)[0]
    return dict(sequences), issues


def _objective_names_are_compatible(episodes: list[EpisodeSummary], objective_names: list[str]) -> bool:
    names_by_index: dict[int, set[str]] = defaultdict(set)
    objective_name_set = set(objective_names)
    for episode in episodes:
        for event in _progress_events(episode.record):
            objective_name = _event_objective_name(event)
            index = _as_int(event.get("predicate_index"))
            if objective_name in objective_name_set and index is not None:
                names_by_index[index].add(_base_predicate_name(event.get("predicate_name", "")))
    return all(len(names) <= 1 for names in names_by_index.values())


def _progress(record: dict[str, Any]) -> dict[str, Any]:
    progress = record.get("progress")
    return progress if isinstance(progress, dict) else {}


def _progress_objectives(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objectives = _progress(record).get("objectives")
    return objectives if isinstance(objectives, dict) else {}


def _progress_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    events = _progress(record).get("events")
    return [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []


def _event_objective_name(event: dict[str, Any]) -> str | None:
    objective = event.get("objective")
    return str(objective) if objective is not None else None


def _event_objectives(record: dict[str, Any]) -> set[str]:
    return {objective for event in _progress_events(record) if (objective := _event_objective_name(event)) is not None}


def _episode_objective_names(episode: EpisodeSummary) -> set[str]:
    return set(_progress_objectives(episode.record)) | _event_objectives(episode.record)


def _objective_names(episodes: list[EpisodeSummary]) -> set[str]:
    names = set()
    for episode in episodes:
        names.update(_episode_objective_names(episode))
    return names


def _events_by_objective_and_index(record: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for event in _progress_events(record):
        objective_name = _event_objective_name(event)
        index = _as_int(event.get("predicate_index"))
        if objective_name is not None and index is not None:
            result.setdefault(objective_name, {})[index] = event
    return result


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _job_name_for_path(path: pathlib.Path, root: pathlib.Path) -> str:
    relative = path.relative_to(root)
    return "" if relative.parent == pathlib.Path(".") else str(relative.parent)


def _validate_record(record: dict[str, Any], path: pathlib.Path, root: pathlib.Path) -> tuple[int, int] | DataIssue:
    display_path = str(path.relative_to(root))
    env_id = _as_int(record.get("env_id"))
    episode = _as_int(record.get("episode_in_env"))
    if env_id is None:
        return DataIssue(display_path, "record missing integer env_id")
    if episode is None:
        return DataIssue(display_path, "record missing integer episode_in_env")
    return env_id, episode


def _scan_results(root: pathlib.Path) -> tuple[dict[str, dict[EpisodeIdentity, dict[str, Any]]], list[DataIssue]]:
    results: dict[str, dict[EpisodeIdentity, dict[str, Any]]] = defaultdict(dict)
    issues: list[DataIssue] = []
    for path in find_episode_results_files(root):
        parsed = parse_episode_results_filename(path.name)
        assert parsed is not None, f"'{path.name}' was matched as a results file but did not parse"
        job = _job_name_for_path(path, root)
        source = str(path.relative_to(root)) if parsed.rank_index is not None else ""
        records, read_issues = read_episode_results(path, root)
        issues.extend(read_issues)
        for record in records:
            validated = _validate_record(record, path, root)
            if isinstance(validated, DataIssue):
                issues.append(validated)
                continue
            env_id, episode_in_env = validated
            identity = EpisodeIdentity(source, parsed.rebuild_index, env_id, episode_in_env)
            if identity in results[job]:
                issues.append(DataIssue(str(path.relative_to(root)), "duplicate episode record ignored"))
                continue
            results[job][identity] = record
    return dict(results), issues


def _scan_videos(
    root: pathlib.Path,
) -> tuple[dict[str, dict[tuple[int, int, int], dict[str, str]]], dict[str, list[str]]]:
    videos: dict[str, dict[tuple[int, int, int], dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    cameras_by_job: dict[str, list[str]] = defaultdict(list)
    for path in find_episode_video_files(root):
        parsed = parse_episode_video_filename(path.name)
        assert parsed is not None, f"'{path.name}' was matched as a video file but did not parse"
        job = _job_name_for_path(path, root)
        key = (parsed.rebuild_index, parsed.env_index, parsed.episode_index)
        videos[job][key][parsed.camera_name] = str(path.relative_to(root))
        if parsed.camera_name not in cameras_by_job[job]:
            cameras_by_job[job].append(parsed.camera_name)
    return {job: dict(entries) for job, entries in videos.items()}, dict(cameras_by_job)


def _scan_jobs(root: pathlib.Path) -> tuple[list[_ScannedJob], list[DataIssue]]:
    root = pathlib.Path(root)
    results, result_issues = _scan_results(root)
    videos, cameras_by_job = _scan_videos(root)
    issues = list(result_issues)
    jobs = []
    for job in sorted(set(results) | set(videos)):
        job_results = results.get(job, {})
        job_videos = videos.get(job, {})
        result_keys_by_video_key: dict[tuple[int, int, int], list[EpisodeIdentity]] = defaultdict(list)
        for identity in job_results:
            result_keys_by_video_key[
                (identity.rebuild_index, identity.env_index, identity.recorder_episode_index)
            ].append(identity)

        episodes_by_env: dict[int, list[tuple[EpisodeIdentity, dict[str, Any], dict[str, str]]]] = defaultdict(list)
        consumed_video_keys: set[tuple[int, int, int]] = set()
        for identity, record in job_results.items():
            video_key = (identity.rebuild_index, identity.env_index, identity.recorder_episode_index)
            same_record_keys = result_keys_by_video_key[video_key]
            if len(same_record_keys) == 1:
                video_by_camera = job_videos.get(video_key, {})
                consumed_video_keys.add(video_key)
            else:
                video_by_camera = {}
                issues.append(
                    DataIssue(
                        job or ".",
                        "multiple rank records share one video key; leaving videos unpaired for that key",
                    )
                )
            episodes_by_env[identity.env_index].append((identity, record, video_by_camera))

        for video_key, video_by_camera in job_videos.items():
            if video_key in consumed_video_keys:
                continue
            rebuild_index, env_index, recorder_episode_index = video_key
            identity = EpisodeIdentity("", rebuild_index, env_index, recorder_episode_index)
            episodes_by_env[env_index].append((identity, {}, video_by_camera))

        episodes = []
        for env_index in sorted(episodes_by_env):
            env_entries = sorted(
                episodes_by_env[env_index],
                key=lambda item: (
                    item[0].rebuild_index,
                    item[0].recorder_episode_index,
                    item[0].result_source,
                ),
            )
            for display_episode_index, (identity, record, video_by_camera) in enumerate(env_entries):
                episodes.append(
                    EpisodeSummary(
                        identity=identity,
                        episode_index=display_episode_index,
                        video_by_camera=video_by_camera,
                        record=record,
                    )
                )
        jobs.append(
            _ScannedJob(
                name=job,
                cameras=sorted(cameras_by_job.get(job, [])),
                episodes=episodes,
                issues=[issue for issue in issues if issue.path == (job or ".")],
            )
        )
    return jobs, issues


def _infer_task_and_policy_labels_with_source(
    job_names: list[str],
    policy_suffixes: tuple[str, ...] = DEFAULT_POLICY_SUFFIXES,
) -> tuple[dict[str, tuple[str, str]] | None, str]:
    labels = _infer_labels_from_explicit_suffixes(job_names, policy_suffixes)
    if labels is not None:
        return labels, "policy_suffixes"
    labels = _infer_labels_from_repeated_final_tokens(job_names)
    if labels is not None:
        return labels, "run_names"
    return None, "none"


def _infer_labels_from_explicit_suffixes(
    job_names: list[str],
    policy_suffixes: tuple[str, ...],
) -> dict[str, tuple[str, str]] | None:
    labels: dict[str, tuple[str, str]] = {}
    suffixes = tuple(sorted((suffix for suffix in policy_suffixes if suffix), key=len, reverse=True))
    if not suffixes:
        return None
    for job_name in job_names:
        for suffix in suffixes:
            marker = f"_{suffix}"
            if job_name.endswith(marker) and len(job_name) > len(marker):
                labels[job_name] = (job_name[: -len(marker)], suffix)
                break
        else:
            return None
    return labels if labels else None


def _infer_labels_from_repeated_final_tokens(job_names: list[str]) -> dict[str, tuple[str, str]] | None:
    final_tokens: dict[str, int] = defaultdict(int)
    split_names: dict[str, tuple[str, str]] = {}
    for job_name in job_names:
        task, separator, policy = job_name.rpartition("_")
        if not separator or not task or not policy:
            return None
        split_names[job_name] = (task, policy)
        final_tokens[policy] += 1
    if len(final_tokens) < 2:
        return None
    policies_by_task: dict[str, set[str]] = defaultdict(set)
    for task, policy in split_names.values():
        policies_by_task[task].add(policy)
    if not any(len(policies) > 1 for policies in policies_by_task.values()):
        return None
    return split_names


def _resolve_job_labels(
    job_names: list[str],
    policy_suffixes: tuple[str, ...] = DEFAULT_POLICY_SUFFIXES,
) -> tuple[dict[str, tuple[str, str]], str]:
    inferred, source = (
        _infer_task_and_policy_labels_with_source(job_names, policy_suffixes) if job_names else (None, "none")
    )
    if inferred is None:
        return {job_name: (job_name or UNGROUPED_TASK, "") for job_name in job_names}, "none"

    labels = {job_name: (job_name or UNGROUPED_TASK, "") for job_name in job_names}
    labels.update(inferred)
    return labels, source


def build_experiment_summary(
    root: str | pathlib.Path,
    title: str,
    run_executions: list[RunExecutionReport] | None = None,
    policy_suffixes: tuple[str, ...] = DEFAULT_POLICY_SUFFIXES,
) -> ExperimentSummary:
    """Scan ``root`` and aggregate recorded results into the report's data model."""
    root = pathlib.Path(root)
    scanned, issues = _scan_jobs(root)
    run_executions = list(run_executions or [])
    failed_run_names = {
        run_execution.run_name for run_execution in run_executions if is_failed_execution(run_execution)
    }
    scanned = [entry for entry in scanned if entry.name not in failed_run_names]

    labels, grouping_source = _resolve_job_labels([entry.name for entry in scanned], policy_suffixes)
    jobs_by_task: dict[str, list[JobSummary]] = {}
    for scanned_job in scanned:
        task, policy = labels[scanned_job.name]
        jobs_by_task.setdefault(task, []).append(
            JobSummary(
                name=scanned_job.name,
                task=task,
                policy=policy,
                cameras=scanned_job.cameras,
                episodes=scanned_job.episodes,
                issues=scanned_job.issues,
            )
        )

    tasks = [
        TaskSummary(name=task, jobs=sorted(jobs_by_task[task], key=lambda job: (job.policy, job.name)))
        for task in sorted(jobs_by_task)
    ]
    policies = sorted({job.policy for task in tasks for job in task.jobs if job.policy})
    return ExperimentSummary(
        title=title,
        tasks=tasks,
        policies=policies,
        run_executions=run_executions,
        grouping_source=grouping_source,
        issues=issues,
    )
