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
from dataclasses import dataclass, field, replace
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
_Group = str | None


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
    def progress_fraction(self) -> float | None:
        # overall_score is recorded already normalized to [0, 1] by the progress tracker.
        score = _as_float(_progress(self.record).get("overall_score"))
        return None if score is None else max(0.0, min(1.0, score))

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
    show_empty: bool = False
    """Display a group that has not recorded any predicate events."""


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
    _family_sequences: dict[tuple[str, _Group], dict[int, str]] = field(init=False, repr=False)
    _progress_episodes: dict[EpisodeIdentity, EpisodeSummary] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._progress_episodes, group_issues = _normalize_progress_groups(self.name, self.episodes)
        progress_episodes = list(self._progress_episodes.values())
        self._objective_family_by_name, family_issues = _build_objective_family_map(self.name, progress_episodes)
        self._family_sequences, sequence_issues = _build_family_sequences(
            self.name, progress_episodes, self._objective_family_by_name
        )
        self.issues.extend(family_issues)
        self.issues.extend(group_issues)
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
    def progress_fractions(self) -> list[float]:
        """Per-episode progress fractions, skipping episodes recorded without a progress score."""
        return [episode.progress_fraction for episode in self.episodes if episode.progress_fraction is not None]

    @property
    def mean_progress(self) -> float | None:
        return _mean(self.progress_fractions)

    @property
    def num_videos(self) -> int:
        return sum(len(episode.video_by_camera) for episode in self.episodes)

    @functools.cached_property
    def funnels(self) -> list[ObjectiveFunnel]:
        instances_by_group: dict[tuple[str, _Group], set[tuple[EpisodeIdentity, str]]] = defaultdict(set)
        reached_by_group_index: dict[tuple[str, _Group, int], set[tuple[EpisodeIdentity, str]]] = defaultdict(set)
        for episode in self._progress_episodes.values():
            for objective_name in _episode_objective_names(episode):
                family = self._objective_family_by_name.get(objective_name, objective_name)
                instance = (episode.identity, objective_name)
                for group in _objective_groups(episode.record, objective_name):
                    instances_by_group[(family, group)].add(instance)
            for event in _progress_events(episode.record):
                objective_name = _event_objective_name(event)
                index = _as_int(event.get("predicate_index"))
                if objective_name is None or index is None:
                    continue
                family = self._objective_family_by_name.get(objective_name, objective_name)
                group = _event_group(event)
                instance = (episode.identity, objective_name)
                reached_by_group_index[(family, group, index)].add(instance)

        funnels = []
        for family, group in sorted(instances_by_group, key=lambda key: (key[0], _group_sort_key(key[1]))):
            sequence = self._family_sequences.get((family, group), {})
            stages = [
                FunnelStage(
                    index=index, name=sequence[index], num_reached=len(reached_by_group_index[(family, group, index)])
                )
                for index in sorted(sequence)
            ]
            multiple_groups = sum(key[0] == family for key in instances_by_group) > 1
            name = f"{family}/{_group_label(group)}" if multiple_groups else family
            funnels.append(
                ObjectiveFunnel(
                    name=name,
                    num_instances=len(instances_by_group[(family, group)]),
                    stages=stages,
                    show_empty=multiple_groups,
                )
            )
        return funnels

    def objectives_for(self, episode: EpisodeSummary) -> list[ObjectiveProgress]:
        episode = self._progress_episodes[episode.identity]
        objectives = _progress_objectives(episode.record)
        results = []
        fired = _events_by_objective_and_index(episode.record)
        for name, detail in objectives.items():
            family = self._objective_family_by_name.get(name, name)
            groups = _objective_groups(episode.record, name)
            active_by_group = detail.get("active_predicates") or {}
            signals = []
            matched_blocked: set[tuple[_Group, str]] = set()
            for group in sorted(groups, key=_group_sort_key):
                sequence = self._family_sequences.get((family, group), {})
                active_name = _base_predicate_name(active_by_group.get(group, ""))
                for index in sorted(sequence):
                    event = fired.get((name, group), {}).get(index)
                    blocked = event is None and sequence[index] == active_name
                    if blocked:
                        matched_blocked.add((group, sequence[index]))
                    signal_name = f"{_group_label(group)}/{sequence[index]}" if len(groups) > 1 else sequence[index]
                    signals.append(
                        PredicateSignal(
                            index=index,
                            name=signal_name,
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
                    blocked_predicates=[
                        f"{_group_label(group)}/{predicate}" if len(groups) > 1 else predicate
                        for group, predicate in (
                            (group, _base_predicate_name(predicate))
                            for group, predicate in active_by_group.items()
                            if predicate
                        )
                        if (group, predicate) not in matched_blocked
                    ],
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

    def mean_progress_for_policy(self, policy: str) -> float | None:
        return _mean([fraction for job in self.jobs if job.policy == policy for fraction in job.progress_fractions])

    def num_episodes_for_policy(self, policy: str) -> int:
        return sum(job.num_episodes for job in self.jobs if job.policy == policy)

    @property
    def overall_success_rate(self) -> float | None:
        scored = sum(job.num_scored_episodes for job in self.jobs)
        return None if scored == 0 else sum(job.num_successes for job in self.jobs) / scored

    @property
    def overall_mean_progress(self) -> float | None:
        return _mean([fraction for job in self.jobs for fraction in job.progress_fractions])

    @property
    def num_progress_episodes(self) -> int:
        return sum(len(job.progress_fractions) for job in self.jobs)


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
) -> tuple[dict[tuple[str, _Group], dict[int, str]], list[DataIssue]]:
    names_by_family_index: dict[tuple[str, _Group, int], set[str]] = defaultdict(set)
    for episode in episodes:
        for event in _progress_events(episode.record):
            objective_name = _event_objective_name(event)
            index = _as_int(event.get("predicate_index"))
            if objective_name is None or index is None:
                continue
            family = family_by_name.get(objective_name, objective_name)
            names_by_family_index[(family, _event_group(event), index)].add(
                _base_predicate_name(event.get("predicate_name", ""))
            )

    issues = []
    sequences: dict[tuple[str, _Group], dict[int, str]] = defaultdict(dict)
    for (family, group, index), names in names_by_family_index.items():
        if len(names) > 1:
            issues.append(
                DataIssue(
                    job_name or ".",
                    f"objective family '{family}' has multiple predicate names at index {index}: {sorted(names)}",
                )
            )
        sequences[(family, group)][index] = sorted(names)[0]
    return dict(sequences), issues


def _objective_names_are_compatible(episodes: list[EpisodeSummary], objective_names: list[str]) -> bool:
    names_by_index: dict[tuple[_Group, int], set[str]] = defaultdict(set)
    objective_name_set = set(objective_names)
    for episode in episodes:
        for event in _progress_events(episode.record):
            objective_name = _event_objective_name(event)
            index = _as_int(event.get("predicate_index"))
            if objective_name in objective_name_set and index is not None:
                names_by_index[(_event_group(event), index)].add(_base_predicate_name(event.get("predicate_name", "")))
    for (_, index), names in names_by_index.items():
        # An unattributed event could belong to any named group at this index.
        possible_names = names | names_by_index.get((None, index), set())
        if len(possible_names) > 1:
            return False
    return True


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


def _event_group(event: dict[str, Any]) -> _Group:
    """Keep missing attribution distinct from an explicitly empty group name."""
    group = event.get("group")
    return None if group is None else str(group)


def _group_sort_key(group: _Group) -> tuple[bool, str]:
    return group is not None, group or ""


def _group_label(group: _Group) -> str:
    return "(unattributed)" if group is None else group


def _normalize_progress_groups(
    job_name: str, episodes: list[EpisodeSummary]
) -> tuple[dict[EpisodeIdentity, EpisodeSummary], list[DataIssue]]:
    """Resolve missing groups within exact objectives without modifying recorded episodes."""
    known_groups: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        for name in _episode_objective_names(episode):
            known_groups[name].update(group for group in _objective_groups(episode.record, name) if group is not None)

    normalized = {}
    issues = []
    for episode in episodes:
        objectives = {}
        events = []
        objective_names = list(_progress_objectives(episode.record))
        objective_names.extend(sorted(_event_objectives(episode.record) - set(objective_names)))
        for name in objective_names:
            detail = dict(_progress_objectives(episode.record).get(name, {}))
            active = dict(detail.get("active_predicates") or {})
            recorded_events = [
                event for event in _progress_events(episode.record) if _event_objective_name(event) == name
            ]
            candidates = set(active) or known_groups[name]
            inferred_group = next(iter(candidates)) if len(candidates) == 1 else None
            needs_inference = any(_event_group(event) is None for event in recorded_events) or (
                not recorded_events and not active
            )
            if needs_inference and len(candidates) > 1:
                issues.append(DataIssue(job_name or ".", f"objective '{name}' has ambiguous missing group attribution"))
            events.extend(
                {**event, "group": inferred_group if _event_group(event) is None else _event_group(event)}
                for event in recorded_events
            )
            if not active and not recorded_events:
                active[inferred_group] = None
            objectives[name] = {**detail, "active_predicates": active}
        record = {
            **episode.record,
            "progress": {**_progress(episode.record), "objectives": objectives, "events": events},
        }
        normalized[episode.identity] = replace(episode, record=record)
    return normalized, issues


def _objective_groups(record: dict[str, Any], objective_name: str) -> set[_Group]:
    """Find recorded groups, including groups that have not emitted an event."""
    event_groups = {
        _event_group(event) for event in _progress_events(record) if _event_objective_name(event) == objective_name
    }
    active_groups = set(_progress_objectives(record).get(objective_name, {}).get("active_predicates") or {})
    return event_groups | active_groups or {None}


def _events_by_objective_and_index(record: dict[str, Any]) -> dict[tuple[str, _Group], dict[int, dict[str, Any]]]:
    result: dict[tuple[str, _Group], dict[int, dict[str, Any]]] = {}
    for event in _progress_events(record):
        objective_name = _event_objective_name(event)
        index = _as_int(event.get("predicate_index"))
        if objective_name is not None and index is not None:
            result.setdefault((objective_name, _event_group(event)), {})[index] = event
    return result


def _mean(values: list[float]) -> float | None:
    """Return the mean of ``values``, or None when there is nothing to average."""
    return None if not values else sum(values) / len(values)


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
