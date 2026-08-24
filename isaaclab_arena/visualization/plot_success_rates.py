# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Plot per-task success rate and average progress for policies encoded in Arena Run names.

Every Run name must end in one of the explicitly supplied ``_<policy>`` suffixes; the remaining prefix
is the task name used to group policy bars. Both metrics are written as grouped bar charts.
"""

from __future__ import annotations

import argparse
import io
import json
import matplotlib
import re
import sys
import tarfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_RESULTS_FILENAME_PATTERN = re.compile(r"^episode_results(?:_rebuild\d+)?(?:_rank\d+)?\.jsonl$")
_RUN_DIRECTORY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
_POLICY_STYLES = (
    ("#0072B2", ""),
    ("#D55E00", "xxx"),
    ("#009E73", "///"),
    ("#CC79A7", "\\\\"),
    ("#E69F00", "..."),
    ("#56B4E9", "++"),
    ("#F0E442", "oo"),
    ("#000000", "--"),
)


@dataclass(frozen=True)
class SuccessRateResult:
    """Successful and total scored episodes for one task-policy pair."""

    successful_episodes: int
    """Number of episodes whose top-level ``success`` field is true."""

    total_episodes: int
    """Number of episodes whose top-level ``success`` field is a boolean."""

    def __post_init__(self) -> None:
        assert self.total_episodes >= 0, "The total number of episodes must not be negative."
        assert (
            0 <= self.successful_episodes <= self.total_episodes
        ), "The successful episode count must be between zero and the total episode count."

    @property
    def success_rate(self) -> float | None:
        """Return the successful fraction, or None when no episode was scored."""
        if self.total_episodes == 0:
            return None
        return self.successful_episodes / self.total_episodes

    @property
    def num_scored_episodes(self) -> int:
        """Number of episodes that contributed to the metric."""
        return self.total_episodes


@dataclass(frozen=True)
class AverageProgressResult:
    """Summed progress score and scored episode count for one task-policy pair."""

    total_progress: float
    """Sum over episodes of the ``progress`` score in ``[0, 1]``."""

    scored_episodes: int
    """Number of episodes whose ``progress`` score is a number in ``[0, 1]``."""

    def __post_init__(self) -> None:
        assert self.scored_episodes >= 0, "The number of scored episodes must not be negative."
        assert self.total_progress >= 0.0, "The summed progress must not be negative."
        assert (
            self.total_progress <= self.scored_episodes
        ), "The summed progress must not exceed the scored episode count."

    @property
    def average_progress(self) -> float | None:
        """Return the mean progress score, or None when no episode was scored."""
        if self.scored_episodes == 0:
            return None
        return self.total_progress / self.scored_episodes

    @property
    def num_scored_episodes(self) -> int:
        """Number of episodes that contributed to the metric."""
        return self.scored_episodes


def _validate_policies(policies: Sequence[str]) -> tuple[str, ...]:
    """Return policies as a validated tuple while preserving their requested order."""
    policy_names = tuple(policies)
    if not policy_names:
        raise ValueError("At least one policy must be supplied.")
    if any(not policy_name for policy_name in policy_names):
        raise ValueError("Policy names must not be empty.")
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("Policy names must be unique.")
    return policy_names


def _is_results_path(path: str | Path) -> bool:
    """Return whether ``path`` has a canonical episode-results filename."""
    return _RESULTS_FILENAME_PATTERN.fullmatch(PurePosixPath(path).name) is not None


def _iter_text_lines(text_stream, source_name: str) -> Iterator[tuple[str, int, str]]:
    """Yield non-empty lines with their source name and one-based line number."""
    for line_number, line in enumerate(text_stream, start=1):
        stripped_line = line.strip()
        if stripped_line:
            yield source_name, line_number, stripped_line


def _iter_directory_result_lines(results_directory: Path) -> Iterator[tuple[str, int, str]]:
    """Yield lines from canonical episode-results files below a directory."""
    results_paths = sorted(
        path for path in results_directory.rglob("*.jsonl") if path.is_file() and _is_results_path(path)
    )
    if not results_paths:
        raise ValueError(f"No episode-results JSONL files found below '{results_directory}'.")

    for results_path in results_paths:
        with results_path.open(encoding="utf-8") as results_file:
            yield from _iter_text_lines(results_file, str(results_path))


def _iter_archive_result_lines(archive_path: Path) -> Iterator[tuple[str, int, str]]:
    """Yield lines from canonical episode-results files in a tar archive."""
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            results_members = [
                member for member in archive.getmembers() if member.isfile() and _is_results_path(member.name)
            ]
            if not results_members:
                raise ValueError(f"No episode-results JSONL files found in '{archive_path}'.")
            dated_run_names = sorted({
                path_component
                for member in results_members
                for path_component in PurePosixPath(member.name).parts[:-1]
                if _RUN_DIRECTORY_PATTERN.fullmatch(path_component)
            })
            if dated_run_names:
                newest_run_name = dated_run_names[-1]
                results_members = [
                    member for member in results_members if newest_run_name in PurePosixPath(member.name).parts[:-1]
                ]
            results_members.sort(key=lambda member: member.name)

            for member in results_members:
                extracted_file = archive.extractfile(member)
                assert extracted_file is not None, f"Could not read archive member '{member.name}'."
                source_name = f"{archive_path}!{member.name}"
                with extracted_file, io.TextIOWrapper(extracted_file, encoding="utf-8") as results_file:
                    yield from _iter_text_lines(results_file, source_name)
    except tarfile.TarError as error:
        raise ValueError(f"'{archive_path}' is not a readable tar archive: {error}") from error


def _iter_result_lines(results_path: Path) -> Iterator[tuple[str, int, str]]:
    """Yield JSONL records from a results directory, JSONL file, or tar archive."""
    if results_path.is_dir():
        yield from _iter_directory_result_lines(results_path)
        return
    if not results_path.is_file():
        raise ValueError(f"Results path '{results_path}' does not exist.")
    if results_path.suffix == ".jsonl":
        with results_path.open(encoding="utf-8") as results_file:
            yield from _iter_text_lines(results_file, str(results_path))
        return
    yield from _iter_archive_result_lines(results_path)


def _resolve_results_path(results_path: Path) -> Path:
    """Descend into the newest reverse-dated experiment directory when present."""
    if not results_path.is_dir():
        return results_path
    run_directories = sorted(
        child for child in results_path.iterdir() if child.is_dir() and _RUN_DIRECTORY_PATTERN.fullmatch(child.name)
    )
    if not run_directories:
        return results_path
    return run_directories[-1]


def _load_record(line: str, source_name: str, line_number: int) -> tuple[dict, str, str]:
    """Return a validated ``(record, job_name, location)`` triple from one JSONL line."""
    location = f"{source_name}:{line_number}"
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON at {location}: {error.msg}.") from error
    if not isinstance(record, dict):
        raise ValueError(f"Episode result at {location} must be a JSON object.")
    job_name = record.get("job_name")
    if not isinstance(job_name, str) or not job_name:
        raise ValueError(f"Episode result at {location} must contain a non-empty string 'job_name'.")
    return record, job_name, location


def _parse_success_record(line: str, source_name: str, line_number: int) -> tuple[str, bool | None]:
    """Return a validated ``(job_name, success)`` pair from one JSONL record."""
    record, job_name, location = _load_record(line, source_name, line_number)
    if "success" not in record:
        raise ValueError(f"Episode result at {location} does not contain 'success'.")
    success = record["success"]
    if success is not None and not isinstance(success, bool):
        raise ValueError(f"Episode result at {location} has 'success' that is not a boolean or null.")
    return job_name, success


def _parse_progress_record(line: str, source_name: str, line_number: int) -> tuple[str, float | None]:
    """Return a validated ``(job_name, progress_fraction)`` pair from one JSONL record."""
    record, job_name, location = _load_record(line, source_name, line_number)
    return job_name, _progress_fraction(record.get("progress"), location)


def _progress_fraction(progress: object, location: str) -> float | None:
    """Return the episode's recorded progress in ``[0, 1]``, or None when it was not recorded."""
    if progress is None:
        return None
    if not isinstance(progress, dict):
        raise ValueError(f"Episode result at {location} has 'progress' that is not an object or null.")
    overall_score = progress.get("overall_score")
    if overall_score is None:
        return None
    if isinstance(overall_score, bool) or not isinstance(overall_score, (int, float)):
        raise ValueError(f"Episode result at {location} has 'progress.overall_score' that is not a number.")
    # overall_score is recorded already normalized to [0, 1]; reject out-of-range values rather than
    # silently clamp them.
    fraction = float(overall_score)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Episode result at {location} has 'progress.overall_score' {fraction} outside [0, 1].")
    return fraction


def _split_task_and_policy(job_name: str, policies: Sequence[str]) -> tuple[str, str]:
    """Split a Run name into its task and longest matching ``_<policy>`` suffix."""
    for policy_name in sorted(policies, key=len, reverse=True):
        policy_suffix = f"_{policy_name}"
        if not job_name.endswith(policy_suffix):
            continue
        task_name = job_name[: -len(policy_suffix)]
        if not task_name:
            raise ValueError(f"Run name '{job_name}' has no task name before policy suffix '{policy_suffix}'.")
        return task_name, policy_name
    expected_suffixes = ", ".join(f"_{policy_name}" for policy_name in policies)
    raise ValueError(f"Run name '{job_name}' does not end with a requested policy suffix: {expected_suffixes}.")


def collect_success_rates(
    results_path: str | Path, policies: Sequence[str]
) -> dict[tuple[str, str], SuccessRateResult]:
    """Collect episode-weighted success rates keyed by derived task name and policy.

    Args:
        results_path: Results directory, JSONL file, or tar archive to read.
        policies: Policy names expected as underscore-delimited Run-name suffixes.

    Returns:
        Success counts keyed by ``(task_name, policy_name)``.
    """
    resolved_results_path = _resolve_results_path(Path(results_path).expanduser())
    policy_names = _validate_policies(policies)
    episode_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    for source_name, line_number, line in _iter_result_lines(resolved_results_path):
        job_name, success = _parse_success_record(line, source_name, line_number)
        task_name, policy_name = _split_task_and_policy(job_name, policy_names)
        counts = episode_counts[(task_name, policy_name)]
        if success is None:
            continue
        counts[0] += int(success)
        counts[1] += 1

    if not episode_counts or not any(counts[1] for counts in episode_counts.values()):
        raise ValueError(f"No scored episodes found under '{results_path}'.")

    return {
        key: SuccessRateResult(successful_episodes=counts[0], total_episodes=counts[1])
        for key, counts in episode_counts.items()
    }


def collect_average_progress(
    results_path: str | Path, policies: Sequence[str]
) -> dict[tuple[str, str], AverageProgressResult]:
    """Collect episode-weighted average progress keyed by derived task name and policy.

    Args:
        results_path: Results directory, JSONL file, or tar archive to read.
        policies: Policy names expected as underscore-delimited Run-name suffixes.

    Returns:
        Summed progress and scored counts keyed by ``(task_name, policy_name)``.
    """
    resolved_results_path = _resolve_results_path(Path(results_path).expanduser())
    policy_names = _validate_policies(policies)
    progress_totals: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])

    for source_name, line_number, line in _iter_result_lines(resolved_results_path):
        job_name, score = _parse_progress_record(line, source_name, line_number)
        task_name, policy_name = _split_task_and_policy(job_name, policy_names)
        totals = progress_totals[(task_name, policy_name)]
        if score is None:
            continue
        totals[0] += score
        totals[1] += 1

    if not progress_totals or not any(totals[1] for totals in progress_totals.values()):
        raise ValueError(f"No scored progress episodes found under '{results_path}'.")

    return {
        key: AverageProgressResult(total_progress=totals[0], scored_episodes=int(totals[1]))
        for key, totals in progress_totals.items()
    }


def _plot_grouped_percentages(
    percentages: dict[tuple[str, str], float | None],
    policies: Sequence[str],
    output_path: str | Path,
    *,
    title: str,
    y_label: str,
) -> None:
    """Render a grouped percentage bar chart of ``(task, policy) -> percent`` and save it.

    Args:
        percentages: Percentages in ``[0, 100]`` keyed by ``(task_name, policy_name)``; a None or
            missing pair is drawn as an ``N/A`` marker.
        policies: Policy names in the desired left-to-right and legend order.
        output_path: Destination image path; its extension selects the output format.
        title: Plot title.
        y_label: Label for the vertical axis.
    """
    policy_names = _validate_policies(policies)
    if not percentages:
        raise ValueError("At least one result is required.")
    task_names = sorted({task_name for task_name, _ in percentages})
    task_positions = list(range(len(task_names)))
    bar_width = 0.8 / len(policy_names)
    width_per_task = max(1.15, 0.36 * len(policy_names))
    figure_width = max(10.0, width_per_task * len(task_names))

    figure, axes = plt.subplots(figsize=(figure_width, 8.0), facecolor="white")
    axes.set_facecolor("#FBFBFB")
    for policy_index, policy_name in enumerate(policy_names):
        color, hatch = _POLICY_STYLES[policy_index % len(_POLICY_STYLES)]
        offsets = [
            task_position + (policy_index - (len(policy_names) - 1) / 2) * bar_width for task_position in task_positions
        ]
        percentage_values = [percentages.get((task_name, policy_name)) for task_name in task_names]
        bars = axes.bar(
            offsets,
            [0.0 if percentage is None else percentage for percentage in percentage_values],
            bar_width,
            label=policy_name,
            color=color,
            edgecolor="#333333",
            hatch=hatch,
            linewidth=0.8,
        )
        for task_name, bar, percentage in zip(task_names, bars, percentage_values):
            if percentage is None:
                bar.set_visible(False)
                axes.annotate(
                    "N/A",
                    (bar.get_x() + bar.get_width() / 2, 0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#777777",
                )
                continue
            axes.annotate(
                f"{percentage:.0f}",
                (bar.get_x() + bar.get_width() / 2, percentage),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

    axes.set_axisbelow(True)
    axes.set_ylabel(y_label)
    axes.set_ylim(0, 105)
    axes.set_yticks(range(0, 101, 20))
    axes.set_xticks(task_positions)
    axes.set_xticklabels(task_names, rotation=45, ha="right")
    axes.set_title(title)
    axes.grid(axis="y", color="#E5E5E5", linewidth=1.0)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    figure.legend(loc="lower center", ncol=len(policy_names), frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_success_rates(
    results: dict[tuple[str, str], SuccessRateResult],
    policies: Sequence[str],
    output_path: str | Path,
    title: str = "Task success rate",
) -> None:
    """Render a grouped success-rate bar chart and save it to ``output_path``.

    Args:
        results: Success counts keyed by ``(task_name, policy_name)``.
        policies: Policy names in the desired left-to-right and legend order.
        output_path: Destination image path; its extension selects the output format.
        title: Plot title.
    """
    if not results:
        raise ValueError("At least one success-rate result is required.")
    percentages = {
        key: None if result.success_rate is None else result.success_rate * 100.0 for key, result in results.items()
    }
    _plot_grouped_percentages(percentages, policies, output_path, title=title, y_label="Success rate (%)")


def plot_average_progress(
    results: dict[tuple[str, str], AverageProgressResult],
    policies: Sequence[str],
    output_path: str | Path,
    title: str = "Average task progress",
) -> None:
    """Render a grouped average-progress bar chart and save it to ``output_path``.

    Args:
        results: Summed progress and scored counts keyed by ``(task_name, policy_name)``.
        policies: Policy names in the desired left-to-right and legend order.
        output_path: Destination image path; its extension selects the output format.
        title: Plot title.
    """
    if not results:
        raise ValueError("At least one average-progress result is required.")
    percentages = {
        key: None if result.average_progress is None else result.average_progress * 100.0
        for key, result in results.items()
    }
    _plot_grouped_percentages(percentages, policies, output_path, title=title, y_label="Average progress (%)")


@dataclass(frozen=True)
class _MetricSpec:
    """Collection and rendering configuration for one plottable metric."""

    collect: Callable[[str | Path, Sequence[str]], dict]
    plot: Callable[[dict, Sequence[str], str | Path, str], None]
    default_basename: str
    default_title: str


_METRICS = {
    "success_rate": _MetricSpec(collect_success_rates, plot_success_rates, "success_rates", "Task success rate"),
    "average_progress": _MetricSpec(
        collect_average_progress, plot_average_progress, "average_progress", "Average task progress"
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Write a success-rate and an average-progress bar chart into the output directory.

    Both metrics are plotted; a metric with no scored episodes is skipped with a warning.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", type=Path, help="Results directory, JSONL file, or tar archive")
    parser.add_argument(
        "--policies",
        nargs="+",
        required=True,
        help="All policy suffixes present in the input, in bar and legend order (for example: pi0 cosmos)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the output images (default: the results directory)",
    )
    parser.add_argument("--title", help="Plot title applied to both charts (default: a per-metric title)")
    args = parser.parse_args(argv)

    requested_results_path = args.results_path.expanduser()
    try:
        resolved_results_path = _resolve_results_path(requested_results_path)
    except OSError as error:
        parser.error(str(error))
    if resolved_results_path != requested_results_path:
        print(f"Using most recent experiment directory: {resolved_results_path}")
    output_dir = args.output_dir or (
        resolved_results_path if resolved_results_path.is_dir() else resolved_results_path.parent
    )

    any_plotted = False
    for metric_name, metric in _METRICS.items():
        try:
            results = metric.collect(resolved_results_path, args.policies)
        except (OSError, UnicodeError, ValueError) as error:
            print(f"[{metric_name}] skipped: {error}", file=sys.stderr)
            continue
        output_path = output_dir / f"{metric.default_basename}.png"
        metric.plot(results, args.policies, output_path, args.title or metric.default_title)
        any_plotted = True

        task_names = {task_name for task_name, _ in results}
        total_episodes = sum(result.num_scored_episodes for result in results.values())
        unavailable_pairs = [
            f"{task_name}/{policy_name}"
            for task_name in sorted(task_names)
            for policy_name in args.policies
            if (task_name, policy_name) not in results or results[(task_name, policy_name)].num_scored_episodes == 0
        ]
        if unavailable_pairs:
            print(f"Unavailable results (bars marked N/A): {', '.join(unavailable_pairs)}", file=sys.stderr)
        print(
            f"[{metric_name}] Read {total_episodes} scored episodes across {len(task_names)} tasks and "
            f"{len(args.policies)} policies."
        )
        print(f"Wrote {output_path}")

    if not any_plotted:
        parser.error(f"no metrics could be plotted from '{requested_results_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
