# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Plot where a rollout step's time goes, from an Arena Experiment timings file.

Timer names are hierarchical: a timer entered inside another records under ``<enclosing>/<name>``.
This script follows that tree, so each bar is one Run's mean step split into the timers nested
below it, and the time a timer does not attribute to its children becomes its own "(other)" slice.
Adding a timer to the rollout therefore adds a slice here without changing this script.
"""

from __future__ import annotations

import argparse
import json
import matplotlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIMER_NAME_SEPARATOR = "/"
DEFAULT_ROOT_TIMER_NAME = "step"

# Time a parent timer does not attribute to its children is allowed to go this far below zero
# before the file is treated as inconsistent, which absorbs float error in the recorded totals.
_UNATTRIBUTED_MS_TOLERANCE = 1e-6

# Okabe-Ito, which stays distinguishable for the most common colour vision deficiencies.
_SEGMENT_COLORS = ("#D55E00", "#009E73", "#E69F00", "#CC79A7", "#0072B2", "#56B4E9", "#F0E442", "#000000")
# Greys for the "(other)" slices, so unattributed time at different nesting depths stays apart.
_UNATTRIBUTED_COLORS = ("#B0B0B0", "#7A7A7A", "#D6D6D6")

# Rows a panel is sized and scaled for even when fewer Runs are plotted, so that a file holding a
# single Run draws a normal-looking bar instead of one filling the panel.
_MIN_PLOTTED_ROWS = 3


@dataclass(frozen=True)
class TimerSegment:
    """One slice of the root timer, as its mean cost per root-timer call."""

    label: str
    """Timer name relative to the root, suffixed with " (other)" for unattributed time."""

    ms_per_call: float
    """Mean milliseconds this slice costs per call of the root timer."""


@dataclass(frozen=True)
class RunTimings:
    """One Run's root-timer breakdown and spread."""

    run_name: str
    """Name of the Run these timings came from."""

    call_count: int
    """Number of root-timer calls, which is the step count for the default root."""

    mean_ms: float
    """Mean duration of the root timer."""

    p50_ms: float | None
    """Approximate median root-timer duration, or None when the file carries no percentiles."""

    p90_ms: float | None
    """Approximate 90th-percentile root-timer duration, or None when the file carries none."""

    segments: tuple[TimerSegment, ...]
    """Slices of the root timer, outermost timer first, each parent's "(other)" after its children."""

    def __post_init__(self) -> None:
        assert self.call_count > 0, "A recorded timer must have been called at least once."


def read_run_timer_records(timings_path: Path) -> dict[str, list[dict]]:
    """Group a timings file's records by Run name, preserving the order they were recorded in.

    Accepts both shapes Arena writes: the Experiment Runner's own file, which is a list of records
    for a single Run and is named after the directory holding it, and the collected Experiment
    file, whose ``runs`` entries each carry a ``run_name``.

    Args:
        timings_path: Path to the timings JSON file to read.

    Returns:
        Timer records keyed by Run name.
    """
    document = json.loads(timings_path.read_text(encoding="utf-8"))
    if isinstance(document, Mapping):
        if "runs" not in document:
            raise ValueError(f"Timings file '{timings_path}' has no 'runs' entry")
        records = document["runs"]
    elif isinstance(document, list):
        records = document
    else:
        raise ValueError(f"Timings file '{timings_path}' must hold a list or an object")

    records_by_run_name: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if "name" not in record or "total_ms" not in record:
            raise ValueError(f"Timings file '{timings_path}' has a record without a name and total")
        records_by_run_name[record.get("run_name", timings_path.parent.name)].append(record)
    if not records_by_run_name:
        raise ValueError(f"Timings file '{timings_path}' holds no timer records")
    return dict(records_by_run_name)


def _child_timer_names(parent_name: str, timer_names: Sequence[str]) -> list[str]:
    """Return the names nested exactly one level below the parent, in recorded order."""
    prefix = parent_name + TIMER_NAME_SEPARATOR
    return [
        timer_name
        for timer_name in timer_names
        if timer_name.startswith(prefix) and TIMER_NAME_SEPARATOR not in timer_name[len(prefix) :]
    ]


def _timer_slice_totals(
    timer_name: str, total_ms_by_name: Mapping[str, float], root_name: str
) -> Iterator[tuple[str, float]]:
    """Yield (label, total ms) per slice of one timer, recursing into children before its leftover."""
    label = timer_name[len(root_name) + 1 :] if timer_name != root_name else timer_name
    child_names = _child_timer_names(timer_name, list(total_ms_by_name))
    if not child_names:
        yield label, total_ms_by_name[timer_name]
        return

    attributed_ms = 0.0
    for child_name in child_names:
        yield from _timer_slice_totals(child_name, total_ms_by_name, root_name)
        attributed_ms += total_ms_by_name[child_name]

    unattributed_ms = total_ms_by_name[timer_name] - attributed_ms
    if unattributed_ms < -_UNATTRIBUTED_MS_TOLERANCE:
        raise ValueError(
            f"Timer '{timer_name}' totals {total_ms_by_name[timer_name]:.3f} ms but its children"
            f" total {attributed_ms:.3f} ms, so the file is inconsistent"
        )
    yield f"{label} (other)", max(unattributed_ms, 0.0)


def build_run_timings(run_name: str, records: Sequence[Mapping], root_timer_name: str) -> RunTimings:
    """Split one Run's records into the slices of its root timer.

    Args:
        run_name: Name to label this Run's bar with.
        records: The Run's timer records, as written by get_timer_stats_json.
        root_timer_name: Timer whose cost the slices add up to, normally the per-step timer.

    Returns:
        The Run's root-timer breakdown, normalized to one root-timer call.
    """
    total_ms_by_name = {str(record["name"]): float(record["total_ms"]) for record in records}
    if root_timer_name not in total_ms_by_name:
        raise ValueError(
            f"Run '{run_name}' has no '{root_timer_name}' timer."
            f" It recorded: {', '.join(sorted(total_ms_by_name)) or '(nothing)'}"
        )

    root_record = next(record for record in records if record["name"] == root_timer_name)
    call_count = int(root_record["count"])
    if call_count <= 0:
        raise ValueError(f"Run '{run_name}' recorded no calls of '{root_timer_name}'")

    segments = tuple(
        TimerSegment(label, total_ms / call_count)
        for label, total_ms in _timer_slice_totals(root_timer_name, total_ms_by_name, root_timer_name)
    )
    return RunTimings(
        run_name=run_name,
        call_count=call_count,
        mean_ms=total_ms_by_name[root_timer_name] / call_count,
        p50_ms=root_record.get("p50_ms"),
        p90_ms=root_record.get("p90_ms"),
        segments=segments,
    )


def collect_run_timings(timings_path: Path, root_timer_name: str) -> list[RunTimings]:
    """Read a timings file and split every Run in it into its root-timer slices.

    Args:
        timings_path: Path to the timings JSON file to read.
        root_timer_name: Timer whose cost the slices add up to.

    Returns:
        One entry per Run, ordered by Run name.
    """
    records_by_run_name = read_run_timer_records(timings_path)
    return [
        build_run_timings(run_name, records_by_run_name[run_name], root_timer_name)
        for run_name in sorted(records_by_run_name)
    ]


def _segment_colors(run_timings: Sequence[RunTimings]) -> dict[str, str]:
    """Assign each distinct segment label a stable colour, greying out unattributed slices."""
    ordered_labels: list[str] = []
    for run in run_timings:
        for segment in run.segments:
            if segment.label not in ordered_labels:
                ordered_labels.append(segment.label)

    colors: dict[str, str] = {}
    measured_label_index = 0
    unattributed_label_index = 0
    for label in ordered_labels:
        if label.endswith(" (other)"):
            colors[label] = _UNATTRIBUTED_COLORS[unattributed_label_index % len(_UNATTRIBUTED_COLORS)]
            unattributed_label_index += 1
        else:
            colors[label] = _SEGMENT_COLORS[measured_label_index % len(_SEGMENT_COLORS)]
            measured_label_index += 1
    return colors


def _set_run_axis(axes, run_timings: Sequence[RunTimings]) -> None:
    """Label one row per Run, top to bottom, holding the row height steady for few Runs."""
    axes.set_yticks(range(len(run_timings)), [run.run_name for run in run_timings])
    axes.set_ylim(max(len(run_timings), _MIN_PLOTTED_ROWS) - 0.5, -0.5)


def _plot_breakdown(axes, run_timings: Sequence[RunTimings], root_timer_name: str) -> None:
    """Draw one stacked bar per Run, splitting its mean root-timer call into slices."""
    colors = _segment_colors(run_timings)
    bar_positions = range(len(run_timings))
    labelled: set[str] = set()
    label_offset_ms = max(run.mean_ms for run in run_timings) * 0.01

    for bar_position, run in zip(bar_positions, run_timings):
        slice_start_ms = 0.0
        for segment in run.segments:
            axes.barh(
                bar_position,
                segment.ms_per_call,
                left=slice_start_ms,
                color=colors[segment.label],
                label=segment.label if segment.label not in labelled else None,
                height=0.62,
            )
            labelled.add(segment.label)
            slice_start_ms += segment.ms_per_call
        axes.text(
            slice_start_ms + label_offset_ms,
            bar_position,
            f"{run.mean_ms:,.0f} ms  ·  {run.call_count:,} calls",
            va="center",
            fontsize=9,
        )

    _set_run_axis(axes, run_timings)
    axes.set_xlabel(f"mean milliseconds per '{root_timer_name}'")
    axes.set_title(f"Where one '{root_timer_name}' goes", loc="left", fontweight="bold", pad=34)
    axes.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8, ncols=3, frameon=False)
    axes.set_xlim(left=0.0)
    axes.margins(x=0.22)


def _plot_spread(axes, run_timings: Sequence[RunTimings], root_timer_name: str) -> None:
    """Draw each Run's median-to-p90 spread, which shows the tail the mean hides."""
    bar_positions = range(len(run_timings))
    label_offset_ms = max(run.p90_ms for run in run_timings) * 0.01
    for bar_position, run in zip(bar_positions, run_timings):
        axes.barh(
            bar_position,
            run.p90_ms - run.p50_ms,
            left=run.p50_ms,
            color=_SEGMENT_COLORS[0],
            height=0.45,
        )
        axes.text(
            run.p90_ms + label_offset_ms,
            bar_position,
            f"p50 {run.p50_ms:,.0f} - p90 {run.p90_ms:,.0f} ms",
            va="center",
            fontsize=9,
        )

    _set_run_axis(axes, run_timings)
    axes.set_xlabel(f"milliseconds per '{root_timer_name}'")
    axes.set_title(f"'{root_timer_name}' spread, median to p90", loc="left", fontweight="bold")
    axes.set_xlim(left=0.0)
    axes.margins(x=0.25)


def plot_run_timings(
    run_timings: Sequence[RunTimings],
    output_path: Path,
    title: str,
    root_timer_name: str = DEFAULT_ROOT_TIMER_NAME,
) -> Path:
    """Write a figure breaking down every Run's root timer and, where recorded, its spread.

    Args:
        run_timings: One entry per Run, in the order the bars should appear.
        output_path: Image file to write. Parent directories are created.
        title: Figure title.
        root_timer_name: Timer the slices add up to, named in the axis labels.

    Returns:
        The path that was written.
    """
    assert len(run_timings) > 0, "There must be at least one Run to plot."

    # Percentiles are dropped when several processes' timings are merged, so only plot the spread
    # when every Run still carries them.
    spread_is_plottable = all(run.p50_ms is not None and run.p90_ms is not None for run in run_timings)
    panel_count = 2 if spread_is_plottable else 1
    plotted_row_count = max(len(run_timings), _MIN_PLOTTED_ROWS)
    figure, axes_list = plt.subplots(
        panel_count,
        1,
        figsize=(12.0, 1.6 + panel_count * (1.1 + 0.42 * plotted_row_count)),
        squeeze=False,
    )
    figure.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left")

    _plot_breakdown(axes_list[0][0], run_timings, root_timer_name)
    if spread_is_plottable:
        _plot_spread(axes_list[1][0], run_timings, root_timer_name)

    for axes in axes_list.flat:
        axes.grid(axis="x", alpha=0.3)
        axes.set_axisbelow(True)
        for spine_name in ("top", "right", "left"):
            axes.spines[spine_name].set_visible(False)

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _default_output_path(timings_path: Path) -> Path:
    """Return the image path to write when none was given, beside the timings file."""
    return timings_path.with_suffix(".png")


def main(argv: Sequence[str] | None = None) -> int:
    """Read a timings file named on the command line and write its plot."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("timings_path", type=Path, help="arena_experiment_timings.json to plot")
    parser.add_argument("--output", type=Path, help="Output image path (default: beside the input)")
    parser.add_argument(
        "--root-timer",
        default=DEFAULT_ROOT_TIMER_NAME,
        help=f"Timer whose cost the slices add up to (default: {DEFAULT_ROOT_TIMER_NAME})",
    )
    parser.add_argument("--title", default="Arena rollout timing", help="Figure title")
    args = parser.parse_args(argv)

    timings_path = args.timings_path.expanduser()
    try:
        run_timings = collect_run_timings(timings_path, args.root_timer)
        output_path = plot_run_timings(
            run_timings,
            args.output or _default_output_path(timings_path),
            args.title,
            args.root_timer,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    total_calls = sum(run.call_count for run in run_timings)
    print(f"Read {total_calls:,} '{args.root_timer}' calls across {len(run_timings)} run(s).")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
