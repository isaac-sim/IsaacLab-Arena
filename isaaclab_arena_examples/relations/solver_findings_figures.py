# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate publication figures for the solver speed and coverage findings."""

from __future__ import annotations

import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_speedup(value: float) -> str:
    return f"{value:.2f}×" if value >= 0.1 else f"{value:.2g}×"


def _save(figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _annotated_heatmap(
    axis,
    values: np.ndarray,
    annotations: list[list[str]],
    x_labels: list[str],
    y_labels: list[str],
    *,
    color_map: str,
    norm=None,
    colorbar_label: str,
) -> None:
    masked = np.ma.masked_invalid(values)
    image = axis.imshow(masked, cmap=color_map, norm=norm, aspect="auto")
    image.cmap.set_bad("#e5e7eb")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "black"
            if np.isfinite(values[row, column]) and norm is not None:
                color = "white" if norm(values[row, column]) < 0.2 or norm(values[row, column]) > 0.8 else "black"
            axis.text(column, row, annotations[row][column], ha="center", va="center", fontsize=8, color=color)
    axis.set_xticks(range(len(x_labels)), x_labels)
    axis.set_yticks(range(len(y_labels)), y_labels)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)


def _write_speed_figure(root: Path, output_directory: Path) -> None:
    object_counts = (2, 5, 10, 20)
    background_counts = (0, 1, 5, 9)
    crossover_values = np.full((len(object_counts), len(background_counts)), np.nan)
    crossover_annotations = [["" for _ in background_counts] for _ in object_counts]
    for row, num_objects in enumerate(object_counts):
        analysis = _load(root / "background_complexity_frontier" / "fixed" / f"analysis_n{num_objects}.json")
        by_background = {item["background_objects"]: item for item in analysis["crossovers"]}
        for column, backgrounds in enumerate(background_counts):
            crossover = by_background[backgrounds]["crossover_batch_size"]
            crossover_annotations[row][column] = ">2048" if crossover is None else str(crossover)
            if crossover is not None:
                crossover_values[row, column] = math.log2(crossover)

    practical = _load(root / "background_practical_generation" / "full" / "analysis.json")
    practical_rows = practical["rows"]
    target_counts = sorted({row["target_layouts"] for row in practical_rows})

    factorial = _load(root / "background_factorial" / "pilot" / "analysis.json")
    condition_labels = [
        "Open",
        "19% scattered",
        "19% corridor",
        "38% scattered",
        "38% corridor",
    ]
    factorial_values = np.full((3, 5), np.nan)
    factorial_annotations = [["censored" for _ in condition_labels] for _ in range(3)]
    factorial_objects = (5, 10, 20)
    for row_index, num_objects in enumerate(factorial_objects):
        rows = [row for row in factorial["rows"] if row["num_objects"] == num_objects]
        keyed = {
            (
                round(row["measured_excluded_fraction"] * 100),
                row["topology"],
            ): row
            for row in rows
        }
        keys = [
            (0, "scattered"),
            (19, "scattered"),
            (19, "corridor"),
            (38, "scattered"),
            (38, "corridor"),
        ]
        for column, key in enumerate(keys):
            speedup = keyed[key]["arena_speedup_scene_median"]
            if speedup is not None:
                factorial_values[row_index, column] = math.log10(speedup)
                factorial_annotations[row_index][column] = _format_speedup(speedup)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    _annotated_heatmap(
        axes[0],
        crossover_values,
        crossover_annotations,
        [str(value) for value in background_counts],
        [str(value) for value in object_counts],
        color_map="viridis_r",
        colorbar_label="log₂(crossover batch)",
    )
    axes[0].set_title("(a) Fixed 600-iteration crossover")
    axes[0].set_xlabel("Fixed background objects")
    axes[0].set_ylabel("Movable objects")

    for backgrounds in background_counts:
        rows = sorted(
            (row for row in practical_rows if row["background_objects"] == backgrounds),
            key=lambda row: row["target_layouts"],
        )
        axes[1].plot(
            [row["target_layouts"] for row in rows],
            [row["arena_speedup"] for row in rows],
            marker="o",
            label=f"{backgrounds} backgrounds",
        )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, label="Equal throughput")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks(target_counts, [str(value) for value in target_counts])
    axes[1].set_xlabel("Requested shared-valid layouts, K")
    axes[1].set_ylabel("Arena / RoboLab throughput")
    axes[1].set_title("(b) Native practical exact-K speedup")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    _annotated_heatmap(
        axes[2],
        factorial_values,
        factorial_annotations,
        condition_labels,
        [str(value) for value in factorial_objects],
        color_map="RdBu",
        norm=TwoSlopeNorm(vmin=-4.0, vcenter=0.0, vmax=1.0),
        colorbar_label="log₁₀(Arena / RoboLab throughput)",
    )
    axes[2].tick_params(axis="x", rotation=28)
    axes[2].set_title("(c) Controlled factorial pilot, K=128")
    axes[2].set_xlabel("Background condition")
    axes[2].set_ylabel("Movable objects")

    figure.suptitle(
        "Solver speed: batching wins fixed work, but not the matched practical pilot",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.03,
        "Panels (a,b): three repetitions. Panel (c): three scene instances × three repetitions; "
        "censored means at least one paired run failed to reach K.",
        ha="center",
        fontsize=8,
    )
    _save(figure, output_directory / "solver_speed_findings")


def _write_fixed_iteration_figure(root: Path, output_directory: Path) -> None:
    object_counts = (2, 5, 10, 20)
    background_counts = (0, 1, 5, 9)
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(background_counts)))
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, constrained_layout=True)
    for axis, num_objects in zip(axes.flat, object_counts, strict=True):
        analysis = _load(root / "background_complexity_frontier" / "fixed" / f"analysis_n{num_objects}.json")
        crossover_by_background = {
            row["background_objects"]: row["crossover_batch_size"] for row in analysis["crossovers"]
        }
        for color, backgrounds in zip(colors, background_counts, strict=True):
            rows = sorted(
                (row for row in analysis["rows"] if row["background_objects"] == backgrounds),
                key=lambda row: row["batch_size"],
            )
            batches = np.asarray([row["batch_size"] for row in rows])
            medians = np.asarray([row["arena_speedup"] for row in rows])
            lower = np.asarray([row["arena_speedup_q25"] for row in rows])
            upper = lower + np.asarray([row["arena_speedup_iqr"] for row in rows])
            axis.plot(
                batches,
                medians,
                marker="o",
                markersize=3,
                color=color,
                label=f"{backgrounds} backgrounds",
            )
            axis.fill_between(batches, lower, upper, color=color, alpha=0.15, linewidth=0)
            crossover = crossover_by_background[backgrounds]
            if crossover is not None:
                crossover_row = next(row for row in rows if row["batch_size"] == crossover)
                axis.scatter(
                    [crossover],
                    [crossover_row["arena_speedup"]],
                    s=55,
                    facecolors="none",
                    edgecolors=color,
                    linewidth=1.5,
                    zorder=4,
                )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.set_title(f"{num_objects} movable objects")
        axis.text(
            0.03,
            0.96,
            "Crossover: "
            + ", ".join(
                f"{backgrounds}→{crossover_by_background[backgrounds] or '>2048'}" for backgrounds in background_counts
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=7,
        )
    for axis in axes[-1]:
        axis.set_xlabel("Layouts optimized in one Arena batch")
    for axis in axes[:, 0]:
        axis.set_ylabel("Arena / serial RoboLab throughput")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    figure.suptitle(
        "Fixed-work scaling: exactly 600 matched-AABB collision iterations",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.015,
        "Curves show median paired speedup; bands span the reported interquartile range. "
        "Open circles mark the first tested batch whose speedup Q25 exceeds 1.",
        ha="center",
        fontsize=8,
    )
    _save(figure, output_directory / "fixed_600_iteration_scaling")


def _write_two_axis_scaling_bars(root: Path, output_directory: Path) -> None:
    object_counts = (5, 10, 15, 20, 25)
    analyses = {
        num_objects: _load(root / "validated_scaling" / f"analysis_n{num_objects}.json")
        for num_objects in object_counts
    }
    environment_counts = (32, 64, 128, 256, 512, 1024, 2048)
    fixed_object_rows = {
        row["batch_size"]: row
        for row in analyses[10]["rows"]
        if row["background_objects"] == 0 and row["batch_size"] in environment_counts
    }
    fixed_environment_rows = {
        num_objects: next(
            row for row in analyses[num_objects]["rows"] if row["background_objects"] == 0 and row["batch_size"] == 512
        )
        for num_objects in object_counts
    }

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, constrained_layout=True)
    arena_color = "#2563eb"
    robolab_color = "#dc2626"
    width = 0.38

    def grouped_bars(axis, categories, arena_values, robolab_values) -> None:
        positions = np.arange(len(categories))
        axis.bar(
            positions - width / 2,
            arena_values,
            width,
            color=arena_color,
            label="Arena",
        )
        axis.bar(
            positions + width / 2,
            robolab_values,
            width,
            color=robolab_color,
            label="RoboLab",
        )
        axis.set_xticks(positions, [str(value) for value in categories])
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    grouped_bars(
        axes[0],
        environment_counts,
        [fixed_object_rows[count]["arena_layouts_per_second"] for count in environment_counts],
        [fixed_object_rows[count]["robolab_layouts_per_second"] for count in environment_counts],
    )
    axes[0].set_title("(a) Environment scaling (10 objects)")
    axes[0].set_xlabel("Number of environments")
    axes[0].set_ylabel("Layouts optimized per second")

    grouped_bars(
        axes[1],
        object_counts,
        [fixed_environment_rows[num_objects]["arena_layouts_per_second"] for num_objects in object_counts],
        [fixed_environment_rows[num_objects]["robolab_layouts_per_second"] for num_objects in object_counts],
    )
    axes[1].set_title("(b) Object scaling (512 environments)")
    axes[1].set_xlabel("Number of objects per environment")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
    )
    _save(figure, output_directory / "environment_object_scaling")


def _write_native_object_scaling_bars(root: Path, output_directory: Path) -> None:
    object_counts = (5, 10, 15, 20, 25)
    rows = [
        _load(root / "validated_native" / f"analysis_n{num_objects}.json")["rows"][0] for num_objects in object_counts
    ]
    positions = np.arange(len(object_counts))
    width = 0.38
    figure, axis = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        [row["arena_unique_valid_layouts_per_second"] for row in rows],
        width,
        color="#2563eb",
        label="Arena",
    )
    axis.bar(
        positions + width / 2,
        [row["robolab_unique_valid_layouts_per_second"] for row in rows],
        width,
        color="#dc2626",
        label="RoboLab",
    )
    axis.set_xticks(positions, [str(value) for value in object_counts])
    axis.set_yscale("log")
    axis.set_xlabel("Objects per layout")
    axis.set_ylabel("Valid layouts generated per second")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    _save(figure, output_directory / "native_valid_layouts_per_second")


def _write_practical_speed_figure(root: Path, output_directory: Path) -> None:
    analysis = _load(root / "background_practical_generation" / "full" / "analysis.json")
    background_counts = (0, 1, 5, 9)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, constrained_layout=True)
    for axis, backgrounds in zip(axes.flat, background_counts, strict=True):
        rows = sorted(
            (row for row in analysis["rows"] if row["background_objects"] == backgrounds),
            key=lambda row: row["target_layouts"],
        )
        target_counts = [row["target_layouts"] for row in rows]
        arena_rates = [row["arena_unique_valid_layouts_per_second"] for row in rows]
        robolab_rates = [row["robolab_unique_valid_layouts_per_second"] for row in rows]
        axis.plot(target_counts, arena_rates, marker="o", linewidth=2, label="Arena")
        axis.plot(target_counts, robolab_rates, marker="s", linewidth=2, label="RoboLab")
        for target, arena_rate, row in zip(target_counts, arena_rates, rows, strict=True):
            axis.annotate(
                _format_speedup(row["arena_speedup"]),
                (target, arena_rate),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(target_counts, [str(value) for value in target_counts])
        axis.grid(alpha=0.2)
        axis.set_title(f"{backgrounds} fixed backgrounds")
    for axis in axes[-1]:
        axis.set_xlabel("Requested unique shared-valid layouts, K")
    for axis in axes[:, 0]:
        axis.set_ylabel("Unique shared-valid layouts/s")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Native practical generation: early stopping, validation, and retries included",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.015,
        "Labels show Arena / RoboLab throughput. Ten movable objects; medians over three repetitions.",
        ha="center",
        fontsize=8,
    )
    _save(figure, output_directory / "native_practical_generation_speed")


def _write_controlled_factorial_figure(root: Path, output_directory: Path) -> None:
    analysis = _load(root / "background_factorial" / "pilot" / "analysis.json")
    object_counts = (5, 10, 20)
    condition_labels = [
        "Open",
        "19% scattered",
        "19% corridor",
        "38% scattered",
        "38% corridor",
    ]
    keys = [
        (0, "scattered"),
        (19, "scattered"),
        (19, "corridor"),
        (38, "scattered"),
        (38, "corridor"),
    ]
    values = np.full((len(object_counts), len(keys)), np.nan)
    annotations = [["censored" for _ in keys] for _ in object_counts]
    for row_index, num_objects in enumerate(object_counts):
        rows = [row for row in analysis["rows"] if row["num_objects"] == num_objects]
        keyed = {(round(row["measured_excluded_fraction"] * 100), row["topology"]): row for row in rows}
        for column, key in enumerate(keys):
            speedup = keyed[key]["arena_speedup_scene_median"]
            if speedup is not None:
                values[row_index, column] = math.log10(speedup)
                annotations[row_index][column] = _format_speedup(speedup)
    figure, axis = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    _annotated_heatmap(
        axis,
        values,
        annotations,
        condition_labels,
        [str(value) for value in object_counts],
        color_map="RdBu",
        norm=TwoSlopeNorm(vmin=-4.0, vcenter=0.0, vmax=1.0),
        colorbar_label="log₁₀(Arena / RoboLab throughput)",
    )
    axis.set_xlabel("Background condition")
    axis.set_ylabel("Movable objects")
    axis.tick_params(axis="x", rotation=20)
    axis.set_title(
        "Controlled matched-AABB practical pilot, K=128\n"
        "RoboLab adaptive margins, iteration overrides, and relaxation disabled"
    )
    figure.text(
        0.5,
        -0.04,
        "Scene-median paired speedup over three scene instances × three repetitions. "
        "Censored cells contain at least one run that did not reach K.",
        ha="center",
        fontsize=8,
    )
    _save(figure, output_directory / "controlled_factorial_pilot_speed")


def _coverage_rows(analysis: dict, representation: str) -> list[dict]:
    return [row for row in analysis["measurements"] if row["representation"] == representation]


def _write_coverage_figure(root: Path, output_directory: Path) -> None:
    analysis = _load(root / "collision_space" / "factorial_full" / "analysis.json")
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    colors = {"arena": "#2563eb", "robolab": "#dc2626", "reference": "#111827"}
    labels = {"arena": "Arena", "robolab": "RoboLab", "reference": "Uniform reference"}

    for axis, representation in zip(axes[:2], ("aabb", "circle"), strict=True):
        rows = _coverage_rows(analysis, representation)
        for algorithm in ("arena", "robolab", "reference"):
            algorithm_rows = sorted(
                (row for row in rows if row["algorithm"] == algorithm),
                key=lambda row: row["obstacle_size_m"],
            )
            axis.plot(
                [row["obstacle_size_m"] * 100 for row in algorithm_rows],
                [row["coverage_1cm"] * 100 for row in algorithm_rows],
                marker="o",
                color=colors[algorithm],
                label=labels[algorithm],
            )
        axis.set_xlabel("Obstacle size (cm)")
        axis.set_ylabel("Free-space probes covered within 1 cm (%)")
        axis.set_title(f"({'a' if representation == 'aabb' else 'b'}) Matched {representation.upper()} representation")
        axis.grid(alpha=0.25)
        axis.set_ylim(45, 95)
    axes[0].legend(fontsize=8)

    effects = analysis["algorithm_effects"]
    for representation, color, marker in (
        ("aabb", colors["arena"], "o"),
        ("circle", "#7c3aed", "s"),
    ):
        rows = sorted(
            (row for row in effects if row["representation"] == representation),
            key=lambda row: row["obstacle_size_m"],
        )
        axes[2].plot(
            [row["obstacle_size_m"] * 100 for row in rows],
            [row["arena_minus_robolab_coverage_1cm"] * 100 for row in rows],
            marker=marker,
            color=color,
            label=f"Matched {representation.upper()}",
        )
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Obstacle size (cm)")
    axes[2].set_ylabel("Arena − RoboLab 1 cm coverage (percentage points)")
    axes[2].set_title("(c) Residual algorithm effect")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    figure.suptitle(
        "Collision-space coverage after matching collision representation",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.03,
        "2,500 generated positions and 2,500 independent exact-valid probes per repetition; "
        "curves show medians over three repetitions.",
        ha="center",
        fontsize=8,
    )
    _save(figure, output_directory / "solver_coverage_findings")


def _write_coverage_radius_figure(root: Path, output_directory: Path) -> None:
    analysis = _load(root / "collision_space" / "factorial_full" / "analysis.json")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True, constrained_layout=True)
    radius_styles = (
        ("coverage_1cm", "1 cm radius", "o"),
        ("coverage_2cm", "2 cm radius", "s"),
        ("coverage_3cm", "3 cm radius", "^"),
    )
    for axis, representation in zip(axes, ("aabb", "circle"), strict=True):
        rows = sorted(
            (row for row in analysis["algorithm_effects"] if row["representation"] == representation),
            key=lambda row: row["obstacle_size_m"],
        )
        for metric, label, marker in radius_styles:
            effect_key = f"arena_minus_robolab_{metric}"
            axis.plot(
                [row["obstacle_size_m"] * 100 for row in rows],
                [row[effect_key] * 100 for row in rows],
                marker=marker,
                label=label,
            )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Obstacle size (cm)")
        axis.set_title(f"Matched {representation.upper()} representation")
    axes[0].set_ylabel("Arena − RoboLab coverage (percentage points)")
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Coverage conclusion depends on spatial resolution",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.025,
        "At 2–3 cm radii, broad-support differences are small; the large matched-AABB effect occurs at 1 cm.",
        ha="center",
        fontsize=8,
    )
    _save(figure, output_directory / "coverage_radius_sensitivity")


def _write_scene_figure(output_directory: Path) -> None:
    centers = (
        (0.0, 0.0),
        (-0.22, 0.0),
        (0.22, 0.0),
        (0.0, -0.22),
        (0.0, 0.22),
        (-0.22, -0.22),
        (-0.22, 0.22),
        (0.22, -0.22),
        (0.22, 0.22),
    )
    conditions = ((0, "Open"), (1, "1 obstacle"), (5, "5 obstacles"), (9, "9 obstacles"))
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    for axis, (count, label) in zip(axes, conditions, strict=True):
        axis.add_patch(Rectangle((-0.5, -0.5), 1.0, 1.0, fill=False, edgecolor="black", linewidth=1.5))
        for x, y in centers[:count]:
            axis.add_patch(
                Rectangle(
                    (x - 0.1, y - 0.1),
                    0.2,
                    0.2,
                    facecolor="#bfdbfe",
                    edgecolor="none",
                    alpha=0.65,
                )
            )
            axis.add_patch(
                Rectangle(
                    (x - 0.06, y - 0.06),
                    0.12,
                    0.12,
                    facecolor="#2563eb",
                    edgecolor="#1e3a8a",
                    linewidth=0.8,
                )
            )
        axis.set_xlim(-0.52, 0.52)
        axis.set_ylim(-0.52, 0.52)
        axis.set_aspect("equal")
        axis.set_title(label)
        axis.set_xlabel("X (m)")
        if axis is axes[0]:
            axis.set_ylabel("Y (m)")
        else:
            axis.set_yticklabels([])
    figure.suptitle(
        "Fixed-background progression used in the native practical benchmark",
        fontsize=13,
    )
    figure.text(
        0.5,
        -0.04,
        "Dark: physical 12 cm obstacle. Light: center-space excluded for an 8 cm movable object.",
        ha="center",
        fontsize=8,
    )
    _save(figure, output_directory / "background_scene_conditions")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/corl_exp1_3"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/corl_exp1_3/figures"),
    )
    args = parser.parse_args()
    _write_speed_figure(args.results_root, args.output_directory)
    _write_fixed_iteration_figure(args.results_root, args.output_directory)
    _write_two_axis_scaling_bars(args.results_root, args.output_directory)
    _write_native_object_scaling_bars(args.results_root, args.output_directory)
    _write_practical_speed_figure(args.results_root, args.output_directory)
    _write_controlled_factorial_figure(args.results_root, args.output_directory)
    _write_coverage_figure(args.results_root, args.output_directory)
    _write_coverage_radius_figure(args.results_root, args.output_directory)
    _write_scene_figure(args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
