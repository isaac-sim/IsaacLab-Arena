# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare valid non-collision layout distributions against rejection sampling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiversitySummary:
    """Diversity statistics for one method and object count."""

    method: str
    num_objects: int
    sample_count: int
    sample_count_per_repetition: int
    repetitions: int
    unique_1cm_rate: float
    unique_1cm_rate_iqr: float
    mean_grid_coverage: float
    mean_grid_coverage_iqr: float
    median_nearest_neighbor_distance: float
    median_nearest_neighbor_distance_iqr: float
    mean_pair_distance: float
    mean_pair_distance_iqr: float
    sliced_wasserstein_to_rejection: float
    sliced_wasserstein_to_rejection_iqr: float


def _layout_groups(run: dict, target_layouts: int) -> dict[int, np.ndarray]:
    names = tuple(sorted(run["object_xy_sizes"]))
    groups = {}
    for sample in run["samples"]:
        if sample["target_layouts"] != target_layouts:
            continue
        rows = []
        for layout in sample["layouts"]:
            by_name = {pose["object_name"]: pose for pose in layout}
            rows.append([coordinate for name in names for coordinate in (by_name[name]["x"], by_name[name]["y"])])
        groups[sample["seed"]] = np.asarray(rows, dtype=np.float64)
    return groups


def _normalize(layouts: np.ndarray, run: dict) -> np.ndarray:
    xmin, xmax, ymin, ymax = run["table_xy_bounds"]
    names = tuple(sorted(run["object_xy_sizes"]))
    normalized = layouts.copy()
    for index, name in enumerate(names):
        width, depth = run["object_xy_sizes"][name]
        normalized[:, 2 * index] = (layouts[:, 2 * index] - (xmin + width / 2)) / (xmax - xmin - width)
        normalized[:, 2 * index + 1] = (layouts[:, 2 * index + 1] - (ymin + depth / 2)) / (ymax - ymin - depth)
    return normalized


def _unique_rate(layouts: np.ndarray, resolution_m: float = 0.01) -> float:
    quantized = np.rint(layouts / resolution_m).astype(np.int64)
    return len({tuple(row) for row in quantized}) / len(quantized)


def _grid_coverage(normalized: np.ndarray, grid_size: int = 16) -> float:
    coverages = []
    for index in range(normalized.shape[1] // 2):
        xy = np.clip(normalized[:, 2 * index : 2 * index + 2], 0.0, np.nextafter(1.0, 0.0))
        cells = np.floor(xy * grid_size).astype(np.int64)
        occupied = len({(int(x), int(y)) for x, y in cells})
        coverages.append(occupied / (grid_size * grid_size))
    return float(np.mean(coverages))


def _distance_statistics(normalized: np.ndarray, seed: int = 0) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    sample_count, dimensions = normalized.shape
    nearest = np.full(sample_count, np.inf)
    for start in range(0, sample_count, 128):
        block = normalized[start : start + 128]
        distances = np.linalg.norm(block[:, None, :] - normalized[None, :, :], axis=2) / math.sqrt(dimensions)
        rows = np.arange(len(block))
        distances[rows, start + rows] = np.inf
        nearest[start : start + len(block)] = distances.min(axis=1)
    left = generator.integers(0, sample_count, size=10_000)
    right = generator.integers(0, sample_count - 1, size=10_000)
    right += right >= left
    pair_distances = np.linalg.norm(normalized[left] - normalized[right], axis=1) / math.sqrt(dimensions)
    return float(np.median(nearest)), float(np.mean(pair_distances))


def _sliced_wasserstein(left: np.ndarray, right: np.ndarray, seed: int = 0, projections: int = 128) -> float:
    count = min(len(left), len(right), 640)
    generator = np.random.default_rng(seed)
    left = left[generator.permutation(len(left))[:count]]
    right = right[generator.permutation(len(right))[:count]]
    directions = generator.normal(size=(projections, left.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    left_projected = np.sort(left @ directions.T, axis=0)
    right_projected = np.sort(right @ directions.T, axis=0)
    return float(np.mean(np.abs(left_projected - right_projected)))


def _median_iqr(values: list[float]) -> tuple[float, float]:
    return float(np.median(values)), float(np.percentile(values, 75) - np.percentile(values, 25))


def _validate_compatible(runs: dict[str, dict], target_layouts: int) -> None:
    reference = runs["random_rejection"]
    required_fields = (
        "workload",
        "table_xy_bounds",
        "object_xy_sizes",
        "max_attempts_per_layout",
        "seeds",
        "target_layout_counts",
    )
    for method, run in runs.items():
        for field in required_fields:
            if run.get(field) != reference.get(field):
                raise ValueError(f"{method} has incompatible {field}")
        if target_layouts not in run["target_layout_counts"]:
            raise ValueError(f"{method} does not contain target_layouts={target_layouts}")
        if not run.get("solver_config"):
            raise ValueError(f"{method} is missing solver_config provenance")


def summarize(
    method: str,
    run: dict,
    reference: dict,
    target_layouts: int,
    sample_count_per_repetition: int,
) -> DiversitySummary:
    """Summarize one valid-layout distribution against rejection sampling."""
    groups = _layout_groups(run, target_layouts)
    reference_groups = _layout_groups(reference, target_layouts)
    metrics = []
    for repeat_index, seed in enumerate(run["seeds"]):
        generator = np.random.default_rng(seed % (2**32))
        layouts = groups[seed][generator.permutation(len(groups[seed]))[:sample_count_per_repetition]]
        reference_layouts = reference_groups[seed][
            generator.permutation(len(reference_groups[seed]))[:sample_count_per_repetition]
        ]
        normalized = _normalize(layouts, run)
        normalized_reference = _normalize(reference_layouts, reference)
        nearest, pair_distance = _distance_statistics(normalized, seed=repeat_index)
        swd_count = sample_count_per_repetition // 2
        if method == "random_rejection":
            distribution_distance = _sliced_wasserstein(
                normalized_reference[:swd_count],
                normalized_reference[swd_count : 2 * swd_count],
                seed=repeat_index,
            )
        else:
            distribution_distance = _sliced_wasserstein(
                normalized[:swd_count],
                normalized_reference[:swd_count],
                seed=repeat_index,
            )
        metrics.append((
            _unique_rate(layouts),
            _grid_coverage(normalized),
            nearest,
            pair_distance,
            distribution_distance,
        ))
    unique_rate, unique_rate_iqr = _median_iqr([row[0] for row in metrics])
    grid_coverage, grid_coverage_iqr = _median_iqr([row[1] for row in metrics])
    nearest_distance, nearest_distance_iqr = _median_iqr([row[2] for row in metrics])
    pair_distance, pair_distance_iqr = _median_iqr([row[3] for row in metrics])
    distribution_distance, distribution_distance_iqr = _median_iqr([row[4] for row in metrics])
    return DiversitySummary(
        method=method,
        num_objects=len(run["object_xy_sizes"]),
        sample_count=sample_count_per_repetition * len(metrics),
        sample_count_per_repetition=sample_count_per_repetition,
        repetitions=len(metrics),
        unique_1cm_rate=unique_rate,
        unique_1cm_rate_iqr=unique_rate_iqr,
        mean_grid_coverage=grid_coverage,
        mean_grid_coverage_iqr=grid_coverage_iqr,
        median_nearest_neighbor_distance=nearest_distance,
        median_nearest_neighbor_distance_iqr=nearest_distance_iqr,
        mean_pair_distance=pair_distance,
        mean_pair_distance_iqr=pair_distance_iqr,
        sliced_wasserstein_to_rejection=distribution_distance,
        sliced_wasserstein_to_rejection_iqr=distribution_distance_iqr,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/corl_exp1_3"))
    parser.add_argument("--target-layouts", type=int, default=256)
    parser.add_argument("--object-count", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    counts = tuple(args.object_count or range(1, 11))
    summaries = []
    inputs = []
    for count in counts:
        paths = {
            "random_rejection": args.root / "rejection" / f"objects_{count}.json",
            "arena": args.root / "arena" / f"objects_{count}.json",
            "robolab": args.root / "robolab" / f"objects_{count}.json",
        }
        runs = {method: json.loads(path.read_text(encoding="utf-8")) for method, path in paths.items()}
        _validate_compatible(runs, args.target_layouts)
        groups = {method: _layout_groups(run, args.target_layouts) for method, run in runs.items()}
        sample_count = min(len(layouts) for method_groups in groups.values() for layouts in method_groups.values())
        assert sample_count >= 2, "diversity metrics require at least two valid layouts per repetition"
        reference = runs["random_rejection"]
        for method in ("random_rejection", "arena", "robolab"):
            run = runs[method]
            summaries.append(summarize(method, run, reference, args.target_layouts, sample_count))
            inputs.append({
                "path": str(paths[method]),
                "method": method,
                "num_objects": count,
                "source_commit": run.get("source_commit"),
                "solver_config": run["solver_config"],
                "seeds": run["seeds"],
                "table_xy_bounds": run["table_xy_bounds"],
                "object_xy_sizes": run["object_xy_sizes"],
            })
    payload = {
        "schema_version": 2,
        "reference_distribution": "uniform independent XY proposals conditioned on exact AABB non-overlap",
        "target_layouts_per_repetition": args.target_layouts,
        "uncertainty": "median and interquartile range across repetitions after equal-count rarefaction",
        "inputs": inputs,
        "summaries": [asdict(summary) for summary in summaries],
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(asdict(summaries[0])))
        writer.writeheader()
        writer.writerows(asdict(summary) for summary in summaries)
    print("| Objects | Method | Samples | Unique@1cm | Grid coverage | NN spread | Pair spread | SWD to rejection |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in summaries:
        print(
            f"| {summary.num_objects} | {summary.method} | {summary.sample_count} | "
            f"{summary.unique_1cm_rate:.3f} | {summary.mean_grid_coverage:.3f} | "
            f"{summary.median_nearest_neighbor_distance:.4f} | {summary.mean_pair_distance:.4f} | "
            f"{summary.sliced_wasserstein_to_rejection:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
