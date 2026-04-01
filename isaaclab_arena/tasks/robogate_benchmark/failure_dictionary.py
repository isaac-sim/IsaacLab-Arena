"""RoboGate Failure Dictionary — HuggingFace 30K dataset integration.

Downloads and analyzes the boundary-focused failure dictionary from
HuggingFace: liveplex/robogate-failure-dictionary (30,720 episodes).

Dataset splits:
    train:      24,576 episodes (80%)
    validation:  3,072 episodes (10%)
    test:        3,072 episodes (10%)

Columns:
    episode_id, scenario_category, variant, seed, success, failure_type,
    cycle_time_s, collision, drop, grasp_miss, obj_x/y/z, target_x/y/z,
    obj_scale, obj_mass, approach_noise, grasp_tol, confidence_score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

HF_DATASET_ID = "liveplex/robogate-failure-dictionary"
EXPECTED_TOTAL = 30_720


@dataclass
class FailureStat:
    """Aggregated failure statistics."""

    total_episodes: int
    success_rate: float
    failure_types: dict[str, int]
    failure_rates: dict[str, float]
    per_category: dict[str, dict[str, Any]]
    boundary_zone_rate: float


def download_dataset(
    cache_dir: str | Path | None = None,
    split: str = "test",
) -> Any:
    """Download the RoboGate failure dictionary from HuggingFace.

    Args:
        cache_dir: Local cache directory for the dataset.
        split: Dataset split to load ("train", "validation", "test").

    Returns:
        HuggingFace Dataset object.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install datasets: pip install datasets huggingface-hub"
        )

    logger.info("Downloading %s (split=%s)...", HF_DATASET_ID, split)
    ds = load_dataset(
        HF_DATASET_ID,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    logger.info("Loaded %d episodes", len(ds))
    return ds


def analyze_failures(dataset: Any) -> FailureStat:
    """Analyze failure patterns in the dataset.

    Args:
        dataset: HuggingFace Dataset with failure dictionary columns.

    Returns:
        FailureStat with aggregated statistics.
    """
    total = len(dataset)
    successes = sum(1 for row in dataset if row["success"])

    # Failure type counts
    failure_types: dict[str, int] = {}
    for row in dataset:
        if not row["success"]:
            ft = row.get("failure_type", "unknown")
            failure_types[ft] = failure_types.get(ft, 0) + 1

    total_failures = total - successes
    failure_rates = {
        ft: count / total_failures if total_failures > 0 else 0.0
        for ft, count in failure_types.items()
    }

    # Per-category breakdown
    categories: dict[str, dict[str, Any]] = {}
    for row in dataset:
        cat = row["scenario_category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "failures": {}}
        categories[cat]["total"] += 1
        if row["success"]:
            categories[cat]["passed"] += 1
        else:
            ft = row.get("failure_type", "unknown")
            categories[cat]["failures"][ft] = (
                categories[cat]["failures"].get(ft, 0) + 1
            )

    for cat_data in categories.values():
        cat_data["success_rate"] = (
            cat_data["passed"] / cat_data["total"]
            if cat_data["total"] > 0
            else 0.0
        )

    # Boundary zone: episodes with confidence score between 40-60
    boundary_count = 0
    for row in dataset:
        cs = row.get("confidence_score", 50)
        if 40 <= cs <= 60:
            boundary_count += 1
    boundary_rate = boundary_count / total if total > 0 else 0.0

    return FailureStat(
        total_episodes=total,
        success_rate=successes / total if total > 0 else 0.0,
        failure_types=failure_types,
        failure_rates=failure_rates,
        per_category=categories,
        boundary_zone_rate=boundary_rate,
    )


def get_boundary_episodes(
    dataset: Any,
    confidence_range: tuple[float, float] = (40.0, 60.0),
) -> list[dict[str, Any]]:
    """Extract boundary-zone episodes (near pass/fail threshold).

    Args:
        dataset: HuggingFace Dataset.
        confidence_range: Confidence score range for boundary zone.

    Returns:
        List of episode dicts in the boundary zone.
    """
    low, high = confidence_range
    episodes = []
    for row in dataset:
        cs = row.get("confidence_score", 50)
        if low <= cs <= high:
            episodes.append(dict(row))
    return episodes


def compute_variant_difficulty(dataset: Any) -> dict[str, float]:
    """Compute per-variant difficulty scores (failure rates).

    Args:
        dataset: HuggingFace Dataset.

    Returns:
        Mapping from variant name to failure rate (0.0-1.0).
    """
    variants: dict[str, dict[str, int]] = {}
    for row in dataset:
        v = row["variant"]
        if v not in variants:
            variants[v] = {"total": 0, "failed": 0}
        variants[v]["total"] += 1
        if not row["success"]:
            variants[v]["failed"] += 1

    return {
        v: data["failed"] / data["total"] if data["total"] > 0 else 0.0
        for v, data in sorted(variants.items(), key=lambda x: -x[1].get("failed", 0))
    }


def print_summary(stats: FailureStat) -> str:
    """Format failure statistics as human-readable text.

    Args:
        stats: FailureStat from analyze_failures().

    Returns:
        Formatted summary string.
    """
    lines = [
        f"RoboGate Failure Dictionary Summary",
        f"{'='*40}",
        f"Total episodes:     {stats.total_episodes:,}",
        f"Success rate:       {stats.success_rate:.1%}",
        f"Boundary zone rate: {stats.boundary_zone_rate:.1%}",
        f"",
        f"Failure Types:",
    ]
    for ft, count in sorted(
        stats.failure_types.items(), key=lambda x: -x[1]
    ):
        rate = stats.failure_rates[ft]
        lines.append(f"  {ft:20s} {count:>6,} ({rate:.1%})")

    lines.append(f"")
    lines.append(f"Per-Category Breakdown:")
    for cat, data in sorted(stats.per_category.items()):
        sr = data["success_rate"]
        lines.append(
            f"  {cat:25s} {data['passed']:>4}/{data['total']:<4} ({sr:.1%})"
        )

    return "\n".join(lines)
