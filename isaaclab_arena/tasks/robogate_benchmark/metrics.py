"""RoboGate 5-metric evaluation suite for Isaac Lab-Arena.

Implements MetricBase-compatible metrics:
    1. GraspSuccessRate  — grasp success ratio (threshold >= 0.92)
    2. CycleTime         — average cycle time in seconds (<= baseline * 1.1)
    3. CollisionCount    — total collisions (== 0)
    4. DropRate           — object drop ratio (<= 0.03)
    5. GraspMissRate      — grasp miss ratio (<= baseline * 1.2)

Each metric follows the Isaac Lab-Arena MetricBase pattern with
recorder term config and compute_metric_from_recording().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# CycleResult — single episode outcome
# ---------------------------------------------------------------------------


@dataclass
class CycleResult:
    """Result of a single simulation episode."""

    scenario_category: str
    scenario_variant: str
    success: bool
    cycle_time: float
    collision: bool
    drop: bool
    grasp_miss: bool


# ---------------------------------------------------------------------------
# Pass criteria (from PRD)
# ---------------------------------------------------------------------------

DEFAULT_PASS_CRITERIA: dict[str, dict[str, Any]] = {
    "grasp_success_rate": {"op": ">=", "value": 0.92, "baseline_relative": False},
    "cycle_time": {
        "op": "<=",
        "value": 1.1,
        "baseline_relative": True,
        "min_baseline": 0.5,
    },
    "collision_count": {"op": "==", "value": 0, "baseline_relative": False},
    "drop_rate": {"op": "<=", "value": 0.03, "baseline_relative": False},
    "grasp_miss_rate": {"op": "<=", "value": 1.2, "baseline_relative": True},
}


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------


def compute_grasp_success_rate(cycles: list[CycleResult]) -> float:
    """Compute grasp success rate (0.0 - 1.0)."""
    if not cycles:
        return 0.0
    return sum(1 for c in cycles if c.success) / len(cycles)


def compute_cycle_time(cycles: list[CycleResult]) -> float:
    """Compute average cycle time in seconds."""
    if not cycles:
        return 0.0
    return float(np.mean([c.cycle_time for c in cycles]))


def compute_collision_count(cycles: list[CycleResult]) -> int:
    """Count total collisions."""
    return sum(1 for c in cycles if c.collision)


def compute_drop_rate(cycles: list[CycleResult]) -> float:
    """Compute drop rate (0.0 - 1.0)."""
    if not cycles:
        return 0.0
    return sum(1 for c in cycles if c.drop) / len(cycles)


def compute_grasp_miss_rate(cycles: list[CycleResult]) -> float:
    """Compute grasp miss rate (0.0 - 1.0)."""
    if not cycles:
        return 0.0
    return sum(1 for c in cycles if c.grasp_miss) / len(cycles)


def compute_all_metrics(cycles: list[CycleResult]) -> dict[str, float | int]:
    """Compute all 5 RoboGate metrics.

    Args:
        cycles: List of episode results.

    Returns:
        Dictionary mapping metric_id to computed value.
    """
    return {
        "grasp_success_rate": compute_grasp_success_rate(cycles),
        "cycle_time": compute_cycle_time(cycles),
        "collision_count": compute_collision_count(cycles),
        "drop_rate": compute_drop_rate(cycles),
        "grasp_miss_rate": compute_grasp_miss_rate(cycles),
    }


# ---------------------------------------------------------------------------
# Pass evaluation
# ---------------------------------------------------------------------------


def evaluate_pass(
    metric_id: str,
    new_value: float,
    baseline_value: float | None = None,
    criteria: dict[str, Any] | None = None,
) -> bool:
    """Evaluate whether a metric passes its threshold.

    Args:
        metric_id: Metric identifier.
        new_value: Current metric value.
        baseline_value: Baseline value for relative criteria.
        criteria: Override criteria dict.

    Returns:
        True if the metric passes.
    """
    if criteria is None:
        criteria = DEFAULT_PASS_CRITERIA.get(metric_id)
    if criteria is None:
        return True

    op = criteria["op"]
    threshold_value = criteria["value"]
    is_relative = criteria.get("baseline_relative", False)

    if is_relative:
        if baseline_value is None:
            return True
        min_baseline = criteria.get("min_baseline")
        if min_baseline is not None and baseline_value < min_baseline:
            return True
        threshold = baseline_value * threshold_value
    else:
        threshold = threshold_value

    if op == ">=":
        return new_value >= threshold
    elif op == "<=":
        return new_value <= threshold
    elif op == "==":
        return new_value == threshold
    return False


def evaluate_all_metrics(
    metrics: dict[str, float | int],
    baseline_metrics: dict[str, float | int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate all metrics against pass criteria.

    Args:
        metrics: Current metric values.
        baseline_metrics: Baseline metric values.

    Returns:
        Dictionary mapping metric_id to evaluation result dict.
    """
    results: dict[str, dict[str, Any]] = {}

    for metric_id, new_value in metrics.items():
        new_val = float(new_value)
        baseline_val = (
            float(baseline_metrics[metric_id])
            if baseline_metrics and metric_id in baseline_metrics
            else None
        )

        # Baseline diff
        delta = None
        delta_pct = None
        if baseline_val is not None:
            delta = new_val - baseline_val
            if baseline_val != 0:
                pct = (delta / abs(baseline_val)) * 100
                sign = "+" if pct >= 0 else ""
                delta_pct = f"{sign}{pct:.1f}%"
            else:
                delta_pct = "+inf%" if delta > 0 else ("+0.0%" if delta == 0 else "-inf%")

        passed = evaluate_pass(metric_id, new_val, baseline_val)

        results[metric_id] = {
            "value": new_val,
            "baseline": baseline_val,
            "delta": delta,
            "delta_pct": delta_pct,
            "passed": passed,
        }

    return results


# ---------------------------------------------------------------------------
# Scenario summary
# ---------------------------------------------------------------------------


@dataclass
class ScenarioSummary:
    """Per-category scenario summary."""

    category: str
    total: int
    passed: int
    failed: int

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total


def compute_scenario_summary(
    cycles: list[CycleResult],
) -> dict[str, ScenarioSummary]:
    """Compute per-category pass/fail summaries.

    Args:
        cycles: List of episode results.

    Returns:
        Dictionary mapping category to ScenarioSummary.
    """
    categories: dict[str, list[CycleResult]] = {}
    for c in cycles:
        categories.setdefault(c.scenario_category, []).append(c)

    summaries: dict[str, ScenarioSummary] = {}
    for cat, cat_cycles in categories.items():
        total = len(cat_cycles)
        passed = sum(1 for c in cat_cycles if c.success)
        summaries[cat] = ScenarioSummary(
            category=cat,
            total=total,
            passed=passed,
            failed=total - passed,
        )

    return summaries


# ---------------------------------------------------------------------------
# Failure evidence
# ---------------------------------------------------------------------------


@dataclass
class FailureEvidence:
    """Aggregated failure evidence."""

    scenario: str
    failure_type: str
    count: int
    description: str
    affected_metrics: list[str]
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"


FAILURE_TYPE_SEVERITY: dict[str, str] = {
    "collision": "HIGH",
    "drop": "MEDIUM",
    "grasp_miss": "MEDIUM",
    "unknown_failure": "LOW",
}

FAILURE_AFFECTED_METRICS: dict[str, list[str]] = {
    "collision": ["collision_count"],
    "drop": ["drop_rate", "grasp_success_rate"],
    "grasp_miss": ["grasp_miss_rate", "grasp_success_rate"],
    "unknown_failure": ["grasp_success_rate"],
}


def _classify_failure(cycle: CycleResult) -> str:
    """Classify primary failure type for a failed cycle."""
    if cycle.collision:
        return "collision"
    if cycle.drop:
        return "drop"
    if cycle.grasp_miss:
        return "grasp_miss"
    return "unknown_failure"


def collect_failure_evidence(cycles: list[CycleResult]) -> list[FailureEvidence]:
    """Collect and aggregate failure evidence from cycle results.

    Args:
        cycles: List of episode results.

    Returns:
        List of FailureEvidence sorted by severity (HIGH first).
    """
    failed = [c for c in cycles if not c.success]
    if not failed:
        return []

    groups: dict[tuple[str, str], list[CycleResult]] = {}
    for c in failed:
        ftype = _classify_failure(c)
        key = (f"{c.scenario_category}/{c.scenario_variant}", ftype)
        groups.setdefault(key, []).append(c)

    evidence_list: list[FailureEvidence] = []
    for (scenario, ftype), group_cycles in groups.items():
        severity = FAILURE_TYPE_SEVERITY.get(ftype, "LOW")
        affected = FAILURE_AFFECTED_METRICS.get(ftype, ["grasp_success_rate"])
        count = len(group_cycles)
        description = f"{scenario}: {ftype} x{count}"

        evidence_list.append(
            FailureEvidence(
                scenario=scenario,
                failure_type=ftype,
                count=count,
                description=description,
                affected_metrics=affected,
                severity=severity,
            )
        )

    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    evidence_list.sort(key=lambda e: severity_order.get(e.severity, 99))

    return evidence_list


# ---------------------------------------------------------------------------
# Isaac Lab-Arena MetricBase adapters
# ---------------------------------------------------------------------------

try:
    from isaaclab_arena.metrics.metric_base import MetricBase

    class RoboGateSuccessRateMetric(MetricBase):
        """Grasp success rate metric for Isaac Lab-Arena."""

        name = "robogate_grasp_success_rate"
        recorder_term_name = "success"

        def get_recorder_term_cfg(self):  # type: ignore[override]
            from isaaclab_arena.metrics.success_rate import SuccessRecorderCfg

            return SuccessRecorderCfg(name=self.recorder_term_name)

        def compute_metric_from_recording(
            self, recorded_metric_data: list[np.ndarray]
        ) -> float:
            if not recorded_metric_data:
                return 0.0
            all_flags = np.concatenate(recorded_metric_data)
            return float(np.mean(all_flags))

    class RoboGateCollisionMetric(MetricBase):
        """Collision count metric for Isaac Lab-Arena."""

        name = "robogate_collision_count"
        recorder_term_name = "collision"

        def get_recorder_term_cfg(self):  # type: ignore[override]
            from isaaclab_arena.metrics.success_rate import SuccessRecorderCfg

            return SuccessRecorderCfg(name=self.recorder_term_name)

        def compute_metric_from_recording(
            self, recorded_metric_data: list[np.ndarray]
        ) -> float:
            if not recorded_metric_data:
                return 0.0
            all_flags = np.concatenate(recorded_metric_data)
            return float(np.sum(all_flags))

except ImportError:
    pass
