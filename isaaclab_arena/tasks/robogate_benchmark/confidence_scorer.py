"""Deployment Confidence Score calculator (0-100).

Weighted sum of 5 component scores:
    grasp_success_rate:   0.30
    cycle_time:           0.20
    collision_count:      0.25
    edge_case_performance: 0.15
    baseline_delta:       0.10

Score interpretation:
    76-100: PASS  — safe to deploy
    51-75:  WARN  — deploy with monitoring
    0-50:   FAIL  — do not deploy
"""

from __future__ import annotations

from typing import Any

from robogate_benchmark.metrics import ScenarioSummary


DEFAULT_WEIGHTS: dict[str, float] = {
    "grasp_success_rate": 0.30,
    "cycle_time": 0.20,
    "collision_count": 0.25,
    "edge_case_performance": 0.15,
    "baseline_delta": 0.10,
}


def _score_grasp_success_rate(value: float) -> float:
    """Score grasp success rate (0-100).

    Maps 0.80-1.00 to 0-100. Below 0.80 = 0.
    """
    if value >= 1.0:
        return 100.0
    if value <= 0.80:
        return 0.0
    return (value - 0.80) / 0.20 * 100.0


def _score_cycle_time(
    value: float, baseline_value: float | None
) -> float:
    """Score cycle time relative to baseline (0-100).

    100 = same or better. 0 = 30%+ slower.
    """
    if baseline_value is None or baseline_value == 0:
        return 50.0
    ratio = value / baseline_value
    if ratio <= 1.0:
        return 100.0
    if ratio >= 1.3:
        return 0.0
    return (1.3 - ratio) / 0.3 * 100.0


def _score_collision_count(value: int) -> float:
    """Score collision count (0-100).

    0 collisions = 100, 1 = 50, 2 = 25, 3+ = 0.
    """
    if value == 0:
        return 100.0
    if value == 1:
        return 50.0
    if value == 2:
        return 25.0
    return 0.0


def _score_edge_case_performance(
    scenario_summaries: dict[str, ScenarioSummary],
) -> float:
    """Score edge case performance (0-100)."""
    edge = scenario_summaries.get("edge_cases")
    if edge is None or edge.total == 0:
        return 50.0
    return edge.pass_rate * 100.0


def _score_baseline_delta(
    metrics: dict[str, dict[str, Any]],
) -> float:
    """Score overall baseline delta (0-100).

    100 = all improved. 0 = all regressed.
    """
    improvements = 0
    regressions = 0
    total = 0

    for metric_id, m in metrics.items():
        delta = m.get("delta")
        if delta is None:
            continue
        total += 1
        # For grasp_success_rate: higher is better
        if metric_id == "grasp_success_rate":
            if delta > 0:
                improvements += 1
            elif delta < 0:
                regressions += 1
        else:
            # For all others: lower is better
            if delta < 0:
                improvements += 1
            elif delta > 0:
                regressions += 1

    if total == 0:
        return 50.0

    ratio = (improvements - regressions) / total
    return (ratio + 1.0) / 2.0 * 100.0


def compute_confidence_score(
    metrics: dict[str, dict[str, Any]],
    scenario_summaries: dict[str, ScenarioSummary],
    baseline_metrics: dict[str, float | int] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute Deployment Confidence Score (0-100).

    Args:
        metrics: Evaluated metric results (from evaluate_all_metrics).
        scenario_summaries: Per-category summaries.
        baseline_metrics: Baseline metric values for cycle_time scoring.
        weights: Override weight dict.

    Returns:
        Dictionary with 'score', 'verdict', and 'components'.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    components: dict[str, float] = {}

    # grasp_success_rate
    gsr = metrics.get("grasp_success_rate", {})
    components["grasp_success_rate"] = _score_grasp_success_rate(
        gsr.get("value", 0.0)
    )

    # cycle_time
    ct = metrics.get("cycle_time", {})
    ct_baseline = (
        float(baseline_metrics["cycle_time"])
        if baseline_metrics and "cycle_time" in baseline_metrics
        else ct.get("baseline")
    )
    components["cycle_time"] = _score_cycle_time(
        ct.get("value", 0.0), ct_baseline
    )

    # collision_count
    cc = metrics.get("collision_count", {})
    components["collision_count"] = _score_collision_count(
        int(cc.get("value", 0))
    )

    # edge_case_performance
    components["edge_case_performance"] = _score_edge_case_performance(
        scenario_summaries
    )

    # baseline_delta
    components["baseline_delta"] = _score_baseline_delta(metrics)

    # Weighted sum
    score = sum(
        weights.get(k, 0.0) * v
        for k, v in components.items()
        if k in weights
    )
    score = max(0.0, min(100.0, round(score, 1)))

    # Verdict
    if score >= 76:
        verdict = "PASS"
    elif score >= 51:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "score": score,
        "verdict": verdict,
        "components": components,
        "weights": weights,
    }
