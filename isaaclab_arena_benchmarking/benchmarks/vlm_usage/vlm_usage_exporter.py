# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export the VLM requests and tokens of a finished Arena Experiment as an InfluxDB KPI record.

The records file is ``kpis_vlm_usage.json``. It holds one ``vlm_usage`` record with the requests and tokens summed
over every Run whose policy wrote VLM inference usage, the same totals per episode of those Runs, and the latency of
each request, measured per request even when parallel environments send them concurrently. The record is
timestamped with the latest Run end and tagged with the Arena git commit SHA, branch, benchmark id (the Experiment
name), and the endpoint and model that served the requests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_SINGLE_RUN_INFERENCE_USAGE_FILENAME,
    ArenaExperimentResult,
)
from isaaclab_arena.utils.git_stamp import GitStamp
from isaaclab_arena_benchmarking.benchmarks.registry import register_kpi_exporter
from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import InfluxdbKpiExporter, InfluxdbKpiRecord

VLM_USAGE_RECORDS_FILENAME = "kpis_vlm_usage.json"
"""Name of the VLM usage KPI records file in an Experiment output directory."""

VLM_USAGE_KEYS = (
    "endpoint",
    "model",
    "num_requests",
    "num_responses_without_usage",
    "request_latency_total_ms",
    "request_latency_max_ms",
)
"""Keys an inference usage file must hold to count as VLM usage; the token totals are read alongside them."""


@dataclass(kw_only=True)
class VLMUsageRecord(InfluxdbKpiRecord):
    """VLM requests and tokens summed over the Runs that used a VLM, in total and per episode."""

    record_type: ClassVar[str] = "vlm_usage"

    endpoint: str
    """Inference endpoint that served the requests."""

    model: str
    """Model that served the requests."""

    num_requests: int
    """VLM requests sent, including retries and requests that failed."""

    num_episodes: int
    """Episodes of the Runs that used a VLM."""

    requests_per_episode: float | None = None
    """VLM requests per episode, or None when there were no episodes."""

    request_latency_mean_ms: float | None = None
    """Mean wall time of one request in milliseconds, or None when no request was sent."""

    request_latency_max_ms: float | None = None
    """Wall time of the slowest request in milliseconds, or None when no request was sent."""

    prompt_tokens: int | None = None
    """Prompt tokens, or None when a response reported no token usage."""

    completion_tokens: int | None = None
    """Completion tokens, or None when a response reported no token usage."""

    tokens_per_episode: float | None = None
    """Prompt and completion tokens per episode, or None when tokens or episodes are missing."""


def read_vlm_inference_usage(run_output_directory: Path) -> list[dict[str, Any]]:
    """Return the VLM inference usage a Run's policy wrote, one per rebuild; empty for a Run without VLM usage.

    Args:
        run_output_directory: Output directory of one Run.
    """
    usage_path = run_output_directory / ARENA_SINGLE_RUN_INFERENCE_USAGE_FILENAME
    if not usage_path.exists():
        return []
    usage_per_rebuild = []
    for usage in json.loads(usage_path.read_text(encoding="utf-8")):
        if all(key in usage for key in VLM_USAGE_KEYS):
            usage_per_rebuild.append(usage)
    return usage_per_rebuild


def build_vlm_usage_record(
    usage_per_rebuild: list[dict[str, Any]],
    num_episodes: int,
    *,
    timestamp_ns: int,
    git_stamp: GitStamp,
    benchmark_id: str,
) -> VLMUsageRecord:
    """Return the record of the summed VLM usage; every entry must name the same endpoint and model.

    Args:
        usage_per_rebuild: VLM inference usage of every rebuild of the Runs that used a VLM.
        num_episodes: Episodes of those Runs.
        timestamp_ns: Point time in Unix epoch nanoseconds (UTC).
        git_stamp: Arena git commit and branch the Experiment ran with.
        benchmark_id: Benchmark the Experiment belongs to.

    Returns:
        The VLM usage record; token fields are None when any response reported no token usage.
    """
    endpoints = {usage["endpoint"] for usage in usage_per_rebuild}
    models = {usage["model"] for usage in usage_per_rebuild}
    assert len(endpoints) == 1 and len(models) == 1, (
        "VLM usage KPIs compare one endpoint and model per Experiment, got endpoints"
        f" {sorted(endpoints)} and models {sorted(models)}"
    )
    num_requests = sum(usage["num_requests"] for usage in usage_per_rebuild)
    request_latency_mean_ms = None
    request_latency_max_ms = None
    if num_requests > 0:
        request_latency_total_ms = sum(usage["request_latency_total_ms"] for usage in usage_per_rebuild)
        request_latency_mean_ms = request_latency_total_ms / num_requests
        request_latency_max_ms = max(usage["request_latency_max_ms"] for usage in usage_per_rebuild)
    num_responses_without_usage = sum(usage["num_responses_without_usage"] for usage in usage_per_rebuild)
    prompt_tokens = None
    completion_tokens = None
    tokens_per_episode = None
    if num_responses_without_usage == 0:
        prompt_tokens = sum(usage["prompt_tokens"] for usage in usage_per_rebuild)
        completion_tokens = sum(usage["completion_tokens"] for usage in usage_per_rebuild)
        if num_episodes > 0:
            tokens_per_episode = (prompt_tokens + completion_tokens) / num_episodes
    return VLMUsageRecord(
        timestamp_ns=timestamp_ns,
        git_sha=git_stamp.git_sha,
        git_branch=git_stamp.git_branch,
        benchmark_id=benchmark_id,
        endpoint=endpoints.pop(),
        model=models.pop(),
        num_requests=num_requests,
        num_episodes=num_episodes,
        requests_per_episode=num_requests / num_episodes if num_episodes > 0 else None,
        request_latency_mean_ms=request_latency_mean_ms,
        request_latency_max_ms=request_latency_max_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_episode=tokens_per_episode,
    )


@register_kpi_exporter("vlm_usage")
class VLMUsageKpiExporter(InfluxdbKpiExporter):
    """Exports the VLM requests and tokens of a finished Arena Experiment output directory."""

    record_types: ClassVar[tuple[str, ...]] = (VLMUsageRecord.record_type,)

    def get_records_filename(self) -> str:
        return VLM_USAGE_RECORDS_FILENAME

    def build_records_from_output_dir(self, output_dir: Path) -> list[InfluxdbKpiRecord]:
        """Build the VLM usage record of a finished Experiment output directory.

        Reads ``arena_experiment_result.json`` for the episodes, timestamp, git stamp and Experiment name, and each
        Run's inference usage file for its VLM usage.

        Args:
            output_dir: Experiment output root holding the result file and one directory per Run.

        Returns:
            One record over the Runs that used a VLM; empty when no Run did.
        """
        output_dir = Path(output_dir)
        result = ArenaExperimentResult.from_experiment_output(output_dir)
        usage_per_rebuild: list[dict[str, Any]] = []
        num_episodes = 0
        for run_name, run_result in result.runs.items():
            run_usage = read_vlm_inference_usage(output_dir / run_name)
            if not run_usage:
                continue
            usage_per_rebuild.extend(run_usage)
            for rebuild in run_result["rebuilds"]:
                num_episodes += len(rebuild["episodes"])
        if not usage_per_rebuild:
            return []
        records: list[InfluxdbKpiRecord] = [
            build_vlm_usage_record(
                usage_per_rebuild,
                num_episodes,
                timestamp_ns=result.latest_run_end_timestamp_ns,
                git_stamp=result.git_stamp,
                benchmark_id=result.experiment_name,
            )
        ]
        return records
