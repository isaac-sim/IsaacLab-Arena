# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types

if TYPE_CHECKING:
    from isaaclab_arena.evaluation.job_manager import Job


# Bumped when the on-disk record shape changes so readers can branch on it.
SCHEMA_VERSION = 1


def write_episode_summaries(env, job: Job, output_path: str | Path) -> int:
    """Append one JSONL row per recorded episode for the just-completed job.

    The file is self-describing: its first line is a ``"record": "meta"`` header (written once,
    by whichever job reaches an empty/absent file) carrying the run-level slice identity, the
    enabled-variation factor schema, and the outcome names. Every later line is an episode::

        {
          "record": "episode",
          "job_name": "<job.name>",
          "episode_idx": <episode index in the recorded dataset>,
          "arena_env_args": <full job.arena_env_args_dict>,
          "variation_draws": <realized <asset>.<variation> values this build sampled>,
          "outcomes": <per-episode metric values>
        }

    Per-episode metric values come from the env's ``MetricsManager`` (the same machinery that
    backs ``compute_metrics``), so all HDF5/metric access stays in the metrics layer. The
    variation factor schema and draws are read off the env where ``load_env`` attached them.

    Args:
        env: The (possibly gym-wrapped) Arena env that just finished its rollout. Its
            ``MetricsManager`` provides the per-episode metric values.
        job: The Job that ran. Its ``arena_env_args_dict`` is logged verbatim under
            ``arena_env_args``.
        output_path: JSONL file to append to. Created (with parent dirs) if absent.

    Returns:
        Number of episode rows written.
    """
    unwrapped_env = env.unwrapped
    if not hasattr(unwrapped_env.cfg, "metrics") or unwrapped_env.cfg.metrics is None:
        return 0

    per_episode_metrics = unwrapped_env.metrics_manager.compute_per_episode()
    arena_env_args_snapshot = dict(job.arena_env_args_dict)
    # Build-time draws are fixed for this env, so the same dict applies to every episode it ran.
    variation_draws = dict(getattr(unwrapped_env, "arena_variation_draws", {}) or {})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_run_header_if_new(output_path, unwrapped_env, job)

    with open(output_path, "a", encoding="utf-8") as jsonl_output:
        for episode_idx, episode_metrics in enumerate(per_episode_metrics):
            summary_row = {
                "record": "episode",
                "job_name": job.name,
                "episode_idx": episode_idx,
                "arena_env_args": arena_env_args_snapshot,
                "variation_draws": variation_draws,
                "outcomes": metrics_to_plain_python_types(episode_metrics),
            }
            jsonl_output.write(json.dumps(summary_row) + "\n")

    return len(per_episode_metrics)


def _write_run_header_if_new(output_path: Path, unwrapped_env, job: Job) -> None:
    """Write the one-time ``"record": "meta"`` header iff ``output_path`` is empty/absent.

    Made idempotent by the empty-file check so it fires exactly once per run — including across
    the per-chunk subprocesses that append to a shared file (chunks run sequentially, so the
    first writes the header and the rest see a non-empty file and skip).
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    header = {
        "record": "meta",
        "schema_version": SCHEMA_VERSION,
        "slice": _slice_identity(job),
        "factors": dict(getattr(unwrapped_env, "arena_variation_factor_schema", {}) or {}),
        "outcome_names": list(unwrapped_env.metrics_manager.active_terms),
    }
    with open(output_path, "a", encoding="utf-8") as jsonl_output:
        jsonl_output.write(json.dumps(header) + "\n")


def _slice_identity(job: Job) -> dict:
    """The (policy, task, embodiment) tuple this run analyses — MNPE assumes a single source."""
    arena_env_args = job.arena_env_args_dict
    return {
        "environment": arena_env_args.get("environment"),
        "embodiment": arena_env_args.get("embodiment"),
        "policy_type": job.policy_type,
    }
