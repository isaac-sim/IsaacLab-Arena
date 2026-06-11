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


def write_episode_summaries(env, job: Job, output_path: str | Path) -> int:
    """Append one JSONL row per recorded episode for the just-completed job.

    Each row has shape::

        {
          "job_name": "<job.name>",
          "episode_idx": <episode index in the recorded dataset>,
          "arena_env_args": <full job.arena_env_args_dict>,
          "outcomes": <per-episode metric values>
        }

    Per-episode metric values come from the env's ``MetricsManager`` (the same machinery
    that backs ``compute_metrics``), so all HDF5/metric access stays in the metrics layer.

    Args:
        env: The (possibly gym-wrapped) Arena env that just finished its rollout. Its
            ``MetricsManager`` provides the per-episode metric values.
        job: The Job that ran. Its ``arena_env_args_dict`` is logged verbatim under
            ``arena_env_args``.
        output_path: JSONL file to append to. Created (with parent dirs) if absent.

    Returns:
        Number of rows written.
    """
    unwrapped_env = env.unwrapped
    if not hasattr(unwrapped_env.cfg, "metrics") or unwrapped_env.cfg.metrics is None:
        return 0

    per_episode_metrics = unwrapped_env.metrics_manager.compute_per_episode()
    arena_env_args_snapshot = dict(job.arena_env_args_dict)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as jsonl_output:
        for episode_idx, episode_metrics in enumerate(per_episode_metrics):
            summary_row = {
                "job_name": job.name,
                "episode_idx": episode_idx,
                "arena_env_args": arena_env_args_snapshot,
                "outcomes": metrics_to_plain_python_types(episode_metrics),
            }
            jsonl_output.write(json.dumps(summary_row) + "\n")

    return len(per_episode_metrics)
