# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-episode summary writer for sensitivity analysis.

``write_episode_summaries`` appends one JSONL row per recorded demo for a just-completed
job. Each row carries:

  - ``job_name`` and ``episode_idx`` for traceability,
  - ``arena_env_args`` — the *entire* job.arena_env_args_dict, i.e. every value that
    parameterized the env for this episode,
  - ``outcomes`` — per-episode outcome values from the task's registered metrics, extracted
    from the recorded hdf5 demos via each metric's ``compute_metric_from_recording``.

The eval-side writer is intentionally analysis-agnostic: it logs all env state, and the
analyzer's ``factors.yaml`` decides which subset of those keys to treat as factors. This
keeps the writer free of any "what counts as a factor?" knowledge.

Import-order note: this module legitimately touches pxr at import time via
``isaaclab_arena.metrics.metrics`` (which imports ``isaaclab.envs.manager_based_rl_env``).
Like ``metrics`` itself, callers must defer importing this module until *after*
``SimulationAppContext`` is active — see ``policy_runner.py`` (which uses the same pattern
for ``compute_metrics``) and ``eval_runner.py``'s per-job try block for examples.
"""

from __future__ import annotations

import h5py
import json
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.metrics.metrics import get_metric_recorder_dataset_path
from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types

if TYPE_CHECKING:
    from isaaclab_arena.evaluation.job_manager import Job


def write_episode_summaries(env, job: Job, output_path: str | Path) -> int:
    """Append one JSONL row per recorded demo for the just-completed job.

    Each row has shape::

        {
          "job_name": "<job.name>",
          "episode_idx": <demo index in the hdf5>,
          "arena_env_args": <full job.arena_env_args_dict>,
          "outcomes": <per-metric value computed from the demo>
        }

    Args:
        env: The (possibly gym-wrapped) Arena env that just finished its rollout. The hdf5
            path and registered metrics are read from ``env.unwrapped.cfg``.
        job: The Job that ran. Its ``arena_env_args_dict`` is logged verbatim under
            ``arena_env_args``.
        output_path: JSONL file to append to. Created (with parent dirs) if absent.

    Returns:
        Number of rows written.
    """
    unwrapped_env = env.unwrapped
    if not hasattr(unwrapped_env.cfg, "metrics") or unwrapped_env.cfg.metrics is None:
        return 0

    arena_env_args_snapshot = dict(job.arena_env_args_dict)

    hdf5_dataset_path = get_metric_recorder_dataset_path(unwrapped_env)
    registered_metrics = unwrapped_env.cfg.metrics
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    env_step_dt = float(unwrapped_env.step_dt)
    with h5py.File(hdf5_dataset_path, "r") as hdf5_file:
        recorded_demos = hdf5_file["data"]
        with open(output_path, "a", encoding="utf-8") as jsonl_output:
            for demo_index, demo_name in enumerate(recorded_demos):
                demo_group = recorded_demos[demo_name]
                raw_outcome_values = {}
                # Find the demo's actual step count by taking the max length across all
                # recorded metric arrays. Per-step time-series recorders (e.g. the
                # ObjectVelocityRecorder for `object_moved_rate`) produce a (T, …) array
                # whose first dim is the step count. Per-episode scalar recorders (e.g.
                # the SuccessRecorder) produce a (1,) array regardless of episode length —
                # using `len()` on the wrong one collapses task_duration to a single step.
                demo_step_count = 0
                # cfg.metrics is a MetricsCfg configclass: one field per metric, each a
                # MetricTermCfg(compute_metric_func, params, recorder_term_name). Iterate its
                # fields the same way MetricsManager does (metrics_manager.py). The per-demo
                # value comes from feeding compute_metric_func a single-element list.
                for metric_name, metric_cfg in registered_metrics.__dict__.items():
                    recorded_metric_data = demo_group[metric_cfg.recorder_term_name][:]
                    raw_outcome_values[metric_name] = metric_cfg.compute_metric_func(
                        [recorded_metric_data], **metric_cfg.params
                    )
                    demo_step_count = max(demo_step_count, len(recorded_metric_data))
                # task_duration: wall-clock-equivalent seconds spent on this episode before
                # termination. Short for fast successes / early failures, max_episode_length
                # for timeouts. Provides a continuous outcome that carries information beyond
                # binary success metrics.
                if demo_step_count > 0:
                    raw_outcome_values["task_duration"] = float(demo_step_count) * env_step_dt
                outcome_values = metrics_to_plain_python_types(raw_outcome_values)
                summary_row = {
                    "job_name": job.name,
                    "episode_idx": demo_index,
                    "arena_env_args": arena_env_args_snapshot,
                    "outcomes": outcome_values,
                }
                jsonl_output.write(json.dumps(summary_row) + "\n")
                rows_written += 1

    return rows_written
