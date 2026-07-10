# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run an Arena Experiment locally."""

from __future__ import annotations

from isaaclab_arena.evaluation.eval_runner_cli import parse_eval_runner_cfg
from isaaclab_arena.evaluation.local_evaluation import run_local_experiment


def main(cli_args: list[str] | None = None) -> int:
    """Parse one eval-runner invocation and return its process exit code."""
    return run_local_experiment(parse_eval_runner_cfg(cli_args))


if __name__ == "__main__":
    raise SystemExit(main())
