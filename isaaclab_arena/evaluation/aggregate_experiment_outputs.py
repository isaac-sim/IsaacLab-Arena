# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Combine independently executed Arena Run outputs into one Experiment report."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from isaaclab_arena.visualization.report import build_report

_FIRST_REBUILD_RESULTS_FILENAME = "episode_results_rebuild0.jsonl"


def aggregate_experiment_outputs(run_input_dirs: Mapping[str, str | Path], output_dir: str | Path) -> Path:
    """Copy one output directory per Run and build their combined report.

    Args:
        run_input_dirs: Run name to OSMO task-output directory.
        output_dir: Destination for the combined Run directories and report.

    Returns:
        Path to the generated Experiment report.
    """
    assert run_input_dirs, "At least one Run output is required"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for run_name, input_dir in run_input_dirs.items():
        result_paths = sorted(Path(input_dir).rglob(_FIRST_REBUILD_RESULTS_FILENAME))
        assert len(result_paths) == 1, (
            f"Run '{run_name}' must have exactly one {_FIRST_REBUILD_RESULTS_FILENAME} below '{input_dir}', "
            f"found {len(result_paths)}"
        )
        run_output_dir = output_dir / run_name
        assert not run_output_dir.exists(), f"Run output destination already exists: '{run_output_dir}'"
        run_output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(result_paths[0].parent, run_output_dir)

    return build_report(output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine Arena Run outputs and build an Experiment report.")
    parser.add_argument("--run-inputs", required=True, type=Path, help="JSON mapping of Run names to input paths")
    parser.add_argument("--output-dir", required=True, type=Path, help="Combined Experiment output directory")
    return parser.parse_args()


def main() -> None:
    """Load the Run input mapping and aggregate its outputs."""
    args = _parse_args()
    with args.run_inputs.open(encoding="utf-8") as run_inputs_file:
        run_input_dirs = json.load(run_inputs_file)
    assert isinstance(run_input_dirs, dict), "Run inputs must be a JSON mapping"
    aggregate_experiment_outputs(run_input_dirs, args.output_dir)


if __name__ == "__main__":
    main()
