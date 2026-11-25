# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare physical deliveries on matched layouts for one task and one training seed."""

import argparse
import json
import numpy as np
from pathlib import Path

METHODS = ("oracle", "cloud", "current_depth", "history", "recurrent", "compact_route", "swapped_cloud")
"""Episode-record filenames required for a full comparison."""


def read_trials(path):
    """Index finished episodes by seed, environment and episode number, rejecting duplicate trials."""
    records = {}
    with path.open() as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                key = (record["seed"], record["env_id"], record["episode_in_env"])
                if key in records:
                    raise ValueError(f"{path}: duplicate episode {key}")
                records[key] = record
    return records


def paired_report(folder):
    """Measure per-layout delivery differences and paired bootstrap intervals on 50 held-out layouts."""
    records = {name: read_trials(folder / f"{name}.jsonl") for name in METHODS}
    reference = records["cloud"]
    keys = sorted(reference)
    if len(keys) != 50:
        raise ValueError(f"Expected 50 cloud trials, received {len(keys)}.")
    tasks = {record["spatial_memory"]["task"] for record in reference.values()}
    if len(tasks) != 1:
        raise ValueError("Use a separate directory for each task.")
    methods = {}
    for name, trials in records.items():
        if trials.keys() != reference.keys():
            raise ValueError(f"{name} has different trial IDs from cloud.")
        for key in keys:
            memory = trials[key]["spatial_memory"]
            expected = reference[key]["spatial_memory"]
            if memory["task"] != expected["task"] or memory["layout"] != expected["layout"]:
                raise ValueError(f"{name} and cloud have different layouts in {key}.")
        methods[name] = np.asarray([trials[key]["success"] for key in keys], dtype=int)
    draws = np.random.default_rng(7611).integers(0, len(keys), (10000, len(keys)))
    comparisons = {}
    for name, successes in methods.items():
        difference = methods["cloud"] - successes
        lower, upper = np.quantile(difference[draws].mean(axis=1), (0.025, 0.975))
        comparisons[name] = {
            "deliveries": int(successes.sum()),
            "cloud_only_wins": int((difference > 0).sum()),
            "baseline_only_wins": int((difference < 0).sum()),
            "cloud_advantage_percentage_points": float(100 * difference.mean()),
            "paired_95_percent_interval": [float(100 * lower), float(100 * upper)],
        }
    return {"task": tasks.pop(), "layouts": len(keys), "methods": comparisons}


def main():
    """Print the matched delivery results for the supplied directory of episode records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help=f"Directory with one JSONL file per method: {', '.join(METHODS)}")
    args = parser.parse_args()
    print(json.dumps(paired_report(args.folder), indent=2))


if __name__ == "__main__":
    main()
