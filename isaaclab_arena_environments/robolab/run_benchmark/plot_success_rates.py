# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Plot per-task success rates from a download_osmo_workflows.py results folder.

Walks a ``<tag>/<policy>/<task>/`` results tree, reads the per-episode ``success`` field from each
task's ``episode_results*.jsonl``, and renders a grouped bar chart with one bar per policy per task.
"""

from __future__ import annotations

import argparse
import json
import matplotlib
import sys
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _success_rate(task_dir: Path) -> tuple[float, int] | None:
    """Return ``(success_rate, num_episodes)`` for a task, or None if it has no scored episodes."""
    successes = 0
    total = 0
    for results_path in sorted(task_dir.rglob("episode_results*.jsonl")):
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            success = json.loads(line).get("success")
            if success is None:
                continue
            successes += int(bool(success))
            total += 1
    if total == 0:
        return None
    return successes / total, total


def _collect_rates(results_dir: Path) -> tuple[list[str], list[str], dict[tuple[str, str], tuple[float, int]]]:
    """Return sorted policies, sorted tasks, and a ``(policy, task) -> (rate, num_episodes)`` map."""
    policies = sorted(child.name for child in results_dir.iterdir() if child.is_dir())
    tasks: set[str] = set()
    rates: dict[tuple[str, str], tuple[float, int]] = {}
    for policy in policies:
        for task_dir in sorted(child for child in (results_dir / policy).iterdir() if child.is_dir()):
            result = _success_rate(task_dir)
            if result is None:
                print(f"No scored episodes for {policy}/{task_dir.name}; skipping.", file=sys.stderr)
                continue
            tasks.add(task_dir.name)
            rates[(policy, task_dir.name)] = result
    return policies, sorted(tasks), rates


def _plot(
    policies: list[str],
    tasks: list[str],
    rates: dict[tuple[str, str], tuple[float, int]],
    output_path: Path,
    title: str,
) -> None:
    """Render the grouped success-rate bar chart and save it to ``output_path``."""
    task_positions = np.arange(len(tasks))
    bar_width = 0.8 / len(policies)
    colors = plt.get_cmap("tab10").colors

    figure, ax = plt.subplots(figsize=(max(8.0, 1.1 * len(tasks) * len(policies)), 6.0))
    for policy_index, policy in enumerate(policies):
        offsets = task_positions + (policy_index - (len(policies) - 1) / 2) * bar_width
        heights = [rates.get((policy, task), (0.0, 0))[0] for task in tasks]
        bars = ax.bar(offsets, heights, bar_width, label=policy, color=colors[policy_index % len(colors)])
        for bar, task in zip(bars, tasks):
            rate, num_episodes = rates.get((policy, task), (None, 0))
            if rate is None:
                continue
            ax.annotate(
                f"{rate:.0%}\nn={num_episodes}",
                (bar.get_x() + bar.get_width() / 2, rate),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(task_positions)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_title(title)
    ax.legend(title="policy")
    ax.grid(alpha=0.3, axis="y")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Tag folder produced by download_osmo_workflows.py")
    parser.add_argument("--output", type=Path, help="Output image path (default: <results_dir>/success_rates.png)")
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        raise SystemExit(f"Results directory '{args.results_dir}' does not exist.")

    policies, tasks, rates = _collect_rates(args.results_dir)
    if not rates:
        raise SystemExit(f"No episode results found under '{args.results_dir}'.")

    output_path = args.output or args.results_dir / "success_rates.png"
    _plot(policies, tasks, rates, output_path, title=f"Success rate by task — {args.results_dir.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
