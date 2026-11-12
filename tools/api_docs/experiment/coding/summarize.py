# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Merge grader scores and usage into results_coding.csv and print Markdown tables.

Usage: python3 summarize.py <scratch_coding_dir> <results_csv>
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tasks import TASKS

CONDS = ["c1", "c2", "c3", "c4"]
LABELS = {
    "c1": "original docstrings",
    "c2": "no docstrings",
    "c3": "no docstrings, no comments, obfuscated names",
    "c4": "original docstrings + generated API reference",
}


def table(headers, rows, right):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    fmt = lambda r: "| " + " | ".join(  # noqa: E731
        str(v).rjust(w) if i in right else str(v).ljust(w) for i, (v, w) in enumerate(zip(r, widths))
    ) + " |"
    sep = "|" + "|".join(("-" * (w + 1) + ":") if i in right else ("-" * (w + 2)) for i, w in enumerate(widths)) + "|"
    return "\n".join([fmt(headers), sep] + [fmt(r) for r in rows])


def main(coding: Path, out_csv: Path) -> None:
    usage = {}
    for r in csv.DictReader((coding / "usage.csv").open()):
        usage[f"{r['variant']}-{r['id']}"] = r
    runs = []
    for task in TASKS:
        for r in csv.DictReader((coding / "scores" / f"{task['id']}.csv").open()):
            r.pop(None, None)
            u = usage[r["run"]]
            cond = r["run"].split("-")[0]
            runs.append(
                {
                    "cond": cond,
                    "task": task["id"],
                    "category": task["category"],
                    "tokens": int(u["tokens"]),
                    "tools": int(u["tool_uses"]),
                    "secs": float(u["duration_s"]),
                    "score": float(r["checks_hit"]) / float(r["checks_total"]),
                    "defects": int(float(r["defects"])),
                    "note": r["note"],
                }
            )
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)

    def agg(rows):
        n = len(rows)
        return [
            n,
            f"{sum(r['tokens'] for r in rows) / n:,.0f}",
            f"{sum(r['tools'] for r in rows) / n:.1f}",
            f"{sum(r['secs'] for r in rows) / n:.0f}",
            f"{100 * sum(r['score'] for r in rows) / n:.1f}",
            f"{sum(r['defects'] for r in rows) / n:.1f}",
        ]

    headers = ["What the agent sees", "Runs", "Mean tokens", "Mean tools", "Mean secs", "Score %", "Defects/run"]
    print("## Aggregate over 10 tasks\n")
    print(table(headers, [[LABELS[c]] + agg([r for r in runs if r["cond"] == c]) for c in CONDS], {1, 2, 3, 4, 5, 6}))
    print("\n## By category\n")
    cats = sorted({t["category"] for t in TASKS})
    for cat in cats:
        print(f"\n### {cat}\n")
        print(
            table(
                headers,
                [[LABELS[c]] + agg([r for r in runs if r["cond"] == c and r["category"] == cat]) for c in CONDS],
                {1, 2, 3, 4, 5, 6},
            )
        )
    print("\n## Per task\n")
    th = ["Task", "Category", "Cond", "Tokens", "Tools", "Secs", "Score %", "Defects"]
    rows = []
    for t in TASKS:
        for c in CONDS:
            r = next(x for x in runs if x["task"] == t["id"] and x["cond"] == c)
            rows.append(
                [t["id"], t["category"], c, f"{r['tokens']:,}", r["tools"], f"{r['secs']:.0f}",
                 f"{100 * r['score']:.0f}", r["defects"]]
            )
    print(table(th, rows, {3, 4, 5, 6, 7}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
