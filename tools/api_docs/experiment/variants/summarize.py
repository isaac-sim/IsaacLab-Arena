# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Print the per-tier tables and the total aggregate of the experiment as markdown.

Usage: ``python summarize.py <experiment_dir>`` (reads results.csv, results_tier2.csv, results_tier3.csv and
results_variants.csv).
"""

import csv
import statistics as st
import sys
from pathlib import Path

GRADE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def table(header, body, numeric):
    widths = [max(len(header[i]), *(len(r[i]) for r in body)) for i in range(len(header))]

    def line(r):
        return (
            "| "
            + " | ".join((r[i].rjust(widths[i]) if i in numeric else r[i].ljust(widths[i])) for i in range(len(r)))
            + " |"
        )

    sep = (
        "|"
        + "|".join(("-" * (widths[i] + 1) + ":" if i in numeric else "-" * (widths[i] + 2)) for i in range(len(widths)))
        + "|"
    )
    return "\n".join([line(header), sep] + [line(r) for r in body])


def stats(rows):
    tokens = [int(r["tokens"]) for r in rows]
    tools = [int(r["tool_uses"]) for r in rows]
    secs = [float(r["duration_s"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    return [
        str(len(rows)),
        "%.0f" % st.mean(tokens),
        "%.2f" % st.mean(tools),
        "%.1f" % st.mean(secs),
        "%.1f" % (100 * st.mean(scores)),
    ]


def load(path, arm, tier1):
    out = []
    for r in csv.DictReader(Path(path).open()):
        if r["arm"] != arm:
            continue
        score = GRADE[r["grade"]] if tier1 else float(r["facts_hit"]) / float(r["facts_total"])
        out.append({"tokens": r["tokens"], "tool_uses": r["tool_uses"], "duration_s": r["duration_s"], "score": score})
    return out


def main(d: Path) -> None:
    variants: dict[tuple[str, str], list] = {}
    for r in csv.DictReader((d / "results_variants.csv").open()):
        variants.setdefault((r["variant"], r["tier"]), []).append(r)
    arms = {
        "1": (load(d / "results.csv", "A", True), load(d / "results.csv", "B", True)),
        "2": (load(d / "results_tier2.csv", "A", False), load(d / "results_tier2.csv", "B", False)),
        "3": (load(d / "results_tier3.csv", "A", False), load(d / "results_tier3.csv", "B", False)),
    }
    header = ["Cond", "What the agent sees", "Runs", "Mean tokens", "Mean tools", "Mean secs", "Score %"]
    numeric = {2, 3, 4, 5, 6}
    descriptions = {
        "A": "new docstrings + generated API reference + full repo",
        "B": "new docstrings, full repo (docs, md, yaml, py)",
        "c0": "new docstrings, python only",
        "c1": "original docstrings, python only",
        "c2": "no docstrings, python only",
        "c3": "no docstrings, no comments, obfuscated names, python only",
        "c4": "original docstrings + generated API reference, python only",
    }
    titles = {
        "1": "Tier 1 (20 lookup questions)",
        "2": "Tier 2 (20 discovery questions)",
        "3": "Tier 3 (20 extra-difficult questions)",
    }
    by_cond: dict[str, list] = {c: [] for c in descriptions}
    for tier in ("1", "2", "3"):
        a, b = arms[tier]
        by_cond["A"] += a
        by_cond["B"] += b
        rows = [["A", descriptions["A"]] + stats(a), ["B", descriptions["B"]] + stats(b)]
        for v in ("c0", "c1", "c2", "c3", "c4"):
            tier_rows = variants[(v, tier)]
            by_cond[v] += [
                {
                    "tokens": r["tokens"],
                    "tool_uses": r["tool_uses"],
                    "duration_s": r["duration_s"],
                    "score": float(r["score"]),
                }
                for r in tier_rows
            ]
            rows.append([v, descriptions[v]] + stats(tier_rows))
        print(f"## {titles[tier]}\n")
        print(table(header, rows, numeric))
        print()
    print("## Total aggregate (tiers 1+2+3)\n")
    total = [[c, descriptions[c]] + stats(by_cond[c]) for c in descriptions]
    print(table(header, total, numeric))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
