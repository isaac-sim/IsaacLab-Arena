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
    variants: dict[str, list] = {}
    for r in csv.DictReader((d / "results_variants.csv").open()):
        variants.setdefault(r["variant"], []).append(r)
    a1, b1 = load(d / "results.csv", "A", True), load(d / "results.csv", "B", True)
    a2, b2 = load(d / "results_tier2.csv", "A", False), load(d / "results_tier2.csv", "B", False)
    a3, b3 = load(d / "results_tier3.csv", "A", False), load(d / "results_tier3.csv", "B", False)
    header = ["Cond", "What the agent sees", "Runs", "Mean tokens", "Mean tools", "Mean secs", "Score %"]
    numeric = {2, 3, 4, 5, 6}
    A = "new docstrings + generated API reference + full repo"
    B = "new docstrings, full repo (docs, md, yaml, py)"
    print("## Tier 1 (20 lookup questions)\n")
    rows = [["A", A] + stats(a1), ["B", B] + stats(b1)]
    for v, desc in (
        ("c0", "new docstrings, python only"),
        ("c1", "original docstrings, python only"),
        ("c2", "no docstrings, python only"),
        ("c3", "no docstrings, no comments, obfuscated names, python only"),
    ):
        rows.append([v, desc] + stats(variants[v]))
    print(table(header, rows, numeric))
    for name, a, b in (("Tier 2 (20 discovery questions)", a2, b2), ("Tier 3 (20 extra-difficult questions)", a3, b3)):
        print(f"\n## {name}\n")
        print(table(header, [["A", A] + stats(a), ["B", B] + stats(b)], numeric))
    print("\n## Total aggregate\n")
    total = [["A", "tiers 1+2+3"] + stats(a1 + a2 + a3), ["B", "tiers 1+2+3"] + stats(b1 + b2 + b3)]
    for v in ("c0", "c1", "c2", "c3"):
        total.append([v, "tier 1 only"] + stats(variants[v]))
    print(table(["Cond", "Tiers", "Runs", "Mean tokens", "Mean tools", "Mean secs", "Score %"], total, numeric))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
