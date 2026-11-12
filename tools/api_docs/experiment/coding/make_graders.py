# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write the diff of every run against its pristine tree and one grader prompt per task.

Usage: python3 make_graders.py <scratch_dir> <variants_dir>
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from setup_runs import CONDITIONS
from tasks import TASKS


def main(scratch: Path, variants: Path) -> None:
    coding = scratch / "coding"
    (coding / "diffs").mkdir(exist_ok=True)
    (coding / "scores").mkdir(exist_ok=True)
    for cond, tree in CONDITIONS.items():
        for task in TASKS:
            name = f"{cond}-{task['id']}"
            result = subprocess.run(
                ["diff", "-ruN", "-x", "__pycache__", str(variants / tree), str(coding / "ws" / name)],
                capture_output=True,
                text=True,
            )
            text = result.stdout.replace(str(variants / tree), "a").replace(str(coding / "ws" / name), "b")
            (coding / "diffs" / f"{name}.diff").write_text(text)

    for task in TASKS:
        checks = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(task["rubric"]))
        names = [f"{c}-{task['id']}" for c in CONDITIONS]
        files = "\n".join(f"  - {coding}/diffs/{n}.diff" for n in names)
        text = f"""You are grading four attempts at one coding task on the Isaac Lab Arena python source.

Task given to the agents (backticked names in the c3 attempt were translated to obfuscated names):
{task['prompt']}

Rubric (score each check per attempt: 1 = fully met, 0.5 = partly met, 0 = not met):
{checks}

The attempts are unified diffs against the pristine tree (a/ = pristine, b/ = attempt):
{files}

Notes:
- Judge from the diffs. You may read the pristine source tree {variants}/c1 (original names) to check claims such as
  whether an attribute exists or which call sites exist. Do not run anything and do not edit any source.
- The c3 attempt uses obfuscated identifiers (prefixes C_, f_, a_, v_, p_ plus 6 hex digits). Translate them back with
  {variants}/obfuscation_map.json (keys are the original names, values the obfuscated names) via a small python
  snippet before judging. The c3 source tree is {variants}/c3.
- Nothing could be executed by the agents, so judge correctness by reading: would the code work and the tests pass?
- Also count `defects`: real bugs, broken imports/references, or behaviour changes the attempt introduced beyond the task.
  Do not count style nits. An empty diff scores 0 on every check.

Write a CSV with the Write tool to {coding}/scores/{task['id']}.csv with header
run,checks_hit,checks_total,defects,note
and one row per attempt (run is one of {', '.join(names)}), checks_hit = the sum of your check scores,
checks_total = {len(task['rubric'])}, note = at most 12 words without commas. Then reply with the single word: done.
"""
        (coding / f"grade_{task['id']}.md").write_text(text)
    print("diffs and graders written")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
