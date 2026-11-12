# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Create one workspace copy and one prompt file per (condition, task) of the coding-task experiment.

Usage: python3 setup_runs.py <scratch_dir> <variants_dir> <api_docs_dir>
"""

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tasks import TASKS

CONDITIONS = {"c1": "c1", "c2": "c2", "c3": "c3", "c4": "c1"}  # condition -> source tree

TEMPLATE = """You are a software engineer. Do this coding task in a copy of the Isaac Lab Arena python source.

Working tree (edit files only inside it): {ws}
{docs_line}
Constraints:
- Only the python source is available. The simulator and torch are not installed, so you cannot run the code or the
  tests. You can use shell tools, python3 -m py_compile and pytest on pure-python code if that helps.
- Edit files in the working tree with your tools. Create new files where needed. Do not touch anything outside it.
- Follow the conventions of the surrounding code (docstring style, assert style, naming, license header on new files).
- Be efficient: do not do more than the task asks.

Task:
{task}

When you are done reply with one sentence per changed file. Do not write any other file.
"""

DOCS_LINE = (
    "A generated API reference of this code (text files, one per module, start at {docs}/index.txt) is available "
    "read-only. Start from the API reference; open source files only if you need to.\n"
)


def translate(text: str, mapping: dict) -> str:
    """Replace backticked identifiers with their obfuscated names."""

    def repl(m):
        word = m.group(1)
        if "." in word:
            return "`" + ".".join(mapping.get(p, p) for p in word.split(".")) + "`"
        return "`" + mapping.get(word, word) + "`"

    return re.sub(r"`([A-Za-z_][A-Za-z_0-9.]*)`", repl, text)


def main(scratch: Path, variants: Path, docs: Path) -> None:
    mapping = json.loads((variants / "obfuscation_map.json").read_text())
    ws_root = scratch / "coding" / "ws"
    prompts = scratch / "coding" / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    for cond, tree in CONDITIONS.items():
        for task in TASKS:
            name = f"{cond}-{task['id']}"
            ws = ws_root / name
            if not ws.exists():
                shutil.copytree(variants / tree, ws, ignore=shutil.ignore_patterns("__pycache__"))
            text = task["prompt"]
            if cond == "c3":
                text = translate(text, mapping)
            docs_line = DOCS_LINE.format(docs=docs) if cond == "c4" else ""
            (prompts / f"{name}.md").write_text(TEMPLATE.format(ws=ws, docs_line=docs_line, task=text))
    print("workspaces:", len(list(ws_root.iterdir())))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
