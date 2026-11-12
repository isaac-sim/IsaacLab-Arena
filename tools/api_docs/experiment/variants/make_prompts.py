# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write one prompt file per (variant, question) for the source-variant experiment.

Usage: ``python make_prompts.py <variants_dir> <prompts_dir> <experiment_dir> <tier>`` where tier is 1, 2 or 3.
For variant c3 the backticked identifiers of the question are replaced by their obfuscated names.
"""

import json
import re
import sys
from pathlib import Path

HEADER = (
    "You are answering a question about a Python codebase. Answer from what you find in the files; do not guess. "
    "Be efficient: stop as soon as you can answer with evidence. Do not modify anything.\n\n"
)
FORMAT = (
    "Final message format, nothing else: `ANSWER:` the answer (concise but covering every requested fact), then "
    "`EVIDENCE:` the path:line locations you relied on.\n"
)


def load_questions(experiment_dir: Path, tier: str) -> dict[str, str]:
    questions: dict[str, str] = {}
    if tier == "1":
        for line in (experiment_dir / "questions.md").read_text().split("\n"):
            m = re.match(r"\|\s*(Q\d\d)\s*\|[^|]*\|\s*(.*?)\s*\|\s*$", line)
            if m:
                questions[m.group(1)] = m.group(2)
    else:
        text = (experiment_dir / f"questions_tier{tier}.md").read_text()
        for m in re.finditer(r"^## (\w+) \([^)]*\)\n\n(.*?)\n\n", text, re.M | re.S):
            questions[m.group(1)] = m.group(2).strip()
    return questions


def translate(question: str, mapping: dict[str, str]) -> str:
    def repl_span(span: re.Match) -> str:
        inner = span.group(1)
        inner = re.sub(r"[A-Za-z_][A-Za-z_0-9]*", lambda t: mapping.get(t.group(0), t.group(0)), inner)
        return f"`{inner}`"

    return re.sub(r"`([^`]*)`", repl_span, question)


def main(variants_dir: Path, prompts_dir: Path, experiment_dir: Path, tier: str) -> None:
    prompts_dir.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((variants_dir / "obfuscation_map.json").read_text())
    questions = load_questions(experiment_dir, tier)
    for variant in ("c0", "c1", "c2", "c3"):
        repo = variants_dir / variant
        para = (
            f"The repository is at {repo} . Only read files inside that directory (apart from this prompt file); do not"
            " look elsewhere on the filesystem.\n\n"
        )
        for qid, text in questions.items():
            q = translate(text, mapping) if variant == "c3" else text
            (prompts_dir / f"{variant.upper()}-{qid}.md").write_text(HEADER + para + "Question: " + q + "\n\n" + FORMAT)
    print(len(questions), "questions x 4 variants")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
