# Source-variant experiment: docstrings and names, all three tiers

Does the quality of the docstrings and names in the source change how fast and how accurately a fresh agent answers
lookup questions? Five python-only source variants were compared on all 60 questions (tiers 1, 2 and 3), next to the
earlier A and B conditions.

| Cond | Source the agent can read                                                              |
|------|----------------------------------------------------------------------------------------|
| A    | full repo, new docstrings, plus the generated API reference                            |
| B    | full repo, new docstrings (docs, markdown, yaml and python)                            |
| c0   | python files only, new docstrings                                                      |
| c1   | python files only, original docstrings                                                 |
| c2   | python files only, all docstrings removed (comments kept)                              |
| c3   | python files only, no docstrings, no comments, classes/functions/params renamed        |
| c4   | python files only, original docstrings, plus a generated API reference built from them |

Each run is one fresh Sonnet agent with no shared context, one question, and no access to anything outside its tree.
Mean tokens are the final context size plus all output tokens (`variants/collect_usage.py`, read from the subagent
transcripts). Score is the fraction of ground-truth facts in the answer (tier 1: correct = 1, partial = 0.5, wrong = 0).

## Tier 1 (20 lookup questions)

| Cond | What the agent sees                                        | Runs | Mean tokens | Mean tools | Mean secs | Score % |
|------|------------------------------------------------------------|-----:|------------:|-----------:|----------:|--------:|
| A    | new docstrings + generated API reference + full repo       |   20 |       52904 |       3.70 |      10.5 |    90.0 |
| B    | new docstrings, full repo (docs, md, yaml, py)             |   20 |       51733 |       3.00 |       9.0 |   100.0 |
| c0   | new docstrings, python only                                |   20 |       52573 |       4.10 |      11.4 |   100.0 |
| c1   | original docstrings, python only                           |   20 |       52508 |       4.05 |      10.9 |    97.5 |
| c2   | no docstrings, python only                                 |   20 |       52413 |       4.05 |      11.0 |    95.0 |
| c3   | no docstrings, no comments, obfuscated names, python only  |   20 |       53178 |       4.65 |      13.3 |    97.5 |
| c4   | original docstrings + generated API reference, python only |   20 |       54855 |       4.75 |      14.4 |   100.0 |

## Tier 2 (20 discovery questions)

| Cond | What the agent sees                                        | Runs | Mean tokens | Mean tools | Mean secs | Score % |
|------|------------------------------------------------------------|-----:|------------:|-----------:|----------:|--------:|
| A    | new docstrings + generated API reference + full repo       |   20 |       55773 |       5.70 |      16.8 |    90.9 |
| B    | new docstrings, full repo (docs, md, yaml, py)             |   20 |       56001 |       5.20 |      16.1 |    92.6 |
| c0   | new docstrings, python only                                |   20 |       56703 |       5.40 |      16.9 |    93.4 |
| c1   | original docstrings, python only                           |   20 |       55476 |       5.15 |      15.8 |    95.5 |
| c2   | no docstrings, python only                                 |   20 |       56025 |       5.20 |      15.4 |    93.9 |
| c3   | no docstrings, no comments, obfuscated names, python only  |   20 |       56859 |       5.70 |      16.9 |    89.5 |
| c4   | original docstrings + generated API reference, python only |   20 |       56908 |       5.85 |      15.9 |    92.1 |

## Tier 3 (20 extra-difficult questions)

| Cond | What the agent sees                                        | Runs | Mean tokens | Mean tools | Mean secs | Score % |
|------|------------------------------------------------------------|-----:|------------:|-----------:|----------:|--------:|
| A    | new docstrings + generated API reference + full repo       |   20 |       62156 |       6.90 |      25.6 |    86.0 |
| B    | new docstrings, full repo (docs, md, yaml, py)             |   20 |       61408 |       6.50 |      25.9 |    86.6 |
| c0   | new docstrings, python only                                |   20 |       62822 |       6.45 |      21.9 |    83.5 |
| c1   | original docstrings, python only                           |   20 |       62661 |       6.40 |      21.9 |    83.0 |
| c2   | no docstrings, python only                                 |   20 |       62812 |       6.90 |      22.9 |    87.9 |
| c3   | no docstrings, no comments, obfuscated names, python only  |   20 |       63705 |       7.05 |      25.4 |    88.3 |
| c4   | original docstrings + generated API reference, python only |   20 |       64385 |       7.65 |      23.7 |    84.1 |

## Total aggregate (tiers 1+2+3)

| Cond | What the agent sees                                        | Runs | Mean tokens | Mean tools | Mean secs | Score % |
|------|------------------------------------------------------------|-----:|------------:|-----------:|----------:|--------:|
| A    | new docstrings + generated API reference + full repo       |   60 |       56944 |       5.43 |      17.7 |    88.9 |
| B    | new docstrings, full repo (docs, md, yaml, py)             |   60 |       56381 |       4.90 |      17.0 |    93.1 |
| c0   | new docstrings, python only                                |   60 |       57366 |       5.32 |      16.7 |    92.3 |
| c1   | original docstrings, python only                           |   60 |       56882 |       5.20 |      16.2 |    92.0 |
| c2   | no docstrings, python only                                 |   60 |       57083 |       5.38 |      16.4 |    92.3 |
| c3   | no docstrings, no comments, obfuscated names, python only  |   60 |       57914 |       5.80 |      18.5 |    91.8 |
| c4   | original docstrings + generated API reference, python only |   60 |       58716 |       6.08 |      18.0 |    92.1 |

## What the numbers say

- Docstrings barely matter for these questions. c0 (new), c1 (original) and c2 (none) are within noise of each other
  on tokens (56.9k to 57.4k), tool calls (5.2 to 5.4), time (16.2 to 16.7 s) and accuracy (92.0 to 92.3 %).
- Obfuscating the names (c3) is the only change with a visible cost: about +0.5 tool calls and +2 s per question
  against c2, with accuracy about half a point lower in total and 4 points lower on tier 2 (discovery questions).
  Tier 3 shows c3 as the best score, which is within the noise of 20 hand-graded questions.
- Adding the generated API reference to the python-only original-docstring tree (c4 against c1) does not help
  either: +0.9 tool calls, +1.8k tokens and +1.8 s per question for the same accuracy (92.1 % against 92.0 %).
  On tier 1 it is slower than c1 (4.75 against 4.05 tool calls, 14.4 against 10.9 s) at 100 % against 97.5 %.
- None of the source variants clearly beats B (full repo): B has the fewest tool calls in total (4.90 against 5.20 to
  5.80) and the best total score. The generated API reference (A) does not help against B either (88.9 % against
  93.1 %, 5.43 against 4.90 tool calls).
- The differences between variants are mostly smaller than the question-to-question spread, so none of them is
  significant on 60 questions.

## Caveats

- c0 to c4 see a python-only tree, while A and B see the full repository, so A/B and c0 to c3 differ in more than
  docstrings. Compare c0 to c3 with each other, c4 with c1, and A with B.
- The prompt is read from a file, which adds one tool call (about 3k tokens) to every run.
- From c1 tier 2 M19 onwards (and for all c4 runs) the agents also write their answer to a file, which adds one Write call; it is
  subtracted from the tool counts here. Earlier runs did not write a file. Tokens include that extra Write turn.
- Tokens and tool calls for all rows were recomputed from the transcripts with the same definition. The token figure
  is within about 1 % of the figure the harness reports.
- Answers are graded by agents against atomic ground-truth facts (c3 answers are translated back with
  `obfuscation_map.json`). Tier 1 and the c0 and c1 tier 2 scores were graded by hand in the main session.
- Some c0 and c1 tier 2 agents saw a context warning from the main session and ignored it.
- Re-run after a missing answer file: c2 tier 3 H12 (the first run did not write its file). c2 tier 2 M19 and M20
  were launched late. Both were run once and graded like the rest.
- 20 questions per cell is a small sample; differences under about 5 points of score or 0.5 tool calls should not be
  read as real.

Raw answers of the runs that wrote an answer file (tiers 2 and 3, and c4 tier 1) are in `variants/answers.tar.gz`; the per-question grader
scores are in `variants/scores/`.

Tables are produced by `variants/summarize.py` from `results.csv`, `results_tier2.csv`, `results_tier3.csv` and
`results_variants.csv`.
