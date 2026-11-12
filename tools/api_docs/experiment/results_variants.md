# Results: docstring and naming variants, and aggregate

Conditions c0 to c3 are source-only runs on python-only variants of the repository (see `variants/build_variants.py`),
tier 1 questions only. For c3 the backticked identifiers in the questions are translated to the obfuscated names.
Score: tier 1 correct = 1, partial = 0.5, wrong = 0; tiers 2 and 3 share of ground-truth facts covered. A and B are the
earlier docs / no-docs arms. They ran without a prompt-file read and with the whole repository visible; c0 to c3 read a
prompt file first (one extra tool call, about 3k tokens) and see python files only. Regenerate the tables with
`python3 variants/summarize.py .`.

## Findings

- On tier 1 the four source variants are within one question of each other (100, 97.5, 95 and 97.5 percent) and cost
  the same: about 52k tokens and 4 to 4.7 tool calls. Removing docstrings added no tool calls; obfuscating names added
  about 0.6 calls and 2 s per question.
- Score losses were not caused by missing docstrings: a miscounted list (c1 and c2; the list itself was right), a
  constraint that only the docstring stated (c2 Q11), and a mechanism the agent could not confirm because the method
  name was obfuscated (c3 Q17).
- With obfuscated names, agents rebuilt meaning from code structure, string literals, defaults and unrenamed library
  calls, and said so explicitly.
- Tiers 2 and 3 were not run on the variants (time and context budget), so the effect of docstrings and names on
  discovery and multi-file questions is unmeasured. The tier 1 variant rows are not comparable in difficulty with the A/B
  aggregate, which includes the harder tiers.
- c3 keeps file names, string literals and external library names. The renamer is a repo-wide tokenizer-based rename
  written for this experiment, because python-minifier and similar tools only rename within one module.
- Hand-graded, one sample per cell.

## Tier 1 (20 lookup questions)

| Cond | What the agent sees                                       | Runs | Mean tokens | Mean tools | Mean secs | Score % |
|------|-----------------------------------------------------------|-----:|------------:|-----------:|----------:|--------:|
| A    | new docstrings + generated API reference + full repo      |   20 |       52904 |       3.70 |      10.5 |    90.0 |
| B    | new docstrings, full repo (docs, md, yaml, py)            |   20 |       51733 |       3.00 |       9.0 |   100.0 |
| c0   | new docstrings, python only                               |   20 |       52103 |       4.10 |      11.4 |   100.0 |
| c1   | original docstrings, python only                          |   20 |       52051 |       4.05 |      11.0 |    97.5 |
| c2   | no docstrings, python only                                |   20 |       51959 |       4.05 |      11.1 |    95.0 |
| c3   | no docstrings, no comments, obfuscated names, python only |   20 |       52586 |       4.65 |      13.3 |    97.5 |

## Tier 2 (20 discovery questions)

| Cond | What the agent sees                                  | Runs | Mean tokens | Mean tools | Mean secs | Score % |
|------|------------------------------------------------------|-----:|------------:|-----------:|----------:|--------:|
| A    | new docstrings + generated API reference + full repo |   20 |       55773 |       5.70 |      16.8 |    90.9 |
| B    | new docstrings, full repo (docs, md, yaml, py)       |   20 |       56001 |       5.20 |      16.1 |    92.6 |

## Tier 3 (20 extra-difficult questions)

| Cond | What the agent sees                                  | Runs | Mean tokens | Mean tools | Mean secs | Score % |
|------|------------------------------------------------------|-----:|------------:|-----------:|----------:|--------:|
| A    | new docstrings + generated API reference + full repo |   20 |       62156 |       6.90 |      25.6 |    86.0 |
| B    | new docstrings, full repo (docs, md, yaml, py)       |   20 |       61408 |       6.50 |      25.9 |    86.6 |

## Total aggregate

| Cond | Tiers       | Runs | Mean tokens | Mean tools | Mean secs | Score % |
|------|-------------|-----:|------------:|-----------:|----------:|--------:|
| A    | tiers 1+2+3 |   60 |       56944 |       5.43 |      17.7 |    88.9 |
| B    | tiers 1+2+3 |   60 |       56381 |       4.90 |      17.0 |    93.1 |
| c0   | tier 1 only |   20 |       52103 |       4.10 |      11.4 |   100.0 |
| c1   | tier 1 only |   20 |       52051 |       4.05 |      11.0 |    97.5 |
| c2   | tier 1 only |   20 |       51959 |       4.05 |      11.1 |    95.0 |
| c3   | tier 1 only |   20 |       52586 |       4.65 |      13.3 |    97.5 |
