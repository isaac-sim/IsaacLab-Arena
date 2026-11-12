# Results

Arm A had the generated API reference (plain text) plus the source; arm B had the source only. Both arms used the
same repository snapshot (docstrings included). 20 questions, one fresh Sonnet agent per question and arm, so 40
runs. Tokens, tool calls and wall time are the figures the harness reported for each agent. Grading is against
`answers.md`.

## Summary

| Arm | Runs | Mean tokens | Median tokens | Mean tools | Mean secs | Correct | Partial | Wrong |
|-----|-----:|------------:|--------------:|-----------:|----------:|--------:|--------:|------:|
| A   |   20 |       52904 |         51870 |       3.70 |      10.5 |      17 |       2 |     1 |
| B   |   20 |       51733 |         51303 |       3.00 |       9.0 |      20 |       0 |     0 |

Paired over the 20 questions, arm A minus arm B: +1171 tokens, +0.7 tool calls and +1.6 s on average. Arm A was
cheaper than arm B on 7 questions for tokens, 3 for tool calls and 7 for time.

## Per question

| ID  | A tokens | B tokens | A tools | B tools | A secs | B secs | A grade | B grade |
|-----|---------:|---------:|--------:|--------:|-------:|-------:|---------|---------|
| Q01 |    53085 |    52011 |       3 |       3 |    9.8 |    8.8 | correct | correct |
| Q02 |    50577 |    51287 |       3 |       4 |    7.4 |   10.6 | wrong   | correct |
| Q03 |    53757 |    51046 |       7 |       3 |   17.0 |    8.2 | correct | correct |
| Q04 |    51625 |    51030 |       4 |       3 |   13.1 |    9.9 | correct | correct |
| Q05 |    52038 |    51106 |       5 |       3 |   13.8 |    9.5 | correct | correct |
| Q06 |    60887 |    52559 |       5 |       2 |   17.4 |   11.3 | correct | correct |
| Q07 |    55462 |    50850 |       4 |       3 |   10.8 |    7.5 | correct | correct |
| Q08 |    50770 |    50756 |       3 |       3 |    7.9 |    8.3 | correct | correct |
| Q09 |    51329 |    53413 |       4 |       3 |    9.8 |    8.2 | correct | correct |
| Q10 |    56121 |    53846 |       4 |       4 |   10.3 |   12.2 | correct | correct |
| Q11 |    51076 |    52473 |       3 |       3 |    8.8 |    8.7 | correct | correct |
| Q12 |    51141 |    52821 |       3 |       3 |    7.6 |    9.7 | correct | correct |
| Q13 |    51375 |    51590 |       5 |       3 |   13.3 |    8.6 | correct | correct |
| Q14 |    54150 |    51319 |       3 |       3 |    8.1 |    8.6 | correct | correct |
| Q15 |    53407 |    52772 |       2 |       3 |    7.4 |    8.9 | partial | correct |
| Q16 |    54102 |    51189 |       3 |       3 |    9.2 |    8.5 | correct | correct |
| Q17 |    51418 |    50728 |       3 |       3 |    8.7 |    8.5 | correct | correct |
| Q18 |    51701 |    51872 |       4 |       3 |   11.2 |    8.6 | correct | correct |
| Q19 |    50663 |    50871 |       2 |       3 |    5.8 |    8.4 | partial | correct |
| Q20 |    53400 |    51123 |       4 |       2 |   13.3 |    6.6 | correct | correct |

## What happened

- The docs did not make these lookups cheaper or faster. Every run costs about 50k tokens, which is almost all
  fixed agent overhead, and the source-only agents found the symbol with one or two greps and a single file read
  (typically 3 tool calls, 9 s).
- Arm A was less accurate. Q15 and Q19 stopped at the docs, which did not state a fact (the numeric threshold, the
  behaviour on a violated precondition), and answered partially without opening the source. Q02 followed the docs
  to a different but plausible enum (`Axis`); the question is ambiguous (see below).
- The generated reference was missing or thin on some facts that the source states plainly. Several arm A agents
  said so and fell back to the source: CLI flags (Q10), environment variable names (Q14), asserts and error
  behaviour (Q07, Q19), class base lists (Q06), and exact values in function bodies (Q04).
- Arm B never accessed the docs directory (no transcript mentions its path).

## Caveats

- The questions are mostly about a named symbol, which `grep` resolves immediately. Questions that require
  discovery without a known name (for example "where is X handled") or that span many files are where a reference
  could help more; this set does not measure that.
- Both arms read the improved docstrings in the source. The experiment measures the generated reference on top of
  well-documented source; it does not isolate the value of the docstrings themselves (that would need a third arm
  on the original source).
- Q02 is ambiguous: `Side` (positive/negative X/Y) and `Axis` (X/Y/Z) both fit "axis directions". Excluding it,
  arm A is 17/19 correct.
- Runs in a batch were executed in parallel, so wall times carry noise; token counts include the agent's fixed
  system prompt. One sample per cell.
- Q20-B flagged the `register_asset_factory` mapping as unverified; it happened to be right.
